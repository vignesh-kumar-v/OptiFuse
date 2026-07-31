#!/usr/bin/env python3
"""Export the fine-tuned SegFormer to ONNX and validate against PyTorch.

    python3 export.py
    python3 export.py --half        # FP16 graph, for a TensorRT FP16 engine

The exported graph bundles upsample + argmax, so it emits a finished uint8
class map at full input resolution rather than raw logits. That is both what a
serving client actually wants and ~16x less data on the wire: a (1,640,960)
uint8 mask is 614KB, where (1,4,160,240) float32 logits would be 9.8MB per
frame -- meaningful over gRPC at streaming rates.

Because the output is discrete class indices, equivalence is measured as
pixel agreement rather than a float tolerance. Boundary pixels can legitimately
flip between backends when two classes are near-tied, so the bar is
>=99.9% agreement, not bit-identity.
"""
import argparse
import glob
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from waymo_dataset import WaymoImages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, LUT


class SegForExport(torch.nn.Module):
    """ImageNet-normalize + SegFormer + upsample + argmax -> uint8 class map.

    Normalization is baked into the graph deliberately: the served input is then
    plain 0-1 RGB, so a serving client cannot get the mean/std wrong. Those
    constants live in exactly one place, which is the same reasoning behind
    keeping one verified letterbox implementation for the detector.
    """

    def __init__(self, model, out_hw):
        super().__init__()
        self.model = model
        self.out_hw = out_hw
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, pixel_values):
        x = (pixel_values - self.mean) / self.std
        logits = self.model(pixel_values=x).logits
        logits = F.interpolate(logits, size=self.out_hw, mode="bilinear", align_corners=False)
        return logits.argmax(1).to(torch.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="runs/seg_v1/best")
    ap.add_argument("--cache", default="../../cache/local")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--min-agreement", type=float, default=0.999)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    weights = os.path.normpath(os.path.join(here, args.weights))
    cache = os.path.normpath(os.path.join(here, args.cache))
    if not os.path.isdir(weights):
        raise SystemExit(f"no fine-tuned weights at {weights} -- run train.py first")

    from transformers import SegformerForSemanticSegmentation
    base = SegformerForSemanticSegmentation.from_pretrained(weights)
    model = SegForExport(base, (args.height, args.width)).eval()

    dtype = torch.float16 if args.half else torch.float32
    if args.half:
        model = model.half()

    onnx_path = os.path.join(weights, "model_fp16.onnx" if args.half else "model.onnx")
    dummy = torch.randn(1, 3, args.height, args.width, dtype=dtype)
    torch.onnx.export(model, (dummy,), onnx_path,
                      input_names=["pixel_values"], output_names=["class_map"],
                      opset_version=args.opset, dynamo=False)
    print(f"Exported: {onnx_path}  ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    onnx.checker.check_model(onnx.load(onnx_path))
    print("onnx.checker.check_model: PASSED")

    shards = sorted(glob.glob(os.path.join(cache, "*.tar")))
    ds = WaymoImages(cache, task="segmentation", size=(args.width, args.height), shards=shards)
    samples = []
    for s in ds:
        samples.append(s["image"])
        if len(samples) >= args.n_samples:
            break
    if not samples:
        raise SystemExit("no segmentation samples found in cache")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    print(f"\nComparing PyTorch vs ONNX Runtime (CPU) class maps on {len(samples)} samples "
          f"({'FP16' if args.half else 'FP32'}, min agreement {args.min_agreement:.3%}):")
    worst = 1.0
    for i, img in enumerate(samples):
        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).unsqueeze(0).to(dtype) / 255.0
        with torch.no_grad():
            pt = model(x).numpy()
        on = sess.run(None, {iname: x.numpy()})[0]
        agree = float((pt == on).mean())
        worst = min(worst, agree)
        diff_classes = sorted(set(np.unique(pt[pt != on]).tolist())) if agree < 1.0 else []
        note = f"  (mismatch in classes {diff_classes})" if diff_classes else ""
        print(f"  sample {i}: agreement={agree:.5%}"
              f"  {'OK' if agree >= args.min_agreement else 'MISMATCH'}{note}")

    print(f"\nWorst agreement: {worst:.5%} (threshold {args.min_agreement:.3%})")
    if worst >= args.min_agreement:
        print("VALIDATION PASSED: ONNX class maps match PyTorch.")
    else:
        print("VALIDATION FAILED: do not proceed to TensorRT.")
        sys.exit(1)


if __name__ == "__main__":
    main()
