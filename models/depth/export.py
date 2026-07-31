#!/usr/bin/env python3
"""Export the fine-tuned depth model to ONNX and validate against PyTorch.

    python3 export.py
    python3 export.py --half        # FP16 graph, for a TensorRT FP16 engine

Like the segmentation export, ImageNet normalization is baked into the graph so
the served input is plain 0-1 RGB and a client cannot get the constants wrong.
Output is metric depth in metres at input resolution.

Tolerance is expressed in metres over the range we actually care about
(MIN_DEPTH..MAX_DEPTH). A relative tolerance would be misleading here: a 0.1m
error is negligible at 60m and significant at 2m, so absolute metres plus a
relative summary are both reported.
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
from train import IMAGENET_MEAN, IMAGENET_STD, MAX_DEPTH, MIN_DEPTH


class DepthForExport(torch.nn.Module):
    """ImageNet-normalize + Depth Anything -> metric depth at input resolution."""

    def __init__(self, model, out_hw):
        super().__init__()
        self.model = model
        self.out_hw = out_hw
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, pixel_values):
        x = (pixel_values - self.mean) / self.std
        d = self.model(pixel_values=x).predicted_depth
        if d.dim() == 3:
            d = d.unsqueeze(1)
        if d.shape[-2:] != self.out_hw:
            d = F.interpolate(d, size=self.out_hw, mode="bilinear", align_corners=False)
        return d.squeeze(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="runs/depth_v1/best")
    ap.add_argument("--cache", default="../../cache/local")
    ap.add_argument("--width", type=int, default=966, help="multiple of 14")
    ap.add_argument("--height", type=int, default=644, help="multiple of 14")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--atol", type=float, default=None,
                    help="max allowed depth difference in metres [0.05, or 0.5 with --half]")
    args = ap.parse_args()

    atol = args.atol if args.atol is not None else (0.5 if args.half else 0.05)
    for name, v in (("width", args.width), ("height", args.height)):
        if v % 14:
            raise SystemExit(f"--{name}={v} must be a multiple of 14 (DINOv2 patch size)")

    here = os.path.dirname(os.path.abspath(__file__))
    weights = os.path.normpath(os.path.join(here, args.weights))
    cache = os.path.normpath(os.path.join(here, args.cache))
    if not os.path.isdir(weights):
        raise SystemExit(f"no fine-tuned weights at {weights} -- run train.py first")

    from transformers import AutoModelForDepthEstimation
    base = AutoModelForDepthEstimation.from_pretrained(weights)
    model = DepthForExport(base, (args.height, args.width)).eval()
    dtype = torch.float16 if args.half else torch.float32
    if args.half:
        model = model.half()

    onnx_path = os.path.join(weights, "model_fp16.onnx" if args.half else "model.onnx")
    dummy = torch.randn(1, 3, args.height, args.width, dtype=dtype)
    torch.onnx.export(model, (dummy,), onnx_path,
                      input_names=["pixel_values"], output_names=["depth"],
                      opset_version=args.opset, dynamo=False)
    print(f"Exported: {onnx_path}  ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    onnx.checker.check_model(onnx.load(onnx_path))
    print("onnx.checker.check_model: PASSED")

    shards = sorted(glob.glob(os.path.join(cache, "*.tar")))
    ds = WaymoImages(cache, task="depth", size=(args.width, args.height), shards=shards)
    samples = []
    for s in ds:
        samples.append(s["image"])
        if len(samples) >= args.n_samples:
            break
    if not samples:
        raise SystemExit("no depth samples found in cache")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    print(f"\nComparing PyTorch vs ONNX Runtime (CPU) depth on {len(samples)} samples "
          f"({'FP16' if args.half else 'FP32'}, atol={atol}m):")
    worst = 0.0
    for i, img in enumerate(samples):
        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).unsqueeze(0).to(dtype) / 255.0
        with torch.no_grad():
            pt = model(x).float().numpy()
        on = sess.run(None, {iname: x.numpy()})[0].astype(np.float32)

        valid = (pt > MIN_DEPTH) & (pt < MAX_DEPTH)
        if not valid.any():
            print(f"  sample {i}: no in-range depth predicted -- skipped")
            continue
        d = np.abs(pt[valid] - on[valid])
        rel = (d / np.clip(pt[valid], MIN_DEPTH, None)).mean()
        mx = float(d.max())
        worst = max(worst, mx)
        print(f"  sample {i}: max_diff={mx:.4f}m  mean_rel={rel:.6f}  "
              f"range={pt[valid].min():.1f}-{pt[valid].max():.1f}m  "
              f"{'OK' if mx <= atol else 'MISMATCH'}")

    print(f"\nWorst absolute difference: {worst:.4f}m (tolerance {atol}m)")
    if worst <= atol:
        print("VALIDATION PASSED: ONNX depth matches PyTorch.")
    else:
        print("VALIDATION FAILED: do not proceed to TensorRT.")
        sys.exit(1)


if __name__ == "__main__":
    main()
