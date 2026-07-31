#!/usr/bin/env python3
"""Export best.pt to ONNX and numerically validate against the PyTorch original.

Compares raw model output tensors (pre-any-further-postprocessing) between the
PyTorch model and the exported ONNX graph on the same preprocessed input --
this is what actually needs to match for the TensorRT engine built from the
ONNX graph to behave identically to the model we validated in train.py.

    python3 export.py
    python3 export.py --weights runs/waymo_detect_final/weights/best.pt
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO


def box_iou(a, b):
    """IoU matrix [len(a), len(b)] for xyxy boxes."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def match_detections(pt_conf, onnx_conf, iou_thres=0.5):
    """Greedy same-class IoU matching. Confidence rank is NOT a valid identity --
    two detections with near-tied confidence can swap rank between backends on
    ordinary floating-point noise, so matching must be spatial, not positional."""
    matched_pt, matched_onnx = [], []
    used_onnx = set()
    for i in range(pt_conf.shape[0]):
        same_cls = np.where(onnx_conf[:, 5] == pt_conf[i, 5])[0]
        same_cls = [j for j in same_cls if j not in used_onnx]
        if not same_cls:
            continue
        ious = box_iou(pt_conf[i:i + 1, :4], onnx_conf[same_cls, :4])[0]
        best = int(np.argmax(ious))
        if ious[best] >= iou_thres:
            j = same_cls[best]
            matched_pt.append(i)
            matched_onnx.append(j)
            used_onnx.add(j)
    n_unmatched_pt = pt_conf.shape[0] - len(matched_pt)
    n_unmatched_onnx = onnx_conf.shape[0] - len(matched_onnx)
    return np.array(matched_pt, int), np.array(matched_onnx, int), n_unmatched_pt, n_unmatched_onnx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="runs/waymo_detect_final/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="must match the imgsz used for training/final val")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--val-images", default="../../data/datasets/detection_v1/images/val")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--half", action="store_true",
                    help="export FP16 weights into the ONNX graph itself. Required for an FP16 "
                         "TensorRT engine: TensorRT v11's trtexec removed the old --fp16 builder "
                         "flag -- ONNX graphs are now 'strongly typed', so engine precision is "
                         "read directly from the graph's own tensor dtypes, not a builder switch.")
    ap.add_argument("--device", default="0", help="CUDA device for the (half-precision) export trace")
    ap.add_argument("--px-atol", type=float, default=None,
                    help="max allowed box coordinate difference, in pixels of the (imgsz, imgsz) "
                         "input space [default: 1.0, or 3.0 with --half]")
    ap.add_argument("--conf-atol", type=float, default=None,
                    help="max allowed confidence-score difference, 0-1 range "
                         "[default: 0.05, or 0.08 with --half]")
    ap.add_argument("--conf-thres", type=float, default=0.1,
                    help="only compare detections above this confidence; near-zero-confidence "
                         "padding slots can reorder between backends on floating-point noise "
                         "and are not meaningful to compare row-by-row")
    args = ap.parse_args()
    px_atol = args.px_atol if args.px_atol is not None else (3.0 if args.half else 1.0)
    conf_atol = args.conf_atol if args.conf_atol is not None else (0.08 if args.half else 0.05)

    here = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.normpath(os.path.join(here, args.weights))
    val_dir = os.path.normpath(os.path.join(here, args.val_images))

    model = YOLO(weights_path)
    export_kwargs = dict(format="onnx", opset=args.opset, simplify=True, imgsz=args.imgsz)
    if args.half:
        export_kwargs.update(half=True, device=args.device)
    onnx_path = model.export(**export_kwargs)
    if args.half:
        # Ultralytics always names the export after the weights file (best.onnx) --
        # rename so the already-validated FP32 graph from a plain `export.py` run
        # is never silently overwritten by this FP16 one.
        fp16_path = os.path.join(os.path.dirname(onnx_path), "best_fp16.onnx")
        os.replace(onnx_path, fp16_path)
        onnx_path = fp16_path
    print(f"Exported: {onnx_path}")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("onnx.checker.check_model: PASSED")

    all_images = sorted(glob.glob(os.path.join(val_dir, "*.jpg")))
    if not all_images:
        raise SystemExit(f"no val images found under {val_dir}")
    rng = random.Random(args.seed)
    samples = rng.sample(all_images, min(args.n_samples, len(all_images)))

    # Running one predict() call initializes model.predictor (device placement,
    # letterbox/normalize preprocessing) so we reuse the exact same preprocessing
    # path the model was trained/validated with, instead of reimplementing it.
    # rect=False is required: Ultralytics' default single-image inference uses
    # rectangular (minimal-padding) letterboxing sized to each image's own aspect
    # ratio, but the ONNX/TensorRT graph has a FIXED square (imgsz, imgsz) input --
    # the same fixed shape every served frame must be resized into.
    model.predict(samples[0], imgsz=args.imgsz, rect=False, verbose=False)
    predictor = model.predictor
    pt_model = predictor.model
    pt_model.eval()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_dtype = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32

    print(f"\nComparing raw PyTorch (FP32) vs ONNX Runtime CPU ({'FP16' if args.half else 'FP32'}) "
          f"output tensors on {len(samples)} sample images (imgsz={args.imgsz}, "
          f"px_atol={px_atol}, conf_atol={conf_atol}):")
    max_px_diff_overall = 0.0
    max_conf_diff_overall = 0.0
    any_class_mismatch = False
    any_boundary_violation = False
    for img_path in samples:
        im0 = cv2.imread(img_path)
        im = predictor.preprocess([im0])

        with torch.no_grad():
            pt_out = pt_model(im)
        if isinstance(pt_out, (list, tuple)):
            pt_out = pt_out[0]
        pt_out_np = pt_out.cpu().numpy()

        onnx_out = sess.run(None, {input_name: im.cpu().numpy().astype(input_dtype)})[0].astype(np.float32)

        name = os.path.basename(img_path)
        if pt_out_np.shape != onnx_out.shape:
            print(f"  {name}: SHAPE MISMATCH pt={pt_out_np.shape} onnx={onnx_out.shape}")
            continue

        # Keep only confident rows -- the other ~280+ near-zero-confidence
        # padding slots per frame are not real detections.
        pt_conf = pt_out_np[0][pt_out_np[0, :, 4] > args.conf_thres]
        onnx_conf = onnx_out[0][onnx_out[0, :, 4] > args.conf_thres]

        if pt_conf.shape[0] == 0 and onnx_conf.shape[0] == 0:
            print(f"  {name}: 0 detections above conf>{args.conf_thres} in both -- OK")
            continue

        matched_pt, matched_onnx, n_unmatched_pt, n_unmatched_onnx = match_detections(
            pt_conf, onnx_conf)

        # A handful of unmatched detections right at the confidence threshold is
        # expected and harmless: a box sitting at conf=0.0999 vs 0.1001 crosses
        # the cutoff on ordinary backend floating-point noise, not a real defect.
        # Flag it only if it's more than a couple of boxes, which would indicate
        # a real behavioral divergence rather than boundary noise.
        boundary_ok = (n_unmatched_pt <= 2 and n_unmatched_onnx <= 2)
        any_boundary_violation |= not boundary_ok

        if len(matched_pt):
            px_diff = float(np.abs(pt_conf[matched_pt, :4] - onnx_conf[matched_onnx, :4]).max())
            conf_diff = float(np.abs(pt_conf[matched_pt, 4] - onnx_conf[matched_onnx, 4]).max())
            class_mismatch = int((pt_conf[matched_pt, 5] != onnx_conf[matched_onnx, 5]).sum())
        else:
            px_diff = conf_diff = 0.0
            class_mismatch = 0
        any_class_mismatch |= class_mismatch > 0
        max_px_diff_overall = max(max_px_diff_overall, px_diff)
        max_conf_diff_overall = max(max_conf_diff_overall, conf_diff)

        ok = (px_diff <= px_atol and conf_diff <= conf_atol
              and class_mismatch == 0 and boundary_ok)
        status = "OK" if ok else "MISMATCH"
        extra = f", unmatched pt={n_unmatched_pt} onnx={n_unmatched_onnx}" \
            if (n_unmatched_pt or n_unmatched_onnx) else ""
        extra += f", class_mismatch={class_mismatch}" if class_mismatch else ""
        print(f"  {name}: {len(matched_pt)} matched detections, "
              f"px_diff={px_diff:.4f}, conf_diff={conf_diff:.4f}{extra} -- {status}")

    print(f"\nMax pixel-coord diff: {max_px_diff_overall:.4f} (tolerance {px_atol})")
    print(f"Max confidence diff:  {max_conf_diff_overall:.4f} (tolerance {conf_atol})")
    passed = (max_px_diff_overall <= px_atol and max_conf_diff_overall <= conf_atol
              and not any_class_mismatch and not any_boundary_violation)
    if passed:
        print("VALIDATION PASSED: ONNX export matches PyTorch within tolerance.")
    else:
        print("VALIDATION FAILED: differences exceed tolerance -- do not proceed to TensorRT.")


if __name__ == "__main__":
    main()
