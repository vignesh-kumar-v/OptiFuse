#!/usr/bin/env python3
"""Stream Waymo camera frames through the Triton-served TensorRT detector.

    tfrecord --(waymo_lib)--> raw JPEG --(letterbox)--> Triton (TensorRT engine,
    GPU) --(undo letterbox)--> boxes --(PIL)--> annotated frame --> HTML viewer

No TensorFlow / waymo-open-dataset anywhere in this path, and no live PyTorch
at serving time -- inference goes through the exported ONNX -> TensorRT ->
Triton path only, exactly as built in export.py / build_engine.sh.

Defaults to the segment held out of training entirely (see VAL_SEGMENT in
data/prepare_detection_dataset.py), so a bare run demonstrates the model on
genuinely unseen frames.

    python3 client.py --limit 60                       # FRONT camera, held-out segment
    python3 client.py --tfrecord ../../waymo_data/segment-X.tfrecord --cameras all --limit 30
"""
import argparse
import base64
import glob
import io
import json
import os
import sys
import time

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import waymo_lib as wl

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "viewer_template.html")

CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]
# Matches the fixed categorical palette used for the BEV lidar box overlays
# earlier in this project -- one color per class, consistent everywhere.
BOX_COLORS = {0: "#3987e5", 1: "#d95926", 2: "#c98500"}

DEFAULT_TFRECORD = os.path.join(
    HERE, "..", "..", "waymo_data",
    "segment-10689101165701914459_2072_300_2092_300_with_camera_labels.tfrecord")


def letterbox(im_bgr, size, pad_value=preprocessing.PAD_VALUE):
    """Thin shim over preprocessing.letterbox, kept so this module's existing
    (int size) -> (padded, ratio, (px, py)) signature still holds.

    The implementation lives in serving/preprocessing.py as the single source of
    truth -- it is shared with the segmentation and depth paths, and is the piece
    verified to match Ultralytics to 5.96e-08. Duplicating it here is exactly how
    train/serve skew creeps in.
    """
    padded, info = preprocessing.letterbox(im_bgr, (size, size), pad_value)
    return padded, info.ratio, (info.pad_x, info.pad_y)


def preprocess(im_bgr, size):
    tensor, info = preprocessing.preprocess(im_bgr, (size, size), dtype=np.float16)
    return tensor, info.ratio, (info.pad_x, info.pad_y)


def postprocess(raw, ratio, pad, orig_w, orig_h, conf_thres):
    """raw: [300,6] (x1,y1,x2,y2,conf,cls) in the padded (size,size) space ->
    confident detections in original-image pixel coordinates."""
    det = raw[raw[:, 4] > conf_thres].copy()
    if len(det) == 0:
        return det
    px, py = pad
    det[:, [0, 2]] = np.clip((det[:, [0, 2]] - px) / ratio, 0, orig_w)
    det[:, [1, 3]] = np.clip((det[:, [1, 3]] - py) / ratio, 0, orig_h)
    return det


def draw_boxes_svg(dets, disp_w, disp_h, orig_w, orig_h):
    """SVG overlay markup for one frame's detections, scaled to the *displayed*
    thumbnail size -- boxes stay crisp and don't need to be baked into pixels."""
    sx, sy = disp_w / orig_w, disp_h / orig_h
    parts = []
    for x1, y1, x2, y2, conf, cls in dets:
        cls = int(cls)
        color = BOX_COLORS.get(cls, "#8a97a3")
        x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        parts.append(
            f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{(x2 - x1):.1f}" height="{(y2 - y1):.1f}" '
            f'fill="none" stroke="{color}" stroke-width="2"/>'
            f'<text x="{x1 + 2:.1f}" y="{max(10, y1 - 4):.1f}" fill="{color}" '
            f'font-size="12" font-family="ui-monospace,monospace">{label}</text>'
        )
    return "".join(parts)


def build_viewer(frame_records, meta, out_path):
    with open(TEMPLATE) as fh:
        html = fh.read()
    html = html.replace("/*__META__*/ null", json.dumps(meta, separators=(",", ":")))
    html = html.replace("/*__FRAMES__*/ null", json.dumps(frame_records, separators=(",", ":")))
    with open(out_path, "w") as fh:
        fh.write(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tfrecord", default=DEFAULT_TFRECORD,
                    help="defaults to the segment held out of training")
    ap.add_argument("--cameras", default="FRONT",
                    help="comma-separated camera names, or 'all' [FRONT]")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame [1]")
    ap.add_argument("--limit", type=int, default=0, help="stop after N kept frames [all]")
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280, help="must match the served engine")
    ap.add_argument("--img-width", type=int, default=800, help="thumbnail width embedded in the viewer")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--triton-url", default="localhost:8001")
    ap.add_argument("--model-name", default="vehicle_detector")
    ap.add_argument("--out", default=None, help="output .html [default: alongside this script]")
    args = ap.parse_args()

    if not os.path.exists(args.tfrecord):
        raise SystemExit(f"tfrecord not found: {args.tfrecord}")

    want_cams = None if args.cameras == "all" else {c.strip().upper() for c in args.cameras.split(",")}

    client = grpcclient.InferenceServerClient(url=args.triton_url)
    if not client.is_server_ready():
        raise SystemExit(f"Triton server at {args.triton_url} is not ready -- "
                          f"is `serving/run_triton.sh -d` running?")
    if not client.is_model_ready(args.model_name):
        raise SystemExit(f"model '{args.model_name}' is not ready on the Triton server")
    print(f"Connected to Triton at {args.triton_url}, model '{args.model_name}' READY")

    out_path = args.out or os.path.join(HERE, os.path.basename(args.tfrecord)
                                        .replace(".tfrecord", "") + "_detections.html")

    frame_records = []
    latencies_ms = []
    class_counts = {n: 0 for n in CLASS_NAMES}
    cams_seen = set()
    t_start = time.time()
    n_kept = 0

    for i, payload in wl.records(args.tfrecord, stride=args.stride):
        if args.limit and n_kept >= args.limit:
            break
        F = wl.Frame(payload)

        for im in F.images():
            if want_cams and im["name"] not in want_cams:
                continue

            im_bgr = cv2.imdecode(np.frombuffer(im["jpeg"], np.uint8), cv2.IMREAD_COLOR)
            orig_h, orig_w = im_bgr.shape[:2]

            tensor, ratio, pad = preprocess(im_bgr, args.imgsz)

            inp = grpcclient.InferInput("images", tensor.shape, "FP16")
            inp.set_data_from_numpy(tensor)
            out = grpcclient.InferRequestedOutput("output0")

            t0 = time.time()
            result = client.infer(args.model_name, inputs=[inp], outputs=[out])
            latency_ms = (time.time() - t0) * 1000
            latencies_ms.append(latency_ms)

            raw = result.as_numpy("output0")[0]
            dets = postprocess(raw, ratio, pad, orig_w, orig_h, args.conf_thres)
            for cls in dets[:, 5].astype(int):
                class_counts[CLASS_NAMES[cls]] += 1

            disp_w = args.img_width
            disp_h = round(orig_h * disp_w / orig_w)
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            thumb = cv2.resize(im_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            jpeg_b64 = base64.b64encode(buf.tobytes()).decode()

            cams_seen.add(im["name"])
            frame_records.append({
                "frame": i, "cam": im["name"], "t": round(F.timestamp, 3),
                "w": disp_w, "h": disp_h,
                "jpg": jpeg_b64,
                "svg": draw_boxes_svg(dets, disp_w, disp_h, orig_w, orig_h),
                "n": len(dets), "ms": round(latency_ms, 2),
            })

        n_kept += 1
        if n_kept % 10 == 0:
            el = time.time() - t_start
            print(f"\r  frame {n_kept}{'/' + str(args.limit) if args.limit else ''}  "
                  f"{len(frame_records)} images  {el:.0f}s elapsed", end="")

    print()
    if not frame_records:
        raise SystemExit("no frames processed -- check --cameras / --tfrecord")

    lat = np.array(latencies_ms)
    meta = {
        "segment": os.path.basename(args.tfrecord),
        "cameras": sorted(cams_seen),
        "n_frames": n_kept,
        "n_images": len(frame_records),
        "conf_thres": args.conf_thres,
        "class_counts": class_counts,
        "latency_ms": {
            "mean": round(float(lat.mean()), 2),
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p90": round(float(np.percentile(lat, 90)), 2),
            "p99": round(float(np.percentile(lat, 99)), 2),
        },
    }
    build_viewer(frame_records, meta, out_path)

    print(f"\n{len(frame_records)} images across {n_kept} frames, "
          f"{sum(class_counts.values())} detections above conf>{args.conf_thres}:")
    for name, count in class_counts.items():
        print(f"  {name:10} {count}")
    print(f"\nClient-observed round-trip latency (preprocess + network + Triton + postprocess):")
    print(f"  mean={meta['latency_ms']['mean']}ms  p50={meta['latency_ms']['p50']}ms  "
          f"p90={meta['latency_ms']['p90']}ms  p99={meta['latency_ms']['p99']}ms")
    print(f"  (GPU-only TensorRT engine compute time, from trtexec, was ~2.5ms/frame -- "
          f"the gap here is the client<->server round trip)")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
