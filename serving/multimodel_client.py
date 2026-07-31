#!/usr/bin/env python3
"""Stream Waymo frames through all three Triton-served models concurrently.

    python3 multimodel_client.py --limit 40
    python3 multimodel_client.py --tfrecord ../../waymo_data/<night>.tfrecord --limit 60

Each frame's JPEG is decoded **once** (serving/preprocessing.py), fanned out to
the detector, segmentation and depth engines with all three requests in flight
at the same time, then joined on frame id and rendered into one synchronized
three-panel viewer.

Concurrency is the point of this file, so it is measured rather than asserted:
`--benchmark` times the same frames dispatched serially vs concurrently and
prints the speedup. Triton executes independent models in parallel on one GPU,
so the concurrent wall clock should sit near the slowest single model rather
than the sum of all three.
"""
import argparse
import base64
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing
from preprocessing import FramePreprocessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import waymo_lib as wl

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "multimodel_viewer.html")

DET_CLASSES = ["vehicle", "pedestrian", "cyclist"]
# Categorical slots 1/2/4 of the reference palette (dark steps), unchanged from
# Milestone 1 so a box means the same colour everywhere in this project.
DET_COLORS = {0: "#3987e5", 1: "#d95926", 2: "#c98500"}

SEG_CLASSES = ["background", "road", "marking", "sidewalk"]
# Slots 1/3/4 (blue / aqua / yellow). Validated all-pairs on the dark surface:
# worst CVD dE 8.4, normal-vision 19.8, all >=3:1 contrast. Orange is avoided
# here because the orange+yellow pair fails the all-pairs floors. Background is
# never painted, so it has no colour.
SEG_COLORS = {1: (0x39, 0x87, 0xE5), 2: (0xC9, 0x85, 0x00), 3: (0x19, 0x9E, 0x70)}

MODELS = {"detector": "vehicle_detector", "segmentation": "drivable_seg", "depth": "depth_estimator"}
OUTPUTS = {"detector": "output0", "segmentation": "class_map", "depth": "depth"}
INPUTS = {"detector": "images", "segmentation": "pixel_values", "depth": "pixel_values"}

DEPTH_MIN, DEPTH_MAX = 2.0, 60.0


def make_inputs(name, tensor):
    inp = grpcclient.InferInput(INPUTS[name], tensor.shape, "FP16")
    inp.set_data_from_numpy(tensor)
    return [inp], [grpcclient.InferRequestedOutput(OUTPUTS[name])]


def infer_serial(client, batch):
    out, per = {}, {}
    for name, (tensor, _info) in batch.items():
        ins, outs = make_inputs(name, tensor)
        t0 = time.perf_counter()
        r = client.infer(MODELS[name], inputs=ins, outputs=outs)
        per[name] = (time.perf_counter() - t0) * 1000
        out[name] = r.as_numpy(OUTPUTS[name])
    return out, per


def infer_concurrent(client, batch):
    """Fire every model's request before waiting on any of them."""
    results, errors, per = {}, {}, {}
    done = threading.Event()
    lock = threading.Lock()
    started = {}
    remaining = [len(batch)]

    def make_cb(name):
        def cb(result, error):
            with lock:
                per[name] = (time.perf_counter() - started[name]) * 1000
                if error is not None:
                    errors[name] = error
                else:
                    results[name] = result.as_numpy(OUTPUTS[name])
                remaining[0] -= 1
                if remaining[0] == 0:
                    done.set()
        return cb

    for name, (tensor, _info) in batch.items():
        ins, outs = make_inputs(name, tensor)
        started[name] = time.perf_counter()
        client.async_infer(MODELS[name], inputs=ins, callback=make_cb(name), outputs=outs)

    if not done.wait(timeout=60):
        raise TimeoutError(f"timed out; got {sorted(results)} of {sorted(batch)}")
    if errors:
        raise RuntimeError(f"inference errors: {errors}")
    return results, per


def decode_detections(raw, info, conf_thres):
    det = raw[0]
    det = det[det[:, 4] > conf_thres]
    if len(det) == 0:
        return det
    det[:, :4] = info.to_original(det[:, :4])
    return det


def boxes_svg(dets, disp_w, disp_h, info):
    sx, sy = disp_w / info.orig_w, disp_h / info.orig_h
    parts = []
    for x1, y1, x2, y2, conf, cls in dets:
        c = DET_COLORS.get(int(cls), "#8a97a3")
        x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        parts.append(
            f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2-x1:.1f}" height="{y2-y1:.1f}" '
            f'fill="none" stroke="{c}" stroke-width="2"/>'
            f'<text x="{x1+2:.1f}" y="{max(10,y1-4):.1f}" fill="{c}" font-size="12" '
            f'font-family="ui-monospace,monospace">{DET_CLASSES[int(cls)]} {conf:.2f}</text>')
    return "".join(parts)


def seg_overlay(class_map, im_bgr, info, disp_w, disp_h, alpha=0.45):
    """Un-letterbox the class map, tint road/marking/sidewalk over the frame."""
    h, w = class_map.shape
    inner = class_map[info.pad_y:h - info.pad_y or None, info.pad_x:w - info.pad_x or None]
    inner = cv2.resize(inner, (info.orig_w, info.orig_h), interpolation=cv2.INTER_NEAREST)
    base = cv2.resize(im_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    small = cv2.resize(inner, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

    tint = np.zeros_like(base)
    any_px = np.zeros(small.shape, bool)
    for cls, (r, g, b) in SEG_COLORS.items():
        m = small == cls
        if m.any():
            tint[m] = (b, g, r)      # BGR for cv2
            any_px |= m
    out = base.copy()
    out[any_px] = (base[any_px] * (1 - alpha) + tint[any_px] * alpha).astype(np.uint8)
    frac = {SEG_CLASSES[c]: float((small == c).mean()) for c in SEG_COLORS}
    return out, frac


def depth_panel(depth, im_bgr, info, disp_w, disp_h):
    """Un-letterbox depth and colorize with a monotonic-lightness ramp.

    A multi-hue ramp is used rather than a single hue because depth is the
    documented "semantic heat" exception -- and it ships with a metre scale
    legend in the viewer, which that exception requires. MAGMA is chosen over
    JET specifically because its lightness increases monotonically; JET's does
    not, which invents banding that is not in the data.
    """
    h, w = depth.shape
    inner = depth[info.pad_y:h - info.pad_y or None, info.pad_x:w - info.pad_x or None]
    # The engine emits FP16; cv2.resize has no float16 kernel, so widen first.
    inner = cv2.resize(inner.astype(np.float32), (disp_w, disp_h),
                       interpolation=cv2.INTER_LINEAR)
    norm = np.clip((inner - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN), 0, 1)
    # near = bright: invert so close surfaces read as "hot"
    u8 = ((1.0 - norm) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA), float(inner.min()), float(inner.max())


def jpg_b64(im_bgr, quality=82):
    ok, buf = cv2.imencode(".jpg", im_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tfrecord", default=os.path.join(
        HERE, "..", "..", "waymo_data",
        "segment-10689101165701914459_2072_300_2092_300_with_camera_labels.tfrecord"))
    ap.add_argument("--camera", default="FRONT")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--conf-thres", type=float, default=0.20,
                    help="0.20 is where F1 peaked on the val sweep [0.20]")
    ap.add_argument("--img-width", type=int, default=760)
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--benchmark", type=int, default=10,
                    help="frames to time serially vs concurrently (0 to skip)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.tfrecord):
        raise SystemExit(f"tfrecord not found: {args.tfrecord}")

    client = grpcclient.InferenceServerClient(url=args.url)
    if not client.is_server_ready():
        raise SystemExit(f"Triton not ready at {args.url} -- start serving/run_triton.sh -d")
    missing = [n for n, m in MODELS.items() if not client.is_model_ready(m)]
    if missing:
        raise SystemExit(f"model(s) not READY on Triton: "
                         f"{[MODELS[m] for m in missing]}. Build their engines first.")
    print(f"Triton {args.url}: all 3 models READY -> {', '.join(MODELS.values())}")

    pre = FramePreprocessor(dtype=np.float16)
    out_path = args.out or os.path.join(
        HERE, os.path.basename(args.tfrecord).replace(".tfrecord", "") + "_multimodel.html")

    frames, lat_concurrent = [], []
    per_model_ms = {k: [] for k in MODELS}
    bench = None
    t_start = time.time()

    for i, payload in wl.records(args.tfrecord, stride=args.stride):
        if len(frames) >= args.limit:
            break
        F = wl.Frame(payload)
        for im in F.images():
            if im["name"] != args.camera:
                continue
            im_bgr = preprocessing.decode(im["jpeg"])
            batch = pre.from_image(im_bgr)

            if bench is None and args.benchmark:
                ser_t, con_t = [], []
                for _ in range(args.benchmark):
                    t0 = time.perf_counter(); infer_serial(client, batch)
                    ser_t.append((time.perf_counter() - t0) * 1000)
                    t0 = time.perf_counter(); infer_concurrent(client, batch)
                    con_t.append((time.perf_counter() - t0) * 1000)
                bench = {"serial_ms": float(np.median(ser_t)),
                         "concurrent_ms": float(np.median(con_t)),
                         "n": args.benchmark}
                bench["speedup"] = bench["serial_ms"] / max(bench["concurrent_ms"], 1e-6)
                print(f"\nbenchmark over {args.benchmark} frames (median):")
                print(f"  serial     {bench['serial_ms']:7.2f} ms")
                print(f"  concurrent {bench['concurrent_ms']:7.2f} ms"
                      f"   -> {bench['speedup']:.2f}x\n")

            t0 = time.perf_counter()
            res, per = infer_concurrent(client, batch)
            lat_concurrent.append((time.perf_counter() - t0) * 1000)
            for k, v in per.items():
                per_model_ms[k].append(v)

            det_info = batch["detector"][1]
            dets = decode_detections(res["detector"], det_info, args.conf_thres)

            disp_w = args.img_width
            disp_h = round(det_info.orig_h * disp_w / det_info.orig_w)

            rgb_panel = cv2.resize(im_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            seg_img, seg_frac = seg_overlay(res["segmentation"][0], im_bgr,
                                            batch["segmentation"][1], disp_w, disp_h)
            dep_img, dmin, dmax = depth_panel(res["depth"][0], im_bgr,
                                              batch["depth"][1], disp_w, disp_h)

            frames.append({
                "frame": i, "t": round(F.timestamp, 3), "w": disp_w, "h": disp_h,
                "rgb": jpg_b64(rgb_panel), "seg": jpg_b64(seg_img), "depth": jpg_b64(dep_img),
                "svg": boxes_svg(dets, disp_w, disp_h, det_info),
                "n_det": int(len(dets)),
                "seg_frac": {k: round(v, 4) for k, v in seg_frac.items()},
                "depth_range": [round(dmin, 1), round(dmax, 1)],
                "ms": round(lat_concurrent[-1], 2),
            })
            if len(frames) % 10 == 0:
                print(f"\r  {len(frames)}/{args.limit} frames  "
                      f"{time.time()-t_start:.0f}s", end="")
    print()

    if not frames:
        raise SystemExit("no frames processed -- check --camera / --tfrecord")

    lat = np.array(lat_concurrent)
    meta = {
        "segment": os.path.basename(args.tfrecord),
        "camera": args.camera,
        "n_frames": len(frames),
        "conf_thres": args.conf_thres,
        "det_classes": DET_CLASSES, "det_colors": DET_COLORS,
        "seg_classes": SEG_CLASSES,
        "seg_colors": {str(k): "#%02x%02x%02x" % v for k, v in SEG_COLORS.items()},
        "depth_min": DEPTH_MIN, "depth_max": DEPTH_MAX,
        "benchmark": bench,
        "latency_ms": {"mean": round(float(lat.mean()), 2),
                       "p50": round(float(np.percentile(lat, 50)), 2),
                       "p90": round(float(np.percentile(lat, 90)), 2)},
        "per_model_ms": {k: round(float(np.mean(v)), 2) for k, v in per_model_ms.items() if v},
    }

    with open(TEMPLATE) as fh:
        html = fh.read()
    html = (html.replace("/*__META__*/ null", json.dumps(meta, separators=(",", ":")))
                .replace("/*__FRAMES__*/ null", json.dumps(frames, separators=(",", ":"))))
    with open(out_path, "w") as fh:
        fh.write(html)

    print(f"\n{len(frames)} frames, all three models per frame")
    print(f"  per-model mean: " + "  ".join(f"{k}={v}ms" for k, v in meta["per_model_ms"].items()))
    print(f"  concurrent round trip: mean={meta['latency_ms']['mean']}ms  "
          f"p50={meta['latency_ms']['p50']}ms  p90={meta['latency_ms']['p90']}ms")
    if bench:
        print(f"  serial {bench['serial_ms']:.1f}ms vs concurrent "
              f"{bench['concurrent_ms']:.1f}ms = {bench['speedup']:.2f}x")
    print(f"\nWrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
