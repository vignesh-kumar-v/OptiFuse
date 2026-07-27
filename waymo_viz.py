#!/usr/bin/env python3
"""Export one Waymo .tfrecord segment to a single self-contained HTML viewer.

    python3 waymo_viz.py waymo_perception/training/segment-....tfrecord

Output opens straight off the filesystem: no server, no CDN, no network. All
geometry and imagery are quantised into one binary blob, base64'd into the page.

Size is the whole design constraint: 198 frames x 183k points is 36M points, so
points are subsampled and stored as int16 centimetres, and the JPEGs are
downscaled. Tune with --points / --img-width / --stride and watch the printed
budget.
"""
import argparse
import base64
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image

from waymo_lib import CAM, OBJ, Frame, count_records, records

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "viewer.html")

# Point record: int16 x,y,z (cm) | u8 intensity | u8 elongation | u8 r,g,b | pad.
# 12 bytes keeps every attribute 4-byte aligned for WebGL.
PT_DTYPE = np.dtype([("xyz", "<i2", 3), ("ie", "u1", 2), ("rgb", "u1", 3), ("_pad", "u1")])


def quantise(pts, imgs_rgb, cams, n_target):
    """Subsample + pack one frame's points into the 12-byte record."""
    n = len(pts["xyz"])
    if n == 0:
        return np.zeros(0, PT_DTYPE), 0
    # Even stride, not random: preserves the visible scan-ring structure.
    idx = np.arange(n) if n <= n_target else np.linspace(0, n - 1, n_target).astype(np.int64)

    rec = np.zeros(len(idx), PT_DTYPE)
    xyz = np.clip(pts["xyz"][idx] * 100.0, -32767, 32767)
    rec["xyz"] = xyz.astype(np.int16)
    # Intensity is unbounded (seen up to 29.5); sqrt keeps the low end visible.
    rec["ie"][:, 0] = np.clip(np.sqrt(np.clip(pts["intensity"][idx], 0, None)) * 255, 0, 255)
    rec["ie"][:, 1] = np.clip(pts["elongation"][idx] * 170, 0, 255)

    # Colour from the projection channels the dataset already computed.
    cam_id, uv = pts["cam"][idx], pts["uv"][idx]
    for cid, name in CAM.items():
        if cid == 0 or name not in imgs_rgb:
            continue
        sel = cam_id == cid
        if not sel.any():
            continue
        arr = imgs_rgb[name]
        h, w = arr.shape[:2]
        u = np.clip(uv[sel, 0], 0, w - 1)
        v = np.clip(uv[sel, 1], 0, h - 1)
        rec["rgb"][sel] = arr[v, u]
    return rec, n


def stamp(meta_json, payload_b64):
    with open(TEMPLATE) as fh:
        html = fh.read()
    return (html.replace("/*__META__*/ null", meta_json)
                .replace("/*__PAYLOAD__*/", payload_b64))


def restamp(path):
    """Rebuild an exported viewer against the current template, reusing its data.

    Viewer changes (styling, controls, bug fixes) otherwise mean re-decoding the
    whole .tfrecord, which is the slow part. This is seconds instead.
    """
    html = open(path).read()
    m0 = html.index("const META = ") + len("const META = ")
    m1 = html.index('\nconst PAYLOAD_B64 = "')
    p0 = m1 + len('\nconst PAYLOAD_B64 = "')
    p1 = html.index('";', p0)
    meta_json = html[m0:m1].rstrip().rstrip(";")
    if meta_json == "/*__META__*/ null":
        raise SystemExit(f"{path} is the empty template, nothing to restamp")
    with open(path, "w") as fh:
        fh.write(stamp(meta_json, html[p0:p1]))
    print(f"restamped {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tfrecord", nargs="+",
                    help="input .tfrecord, or exported .html file(s) with --restamp")
    ap.add_argument("--restamp", action="store_true",
                    help="re-apply the current viewer.html to already-exported viewers")
    ap.add_argument("-o", "--out", help="output .html (default: alongside input)")
    ap.add_argument("--points", type=int, default=9000, help="points kept per frame [9000]")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame [1]")
    ap.add_argument("--frames", type=int, default=0, help="stop after N kept frames [all]")
    ap.add_argument("--img-width", type=int, default=480, help="camera image width, 0=skip [480]")
    ap.add_argument("--quality", type=int, default=55, help="JPEG quality [55]")
    ap.add_argument("--cams", default="all",
                    help="comma-separated camera names, or 'all' / 'front' [all]")
    ap.add_argument("--returns", default="1,2", help="lidar returns to include [1,2]")
    args = ap.parse_args()

    if args.restamp:
        for path in args.tfrecord:
            restamp(path)
        return
    if len(args.tfrecord) != 1:
        raise SystemExit("pass exactly one .tfrecord (multiple inputs are --restamp only)")
    args.tfrecord = args.tfrecord[0]

    want_cams = None
    if args.cams == "front":
        want_cams = {"FRONT"}
    elif args.cams != "all":
        want_cams = {c.strip().upper() for c in args.cams.split(",")}
    returns = tuple(1 + int(r) for r in args.returns.split(","))   # 1->field 2, 2->field 3

    total = count_records(args.tfrecord)
    n_keep = len(range(0, total, args.stride))
    if args.frames:
        n_keep = min(n_keep, args.frames)
    print(f"{os.path.basename(args.tfrecord)}: {total} frames, exporting {n_keep} "
          f"(stride {args.stride}), {args.points} pts/frame")

    blob = bytearray()
    frames_meta = []
    meta = {}
    types_seen = set()
    t_start = time.time()

    for k, (i, payload) in enumerate(records(args.tfrecord, stride=args.stride)):
        if args.frames and k >= args.frames:
            break
        F = Frame(payload)

        if not meta:
            cc = F.camera_calibrations()
            meta = {
                "name": F.context_name,
                "stats": F.stats,
                "t0": F.timestamp,
                "cams": {n: {"w": c["width"], "h": c["height"]}
                         for n, c in cc.items() if not want_cams or n in want_cams},
            }

        # full-res decode first: needed to sample per-point colour
        imgs_rgb, img_recs = {}, {}
        for im in F.images():
            if want_cams and im["name"] not in want_cams:
                continue
            full = Image.open(io.BytesIO(im["jpeg"])).convert("RGB")
            imgs_rgb[im["name"]] = np.asarray(full)
            if args.img_width:
                w = args.img_width
                h = round(full.height * w / full.width)
                buf = io.BytesIO()
                full.resize((w, h), Image.BILINEAR).save(buf, "JPEG", quality=args.quality)
                data = buf.getvalue()
                img_recs[im["name"]] = [len(blob), len(data), w, h]
                blob += data

        pts = F.points(returns=returns)
        rec, n_total = quantise(pts, imgs_rgb, meta["cams"], args.points)
        p_off = len(blob)
        blob += rec.tobytes()

        boxes, b2, b2p = [], {}, {}
        for lb in F.laser_labels():
            types_seen.add(lb["type"])
            boxes.append({"c": [round(v, 2) for v in lb["center"]],
                          "s": [round(v, 2) for v in lb["size"]],
                          "h": round(lb["heading"], 3),
                          "t": lb["type"], "n": lb["num_points"],
                          "i": lb["id"], "v": round(lb["speed"], 2)})
        for dst, field in ((b2, 8), (b2p, 9)):
            for name, bs in F.camera_labels(field).items():
                if want_cams and name not in want_cams:
                    continue
                dst[name] = [[round(b["cx"], 1), round(b["cy"], 1),
                              round(b["w"], 1), round(b["h"], 1), b["type"]] for b in bs]

        frames_meta.append({"t": round(F.timestamp, 6), "po": p_off, "pn": len(rec),
                            "ptotal": n_total, "b": boxes, "b2": b2, "b2p": b2p,
                            "im": img_recs})

        if k % 10 == 0 or k == n_keep - 1:
            mb = len(blob) / 1e6
            el = time.time() - t_start
            eta = el / (k + 1) * (n_keep - k - 1)
            print(f"\r  frame {k+1}/{n_keep}  blob {mb:6.1f} MB  eta {eta:4.0f}s", end="")
    print()

    ts = [f["t"] for f in frames_meta]
    meta.update({
        "frames": frames_meta,
        "dur": round(ts[-1] - ts[0], 3) if len(ts) > 1 else 0.0,
        "hz": round((len(ts) - 1) / (ts[-1] - ts[0]), 3) if len(ts) > 1 and ts[-1] > ts[0] else 0.0,
        "types": sorted(types_seen),
        "subsample": f"{args.points} pts/frame, returns {args.returns}, img {args.img_width}px",
    })

    html = stamp(json.dumps(meta, separators=(",", ":")),
                 base64.b64encode(bytes(blob)).decode())

    out = args.out or os.path.splitext(args.tfrecord)[0] + "_viewer.html"
    with open(out, "w") as fh:
        fh.write(html)

    size = os.path.getsize(out)
    print(f"wrote {out}")
    print(f"  {size/1e6:.1f} MB  (blob {len(blob)/1e6:.1f} MB, "
          f"meta {len(json.dumps(meta))/1e6:.1f} MB)  in {time.time()-t_start:.0f}s")
    if size > 120e6:
        print("  NOTE: heavy for a browser tab. Try --stride 2 --points 6000 --img-width 400")


if __name__ == "__main__":
    sys.exit(main())
