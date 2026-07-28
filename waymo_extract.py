#!/usr/bin/env python3
"""Extract Waymo tfrecords into per-segment tar shards for image-task training.

    python3 waymo_extract.py waymo_perception/training/*.tfrecord -o cache/training

One tar per input segment, WebDataset layout: every member is
`{segment}_{frame:04d}_{CAM}.{ext}`, so a shard is read sequentially and the
sample key groups the parts.

    .jpg        camera image, source bytes copied verbatim (no re-encode)
    .json       2D boxes, intrinsics, extrinsic, poses, timing
    .depth.npz  sparse lidar depth: u,v int16 + z uint16 centimetres
    .seg.png    panoptic label, uint16, sem = v // 1000  (only ~7% of frames)

Images are kept at native resolution because `camera_projection` uv is in
full-res pixels; resizing here would bake a scaling step into the cache. The
dataloader resizes image and uv together instead.

Depth is the one thing decoded rather than copied, and it dominates runtime
(~775 ms/frame vs ~2 ms to parse the proto). --no-depth skips it entirely,
which is the right call for detection- or segmentation-only extractions.
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from waymo_lib import CAM, Frame, msg, records

CAM_ID = {v: k for k, v in CAM.items()}
# uint16 centimetres tops out at 655.35 m; lidar reaches ~76 m, so the clip
# never fires on real data and only guards against a decode going wrong.
MAX_CM = 65535


def camera_depth(frame_pose, image_pose, extrinsic, xyz, uv):
    """Sparse depth for one camera: (uv int16, z uint16 cm), points behind dropped.

    xyz arrives in the *frame* vehicle frame, but the image was triggered up to
    ~50 ms after the frame timestamp and takes another ~49 ms to read out. At
    69 km/h that is ~1.9 m of ego motion, which lands entirely on moving
    objects. Route through global and back out via the image's own pose so the
    depth is measured from where the camera actually was.

        vehicle@frame -> global -> vehicle@image -> camera
    """
    M = np.linalg.inv(extrinsic) @ np.linalg.inv(image_pose) @ frame_pose
    p = xyz @ M[:3, :3].T + M[:3, 3]
    z = p[:, 0]                       # Waymo camera frame: +x is the optical axis
    keep = z > 0
    cm = np.clip(z[keep] * 100.0, 0, MAX_CM)
    return uv[keep].astype(np.int16), cm.astype(np.uint16)


def panoptic_png(image_msg):
    """Raw PNG bytes of CameraSegmentationLabel, or None. Divisor is always 1000."""
    if 10 not in image_msg:
        return None
    return msg(image_msg[10][0])[2][0]


def frame_samples(F, want_cams, with_depth):
    """Yield (cam_name, {ext: bytes}) for one frame."""
    calib = F.camera_calibrations()
    labels = F.camera_labels(8)
    fp = F.pose
    pts = F.points(returns=(2,)) if with_depth else None

    for raw, im in zip(F.f.get(4, []), F.images()):
        name = im["name"]
        if name not in want_cams:
            continue
        c = calib[name]

        parts = {"jpg": im["jpeg"]}

        seg = panoptic_png(msg(raw))
        if seg is not None:
            parts["seg.png"] = seg

        if with_depth:
            sel = pts["cam"] == CAM_ID[name]
            uv, cm = camera_depth(fp, im["pose"], c["extrinsic"],
                                  pts["xyz"][sel], pts["uv"][sel])
            buf = io.BytesIO()
            np.savez_compressed(buf, uv=uv, z_cm=cm)
            parts["depth.npz"] = buf.getvalue()

        parts["json"] = json.dumps({
            "camera": name,
            "width": c["width"], "height": c["height"],
            "intrinsic": [float(x) for x in c["intrinsic"]],
            "extrinsic": c["extrinsic"].tolist(),
            "rolling_shutter": c["rolling_shutter"],
            "frame_pose": fp.tolist(),
            "image_pose": im["pose"].tolist(),
            "frame_timestamp": F.timestamp,
            "trigger": im["trigger"],
            "readout_done": im["readout_done"],
            "shutter": im["shutter"],
            # field 8: native 2D, {VEHICLE, PEDESTRIAN, CYCLIST}. No SIGN --
            # SIGN exists only in the projected-3D labels (field 9).
            "boxes": [{"cx": b["cx"], "cy": b["cy"], "w": b["w"], "h": b["h"],
                       "type": b["type"], "id": b["id"]}
                      for b in labels.get(name, [])],
        }, separators=(",", ":")).encode()

        yield name, parts


def has_segmentation(F):
    return any(10 in msg(im) for im in F.f.get(4, []))


def extract_segment(path, out_dir, stride, want_cams, with_depth, limit=0):
    """One tfrecord -> one tar. Returns a stats dict.

    Every frame is parsed (~2 ms) even when strided out, because camera
    segmentation labels are not spread evenly -- they arrive in bursts at
    indices like 23,27,29,31,35, 73,77,... A uniform stride aliases against
    that pattern and silently destroys them (stride 10 keeps *zero*). So the
    kept set is (stride-selected UNION seg-labelled). Only kept frames pay the
    ~775 ms lidar decode, which is what actually costs anything.
    """
    seg = os.path.basename(path).replace(".tfrecord", "")
    short = seg.replace("segment-", "").replace("_with_camera_labels", "")
    out = os.path.join(out_dir, short + ".tar")
    tmp = out + ".partial"

    n_img = n_seg = n_frames = n_rescued = 0
    t0 = time.time()
    with tarfile.open(tmp, "w") as tar:
        for i, payload in records(path):
            F = Frame(payload)
            on_stride = i % stride == 0
            if not on_stride:
                if not has_segmentation(F):
                    continue
                n_rescued += 1
            if limit and n_frames >= limit:
                break
            n_frames += 1
            for name, parts in frame_samples(F, want_cams, with_depth):
                key = f"{short}_{i:04d}_{name}"
                for ext, data in sorted(parts.items()):
                    info = tarfile.TarInfo(f"{key}.{ext}")
                    info.size = len(data)
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
                n_img += 1
                n_seg += "seg.png" in parts

    os.replace(tmp, out)          # only a complete tar ever appears at `out`
    return {"segment": short, "tar": out, "frames": n_frames, "images": n_img,
            "seg_images": n_seg, "rescued": n_rescued,
            "bytes": os.path.getsize(out), "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tfrecord", nargs="+")
    ap.add_argument("-o", "--out", required=True, help="output directory for tar shards")
    ap.add_argument("--stride", type=int, default=5,
                    help="keep every Nth frame; 5 = 2 Hz [5]")
    ap.add_argument("--cameras", default="all",
                    help="comma-separated names, or 'all' / 'front' [all]")
    ap.add_argument("--no-depth", action="store_true",
                    help="skip lidar decode (~775 ms/frame). Detection/seg only.")
    ap.add_argument("--frames", type=int, default=0, help="stop after N kept frames [all]")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                    help="parallel segments")
    ap.add_argument("--force", action="store_true", help="re-extract existing tars")
    args = ap.parse_args()

    if args.cameras == "all":
        want = set(CAM.values()) - {"UNKNOWN", "REAR", "REAR_LEFT", "REAR_RIGHT"}
    elif args.cameras == "front":
        want = {"FRONT"}
    else:
        want = {c.strip().upper() for c in args.cameras.split(",")}

    os.makedirs(args.out, exist_ok=True)
    todo = args.tfrecord
    if not args.force:
        todo = [p for p in todo if not os.path.exists(os.path.join(
            args.out, os.path.basename(p).replace(".tfrecord", "")
            .replace("segment-", "").replace("_with_camera_labels", "") + ".tar"))]
        skipped = len(args.tfrecord) - len(todo)
        if skipped:
            print(f"skipping {skipped} already-extracted segment(s); --force to redo")
    if not todo:
        print("nothing to do")
        return 0

    print(f"{len(todo)} segment(s)  stride {args.stride}  cameras {sorted(want)}  "
          f"depth {'off' if args.no_depth else 'on'}  jobs {args.jobs}")

    done, t0 = [], time.time()
    with ProcessPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(extract_segment, p, args.out, args.stride, want,
                          not args.no_depth, args.frames): p for p in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as exc:
                print(f"  FAILED {os.path.basename(futs[f])}: {exc}")
                continue
            done.append(r)
            el = time.time() - t0
            print(f"  [{len(done)}/{len(todo)}] {r['segment'][:26]:26} "
                  f"{r['frames']:3}f {r['images']:4}img {r['bytes']/1e6:6.1f}MB "
                  f"{r['seconds']:5.0f}s  eta {el/len(done)*(len(todo)-len(done)):5.0f}s")

    if not done:
        print("no segments extracted")
        return 1
    tot = sum(r["bytes"] for r in done)
    print(f"\n{len(done)} tars  {sum(r['frames'] for r in done)} frames  "
          f"{sum(r['images'] for r in done)} images  "
          f"({sum(r['seg_images'] for r in done)} with segmentation, "
          f"{sum(r['rescued'] for r in done)} frames kept off-stride for it)")
    print(f"{tot/1e9:.2f} GB in {time.time()-t0:.0f}s  ->  {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
