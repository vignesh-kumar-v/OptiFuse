#!/usr/bin/env python3
"""Extract a YOLO-format detection dataset from Waymo .tfrecord segments.

Uses waymo_lib.py (no TensorFlow / waymo-open-dataset). Reads native 2D
labels (Frame.camera_labels(field=8)) -- genuine human-drawn boxes, not the
looser field-9 boxes derived from projected 3D points.

Split is at the SEGMENT level, not a random per-image shuffle: frames within
one segment are ~0.1s apart (~10Hz), so consecutive frames are near-duplicate
scenes and a random split would leak near-identical images between train and
val. One full segment is held out entirely for validation.

    python3 prepare_detection_dataset.py
"""
import argparse
import io
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import waymo_lib as wl

# Waymo Label.Type enum (see waymo_lib.OBJ) -> YOLO class index.
# TYPE_SIGN (3) never appears in native 2D labels, so it is intentionally absent.
CLASS_MAP = {1: 0, 2: 1, 4: 2}   # VEHICLE, PEDESTRIAN, CYCLIST
CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]

DEFAULT_SEGMENTS_DIR = "../../waymo_data"
DEFAULT_OUT_DIR = "datasets/detection_v1"

# Held out entirely for validation. Of the 5 segments, only this one and
# 1024360143612057520 carry all three classes (VEHICLE/PEDESTRIAN/CYCLIST) --
# the other three are missing PEDESTRIAN or CYCLIST entirely (see the
# per-segment class-count probe in the milestone plan). This one is used
# because it is the smaller of the two, keeping more data in train.
VAL_SEGMENT = "segment-10689101165701914459_2072_300_2092_300_with_camera_labels.tfrecord"


def write_split(segment_paths, images_dir, labels_dir, jpeg_quality=90):
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    n_images = 0
    n_boxes = 0
    class_counts = {c: 0 for c in CLASS_NAMES}

    for seg_path in segment_paths:
        seg_name = os.path.basename(seg_path).replace(".tfrecord", "")
        for frame_idx, payload in wl.records(seg_path):
            F = wl.Frame(payload)
            images = {im["name"]: im["jpeg"] for im in F.images()}
            cam_labels = F.camera_labels(field=8)

            for cam_name, jpeg_bytes in images.items():
                boxes = cam_labels.get(cam_name, [])
                img = Image.open(io.BytesIO(jpeg_bytes))
                w_img, h_img = img.size  # per-image: FRONT* is 1920x1280, SIDE* is 1920x886

                stem = f"{seg_name}_f{frame_idx:04d}_{cam_name}"
                img_path = os.path.join(images_dir, stem + ".jpg")
                lbl_path = os.path.join(labels_dir, stem + ".txt")

                # Re-encode via PIL instead of writing raw jpeg_bytes: guarantees
                # a decodable, consistently-oriented JPEG regardless of the
                # source encoder, at negligible quality cost.
                img.convert("RGB").save(img_path, "JPEG", quality=jpeg_quality)

                lines = []
                for b in boxes:
                    cls = CLASS_MAP.get(b["type"])
                    if cls is None:
                        continue  # e.g. TYPE_UNKNOWN; not expected in field 8, but be safe
                    cx, cy, w, h = b["cx"] / w_img, b["cy"] / h_img, b["w"] / w_img, b["h"] / h_img
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    class_counts[CLASS_NAMES[cls]] += 1
                    n_boxes += 1

                # Write the label file even when empty (0 bytes) -- YOLO treats
                # a present-but-empty label file as a genuine background image,
                # which is what an unlabeled Waymo camera entry actually means.
                with open(lbl_path, "w") as f:
                    f.write("\n".join(lines))

                n_images += 1

    return n_images, n_boxes, class_counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments-dir", default=DEFAULT_SEGMENTS_DIR,
                    help=f"directory containing .tfrecord segments [{DEFAULT_SEGMENTS_DIR}]")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help=f"output dataset directory [{DEFAULT_OUT_DIR}]")
    ap.add_argument("--val-segment", default=VAL_SEGMENT,
                    help="filename of the segment to hold out entirely for validation")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    segments_dir = os.path.normpath(os.path.join(here, args.segments_dir))
    out_dir = os.path.normpath(os.path.join(here, args.out_dir))

    all_segments = sorted(
        os.path.join(segments_dir, f)
        for f in os.listdir(segments_dir)
        if f.endswith(".tfrecord")
    )
    if not all_segments:
        raise SystemExit(f"no .tfrecord files found in {segments_dir}")

    val_segments = [p for p in all_segments if os.path.basename(p) == args.val_segment]
    train_segments = [p for p in all_segments if os.path.basename(p) != args.val_segment]
    if not val_segments:
        raise SystemExit(f"val segment {args.val_segment} not found in {segments_dir}")

    print(f"Train segments ({len(train_segments)}):")
    for p in train_segments:
        print(f"  {os.path.basename(p)}")
    print(f"Val segment: {os.path.basename(val_segments[0])}\n")

    for split_name, segs in (("train", train_segments), ("val", val_segments)):
        images_dir = os.path.join(out_dir, "images", split_name)
        labels_dir = os.path.join(out_dir, "labels", split_name)
        print(f"Extracting {split_name} split...")
        n_images, n_boxes, class_counts = write_split(segs, images_dir, labels_dir)
        print(f"  {n_images} images, {n_boxes} boxes, class counts: {class_counts}\n")

    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
