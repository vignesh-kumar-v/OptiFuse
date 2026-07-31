#!/usr/bin/env python3
"""Sanity check: draw YOLO-format label boxes over sample images and save them.

Mandatory gate before spending any training compute -- catches cx/cy/w/h or
class-index bugs immediately, visually, rather than discovering them 30
epochs into a training run.

    python3 visualize_labels.py
"""
import argparse
import glob
import os
import random

from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]
CLASS_COLORS = ["#3987e5", "#d95926", "#c98500"]  # blue, orange, yellow -- matches earlier BEV plot palette


def draw_labels(img_path, lbl_path, out_path):
    img = Image.open(img_path).convert("RGB")
    w_img, h_img = img.size
    draw = ImageDraw.Draw(img)

    n_boxes = 0
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls, cx, cy, w, h = line.split()
                cls = int(cls)
                cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
                x0 = (cx - w / 2) * w_img
                y0 = (cy - h / 2) * h_img
                x1 = (cx + w / 2) * w_img
                y1 = (cy + h / 2) * h_img
                color = CLASS_COLORS[cls % len(CLASS_COLORS)]
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                draw.text((x0 + 2, max(0, y0 - 14)), CLASS_NAMES[cls], fill=color)
                n_boxes += 1

    img.save(out_path, "JPEG", quality=92)
    return n_boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="datasets/detection_v1")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=12, help="number of sample images per class to check")
    ap.add_argument("--out-dir", default="label_check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(here, args.dataset_dir, "images", args.split)
    labels_dir = os.path.join(here, args.dataset_dir, "labels", args.split)
    out_dir = os.path.join(here, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Bucket label files by which classes they contain, so the sample set
    # actually includes pedestrian/cyclist examples rather than being
    # dominated by the much more frequent vehicle class.
    by_class = {c: [] for c in range(len(CLASS_NAMES))}
    all_label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    for lbl_path in all_label_files:
        with open(lbl_path) as f:
            classes_present = {int(line.split()[0]) for line in f if line.strip()}
        for c in classes_present:
            by_class[c].append(lbl_path)

    rng = random.Random(args.seed)
    chosen = set()
    for c, files in by_class.items():
        pick = rng.sample(files, min(args.n, len(files)))
        chosen.update(pick)
        print(f"class '{CLASS_NAMES[c]}': {len(files)} labeled images available, sampled {len(pick)}")

    total_boxes = 0
    for lbl_path in sorted(chosen):
        stem = os.path.splitext(os.path.basename(lbl_path))[0]
        img_path = os.path.join(images_dir, stem + ".jpg")
        out_path = os.path.join(out_dir, stem + ".jpg")
        n_boxes = draw_labels(img_path, lbl_path, out_path)
        total_boxes += n_boxes
        print(f"  {stem}: {n_boxes} boxes -> {out_path}")

    print(f"\nWrote {len(chosen)} annotated images ({total_boxes} boxes total) to {out_dir}")
    print("Inspect them visually: boxes must align tightly on real vehicles/pedestrians/cyclists.")


if __name__ == "__main__":
    main()
