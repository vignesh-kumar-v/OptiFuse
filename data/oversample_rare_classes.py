#!/usr/bin/env python3
"""Oversample rare-class images in the train split by duplicating them on disk.

Cyclist appears in only ~7.7% of training images (490 instances vs 31,991
vehicle / 9,490 pedestrian) -- at batch=16 most batches see zero cyclist
examples, so the model gets almost no gradient signal for that class.
Duplicating the images containing it increases how often the sampler sees
them, without touching the underlying label distribution logic.

Idempotent: skips classes/images already at or above the target duplicate
count (tracked via the "_dupN" suffix), so re-running is safe.

    python3 oversample_rare_classes.py --class-idx 2 --factor 4
"""
import argparse
import glob
import os
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="datasets/detection_v1")
    ap.add_argument("--class-idx", type=int, required=True,
                    help="YOLO class index to oversample (e.g. 2 for cyclist)")
    ap.add_argument("--factor", type=int, required=True,
                    help="total copies per matching image, including the original (e.g. 4 = 3 extra copies)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(here, args.dataset_dir, "images", "train")
    labels_dir = os.path.join(here, args.dataset_dir, "labels", "train")

    matching = []
    for lbl_path in glob.glob(os.path.join(labels_dir, "*.txt")):
        stem = os.path.splitext(os.path.basename(lbl_path))[0]
        if "_dup" in stem:
            continue  # don't oversample already-duplicated copies
        with open(lbl_path) as f:
            classes = {int(line.split()[0]) for line in f if line.strip()}
        if args.class_idx in classes:
            matching.append(stem)

    print(f"Found {len(matching)} original train images containing class {args.class_idx}")

    n_created = 0
    for stem in matching:
        for dup_idx in range(1, args.factor):
            new_stem = f"{stem}_dup{dup_idx}"
            new_img = os.path.join(images_dir, new_stem + ".jpg")
            new_lbl = os.path.join(labels_dir, new_stem + ".txt")
            if os.path.exists(new_img) and os.path.exists(new_lbl):
                continue
            shutil.copyfile(os.path.join(images_dir, stem + ".jpg"), new_img)
            shutil.copyfile(os.path.join(labels_dir, stem + ".txt"), new_lbl)
            n_created += 1

    print(f"Created {n_created} duplicate image/label pairs "
          f"(target: {len(matching)} images x {args.factor - 1} extra copies each)")


if __name__ == "__main__":
    main()
