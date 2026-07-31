#!/usr/bin/env python3
"""Fine-tune SegFormer-B0 on Waymo panoptic labels for drivable-area + lane markings.

    python3 train.py --epochs 20                  # real run
    python3 train.py --epochs 1 --max-steps 30    # smoke test

Waymo's camera labels are full panoptic masks with ~28 semantic classes; we
collapse them to 4 (see CLASS_NAMES). LANE_MARKER(21) is merged into
ROAD_MARKER(22) deliberately: measured on real frames, class 21 alone is
0.00-0.12% of pixels and absent from many frames entirely -- unlearnable in
isolation. Merged, the marking class is ~1.8%.

Even at 1.8%, plain cross-entropy would predict "not marking" everywhere and
score ~98% pixel accuracy while being useless -- the same trap that produced a
0.03 mAP cyclist class in Milestone 1. So the loss is class-weighted CE + Dice
(Dice is insensitive to background dominance) and the reported metric is
per-class IoU, never global pixel accuracy.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from waymo_dataset import IGNORE_INDEX, WaymoImages

CLASS_NAMES = ["background", "road", "marking", "sidewalk"]

# SegformerImageProcessor for this checkpoint has do_rescale=1/255 AND
# do_normalize=True with these ImageNet statistics. Feeding plain 0-1 pixels
# would shift the input distribution away from what the pretrained Cityscapes
# encoder expects and quietly weaken transfer, so apply both.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Waymo CameraSegmentation semantic id -> our class index. Everything not listed
# collapses to 0/background. 255 (letterbox padding) is preserved as ignore.
WAYMO_TO_OURS = {20: 1, 21: 2, 22: 2, 23: 3}


def build_lut():
    """256-entry lookup table so remapping is one vectorized index, not a loop."""
    lut = np.zeros(256, np.uint8)
    for src, dst in WAYMO_TO_OURS.items():
        lut[src] = dst
    lut[IGNORE_INDEX] = IGNORE_INDEX
    return lut


LUT = build_lut()


class SegSamples(IterableDataset):
    """Thin adapter: WaymoImages -> (image CHW float32 0-1, label HW int64)."""

    def __init__(self, shards, size):
        self.shards, self.size = shards, size

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        shards = self.shards if info is None else self.shards[info.id::info.num_workers]
        if not shards:
            return
        ds = WaymoImages(os.path.dirname(shards[0]), task="segmentation",
                         size=self.size, shards=shards)
        for s in ds:
            img = s["image"].astype(np.float32) / 255.0
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            lbl = torch.from_numpy(LUT[s["semantic"]].astype(np.int64))
            yield torch.from_numpy(img.transpose(2, 0, 1).copy()), lbl


def dice_loss(logits, target, n_classes, valid):
    """Soft Dice over valid (non-padding) pixels. Complements CE: Dice measures
    overlap per class, so a class occupying 1.8% of pixels still contributes a
    full-magnitude term instead of being averaged into irrelevance."""
    probs = logits.softmax(1)
    t = target.clone()
    t[~valid] = 0
    onehot = F.one_hot(t, n_classes).permute(0, 3, 1, 2).float()
    v = valid.unsqueeze(1).float()
    probs, onehot = probs * v, onehot * v
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    return (1 - (2 * inter + 1.0) / (denom + 1.0)).mean()


@torch.no_grad()
def evaluate(model, loader, device, n_classes, max_batches=0):
    """Per-class IoU. Global pixel accuracy is deliberately not reported -- it is
    dominated by background and would hide a broken marking class."""
    model.eval()
    inter = torch.zeros(n_classes, dtype=torch.float64, device=device)
    union = torch.zeros(n_classes, dtype=torch.float64, device=device)
    for bi, (img, lbl) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        img, lbl = img.to(device), lbl.to(device)
        logits = model(pixel_values=img).logits
        logits = F.interpolate(logits, size=lbl.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1)
        valid = lbl != IGNORE_INDEX
        for c in range(n_classes):
            p, g = (pred == c) & valid, (lbl == c) & valid
            inter[c] += (p & g).sum()
            union[c] += (p | g).sum()
    model.train()
    iou = (inter / union.clamp(min=1)).cpu().numpy()
    return iou, union.cpu().numpy()


def compute_class_weights(loader, n_classes, n_batches, device):
    """Inverse-sqrt-frequency weights, measured from the data rather than guessed."""
    counts = torch.zeros(n_classes, dtype=torch.float64)
    for bi, (_, lbl) in enumerate(loader):
        if bi >= n_batches:
            break
        v = lbl[lbl != IGNORE_INDEX]
        counts += torch.bincount(v.flatten(), minlength=n_classes).double()
    freq = counts / counts.sum().clamp(min=1)
    w = 1.0 / torch.sqrt(freq.clamp(min=1e-6))
    w = w / w.mean()
    return w.float().to(device), freq.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="../../cache/local", help="dir of .tar shards")
    ap.add_argument("--val-shard", default=None,
                    help="basename of the shard held out for validation "
                         "(default: the last shard containing segmentation)")
    ap.add_argument("--model", default="nvidia/segformer-b0-finetuned-cityscapes-640-1280")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-5, help="SegFormer's standard fine-tune LR")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="cap steps/epoch (smoke tests)")
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/seg_v1")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.normpath(os.path.join(here, args.cache))
    out_dir = os.path.normpath(os.path.join(here, args.out))
    os.makedirs(out_dir, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(cache, "*.tar")))
    if not shards:
        raise SystemExit(f"no .tar shards in {cache} -- run waymo_extract.py first")

    # Only shards that actually carry segmentation are usable here; the rest
    # would contribute zero samples and just slow every epoch down.
    usable = []
    for sh in shards:
        ds = WaymoImages(cache, task="segmentation", size=(args.width, args.height), shards=[sh])
        if any(True for _ in zip(range(1), ds)):
            usable.append(sh)
    if not usable:
        raise SystemExit("no shard contains segmentation labels")

    if args.val_shard:
        val = [s for s in usable if os.path.basename(s) == args.val_shard]
        if not val:
            raise SystemExit(f"--val-shard {args.val_shard} not among segmentation shards")
    else:
        val = [usable[-1]]
    train = [s for s in usable if s not in val]

    # Segment-level split wherever possible. With a single labelled segment that
    # is impossible, so fall back to splitting it and say so plainly -- frames
    # within one segment are ~0.1s apart, so this val set is NOT independent.
    single = not train
    if single:
        train, val = usable, usable
        print("WARNING: only one segmentation-labelled shard available. Train and val "
              "share it, so val IoU is optimistic and NOT a generalization estimate.\n"
              "         Pull more segmentation segments (data/probe_segments.py) before "
              "trusting these numbers.\n")

    print(f"segmentation shards: {len(usable)} usable of {len(shards)}")
    print(f"  train: {[os.path.basename(s) for s in train]}")
    print(f"  val  : {[os.path.basename(s) for s in val]}\n")

    size = (args.width, args.height)
    dl = lambda sh, w: DataLoader(SegSamples(sh, size), batch_size=args.batch, num_workers=w)
    train_loader, val_loader = dl(train, args.workers), dl(val, 0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_classes = len(CLASS_NAMES)

    print("measuring class frequencies...")
    weights, freq = compute_class_weights(dl(train, 0), n_classes, 12, device)
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:11} {100*freq[i]:6.2f}% of labelled pixels   weight={weights[i]:.3f}")
    print()

    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, num_labels=n_classes, ignore_mismatched_sizes=True).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * (args.max_steps or 200)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(total, 1),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = -1.0
    for ep in range(1, args.epochs + 1):
        t0, run, steps = time.time(), 0.0, 0
        for img, lbl in train_loader:
            if args.max_steps and steps >= args.max_steps:
                break
            img, lbl = img.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(pixel_values=img).logits
                logits = F.interpolate(logits, size=lbl.shape[-2:], mode="bilinear",
                                       align_corners=False)
                ce = F.cross_entropy(logits, lbl, weight=weights, ignore_index=IGNORE_INDEX)
                dl_ = dice_loss(logits.float(), lbl, n_classes, lbl != IGNORE_INDEX)
                loss = ce + args.dice_weight * dl_
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            run += loss.item()
            steps += 1

        iou, union = evaluate(model, val_loader, device, n_classes,
                              max_batches=20 if args.max_steps else 0)
        present = union > 0
        miou = float(iou[present].mean()) if present.any() else 0.0
        per = "  ".join(f"{CLASS_NAMES[c][:9]}={iou[c]:.3f}" for c in range(n_classes))
        print(f"epoch {ep:3d}/{args.epochs}  loss={run/max(steps,1):.4f}  "
              f"mIoU={miou:.4f}  {per}  ({time.time()-t0:.0f}s, {steps} steps)")

        if miou > best:
            best = miou
            model.save_pretrained(os.path.join(out_dir, "best"))
    print(f"\nbest mIoU {best:.4f} -> {os.path.join(out_dir,'best')}")
    if single:
        print("Reminder: train and val shared one segment -- treat this as a pipeline "
              "check, not an accuracy result.")


if __name__ == "__main__":
    main()
