#!/usr/bin/env python3
"""Fine-tune Depth Anything V2 (Metric, Outdoor) on Waymo sparse lidar depth.

    python3 train.py --epochs 15                  # real run
    python3 train.py --epochs 1 --max-steps 20    # smoke test

Ground truth is lidar projected into the camera, so only ~0.1-1% of pixels have
a value. Every loss and metric here is therefore **masked to valid pixels** --
computing them densely would average over ~99% zeros and silently produce a
model that predicts near-zero everywhere while the loss looks fine. That masking
is the single most important detail in this file.

The base checkpoint is already metric (predicts metres) and already trained on
outdoor driving data (VKITTI), so no scale alignment step is needed -- which is
why this variant was chosen over the better-known relative-depth V2-Small.

Input geometry: the DINOv2 backbone uses 14x14 patches, so input dims must be
multiples of 14. 966x644 is used because 1920x1280 scales into it at 0.5031 on
both axes -- near-zero letterbox padding for the FRONT-family cameras.
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
from waymo_dataset import WaymoImages

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

MIN_DEPTH = 0.5    # metres; guards log(0)
MAX_DEPTH = 80.0   # lidar tops out ~76m, so anything beyond is noise


class DepthSamples(IterableDataset):
    """WaymoImages -> (image CHW normalized, depth HW metres, mask HW bool)."""

    def __init__(self, shards, size):
        self.shards, self.size = shards, size

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        shards = self.shards if info is None else self.shards[info.id::info.num_workers]
        if not shards:
            return
        ds = WaymoImages(os.path.dirname(shards[0]), task="depth",
                         size=self.size, shards=shards)
        for s in ds:
            img = s["image"].astype(np.float32) / 255.0
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            d = s["depth"].astype(np.float32)
            m = s["depth_mask"] & (d > MIN_DEPTH) & (d < MAX_DEPTH)
            if not m.any():
                continue  # frame with no usable lidar return contributes no gradient
            yield (torch.from_numpy(img.transpose(2, 0, 1).copy()),
                   torch.from_numpy(d), torch.from_numpy(m))


def silog_loss(pred, gt, mask, lam=0.85, alpha=10.0, l1_weight=0.1):
    """Scale-invariant log loss (Eigen et al.), the standard for sparse depth.

    lam<1 leaves some absolute-scale sensitivity, and the extra L1 term anchors
    it further: we want *metric* depth, so a purely scale-invariant objective
    would let the prediction drift away from metres.
    """
    p = pred.clamp(min=MIN_DEPTH)
    g = gt.clamp(min=MIN_DEPTH)
    d = torch.log(p[mask]) - torch.log(g[mask])
    if d.numel() == 0:
        return pred.sum() * 0.0
    silog = torch.sqrt((d ** 2).mean() - lam * (d.mean() ** 2) + 1e-7) * alpha
    return silog + l1_weight * (p[mask] - g[mask]).abs().mean()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    """AbsRel / RMSE / delta<1.25, all over valid lidar pixels only."""
    model.eval()
    n = 0
    absrel = sq = d125 = 0.0
    for bi, (img, gt, mask) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        img, gt, mask = img.to(device), gt.to(device), mask.to(device)
        pred = infer(model, img, gt.shape[-2:])
        p, g = pred[mask].clamp(MIN_DEPTH, MAX_DEPTH), gt[mask]
        if p.numel() == 0:
            continue
        absrel += ((p - g).abs() / g).sum().item()
        sq += ((p - g) ** 2).sum().item()
        d125 += (torch.maximum(p / g, g / p) < 1.25).sum().item()
        n += p.numel()
    model.train()
    if n == 0:
        return dict(absrel=float("nan"), rmse=float("nan"), d125=float("nan"), n=0)
    return dict(absrel=absrel / n, rmse=(sq / n) ** 0.5, d125=d125 / n, n=n)


def infer(model, img, out_hw):
    """Forward pass, resized back to label resolution."""
    d = model(pixel_values=img).predicted_depth
    if d.dim() == 3:
        d = d.unsqueeze(1)
    if d.shape[-2:] != out_hw:
        d = F.interpolate(d, size=out_hw, mode="bilinear", align_corners=False)
    return d.squeeze(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="../../cache/local")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    ap.add_argument("--width", type=int, default=966, help="must be a multiple of 14")
    ap.add_argument("--height", type=int, default=644, help="must be a multiple of 14")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="very low: we are nudging an already-metric model, not retraining it")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--val-shard", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/depth_v1")
    args = ap.parse_args()

    for name, v in (("width", args.width), ("height", args.height)):
        if v % 14:
            raise SystemExit(f"--{name}={v} must be a multiple of 14 (DINOv2 patch size)")

    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.normpath(os.path.join(here, args.cache))
    out_dir = os.path.normpath(os.path.join(here, args.out))
    os.makedirs(out_dir, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(cache, "*.tar")))
    if not shards:
        raise SystemExit(f"no .tar shards in {cache} -- run waymo_extract.py first")

    val = ([s for s in shards if os.path.basename(s) == args.val_shard]
           if args.val_shard else [shards[-1]])
    if not val:
        raise SystemExit(f"--val-shard {args.val_shard} not found")
    train = [s for s in shards if s not in val]

    print(f"depth shards: {len(shards)}")
    print(f"  train: {len(train)}  val: {[os.path.basename(s) for s in val]}\n")

    size = (args.width, args.height)
    mk = lambda sh, w: DataLoader(DepthSamples(sh, size), batch_size=args.batch, num_workers=w)
    train_loader, val_loader = mk(train, args.workers), mk(val, 0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device)
    model.train()

    base = evaluate(model, val_loader, device, max_batches=15)
    print(f"pretrained baseline (before any fine-tuning): "
          f"AbsRel={base['absrel']:.4f}  RMSE={base['rmse']:.3f}m  "
          f"d<1.25={base['d125']:.4f}  over {base['n']:,} lidar points\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0, run, steps = time.time(), 0.0, 0
        for img, gt, mask in train_loader:
            if args.max_steps and steps >= args.max_steps:
                break
            img, gt, mask = (img.to(device, non_blocking=True), gt.to(device, non_blocking=True),
                             mask.to(device, non_blocking=True))
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = infer(model, img, gt.shape[-2:])
                loss = silog_loss(pred.float(), gt, mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            steps += 1

        m = evaluate(model, val_loader, device, max_batches=15 if args.max_steps else 0)
        print(f"epoch {ep:3d}/{args.epochs}  loss={run/max(steps,1):.4f}  "
              f"AbsRel={m['absrel']:.4f}  RMSE={m['rmse']:.3f}m  d<1.25={m['d125']:.4f}  "
              f"({time.time()-t0:.0f}s, {steps} steps)")

        if m["absrel"] < best:
            best = m["absrel"]
            model.save_pretrained(os.path.join(out_dir, "best"))

    print(f"\nbest AbsRel {best:.4f} (baseline was {base['absrel']:.4f}) "
          f"-> {os.path.join(out_dir,'best')}")


if __name__ == "__main__":
    main()
