#!/usr/bin/env python3
"""Fine-tune YOLO26n on the Waymo detection dataset.

    python3 train.py --epochs 10     # smoke test
    python3 train.py --epochs 60     # real run
"""
import argparse
import os

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--data", default="../../data/datasets/detection_v1/dataset.yaml")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="'0' for first GPU, 'cpu' for CPU")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="waymo_detect")
    ap.add_argument("--lr0", type=float, default=0.001,
                    help="initial LR -- lower than Ultralytics' from-scratch default (0.01), "
                         "appropriate for fine-tuning a COCO-pretrained model on a much smaller dataset")
    ap.add_argument("--patience", type=int, default=15,
                    help="early-stop if val mAP50 doesn't improve for this many epochs; "
                         "0 disables early stopping (useful when a rare class may need more "
                         "epochs than the aggregate fitness score, dominated by common classes, "
                         "would otherwise tolerate)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(here, args.data))

    model = YOLO(args.model)
    model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        project=os.path.join(here, "runs"),
        name=args.name,
        lr0=args.lr0,
        cos_lr=True,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
