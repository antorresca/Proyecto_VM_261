from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena YOLO-seg para el dataset ZED.")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="zed_real_yolo_seg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.data).exists():
        raise FileNotFoundError(args.data)
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/segment",
        name=args.name,
        task="segment",
        patience=25,
        workers=4,
        plots=True,
    )


if __name__ == "__main__":
    main()
