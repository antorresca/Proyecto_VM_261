from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta un modelo YOLO a formatos de despliegue.")
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--format",
        choices=["onnx", "torchscript", "engine", "openvino"],
        required=True,
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--out-dir", default="exports/robot_model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)

    model = YOLO(str(weights))
    exported = Path(
        model.export(
            format=args.format,
            imgsz=args.imgsz,
            device=args.device,
            half=args.half,
            dynamic=args.dynamic,
        )
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / exported.name
    if exported.resolve() != destination.resolve():
        if exported.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(exported, destination)
        else:
            shutil.copy2(exported, destination)
    print(destination.resolve())


if __name__ == "__main__":
    main()
