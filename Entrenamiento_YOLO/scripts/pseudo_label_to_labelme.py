from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea JSON de LabelMe usando predicciones YOLO-seg para revision humana."
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-dir", default="data/labelme")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"No pude leer la imagen: {path}")
    height, width = image.shape[:2]
    return width, height


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    weights = Path(args.weights)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    if not weights.exists():
        raise FileNotFoundError(weights)

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    pending = [p for p in images if args.overwrite or not p.with_suffix(".json").exists()]
    if not pending:
        print("No hay imagenes pendientes de pseudo-etiquetar.")
        return

    model = YOLO(str(weights))
    created = 0
    empty = 0
    results = model.predict(
        source=[str(p) for p in pending],
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        stream=True,
        verbose=False,
        retina_masks=True,
    )
    for image_path, result in zip(pending, results):
        width, height = load_image_size(image_path)
        shapes = []
        if result.masks is not None and result.boxes is not None:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
            confidences = result.boxes.conf.detach().cpu().numpy().tolist()
            for cls_id, confidence, polygon in zip(classes, confidences, result.masks.xyn):
                points = [
                    [round(float(x) * width, 2), round(float(y) * height, 2)]
                    for x, y in polygon
                ]
                if len(points) < 3:
                    continue
                shapes.append(
                    {
                        "label": str(result.names[int(cls_id)]),
                        "points": points,
                        "group_id": None,
                        "description": f"pseudo_label_conf={confidence:.3f}",
                        "shape_type": "polygon",
                        "flags": {},
                        "mask": None,
                    }
                )

        if not shapes:
            empty += 1
            continue

        data = {
            "version": "6.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        image_path.with_suffix(".json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        created += 1

    print(f"Imagenes revisadas: {len(pending)}")
    print(f"JSON pseudo-etiquetados creados: {created}")
    print(f"Imagenes sin detecciones al umbral indicado: {empty}")


if __name__ == "__main__":
    main()
