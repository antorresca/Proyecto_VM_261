from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def shape_points(shape: dict) -> list[list[float]]:
    points = shape.get("points", [])
    shape_type = shape.get("shape_type", "polygon")
    if shape_type == "polygon":
        return points if len(points) >= 3 else []
    if shape_type == "rectangle" and len(points) == 2:
        (x1, y1), (x2, y2) = points
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convierte LabelMe polygon a YOLO segmentation.")
    parser.add_argument("--labelme-dir", default="data/labelme")
    parser.add_argument("--images-out", default="data/images/raw_annotated")
    parser.add_argument("--labels-out", default="data/labels/raw")
    parser.add_argument("--classes", default="configs/classes.txt")
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def load_classes(path: Path, discovered: list[str]) -> list[str]:
    existing = []
    if path.exists():
        existing = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    for label in discovered:
        if label not in existing:
            existing.append(label)
    path.write_text("\n".join(existing) + "\n", encoding="utf-8")
    return existing


def main() -> None:
    args = parse_args()
    labelme_dir = Path(args.labelme_dir)
    images_out = Path(args.images_out)
    labels_out = Path(args.labels_out)
    reset_dir(images_out)
    reset_dir(labels_out)
    json_files = sorted(labelme_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No hay .json de LabelMe en {labelme_dir}")

    parsed = []
    discovered = []
    for json_path in json_files:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        parsed.append((json_path, data))
        for shape in data.get("shapes", []):
            if shape_points(shape):
                discovered.append(normalize_label(str(shape["label"])))

    classes = load_classes(Path(args.classes), discovered)
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    converted = 0
    for json_path, data in parsed:
        image_name = data.get("imagePath") or json_path.with_suffix(".jpg").name
        image_path = json_path.parent / image_name
        if not image_path.exists():
            for ext in IMAGE_EXTS:
                candidate = json_path.with_suffix(ext)
                if candidate.exists():
                    image_path = candidate
                    break
        if not image_path.exists():
            raise FileNotFoundError(f"No encontre imagen para {json_path}")
        width = float(data["imageWidth"])
        height = float(data["imageHeight"])
        shutil.copy2(image_path, images_out / image_path.name)
        lines = []
        for shape in data.get("shapes", []):
            points = shape_points(shape)
            if not points:
                continue
            coords = []
            for x, y in points:
                coords.extend([f"{float(x) / width:.6f}", f"{float(y) / height:.6f}"])
            label = normalize_label(str(shape["label"]))
            lines.append(f"{class_to_id[label]} " + " ".join(coords))
        if lines:
            (labels_out / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            converted += 1
    print(f"Anotaciones convertidas: {converted}")


if __name__ == "__main__":
    main()
