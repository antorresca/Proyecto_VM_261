from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Divide dataset YOLO-seg en train/val.")
    parser.add_argument("--images", default="data/images/raw_annotated")
    parser.add_argument("--labels", default="data/labels/raw")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("Faltan carpetas de imagenes o labels.")

    pairs: list[tuple[Path, Path]] = []
    for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        label = label_dir / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
    if not pairs:
        raise RuntimeError("No hay pares imagen/label para dividir.")

    random.seed(args.seed)
    random.shuffle(pairs)
    val_count = max(1, int(len(pairs) * args.val_ratio)) if len(pairs) > 1 else 0
    val = set(pairs[:val_count])

    targets = {
        "train_img": Path("data/images/train"),
        "val_img": Path("data/images/val"),
        "train_lbl": Path("data/labels/train"),
        "val_lbl": Path("data/labels/val"),
    }
    for target in targets.values():
        reset_dir(target)

    for image, label in pairs:
        is_val = (image, label) in val
        shutil.copy2(image, targets["val_img" if is_val else "train_img"] / image.name)
        shutil.copy2(label, targets["val_lbl" if is_val else "train_lbl"] / label.name)
    print(f"Total: {len(pairs)} | train: {len(pairs) - val_count} | val: {val_count}")


if __name__ == "__main__":
    main()
