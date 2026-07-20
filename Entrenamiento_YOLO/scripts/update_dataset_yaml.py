from __future__ import annotations

from pathlib import Path

import yaml


def main() -> None:
    classes = [
        line.strip()
        for line in Path("configs/classes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not classes:
        raise RuntimeError("configs/classes.txt no tiene clases reales.")
    data = {
        "path": str(Path("data").resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(classes)},
    }
    Path("configs/dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"dataset.yaml actualizado con {len(classes)} clases.")


if __name__ == "__main__":
    main()
