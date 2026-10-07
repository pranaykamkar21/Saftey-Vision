from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import yaml


def ensure_dataset_structure(data_root: str | Path, labels: Iterable[str]) -> None:
    """Create a minimal YOLO-style dataset structure and config file."""
    root = Path(data_root)
    split_dirs = ["train", "val", "test"]

    for split in split_dirs:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)

    config = {
        "path": str(root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(list(labels)),
        "names": list(labels),
    }

    with open(root / "data.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)


def split_image_lists(image_dir: str | Path, train_ratio: float = 0.7, val_ratio: float = 0.2) -> dict[str, List[str]]:
    """Example dataset split helper for a directory of images."""
    image_dir = Path(image_dir)
    files = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    total = len(files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    train_files = [str(p) for p in files[:train_count]]
    val_files = [str(p) for p in files[train_count : train_count + val_count]]
    test_files = [str(p) for p in files[train_count + val_count : train_count + val_count + test_count]]

    return {"train": train_files, "val": val_files, "test": test_files}
