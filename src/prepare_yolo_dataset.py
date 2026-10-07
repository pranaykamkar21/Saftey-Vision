from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import yaml

from src.prepare_voc_dataset import CLASS_MAPS, extract_objects


def prepare_yolo_dataset(
    task: str,
    source_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> dict[str, int]:
    if task not in CLASS_MAPS:
        raise ValueError("task must be 'mask' or 'helmet'")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")

    source_dir, output_dir = Path(source_dir), Path(output_dir)
    aliases = CLASS_MAPS[task]
    class_names = list(dict.fromkeys(aliases.values()))
    class_ids = {name: index for index, name in enumerate(class_names)}
    annotated = []
    for xml_path in sorted(source_dir.rglob("*.xml")):
        image_path, objects = extract_objects(xml_path, aliases)
        if image_path is not None and objects:
            annotated.append((image_path, objects))
    if not annotated:
        raise FileNotFoundError(f"No matching Pascal VOC annotations found in {source_dir}")

    random.Random(seed).shuffle(annotated)
    train_end = int(len(annotated) * train_ratio)
    val_end = train_end + int(len(annotated) * val_ratio)
    splits = {
        "train": annotated[:train_end],
        "val": annotated[train_end:val_end],
        "test": annotated[val_end:],
    }
    image_counts: dict[str, int] = {}
    for split, entries in splits.items():
        images_dir = output_dir / split / "images"
        labels_dir = output_dir / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_counts[split] = 0

        for index, (image_path, objects) in enumerate(entries):
            image_name = f"{index:04d}_{image_path.name}"
            label_path = labels_dir / f"{Path(image_name).stem}.txt"
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not decode source image: {image_path}")
            height, width = image.shape[:2]
            shutil.copy2(image_path, images_dir / image_name)
            lines = []
            for label, (raw_x1, raw_y1, raw_x2, raw_y2) in objects:
                x1, x2 = max(0, raw_x1), min(width, raw_x2)
                y1, y2 = max(0, raw_y1), min(height, raw_y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                class_id = class_ids[label]
                center_x = ((x1 + x2) / 2) / width
                center_y = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                lines.append(
                    f"{class_id} {center_x:.6f} {center_y:.6f} "
                    f"{box_width:.6f} {box_height:.6f}"
                )
            if not lines:
                raise ValueError(f"No valid bounding boxes in {image_path}")
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            image_counts[split] += 1

    config = {
        "path": output_dir.resolve().as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": class_names,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    print(f"Prepared {len(annotated)} source images for YOLO under {output_dir}")
    for split, count in image_counts.items():
        print(f"{split}: {count} images")
    return image_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Pascal VOC boxes to a YOLO detection dataset")
    parser.add_argument("--task", choices=["mask", "helmet"], required=True)
    parser.add_argument("--source", required=True, help="Folder containing VOC XMLs and source images")
    parser.add_argument("--output", default=None, help="Output directory; defaults to data/<task>")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = args.output or str(Path("data") / args.task)
    prepare_yolo_dataset(args.task, args.source, output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
