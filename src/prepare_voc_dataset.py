from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2


CLASS_MAPS = {
    "mask": {
        "with mask": "with_mask",
        "with_mask": "with_mask",
        "without mask": "without_mask",
        "without_mask": "without_mask",
        "mask worn incorrectly": "mask_worn_incorrectly",
        "mask_worn_incorrectly": "mask_worn_incorrectly",
        "mask_weared_incorrect": "mask_worn_incorrectly",
        "mask_weared_incorrectly": "mask_worn_incorrectly",
        "mask_weared_incorrect ": "mask_worn_incorrectly",
    },
    "helmet": {
        "with helmet": "helmet",
        "helmet": "helmet",
        "without helmet": "no_helmet",
        "no helmet": "no_helmet",
        "no_helmet": "no_helmet",
        "head": "no_helmet",
    },
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_image(xml_path: Path, filename: str | None) -> Path | None:
    candidates = []
    if filename:
        candidates.append(xml_path.parent / filename)
        candidates.append(xml_path.parent.parent / filename)
        candidates.append(xml_path.parent.parent / "images" / filename)
    candidates.extend(xml_path.parent / f"{xml_path.stem}{ext}" for ext in IMAGE_EXTENSIONS)
    candidates.extend(xml_path.parent.parent / "images" / f"{xml_path.stem}{ext}" for ext in IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for base in (xml_path.parent, xml_path.parent.parent):
        for candidate in base.rglob(f"{xml_path.stem}.*"):
            if candidate.suffix.lower() in IMAGE_EXTENSIONS:
                return candidate
    return None


def extract_objects(xml_path: Path, aliases: dict[str, str]) -> tuple[Path | None, list[tuple[str, tuple[int, int, int, int]]]]:
    root = ET.parse(xml_path).getroot()
    image_path = find_image(xml_path, root.findtext("filename"))
    if image_path is None:
        return None, []
    size = root.find("size")
    image_width = int(float(size.findtext("width", "0"))) if size is not None else 0
    image_height = int(float(size.findtext("height", "0"))) if size is not None else 0

    objects = []
    for obj in root.findall("object"):
        class_name = " ".join((obj.findtext("name") or "").strip().lower().split())
        label = aliases.get(class_name)
        box = obj.find("bndbox")
        if label is None or box is None:
            continue
        try:
            x1 = int(float(box.findtext("xmin", "0")))
            y1 = int(float(box.findtext("ymin", "0")))
            x2 = int(float(box.findtext("xmax", "0")))
            y2 = int(float(box.findtext("ymax", "0")))
        except ValueError:
            continue
        if image_width and image_height and (x2 > image_width or y2 > image_height):
            scaled_box = (x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000)
            if scaled_box[2] <= image_width and scaled_box[3] <= image_height:
                x1, y1, x2, y2 = (round(value) for value in scaled_box)
        if x2 > x1 and y2 > y1:
            objects.append((label, (x1, y1, x2, y2)))
    return image_path, objects


def prepare_dataset(
    task: str,
    source_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> dict[str, Counter[str]]:
    if task not in CLASS_MAPS:
        raise ValueError("task must be 'mask' or 'helmet'")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    xml_files = sorted(source_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No Pascal VOC XML annotation files found in {source_dir}")

    aliases = CLASS_MAPS[task]
    annotated = []
    for xml_path in xml_files:
        image_path, objects = extract_objects(xml_path, aliases)
        if image_path is not None and objects:
            annotated.append((image_path, objects))
    if not annotated:
        raise ValueError(f"No matching labeled images found in {source_dir} for task '{task}'")

    rng = random.Random(seed)
    rng.shuffle(annotated)
    train_end = int(len(annotated) * train_ratio)
    val_end = train_end + int(len(annotated) * val_ratio)
    split_entries = {
        "train": annotated[:train_end],
        "val": annotated[train_end:val_end],
        "test": annotated[val_end:],
    }
    class_names = list(dict.fromkeys(aliases.values()))
    counts = {split: Counter() for split in split_entries}
    for split in split_entries:
        for class_name in class_names:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    for split, entries in split_entries.items():
        for image_number, (image_path, objects) in enumerate(entries):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            for object_number, (label, (x1, y1, x2, y2)) in enumerate(objects):
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                class_dir = output_dir / split / label
                class_dir.mkdir(parents=True, exist_ok=True)
                crop_path = class_dir / f"{image_path.stem}_{image_number}_{object_number}.jpg"
                if cv2.imwrite(str(crop_path), crop):
                    counts[split][label] += 1

    empty_splits = [split for split, split_counts in counts.items() if not split_counts]
    if empty_splits:
        raise ValueError(f"No usable crops created for split(s): {', '.join(empty_splits)}")

    missing_classes = [name for name in class_names if not any(split_counts[name] for split_counts in counts.values())]
    if missing_classes:
        raise ValueError(f"Dataset does not contain crops for class(es): {', '.join(missing_classes)}")

    print(f"Processed {len(annotated)} annotated source images into {output_dir}")
    for split, split_counts in counts.items():
        print(f"{split}: {dict(split_counts)}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Pascal VOC box annotations into MobileNet crop datasets")
    parser.add_argument("--task", choices=["mask", "helmet"], required=True)
    parser.add_argument("--source", required=True, help="Directory containing source images and VOC XML annotations")
    parser.add_argument("--output", default=None, help="Output directory; defaults to data/<task>_crops")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output or str(Path("data") / "classification" / args.task)
    prepare_dataset(args.task, args.source, output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
