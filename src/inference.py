from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def run_inference(model_path: str | Path, source: str | Path, conf: float = 0.5) -> None:
    model = YOLO(str(model_path))
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Input source does not exist: {source_path}")

    result = model(source=str(source_path), conf=conf, save=True)
    print(f"Inference finished for: {source_path}")
    print(f"Saved output to: {source_path.parent / (source_path.stem + '_annotated' + source_path.suffix)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a single image or video")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained YOLO model (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Input image or video path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    run_inference(args.model, args.source, args.conf)


if __name__ == "__main__":
    main()
