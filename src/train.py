from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from config import DATA_DIR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_IMG_SIZE, MASK_CLASSES, MODELS_DIR


def train_model(task: str, data_dir: str | Path, epochs: int, imgsz: int, batch_size: int, model_name: str | None = None) -> str:
    """Train a YOLO detection model for either mask or helmet detection."""
    task = task.lower()
    if task not in {"mask", "helmet"}:
        raise ValueError("task must be either 'mask' or 'helmet'")

    class_names = MASK_CLASSES if task == "mask" else ["helmet", "no_helmet"]
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_path}")

    yaml_path = data_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Expected dataset config at {yaml_path}")

    model_path = model_name or ("yolov8n.pt" if task == "mask" else "yolov8n.pt")
    model = YOLO(model_path)

    result = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=str(MODELS_DIR),
        name=f"{task}_detector",
        pretrained=True,
    )

    best_model = Path(result.save_dir) / "weights" / "best.pt"
    return str(best_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the face mask/helmet detector with transfer learning")
    parser.add_argument("--task", choices=["mask", "helmet"], default="mask", help="Detection task to train")
    parser.add_argument("--data", type=str, default=str(DATA_DIR / "mask"), help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--model-name", type=str, default=None, help="Optional pretrained model name")
    args = parser.parse_args()

    model_path = train_model(
        task=args.task,
        data_dir=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
    print(f"Training complete. Best model saved to: {model_path}")


if __name__ == "__main__":
    main()
