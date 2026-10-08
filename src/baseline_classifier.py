from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image


DEFAULT_IMAGE_SIZE = 160


def build_transforms(image_size: int = DEFAULT_IMAGE_SIZE):
    return build_transform_pipeline(image_size)


def build_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE):
    training_augmentation = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
    ]
    return build_transform_pipeline(image_size, training_augmentation)


def build_transform_pipeline(image_size: int, augmentation: list | None = None):
    operations = [transforms.Resize((image_size, image_size))]
    if augmentation:
        operations.extend(augmentation)
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(
        operations
    )


def build_evaluation_transforms(image_size: int = DEFAULT_IMAGE_SIZE):
    return build_transform_pipeline(image_size)


def get_dataset(task: str, data_root: str | Path, image_size: int = DEFAULT_IMAGE_SIZE):
    root = Path(data_root) / task
    split_datasets = {}
    for split in ("train", "val", "test"):
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing {split} split at {split_root}")
        transform = build_train_transforms(image_size) if split == "train" else build_evaluation_transforms(image_size)
        dataset = datasets.ImageFolder(root=str(split_root), transform=transform)
        if not dataset:
            raise ValueError(f"No images were found under {split_root}")
        split_datasets[split] = dataset

    expected_classes = split_datasets["train"].classes
    for split, dataset in split_datasets.items():
        if dataset.classes != expected_classes:
            raise ValueError(f"Class folders in {split} do not match the train split: {dataset.classes}")
    return split_datasets


def train_baseline_classifier(
    task: str,
    data_root: str | Path,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = DEFAULT_IMAGE_SIZE,
    pretrained: bool = True,
):
    split_datasets = get_dataset(task, data_root, image_size)
    class_names = split_datasets["train"].classes
    random.seed(42)
    torch.manual_seed(42)
    train_loader = DataLoader(split_datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(split_datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(split_datasets["test"], batch_size=batch_size, shuffle=False)

    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_targets = split_datasets["train"].targets
    class_counts = torch.bincount(torch.tensor(train_targets), minlength=len(class_names)).float()
    if (class_counts == 0).any():
        raise ValueError("The training split must contain examples from every class")
    class_weights = (len(train_targets) / (len(class_names) * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    warmup_epochs = min(3, max(1, epochs // 3)) if pretrained else 0
    if pretrained:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    best_state = None
    best_val_loss = float("inf")
    optimizer_parameters = model.classifier.parameters() if pretrained else model.parameters()
    optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate)

    for epoch in range(epochs):
        if pretrained and epoch == warmup_epochs:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * 0.1)
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(split_datasets["train"])
        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(split_datasets["val"])
        val_accuracy = correct / total
        print(
            f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model_variant = "mobilenetv2" if pretrained else "mobilenetv2_scratch"
    save_dir = Path("models") / f"{task}_{model_variant}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    torch.save(model.state_dict(), save_dir / "best_model.pth")

    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    confusion = [[0] * len(class_names) for _ in class_names]
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            for truth, prediction in zip(labels.tolist(), predictions.tolist()):
                class_total[truth] += 1
                class_correct[truth] += int(truth == prediction)
                confusion[truth][prediction] += 1
    test_accuracy = correct / total

    per_class_recall = {
        name: (class_correct[index] / class_total[index] if class_total[index] else None)
        for index, name in enumerate(class_names)
    }
    per_class_precision = {
        name: (
            class_correct[index] / sum(confusion[truth][index] for truth in range(len(class_names)))
            if sum(confusion[truth][index] for truth in range(len(class_names)))
            else 0.0
        )
        for index, name in enumerate(class_names)
    }
    per_class_f1 = {
        name: (
            2 * per_class_precision[name] * per_class_recall[name]
            / (per_class_precision[name] + per_class_recall[name])
            if per_class_precision[name] + per_class_recall[name]
            else 0.0
        )
        for name in class_names
    }
    metadata = {
        "class_names": class_names,
        "task": task,
        "pretrained": pretrained,
        "image_size": image_size,
        "test_accuracy": test_accuracy,
        "test_confusion_matrix": confusion,
        "test_precision_by_class": per_class_precision,
        "test_recall_by_class": per_class_recall,
        "test_f1_by_class": per_class_f1,
    }
    with open(save_dir / "class_names.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test precision by class: {per_class_precision}")
    print(f"Test recall by class: {per_class_recall}")
    print(f"Test F1 by class: {per_class_f1}")
    print(f"Saved MobileNetV2 baseline to: {save_dir}")
    return str(save_dir / "best_model.pth")


def predict_image(model_path: str | Path, image_path: str | Path, task: str | None = None):
    model_path = Path(model_path)
    image_path = Path(image_path)
    labels_path = model_path.parent / "class_names.json"
    with open(labels_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    class_names = metadata["class_names"]
    image_size = metadata.get("image_size", DEFAULT_IMAGE_SIZE)

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = build_transforms(image_size)
    tensor = transform(Image.fromarray(image_rgb)).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        index = int(output.argmax(dim=1).item())
        confidence = float(torch.softmax(output, dim=1)[0, index].item())
    return class_names[index], confidence


def main():
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 transfer-learning baseline for mask or helmet classification")
    parser.add_argument("--task", choices=["mask", "helmet"], required=True)
    parser.add_argument("--data-root", type=str, default="data/classification", help="Root directory containing task train/val/test folders")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--scratch", action="store_true", help="Initialize MobileNetV2 randomly for the transfer-learning comparison")
    args = parser.parse_args()

    train_baseline_classifier(
        task=args.task,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        pretrained=not args.scratch,
    )


if __name__ == "__main__":
    main()
