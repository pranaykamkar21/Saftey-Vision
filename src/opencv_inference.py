from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

from src.baseline_classifier import build_transforms
from src.event_store import EventStore


@dataclass
class Prediction:
    box: tuple[int, int, int, int]
    label: str
    confidence: float
    confirmed: bool = True


class ComplianceDetector:
    """OpenCV proposals plus MobileNetV2 crop classification for one task."""

    def __init__(self, task: str, model_path: str | Path, confidence: float = 0.95):
        if task not in {"mask", "helmet"}:
            raise ValueError("task must be 'mask' or 'helmet'")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        self.task = task
        self.confidence = confidence
        self.model_path = Path(model_path)
        self.yolo_model = None
        self.model = None
        self.metadata = {}
        self.image_size = 160
        if self.model_path.suffix.lower() == ".pt":
            from ultralytics import YOLO

            self.yolo_model = YOLO(str(self.model_path))
            self.class_names = [self.yolo_model.names[index] for index in range(len(self.yolo_model.names))]
        else:
            with open(self.model_path.parent / "class_names.json", encoding="utf-8") as file:
                self.metadata = json.load(file)
            self.class_names = self.metadata["class_names"]
            self.image_size = self.metadata.get("image_size", 160)
            self.model = models.mobilenet_v2(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(self.class_names))
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu", weights_only=True))
            self.model.eval()

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise RuntimeError(f"OpenCV could not load face detector: {cascade_path}")
        self.person_detector = None
        if task == "helmet" and self.yolo_model is None:
            self.person_detector = cv2.HOGDescriptor()
            self.person_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.last_frame_stats = {"candidate_count": 0, "confident_count": 0, "uncertain_count": 0}
        self.last_review_predictions: list[Prediction] = []

    def _candidate_boxes(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = frame.shape[:2]
        assert self.face_detector is not None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_side = max(24, min(height, width) // 18)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.06,
            minNeighbors=4,
            minSize=(min_side, min_side),
        )
        candidates = []
        for x, y, w, h in faces:
            x, y, w, h = int(x), int(y), int(w), int(h)
            if self.task == "helmet":
                # Face proposals include space above the forehead where helmets sit.
                pad_x, pad_top, pad_bottom = int(w * 0.35), int(h * 0.75), int(h * 0.35)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_top)
                x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_bottom)
                candidates.append((x1, y1, x2 - x1, y2 - y1))
            else:
                candidates.append((x, y, w, h))

        if self.task == "helmet":
            assert self.person_detector is not None
            # HOG finds full upright pedestrians; the face detector above also
            # catches riders whose posture/bike prevents a full-person proposal.
            search_frame = frame
            scale = min(1.0, 960 / max(height, width))
            if scale < 1:
                search_frame = cv2.resize(
                    frame,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            people, _weights = self.person_detector.detectMultiScale(
                search_frame,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            inverse_scale = 1 / scale
            for x, y, w, h in people:
                x, y, w, h = (int(value * inverse_scale) for value in (x, y, w, h))
                head_h = max(30, int(h * 0.35))
                x1, y1 = max(0, x - int(w * 0.08)), max(0, y - int(head_h * 0.1))
                x2, y2 = min(width, x + w + int(w * 0.08)), min(height, y + head_h)
                if x2 > x1 and y2 > y1:
                    candidates.append((x1, y1, x2 - x1, y2 - y1))

        return _suppress_overlapping_boxes(candidates, threshold=0.55)

    def predict_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[Prediction]]:
        if frame is None or frame.size == 0:
            raise ValueError("Cannot infer on an empty frame")
        predictions = []
        self.last_review_predictions = []
        if self.yolo_model is not None:
            result = self.yolo_model.predict(frame, conf=0.01, verbose=False, device="cpu")[0]
            for detection in result.boxes:
                x1, y1, x2, y2 = (int(value) for value in detection.xyxy[0].tolist())
                box = (x1, y1, x2 - x1, y2 - y1)
                label = self.class_names[int(detection.cls.item())]
                score = float(detection.conf.item())
                if score >= self.confidence:
                    predictions.append(Prediction(box, label, score))
                elif score >= 0.25:
                    self.last_review_predictions.append(Prediction(box, label, score, confirmed=False))
        else:
            boxes = self._candidate_boxes(frame)
            crops = []
            valid_boxes = []
            for x, y, w, h in boxes:
                crop = frame[y : y + h, x : x + w]
                if crop.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(
                    build_transforms(self.image_size)(Image.fromarray(crop_rgb))
                )
                valid_boxes.append((x, y, w, h))

            if crops:
                batch = torch.stack(crops)
                with torch.inference_mode():
                    assert self.model is not None
                    probabilities = torch.softmax(self.model(batch), dim=1)
                for box, row in zip(valid_boxes, probabilities):
                    index = int(row.argmax().item())
                    score = float(row[index].item())
                    if score >= self.confidence:
                        predictions.append(Prediction(box, self.class_names[index], score))
                    elif score >= 0.25:
                        self.last_review_predictions.append(
                            Prediction(box, self.class_names[index], score, confirmed=False)
                        )

        self.last_frame_stats["confident_count"] = len(predictions)
        self.last_frame_stats["candidate_count"] = len(predictions) + len(self.last_review_predictions)
        self.last_frame_stats["uncertain_count"] = len(self.last_review_predictions)

        annotated = frame.copy()
        for prediction in [*predictions, *self.last_review_predictions]:
            x, y, w, h = prediction.box
            violation = self._is_violation(prediction.label)
            color = (0, 0, 255) if violation else (0, 200, 0)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3, cv2.LINE_AA)
            prefix = "REVIEW " if not prediction.confirmed else ""
            text = f"{prefix}{prediction.label.replace('_', ' ').upper()}  {prediction.confidence:.0%}"
            font_scale = max(0.55, min(0.85, w / 180))
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            label_top = max(0, y - text_height - baseline - 10)
            cv2.rectangle(annotated, (x, label_top), (x + text_width + 16, y), color, -1)
            cv2.putText(
                annotated,
                text,
                (x + 8, max(text_height + 2, y - baseline - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated, predictions

    def count_compliance(self, predictions: list[Prediction]) -> dict[str, int]:
        violations = sum(self._is_violation(p.label) for p in predictions)
        return {
            "compliant": len(predictions) - violations,
            "non_compliant": violations,
            "total": len(predictions),
        }

    def _is_violation(self, label: str) -> bool:
        if self.task == "mask":
            return label != "with_mask"
        return label == "no_helmet"


class EvidenceCapture:
    def __init__(
        self,
        detector: ComplianceDetector,
        store: EventStore,
        cooldown_seconds: float = 8.0,
        crop_padding: float = 0.2,
    ):
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        if not 0 <= crop_padding <= 1:
            raise ValueError("crop_padding must be between 0 and 1")
        self.detector = detector
        self.store = store
        self.cooldown_seconds = cooldown_seconds
        self.crop_padding = crop_padding
        self._tracks: list[dict] = []

    def capture(
        self,
        predictions: list[Prediction],
        original_frame: np.ndarray,
        source: str,
    ) -> int:
        now = time.monotonic()
        if original_frame is None or original_frame.size == 0:
            raise ValueError("Cannot capture evidence from an empty frame")

        self._tracks = [
            track
            for track in self._tracks
            if now - track["last_seen"] <= self.cooldown_seconds
        ]
        available_tracks = set(range(len(self._tracks)))
        accepted: list[tuple[Prediction, dict]] = []
        unique_predictions = _suppress_predictions(predictions)
        for prediction in unique_predictions:
            best_index = None
            best_score = 0.0
            for index in available_tracks:
                track = self._tracks[index]
                score = _track_match_score(prediction.box, track["box"])
                if score > best_score:
                    best_score, best_index = score, index

            if best_index is None:
                track = {
                    "box": prediction.box,
                    "last_seen": now,
                    "event_saved": False,
                }
                self._tracks.append(track)
                best_index = len(self._tracks) - 1
            else:
                available_tracks.remove(best_index)
                track = self._tracks[best_index]
                track["box"] = prediction.box
                track["last_seen"] = now

            if self.detector._is_violation(prediction.label) and not track["event_saved"]:
                accepted.append((prediction, track))

        if not accepted:
            return 0
        kind = "face_crop" if self.detector.task == "mask" else "head_crop"
        saved_ids = []
        grouped_entries: dict[bool, list[tuple[str, float, np.ndarray]]] = {
            True: [],
            False: [],
        }
        for prediction, _track in accepted:
            grouped_entries[prediction.confirmed].append(
                (
                    prediction.label,
                    prediction.confidence,
                    _make_evidence_crop(
                        original_frame,
                        prediction.box,
                        self.detector.task,
                        prediction.label,
                        prediction.confidence,
                        self.crop_padding,
                    ),
                )
            )
        for confirmed, entries in grouped_entries.items():
            if entries:
                evidence_source = source if confirmed else f"{source} (review candidate)"
                saved_ids.extend(
                    self.store.save_evidence_crops(
                        self.detector.task, entries, evidence_source, evidence_kind=kind
                    )
                )
        for _prediction, track in accepted:
            track["event_saved"] = True
        return len(saved_ids)


def process_video(
    detector: ComplianceDetector,
    source: str | Path,
    destination: str | Path,
    evidence_store: EventStore | None = None,
    capture_evidence: bool = True,
    source_label: str = "video",
) -> tuple[dict[str, int], int]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Video has invalid frame dimensions: {source}")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create annotated video: {destination}")

    counts = {
        "compliant": 0,
        "non_compliant": 0,
        "total": 0,
        "candidates": 0,
        "uncertain": 0,
    }
    frame_count = 0
    evidence_capture = EvidenceCapture(detector, evidence_store) if capture_evidence and evidence_store else None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated, predictions = detector.predict_frame(frame)
            frame_counts = detector.count_compliance(predictions)
            for key in counts:
                if key in frame_counts:
                    counts[key] += frame_counts[key]
            counts["candidates"] += detector.last_frame_stats["candidate_count"]
            counts["uncertain"] += detector.last_frame_stats["uncertain_count"]
            if evidence_capture is not None:
                evidence_capture.capture(
                    [*predictions, *detector.last_review_predictions], frame, source_label
                )
            writer.write(annotated)
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        raise ValueError(f"No readable frames in video: {source}")
    return counts, frame_count


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _track_match_score(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    overlap = _box_iou(first, second)
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    center_a = (ax + aw / 2, ay + ah / 2)
    center_b = (bx + bw / 2, by + bh / 2)
    distance = float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))
    size = max(aw, ah, bw, bh, 1)
    if overlap >= 0.12:
        return overlap
    if distance <= size * 0.65:
        return 0.12 * (1 - distance / (size * 0.65))
    return 0.0


def _suppress_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
    threshold: float = 0.45,
) -> list[tuple[int, int, int, int]]:
    selected: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if all(_box_iou(box, kept) < threshold for kept in selected):
            selected.append(box)
    return selected


def _suppress_predictions(
    predictions: list[Prediction],
    threshold: float = 0.3,
) -> list[Prediction]:
    selected: list[Prediction] = []
    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        if all(_box_iou(prediction.box, existing.box) < threshold for existing in selected):
            selected.append(prediction)
    return selected


def _make_evidence_crop(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    task: str,
    label: str,
    confidence: float,
    padding: float,
) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = box
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(frame_width, x + width + pad_x), min(frame_height, y + height + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Detected face/head crop is empty")

    minimum_side = 240
    scale = max(1.0, minimum_side / max(crop.shape[:2]))
    if scale > 1:
        crop = cv2.resize(
            crop,
            (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )
    header_height = 44
    evidence = np.zeros((crop.shape[0] + header_height, crop.shape[1], 3), dtype=np.uint8)
    evidence[:header_height] = (23, 35, 48)
    evidence[header_height:] = crop
    display_label = label.replace("_", " ").upper()
    cv2.putText(
        evidence,
        f"{task.upper()}  |  {display_label}  |  {confidence:.0%}",
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return evidence


def open_default_model(task: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    if task == "helmet":
        detector_runs = list((root / "models").glob("helmet_detector*/weights/best.pt"))
        if detector_runs:
            return max(detector_runs, key=lambda path: path.stat().st_mtime)
    model_path = root / "models" / f"{task}_mobilenetv2" / "best_model.pth"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained {task} model not found: {model_path}")
    return model_path
