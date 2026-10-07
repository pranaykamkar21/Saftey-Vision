from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.event_store import EventStore
from src.opencv_inference import ComplianceDetector, EvidenceCapture, open_default_model, process_video


def run_webcam(
    task: str,
    model_path: str | Path,
    camera_index: int,
    evidence_store: EventStore | None,
) -> None:
    detector = ComplianceDetector(task, model_path, 0.95)
    evidence_capture = EvidenceCapture(detector, evidence_store) if evidence_store else None
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    totals = {"compliant": 0, "non_compliant": 0}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Webcam stopped returning frames")
            annotated, predictions = detector.predict_frame(frame)
            counts = detector.count_compliance(predictions)
            totals["compliant"] += counts["compliant"]
            totals["non_compliant"] += counts["non_compliant"]
            if evidence_capture:
                evidence_capture.capture(
                    [*predictions, *detector.last_review_predictions],
                    frame,
                    f"webcam:{camera_index}",
                )
            status = f"Compliant {counts['compliant']} | Violations {counts['non_compliant']} | q: quit"
            cv2.putText(annotated, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow(f"{task.title()} compliance - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print(f"Detection events: compliant={totals['compliant']}, non_compliant={totals['non_compliant']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenCV image, video, or live webcam compliance inference")
    parser.add_argument("--task", choices=["mask", "helmet"], default="mask")
    parser.add_argument("--model", default=None, help="Path to MobileNetV2 .pth checkpoint (defaults to the trained task model)")
    parser.add_argument("--source", default="0", help="Image/video path or webcam index (default: 0)")
    parser.add_argument("--output", default=None, help="Annotated video output path (video sources only)")
    parser.add_argument("--no-save-evidence", action="store_true", help="Disable local violation photos and SQLite event records")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else open_default_model(args.task)
    evidence_store = None if args.no_save_evidence else EventStore()
    if args.source.isdigit():
        run_webcam(args.task, model_path, int(args.source), evidence_store)
        return

    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    detector = ComplianceDetector(args.task, model_path, 0.95)
    if source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        destination = args.output or str(source.with_name(f"{source.stem}_annotated.mp4"))
        counts, frame_count = process_video(detector, source, destination, evidence_store)
        print(f"Processed {frame_count} frames; counts={counts}; output={destination}")
    else:
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Could not read image: {source}")
        annotated, predictions = detector.predict_frame(frame)
        destination = Path(args.output) if args.output else source.with_name(f"{source.stem}_annotated{source.suffix}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), annotated):
            raise RuntimeError(f"Could not save annotated image: {destination}")
        if evidence_store:
            EvidenceCapture(detector, evidence_store).capture(
                [*predictions, *detector.last_review_predictions],
                frame,
                "image file",
            )
        print(f"counts={detector.count_compliance(predictions)}; output={destination}")


if __name__ == "__main__":
    main()
