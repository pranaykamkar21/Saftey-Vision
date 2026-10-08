from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
import re
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import cv2
import numpy as np

from src.event_store import EventStore
from src.opencv_inference import ComplianceDetector, open_default_model


ROOT = Path(__file__).resolve().parent
PUBLIC_ROOT = ROOT / "public"
LOCAL_DATA = ROOT / ".local-dev"
MAX_REQUEST_BYTES = 5 * 1024 * 1024
MAX_IMAGE_BYTES = 3 * 1024 * 1024
VALID_TASKS = {"mask", "helmet"}
VALID_CLASSES = {
    "mask": {"with_mask", "without_mask", "mask_worn_incorrectly"},
    "helmet": {"helmet", "no_helmet"},
}
detectors: dict[str, ComplianceDetector] = {}
detector_lock = threading.Lock()
event_store = EventStore(
    database_path=LOCAL_DATA / "events.sqlite3",
    evidence_dir=LOCAL_DATA / "evidence",
)


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def event_payload(event: dict) -> dict:
    result = {key: value for key, value in event.items() if key != "evidence_path"}
    result["evidence_url"] = f"/api/evidence?id={quote(event['id'])}"
    return result


class LocalDashboardHandler(BaseHTTPRequestHandler):
    server_version = "SafetyVisionLocal/1.0"

    def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        self._send(status, "application/json; charset=utf-8", json_bytes(payload))

    def _read_json(self, max_bytes: int = MAX_REQUEST_BYTES) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Content-Length must be a valid number.") from error
        if length <= 0 or length > max_bytes:
            raise ValueError(f"Request body must be between 1 byte and {max_bytes} bytes.")
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("Request body must contain valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("Request JSON must be an object.")
        return payload

    def _image_from_data_url(self, value: object, max_bytes: int) -> np.ndarray:
        match = re.fullmatch(r"data:image/(?:jpeg|png|webp);base64,([A-Za-z0-9+/]+=*)", value) if isinstance(value, str) else None
        if not match:
            raise ValueError("image must be a base64 JPEG, PNG or WebP data URL.")
        encoded = match.group(1)
        if len(encoded) > (max_bytes * 4 // 3) + 8:
            raise ValueError("Image exceeds the configured size limit.")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except binascii.Error as error:
            raise ValueError("Image data is not valid base64.") from error
        if not image_bytes or len(image_bytes) > max_bytes:
            raise ValueError(f"Image must be smaller than {max_bytes // (1024 * 1024)} MB.")
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("The uploaded image could not be decoded.")
        return image

    def _get_detector(self, task: str) -> ComplianceDetector:
        with detector_lock:
            if task not in detectors:
                detectors[task] = ComplianceDetector(task, open_default_model(task))
            return detectors[task]

    def _handle_analysis(self) -> None:
        try:
            api_key = os.environ.get("INFERENCE_API_KEY")
            if os.environ.get("SAFETY_REQUIRE_API_KEY") == "true" and not api_key:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Inference API authentication is not configured."})
                return
            if api_key and not secrets.compare_digest(
                self.headers.get("Authorization", ""),
                f"Bearer {api_key}",
            ):
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Invalid inference API key."})
                return
            body = self._read_json()
            task = body.get("task")
            if task not in VALID_TASKS:
                raise ValueError("task must be mask or helmet.")
            frame = self._image_from_data_url(body.get("image"), MAX_IMAGE_BYTES)
            height, width = frame.shape[:2]
            detector = self._get_detector(task)
            with detector_lock:
                _, predictions = detector.predict_frame(frame)
                predictions = [*predictions, *detector.last_review_predictions]
            output = []
            for prediction in predictions:
                x, y, box_width, box_height = prediction.box
                output.append({
                    "label": prediction.label,
                    "confidence": prediction.confidence,
                    "box": [x / width, y / height, box_width / width, box_height / height],
                })
            self._send_json(HTTPStatus.OK, {"predictions": output})
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self.log_error("Local inference failed: %s", error)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Local inference failed: {error}"})

    def _handle_events_get(self, query: dict[str, list[str]]) -> None:
        task_values = query.get("task", [])
        task = task_values[0] if task_values else None
        if task not in {None, *VALID_TASKS}:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "task filter must be mask or helmet."})
            return
        events = [event_payload(event) for event in event_store.recent_events(task=task)]
        self._send_json(HTTPStatus.OK, {"events": events})

    def _handle_event_create(self) -> None:
        try:
            body = self._read_json(2 * 1024 * 1024)
            task = body.get("task")
            class_name = body.get("class_name")
            confidence = body.get("confidence")
            source = body.get("source")
            if task not in VALID_TASKS:
                raise ValueError("task must be mask or helmet.")
            if class_name not in VALID_CLASSES[task]:
                raise ValueError("Unsupported violation class.")
            if (task == "mask") != (class_name != "no_helmet"):
                raise ValueError("The violation class does not match the selected profile.")
            if not isinstance(confidence, (int, float)) or not 0.95 <= confidence <= 1:
                raise ValueError("Only violations at or above 95% confidence can be saved.")
            if not isinstance(source, str) or len(source) > 200:
                raise ValueError("source must be a string of at most 200 characters.")
            image = self._image_from_data_url(body.get("evidence"), 1024 * 1024)
            evidence_kind = "face_crop" if task == "mask" else "head_crop"
            event_store.save_evidence_crops(
                task,
                [(class_name, float(confidence), image)],
                source,
                evidence_kind,
            )
            event = event_store.recent_events(limit=1, task=task)[0]
            self._send_json(HTTPStatus.CREATED, {"event": event_payload(event)})
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self.log_error("Local evidence save failed: %s", error)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Local evidence save failed: {error}"})

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        query = parse_qs(request.query)
        if request.path == "/api/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "mode": "local-model"})
            return
        if request.path == "/api/events":
            self._handle_events_get(query)
            return
        if request.path == "/api/evidence":
            event_id = query.get("id", [""])[0]
            event = next((item for item in event_store.recent_events() if item["id"] == event_id), None)
            if event is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Evidence record not found."})
                return
            evidence_path = event_store.resolve_evidence_path(event["evidence_path"])
            try:
                evidence = evidence_path.read_bytes()
            except OSError as error:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Could not read local evidence: {error}"})
                return
            self._send(HTTPStatus.OK, "image/jpeg", evidence)
            return
        relative_path = "index.html" if request.path == "/" else request.path.lstrip("/")
        file_path = (PUBLIC_ROOT / relative_path).resolve()
        if PUBLIC_ROOT.resolve() not in file_path.parents or not file_path.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, content_type, file_path.read_bytes())

    def do_POST(self) -> None:
        request_path = urlsplit(self.path).path
        if request_path == "/api/analyze":
            self._handle_analysis()
        elif request_path == "/api/events":
            self._handle_event_create()
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_DELETE(self) -> None:
        request = urlsplit(self.path)
        if request.path != "/api/events":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        event_id = parse_qs(request.query).get("id", [""])[0]
        if not event_id:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "An event id is required."})
            return
        if not event_store.delete_event(event_id):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Evidence record not found."})
            return
        self._send_json(HTTPStatus.OK, {"ok": True})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[local] {self.address_string()} - {format % args}")


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "4173"))
    server = ThreadingHTTPServer((host, port), LocalDashboardHandler)
    print(f"Local model dashboard: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
