from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "events.sqlite3"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
RETENTION_DAYS = 30


class EventStore:
    """Local SQLite event database with local photo evidence retention."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DB_PATH,
        evidence_dir: str | Path = EVIDENCE_DIR,
        retention_days: int = RETENTION_DAYS,
    ):
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        self.database_path = Path(database_path)
        self.evidence_dir = Path(evidence_dir)
        self.retention_days = retention_days
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.purge_expired()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    task TEXT NOT NULL CHECK (task IN ('mask', 'helmet')),
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    evidence_path TEXT NOT NULL,
                    source TEXT NOT NULL,
                    evidence_kind TEXT NOT NULL DEFAULT 'head_crop'
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(violations)").fetchall()
            }
            if "evidence_kind" not in columns:
                connection.execute(
                    "ALTER TABLE violations ADD COLUMN evidence_kind TEXT NOT NULL DEFAULT 'full_frame'"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp_utc DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_task ON violations(task, timestamp_utc DESC)"
            )
        self._purge_legacy_full_frames()

    def save_evidence_crops(
        self,
        task: str,
        violations: Iterable[tuple[str, float, np.ndarray]],
        source: str,
        evidence_kind: str,
    ) -> list[str]:
        entries = list(violations)
        if not entries:
            return []
        if task not in {"mask", "helmet"}:
            raise ValueError("task must be 'mask' or 'helmet'")
        if evidence_kind not in {"face_crop", "head_crop"}:
            raise ValueError("evidence_kind must be 'face_crop' or 'head_crop'")
        expected_kind = "face_crop" if task == "mask" else "head_crop"
        if evidence_kind != expected_kind:
            raise ValueError(f"{task} evidence must be stored as {expected_kind}")
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat()
        rows = []
        image_paths = []
        try:
            for entry in entries:
                class_name, confidence, image = entry
                if image is None or image.size == 0:
                    raise ValueError("Cannot save an empty face/head crop")
                relative_path = Path(now.strftime("%Y")) / now.strftime("%m") / f"{uuid.uuid4().hex}.jpg"
                image_path = self.evidence_dir / relative_path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                    raise OSError(                    f"Failed to write face/head evidence crop: {image_path}")
                image_paths.append(image_path)
                try:
                    stored_path = image_path.resolve().relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    stored_path = str(image_path.resolve())
                rows.append(
                    (
                        uuid.uuid4().hex,
                        timestamp,
                        task,
                        class_name,
                        float(confidence),
                        stored_path,
                        source[:500],
                        evidence_kind,
                    )
                )
            with self._connect() as connection:
                connection.executemany(
                    """
                    INSERT INTO violations
                    (id, timestamp_utc, task, class_name, confidence, evidence_path, source, evidence_kind)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except Exception:
            for image_path in image_paths:
                image_path.unlink(missing_ok=True)
            raise
        return [row[0] for row in rows]

    def _purge_legacy_full_frames(self) -> None:
        with self._connect() as connection:
            legacy_paths = [
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT evidence_path FROM violations WHERE evidence_kind = 'full_frame'"
                )
            ]
            connection.execute("DELETE FROM violations WHERE evidence_kind = 'full_frame'")
            remaining_paths = {
                row[0] for row in connection.execute("SELECT DISTINCT evidence_path FROM violations")
            }
        for stored_path in legacy_paths:
            if stored_path not in remaining_paths:
                self._remove_evidence(stored_path)

    def recent_events(self, limit: int = 100, task: str | None = None) -> list[dict]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if task not in {None, "mask", "helmet"}:
            raise ValueError("task must be 'mask', 'helmet', or None")
        query = "SELECT * FROM violations"
        parameters: tuple = ()
        if task:
            query += " WHERE task = ?"
            parameters = (task,)
        query += " ORDER BY timestamp_utc DESC LIMIT ?"
        parameters += (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def resolve_evidence_path(self, stored_path: str) -> Path:
        path = Path(stored_path)
        candidate = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if self.evidence_dir.resolve() not in candidate.parents:
            raise ValueError(f"Evidence path is outside the evidence folder: {stored_path}")
        return candidate

    def delete_event(self, event_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT evidence_path FROM violations WHERE id = ?", (event_id,)
            ).fetchone()
            if row is None:
                return False
            connection.execute("DELETE FROM violations WHERE id = ?", (event_id,))
            still_referenced = connection.execute(
                "SELECT 1 FROM violations WHERE evidence_path = ? LIMIT 1", (row["evidence_path"],)
            ).fetchone()
        if still_referenced is None:
            self._remove_evidence(row["evidence_path"])
        return True

    def summary(self) -> dict:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM violations WHERE timestamp_utc >= ?", (cutoff,)
            ).fetchone()[0]
            by_task = connection.execute(
                """
                SELECT task, COUNT(*) AS count
                FROM violations
                WHERE timestamp_utc >= ?
                GROUP BY task
                """,
                (cutoff,),
            ).fetchall()
            by_class = connection.execute(
                """
                SELECT class_name, COUNT(*) AS count
                FROM violations
                WHERE timestamp_utc >= ?
                GROUP BY class_name
                ORDER BY count DESC
                """,
                (cutoff,),
            ).fetchall()
        return {
            "total": total,
            "by_task": {row["task"]: row["count"] for row in by_task},
            "by_class": {row["class_name"]: row["count"] for row in by_class},
        }

    def purge_expired(self) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        with self._connect() as connection:
            expired_paths = [
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT evidence_path FROM violations WHERE timestamp_utc < ?", (cutoff,)
                )
            ]
            connection.execute("DELETE FROM violations WHERE timestamp_utc < ?", (cutoff,))

        referenced_paths = set()
        with self._connect() as connection:
            referenced_paths = {
                row[0] for row in connection.execute("SELECT DISTINCT evidence_path FROM violations")
            }
        for stored_path in expired_paths:
            if stored_path not in referenced_paths:
                self._remove_evidence(stored_path)
        return len(expired_paths)

    def _remove_evidence(self, stored_path: str) -> None:
        candidate = self.resolve_evidence_path(stored_path)
        candidate.unlink(missing_ok=True)
