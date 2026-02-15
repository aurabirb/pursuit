"""Identification tracking database for measuring real-world performance."""

import json
import os
import sqlite3
import threading
from typing import Optional

from sam3_pursuit.config import Config
from sam3_pursuit.storage.database import retry_on_locked, get_git_version


class IdentificationTracker:
    """Tracks identification requests, matches, and user feedback."""

    _TIMEOUT = 10.0
    _BUSY_TIMEOUT_MS = 15000

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(Config.BASE_DIR, "tracking.db")
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=self._TIMEOUT)
            conn.execute(f"PRAGMA busy_timeout = {self._BUSY_TIMEOUT_MS}")
            self._local.conn = conn
        return conn

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _init_database(self):
        conn = self._connect()
        c = conn.cursor()
        c.execute("PRAGMA journal_mode=WAL")

        c.execute("""
            CREATE TABLE IF NOT EXISTS identification_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_user_id INTEGER,
                telegram_username TEXT,
                telegram_chat_id INTEGER,
                telegram_message_id INTEGER,
                image_path TEXT,
                image_width INTEGER,
                image_height INTEGER,
                num_segments INTEGER,
                num_datasets INTEGER,
                dataset_names TEXT,
                git_version TEXT,
                segmentor_model TEXT,
                segmentor_concept TEXT,
                processing_time_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS identification_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                segment_index INTEGER,
                segment_bbox TEXT,
                segment_confidence REAL,
                dataset_name TEXT,
                embedder TEXT,
                merge_strategy TEXT,
                character_name TEXT,
                match_confidence REAL,
                match_distance REAL,
                matched_post_id TEXT,
                matched_source TEXT,
                rank_in_dataset INTEGER,
                rank_after_merge INTEGER,
                FOREIGN KEY (request_id) REFERENCES identification_requests(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS identification_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                segment_index INTEGER,
                character_name TEXT,
                is_correct INTEGER,
                correct_character TEXT,
                telegram_user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (request_id) REFERENCES identification_requests(id)
            )
        """)

        c.execute("CREATE INDEX IF NOT EXISTS idx_matches_request ON identification_matches(request_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_request ON identification_feedback(request_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_requests_user ON identification_requests(telegram_user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_requests_chat ON identification_requests(telegram_chat_id)")

        conn.commit()

    @retry_on_locked()
    def log_request(
        self,
        telegram_user_id: Optional[int] = None,
        telegram_username: Optional[str] = None,
        telegram_chat_id: Optional[int] = None,
        telegram_message_id: Optional[int] = None,
        image_path: Optional[str] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        num_segments: int = 0,
        num_datasets: int = 0,
        dataset_names: Optional[str] = None,
        git_version: Optional[str] = None,
        segmentor_model: Optional[str] = None,
        segmentor_concept: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
    ) -> int:
        """Log an identification request. Returns the request_id."""
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """INSERT INTO identification_requests
            (telegram_user_id, telegram_username, telegram_chat_id, telegram_message_id,
             image_path, image_width, image_height, num_segments, num_datasets,
             dataset_names, git_version, segmentor_model, segmentor_concept, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                telegram_user_id, telegram_username, telegram_chat_id, telegram_message_id,
                image_path, image_width, image_height, num_segments, num_datasets,
                dataset_names, git_version or get_git_version(), segmentor_model,
                segmentor_concept, processing_time_ms,
            ),
        )
        conn.commit()
        return c.lastrowid

    @retry_on_locked()
    def log_matches(self, request_id: int, matches_data: list[dict]):
        """Batch insert match records for a request."""
        if not matches_data:
            return
        conn = self._connect()
        c = conn.cursor()
        c.executemany(
            """INSERT INTO identification_matches
            (request_id, segment_index, segment_bbox, segment_confidence,
             dataset_name, embedder, merge_strategy, character_name,
             match_confidence, match_distance, matched_post_id, matched_source,
             rank_in_dataset, rank_after_merge)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    request_id,
                    m.get("segment_index"),
                    json.dumps(m["segment_bbox"]) if m.get("segment_bbox") else None,
                    m.get("segment_confidence"),
                    m.get("dataset_name"),
                    m.get("embedder"),
                    m.get("merge_strategy"),
                    m.get("character_name"),
                    m.get("match_confidence"),
                    m.get("match_distance"),
                    m.get("matched_post_id"),
                    m.get("matched_source"),
                    m.get("rank_in_dataset"),
                    m.get("rank_after_merge"),
                )
                for m in matches_data
            ],
        )
        conn.commit()

    @retry_on_locked()
    def add_feedback(
        self,
        request_id: int,
        segment_index: int,
        character_name: str,
        is_correct: Optional[bool],
        correct_character: Optional[str] = None,
        telegram_user_id: Optional[int] = None,
    ):
        """Record user feedback on a match."""
        conn = self._connect()
        c = conn.cursor()
        is_correct_int = None if is_correct is None else (1 if is_correct else 0)
        c.execute(
            """INSERT INTO identification_feedback
            (request_id, segment_index, character_name, is_correct,
             correct_character, telegram_user_id)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (request_id, segment_index, character_name, is_correct_int,
             correct_character, telegram_user_id),
        )
        conn.commit()
        return c.lastrowid
