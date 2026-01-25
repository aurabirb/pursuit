"""SQLite storage for fursuit detection metadata."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sam3_pursuit.config import Config


@dataclass
class Detection:
    id: Optional[int]
    post_id: str
    character_name: Optional[str]
    embedding_id: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    confidence: float
    segmentor_model: str = "unknown"
    created_at: Optional[datetime] = None
    source_filename: Optional[str] = None
    source_url: Optional[str] = None
    is_cropped: bool = False
    segmentation_concept: Optional[str] = None
    preprocessing_info: Optional[str] = None
    crop_path: Optional[str] = None


class Database:
    """SQLite database for detection metadata."""

    _SELECT_FIELDS = """
        id, post_id, character_name, embedding_id, bbox_x, bbox_y,
        bbox_width, bbox_height, confidence, segmentor_model, created_at,
        source_filename, source_url, is_cropped, segmentation_concept,
        preprocessing_info, crop_path
    """

    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT NOT NULL,
                character_name TEXT,
                embedding_id INTEGER UNIQUE NOT NULL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                confidence REAL DEFAULT 0.0,
                segmentor_model TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_filename TEXT,
                source_url TEXT,
                is_cropped INTEGER DEFAULT 0,
                segmentation_concept TEXT,
                preprocessing_info TEXT,
                crop_path TEXT
            )
        """)

        c.execute("CREATE INDEX IF NOT EXISTS idx_post_id ON detections(post_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_character_name ON detections(character_name)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_embedding_id ON detections(embedding_id)")

        # Add missing columns for existing databases
        c.execute("PRAGMA table_info(detections)")
        existing_columns = {row[1] for row in c.fetchall()}

        new_columns = [
            ("source_filename", "TEXT"),
            ("source_url", "TEXT"),
            ("is_cropped", "INTEGER DEFAULT 0"),
            ("segmentation_concept", "TEXT"),
            ("preprocessing_info", "TEXT"),
            ("crop_path", "TEXT"),
        ]

        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                c.execute(f"ALTER TABLE detections ADD COLUMN {col_name} {col_type}")

        conn.commit()
        conn.close()

    def add_detection(self, detection: Detection) -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            INSERT INTO detections
            (post_id, character_name, embedding_id, bbox_x, bbox_y, bbox_width, bbox_height,
             confidence, segmentor_model, source_filename, source_url, is_cropped,
             segmentation_concept, preprocessing_info, crop_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            detection.post_id,
            detection.character_name,
            detection.embedding_id,
            detection.bbox_x,
            detection.bbox_y,
            detection.bbox_width,
            detection.bbox_height,
            detection.confidence,
            detection.segmentor_model,
            detection.source_filename,
            detection.source_url,
            1 if detection.is_cropped else 0,
            detection.segmentation_concept,
            detection.preprocessing_info,
            detection.crop_path
        ))

        row_id = c.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def _row_to_detection(self, row) -> Detection:
        return Detection(
            id=row[0],
            post_id=row[1],
            character_name=row[2],
            embedding_id=row[3],
            bbox_x=row[4],
            bbox_y=row[5],
            bbox_width=row[6],
            bbox_height=row[7],
            confidence=row[8],
            segmentor_model=row[9],
            created_at=row[10],
            source_filename=row[11] if len(row) > 11 else None,
            source_url=row[12] if len(row) > 12 else None,
            is_cropped=bool(row[13]) if len(row) > 13 else False,
            segmentation_concept=row[14] if len(row) > 14 else None,
            preprocessing_info=row[15] if len(row) > 15 else None,
            crop_path=row[16] if len(row) > 16 else None,
        )

    def get_detection_by_embedding_id(self, embedding_id: int) -> Optional[Detection]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(f"SELECT {self._SELECT_FIELDS} FROM detections WHERE embedding_id = ?", (embedding_id,))
        row = c.fetchone()
        conn.close()
        return self._row_to_detection(row) if row else None

    def get_detection_by_id(self, detection_id: int) -> Optional[Detection]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(f"SELECT {self._SELECT_FIELDS} FROM detections WHERE id = ?", (detection_id,))
        row = c.fetchone()
        conn.close()
        return self._row_to_detection(row) if row else None

    def get_detections_by_post_id(self, post_id: str) -> list[Detection]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(f"SELECT {self._SELECT_FIELDS} FROM detections WHERE post_id = ?", (post_id,))
        rows = c.fetchall()
        conn.close()
        return [self._row_to_detection(row) for row in rows]

    def get_detections_by_character(self, character_name: str) -> list[Detection]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(f"SELECT {self._SELECT_FIELDS} FROM detections WHERE character_name = ?", (character_name,))
        rows = c.fetchall()
        conn.close()
        return [self._row_to_detection(row) for row in rows]

    def get_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM detections")
        total = c.fetchone()[0]

        c.execute("SELECT COUNT(DISTINCT character_name) FROM detections WHERE character_name IS NOT NULL")
        unique_chars = c.fetchone()[0]

        c.execute("SELECT COUNT(DISTINCT post_id) FROM detections")
        unique_posts = c.fetchone()[0]

        c.execute("""
            SELECT character_name, COUNT(*) as count FROM detections
            WHERE character_name IS NOT NULL
            GROUP BY character_name ORDER BY count DESC LIMIT 10
        """)
        top_chars = c.fetchall()

        c.execute("""
            SELECT segmentor_model, COUNT(*) as count FROM detections
            GROUP BY segmentor_model ORDER BY count DESC
        """)
        segmentor_breakdown = dict(c.fetchall())

        conn.close()

        return {
            "total_detections": total,
            "unique_characters": unique_chars,
            "unique_posts": unique_posts,
            "top_characters": top_chars,
            "segmentor_breakdown": segmentor_breakdown
        }

    def has_post(self, post_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT 1 FROM detections WHERE post_id = ? LIMIT 1", (post_id,))
        exists = c.fetchone() is not None
        conn.close()
        return exists

    def get_next_embedding_id(self) -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT MAX(embedding_id) FROM detections")
        result = c.fetchone()[0]
        conn.close()
        return 0 if result is None else result + 1
