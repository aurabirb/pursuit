"""SQLite database operations for fursuit detection storage."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sam3_pursuit.config import Config


@dataclass
class Detection:
    """Represents a fursuit detection record."""
    id: Optional[int]
    post_id: str
    character_name: Optional[str]
    embedding_id: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    confidence: float
    segmentor_model: str = "unknown"  # Track which segmentor was used
    created_at: Optional[datetime] = None
    # New fields for source tracking and debugging
    source_filename: Optional[str] = None  # Original filename
    source_url: Optional[str] = None  # Source URL if downloaded
    is_cropped: bool = False  # Whether this is a segmented crop vs full image
    segmentation_concept: Optional[str] = None  # SAM3 concept used (e.g., "fursuiter")
    preprocessing_info: Optional[str] = None  # JSON with preprocessing details
    crop_path: Optional[str] = None  # Path to saved crop for debugging


class Database:
    """SQLite database for storing fursuit detection metadata."""

    def __init__(self, db_path: str = Config.DB_PATH):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
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
        c.execute("CREATE INDEX IF NOT EXISTS idx_segmentor_model ON detections(segmentor_model)")

        # Add missing columns for existing databases (non-destructive)
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
        """Add a detection record to the database.

        Args:
            detection: Detection object to add.

        Returns:
            The row ID of the inserted record.
        """
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

    def add_detections_batch(self, detections: list[Detection]) -> list[int]:
        """Add multiple detection records in a batch.

        Args:
            detections: List of Detection objects to add.

        Returns:
            List of row IDs for inserted records.
        """
        if not detections:
            return []

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        row_ids = []
        for detection in detections:
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
            row_ids.append(c.lastrowid)

        conn.commit()
        conn.close()

        return row_ids

    def _row_to_detection(self, row) -> Detection:
        """Convert a database row to a Detection object."""
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

    _SELECT_FIELDS = """
        id, post_id, character_name, embedding_id, bbox_x, bbox_y,
        bbox_width, bbox_height, confidence, segmentor_model, created_at,
        source_filename, source_url, is_cropped, segmentation_concept,
        preprocessing_info, crop_path
    """

    def get_detection_by_embedding_id(self, embedding_id: int) -> Optional[Detection]:
        """Get a detection by its embedding ID.

        Args:
            embedding_id: The FAISS index ID.

        Returns:
            Detection object or None if not found.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute(f"""
            SELECT {self._SELECT_FIELDS}
            FROM detections WHERE embedding_id = ?
        """, (embedding_id,))

        row = c.fetchone()
        conn.close()

        if row:
            return self._row_to_detection(row)
        return None

    def get_detection_by_id(self, detection_id: int) -> Optional[Detection]:
        """Get a detection by its database row ID.

        Args:
            detection_id: The database row ID.

        Returns:
            Detection object or None if not found.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute(f"""
            SELECT {self._SELECT_FIELDS}
            FROM detections WHERE id = ?
        """, (detection_id,))

        row = c.fetchone()
        conn.close()

        if row:
            return self._row_to_detection(row)
        return None

    def get_detections_by_post_id(self, post_id: str) -> list[Detection]:
        """Get all detections for a post.

        Args:
            post_id: The FurTrack post ID.

        Returns:
            List of Detection objects.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute(f"""
            SELECT {self._SELECT_FIELDS}
            FROM detections WHERE post_id = ?
        """, (post_id,))

        rows = c.fetchall()
        conn.close()

        return [self._row_to_detection(row) for row in rows]

    def get_detections_by_character(self, character_name: str) -> list[Detection]:
        """Get all detections for a character.

        Args:
            character_name: The character name to search for.

        Returns:
            List of Detection objects.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute(f"""
            SELECT {self._SELECT_FIELDS}
            FROM detections WHERE character_name = ?
        """, (character_name,))

        rows = c.fetchall()
        conn.close()

        return [self._row_to_detection(row) for row in rows]

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with total_detections, unique_characters, unique_posts, and segmentor_breakdown.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM detections")
        total_detections = c.fetchone()[0]

        c.execute("SELECT COUNT(DISTINCT character_name) FROM detections WHERE character_name IS NOT NULL")
        unique_characters = c.fetchone()[0]

        c.execute("SELECT COUNT(DISTINCT post_id) FROM detections")
        unique_posts = c.fetchone()[0]

        c.execute("""
            SELECT character_name, COUNT(*) as count
            FROM detections
            WHERE character_name IS NOT NULL
            GROUP BY character_name
            ORDER BY count DESC
            LIMIT 10
        """)
        top_characters = c.fetchall()

        # Segmentor model breakdown
        c.execute("""
            SELECT segmentor_model, COUNT(*) as count
            FROM detections
            GROUP BY segmentor_model
            ORDER BY count DESC
        """)
        segmentor_breakdown = dict(c.fetchall())

        conn.close()

        return {
            "total_detections": total_detections,
            "unique_characters": unique_characters,
            "unique_posts": unique_posts,
            "top_characters": top_characters,
            "segmentor_breakdown": segmentor_breakdown
        }

    def get_embedding_ids_by_segmentor(self, segmentor_model: str) -> list[int]:
        """Get all embedding IDs for a specific segmentor model.

        Args:
            segmentor_model: The segmentor model name (e.g., 'sam2.1_s', 'sam3').

        Returns:
            List of embedding IDs indexed with that segmentor.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute(
            "SELECT embedding_id FROM detections WHERE segmentor_model = ?",
            (segmentor_model,)
        )
        ids = [row[0] for row in c.fetchall()]

        conn.close()
        return ids

    def has_post(self, post_id: str) -> bool:
        """Check if a post has already been indexed.

        Args:
            post_id: The FurTrack post ID.

        Returns:
            True if the post exists in the database.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT 1 FROM detections WHERE post_id = ? LIMIT 1", (post_id,))
        exists = c.fetchone() is not None

        conn.close()
        return exists

    def get_next_embedding_id(self) -> int:
        """Get the next available embedding ID.

        Returns:
            The next embedding ID to use.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT MAX(embedding_id) FROM detections")
        result = c.fetchone()[0]

        conn.close()

        return 0 if result is None else result + 1
