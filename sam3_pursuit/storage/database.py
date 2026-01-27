import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, wraps
from typing import Optional

from sam3_pursuit.config import Config


def retry_on_locked(max_retries: int = 8, base_delay: float = 0.2):
    """Retry on 'database is locked' with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e):
                        last_error = e
                        time.sleep(base_delay * (2 ** attempt))
                    else:
                        raise
            raise last_error
        return wrapper
    return decorator


@lru_cache(maxsize=1)
def get_git_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Config.BASE_DIR,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


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
    git_version: Optional[str] = None


class Database:
    _SELECT_FIELDS = """
        id, post_id, character_name, embedding_id, bbox_x, bbox_y,
        bbox_width, bbox_height, confidence, segmentor_model, created_at,
        source_filename, source_url, is_cropped, segmentation_concept,
        preprocessing_info, crop_path, git_version
    """
    _TIMEOUT = 10.0
    _BUSY_TIMEOUT_MS = 15000

    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_database()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=self._TIMEOUT)
            self._conn.execute(f"PRAGMA busy_timeout = {self._BUSY_TIMEOUT_MS}")
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _init_database(self):
        conn = self._connect()
        c = conn.cursor()
        c.execute("PRAGMA journal_mode=WAL")
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
        c.execute("CREATE INDEX IF NOT EXISTS idx_post_preproc ON detections(post_id, preprocessing_info)")

        c.execute("PRAGMA table_info(detections)")
        existing_columns = {row[1] for row in c.fetchall()}
        new_columns = [
            ("source_filename", "TEXT"),
            ("source_url", "TEXT"),
            ("is_cropped", "INTEGER DEFAULT 0"),
            ("segmentation_concept", "TEXT"),
            ("preprocessing_info", "TEXT"),
            ("crop_path", "TEXT"),
            ("git_version", "TEXT"),
        ]
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                c.execute(f"ALTER TABLE detections ADD COLUMN {col_name} {col_type}")
        conn.commit()

    _INSERT_SQL = """
        INSERT INTO detections
        (post_id, character_name, embedding_id, bbox_x, bbox_y, bbox_width, bbox_height,
         confidence, segmentor_model, source_filename, source_url, is_cropped,
         segmentation_concept, preprocessing_info, crop_path, git_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def _detection_to_tuple(self, d: Detection) -> tuple:
        return (
            d.post_id, d.character_name, d.embedding_id,
            d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
            d.confidence, d.segmentor_model, d.source_filename, d.source_url,
            1 if d.is_cropped else 0, d.segmentation_concept,
            d.preprocessing_info, d.crop_path, d.git_version or get_git_version(),
        )

    @retry_on_locked()
    def add_detections_batch(self, detections: list[Detection]) -> list[int]:
        if not detections:
            return []
        conn = self._connect()
        c = conn.cursor()
        row_ids = []
        for d in detections:
            c.execute(self._INSERT_SQL, self._detection_to_tuple(d))
            row_ids.append(c.lastrowid)
        conn.commit()
        return row_ids

    @retry_on_locked()
    def add_detection(self, detection: Detection) -> int:
        return self.add_detections_batch([detection])[0]

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
            git_version=row[17] if len(row) > 17 else None,
        )

    @retry_on_locked()
    def get_detection_by_embedding_id(self, embedding_id: int) -> Optional[Detection]:
        conn = self._connect()
        c = conn.cursor()
        c.execute(f"SELECT {self._SELECT_FIELDS} FROM detections WHERE embedding_id = ?", (embedding_id,))
        row = c.fetchone()
        return self._row_to_detection(row) if row else None

    @retry_on_locked()
    def get_detection_by_id(self, detection_id: int) -> Optional[Detection]:
        conn = self._connect()
        c = conn.cursor()
        c.execute(f"SELECT {self._SELECT_FIELDS} FROM detections WHERE id = ?", (detection_id,))
        row = c.fetchone()
        return self._row_to_detection(row) if row else None

    @retry_on_locked()
    def get_detections_by_post_id(self, post_id: str) -> list[Detection]:
        conn = self._connect()
        c = conn.cursor()
        c.execute(f"SELECT {self._SELECT_FIELDS} FROM detections WHERE post_id = ?", (post_id,))
        rows = c.fetchall()
        return [self._row_to_detection(row) for row in rows]

    @retry_on_locked()
    def get_detections_by_character(self, character_name: str) -> list[Detection]:
        conn = self._connect()
        c = conn.cursor()
        c.execute(f"SELECT {self._SELECT_FIELDS} FROM detections WHERE character_name = ?", (character_name,))
        rows = c.fetchall()
        return [self._row_to_detection(row) for row in rows]

    @retry_on_locked()
    def get_stats(self) -> dict:
        conn = self._connect()
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

        c.execute("""
            SELECT preprocessing_info, COUNT(*) as count FROM detections
            WHERE preprocessing_info IS NOT NULL
            GROUP BY preprocessing_info ORDER BY count DESC LIMIT 10
        """)
        preprocessing_breakdown = dict(c.fetchall())

        c.execute("""
            SELECT git_version, COUNT(*) as count FROM detections
            GROUP BY git_version ORDER BY count DESC LIMIT 10
        """)
        git_version_breakdown = dict(c.fetchall())

        return {
            "total_detections": total,
            "unique_characters": unique_chars,
            "unique_posts": unique_posts,
            "top_characters": top_chars,
            "segmentor_breakdown": segmentor_breakdown,
            "preprocessing_breakdown": preprocessing_breakdown,
            "git_version_breakdown": git_version_breakdown,
        }

    @retry_on_locked()
    def has_post(self, post_id: str) -> bool:
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT 1 FROM detections WHERE post_id = ? LIMIT 1", (post_id,))
        exists = c.fetchone() is not None
        return exists

    @retry_on_locked()
    def get_posts_needing_update(self, post_ids: list[str], preprocessing_info: str) -> set[str]:
        """Return post_ids that don't have an entry with this preprocessing_info."""
        if not post_ids or not preprocessing_info:
            return set(post_ids)
        conn = self._connect()
        c = conn.cursor()
        placeholders = ",".join("?" * len(post_ids))
        c.execute(
            f"SELECT DISTINCT post_id FROM detections WHERE post_id IN ({placeholders}) AND preprocessing_info = ?",
            (*post_ids, preprocessing_info)
        )
        return set(post_ids) - {row[0] for row in c.fetchall()}

    @retry_on_locked()
    def get_next_embedding_id(self) -> int:
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT MAX(embedding_id) FROM detections")
        result = c.fetchone()[0]
        return 0 if result is None else result + 1

    @retry_on_locked()
    def delete_orphaned_detections(self, max_valid_embedding_id: int) -> int:
        """Delete detections with embedding_id > max_valid_embedding_id."""
        conn = self._connect()
        c = conn.cursor()
        c.execute("DELETE FROM detections WHERE embedding_id > ?", (max_valid_embedding_id,))
        deleted = c.rowcount
        conn.commit()
        return deleted
