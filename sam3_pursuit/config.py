"""Configuration for SAM3 fursuit recognition."""

import os
import torch


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # File paths
    DEFAULT_DB_NAME = "pursuit.db"
    DEFAULT_INDEX_NAME = "pursuit.index"
    DEFAULT_CROPS_DIR = "pursuit_crops"

    DB_PATH = os.path.join(BASE_DIR, DEFAULT_DB_NAME)
    INDEX_PATH = os.path.join(BASE_DIR, DEFAULT_INDEX_NAME)
    CROPS_DIR = os.path.join(BASE_DIR, DEFAULT_CROPS_DIR)
    IMAGES_DIR = os.path.join(BASE_DIR, "furtrack_images")

    # Models
    SAM3_MODEL = "sam3"
    DINOV2_MODEL = "facebook/dinov2-base"
    EMBEDDING_DIM = 768

    # Detection
    DETECTION_CONFIDENCE = 0.5
    MAX_DETECTIONS = 10
    DEFAULT_CONCEPT = "fursuiter"

    # Search
    DEFAULT_TOP_K = 5
    HNSW_M = 32
    HNSW_EF_CONSTRUCTION = 200
    HNSW_EF_SEARCH = 50

    # Batch processing
    DEFAULT_BATCH_SIZE = 16

    @staticmethod
    def get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['FAISS_OPT_LEVEL'] = ''
            return "mps"
        return "cpu"

    @classmethod
    def get_absolute_path(cls, relative_path: str) -> str:
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(cls.BASE_DIR, relative_path)
