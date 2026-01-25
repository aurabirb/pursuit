"""Configuration constants for the SAM3 fursuit recognition system."""

import os
import torch


class Config:
    """Configuration settings for the SAM3 system."""

    # Base paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Default file names
    DEFAULT_DB_NAME = "pursuit.db"
    DEFAULT_INDEX_NAME = "pursuit.index"
    DEFAULT_CROPS_DIR = "pursuit_crops"

    # Database paths
    DB_PATH = os.path.join(BASE_DIR, DEFAULT_DB_NAME)
    INDEX_PATH = os.path.join(BASE_DIR, DEFAULT_INDEX_NAME)
    CROPS_DIR = os.path.join(BASE_DIR, DEFAULT_CROPS_DIR)
    IMAGES_DIR = os.path.join(BASE_DIR, "furtrack_images")

    # Legacy paths (for reference)
    OLD_DB_PATH = os.path.join(BASE_DIR, "furtrack.db")
    LEGACY_DB_PATH = os.path.join(BASE_DIR, "furtrack_sam3.db")
    LEGACY_INDEX_PATH = os.path.join(BASE_DIR, "faiss_sam3.index")

    # Model settings
    SAM3_MODEL = "sam3"  # SAM3 with text prompts

    # DINOv2 for embeddings
    DINOV2_MODEL = "facebook/dinov2-base"  # DINOv2 model name
    EMBEDDING_DIM = 768  # DINOv2 base output dimension

    # Detection settings
    DETECTION_CONFIDENCE = 0.5  # Minimum confidence for detections
    MAX_DETECTIONS = 10  # Maximum detections per image

    # Default concept for fursuit detection (SAM3 text prompt)
    DEFAULT_CONCEPT = "fursuiter"

    # Search settings
    DEFAULT_TOP_K = 5
    HNSW_M = 32  # HNSW index parameter
    HNSW_EF_CONSTRUCTION = 200
    HNSW_EF_SEARCH = 50

    # Batch processing
    DEFAULT_BATCH_SIZE = 16

    # Device selection
    @staticmethod
    def get_device() -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            # Set environment variables for MacOS compatibility
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['FAISS_OPT_LEVEL'] = ''
            return "mps"
        return "cpu"

    @classmethod
    def get_absolute_path(cls, relative_path: str) -> str:
        """Convert relative path to absolute based on BASE_DIR."""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(cls.BASE_DIR, relative_path)
