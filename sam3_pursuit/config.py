"""Configuration for SAM3 fursuit recognition."""

import os
import torch


def sanitize_path_component(name: str) -> str:
    """Make a string safe for use as a single path component.

    Prevents path traversal by removing /, \\, null bytes, and .. sequences.
    """
    name = name.replace("/", "_").replace("\\", "_").replace("\0", "")
    name = name.replace("..", "_")
    name = name.strip(". ")
    name = name[:200]
    return name or "unknown"


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Dataset name (used for db/index filenames and CLI defaults)
    DEFAULT_DATASET = "pursuit"

    # File paths
    DEFAULT_DB_NAME = f"{DEFAULT_DATASET}.db"
    DEFAULT_INDEX_NAME = f"{DEFAULT_DATASET}.index"
    DEFAULT_CROPS_DIR = f"{DEFAULT_DATASET}_crops"
    DEFAULT_MASKS_DIR = f"{DEFAULT_DATASET}_masks"

    DB_PATH = os.path.join(BASE_DIR, DEFAULT_DB_NAME)
    INDEX_PATH = os.path.join(BASE_DIR, DEFAULT_INDEX_NAME)
    CROPS_DIR = os.path.join(BASE_DIR, DEFAULT_CROPS_DIR)
    CROPS_INGEST_DIR = os.path.join(BASE_DIR, DEFAULT_CROPS_DIR, "ingest")
    CROPS_SEARCH_DIR = os.path.join(BASE_DIR, DEFAULT_CROPS_DIR, "search")
    MASKS_DIR = os.path.join(BASE_DIR, DEFAULT_MASKS_DIR)
    MASKS_INGEST_DIR = os.path.join(BASE_DIR, DEFAULT_MASKS_DIR, "ingest")
    MASKS_SEARCH_DIR = os.path.join(BASE_DIR, DEFAULT_MASKS_DIR, "search")

    # Metadata keys (for database metadata table)
    METADATA_KEY_EMBEDDER = "embedder"

    # Models
    DEFAULT_EMBEDDER = "siglip"  # CLI --embedder default
    SAM3_VRAM_REQUIRED_GB = 4  # SAM3 needs ~4GB VRAM
    SAM3_MODEL = "sam3"
    DINOV2_MODEL = "facebook/dinov2-base"
    DINOV2_LARGE_MODEL = "facebook/dinov2-large"
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    CLIP_MODEL_REVISION = "refs/pr/21" # safetensors model to fix a security issue
    SIGLIP_MODEL = "google/siglip-base-patch16-224"
    EMBEDDING_DIM = 768
    COLOR_HIST_BINS = 64

    # Classification
    CLASSIFY_FURSUIT_LABELS = {"a photo of a fursuit", "a photo of an animal costume", "a photo of a mascot"}
    CLASSIFY_OTHER_LABELS = ["a photo of a person", "a photo of nature or objects", "a cartoon or drawing", "digital art", "cropped image"]
    CLASSIFY_LABELS = [*CLASSIFY_FURSUIT_LABELS, *CLASSIFY_OTHER_LABELS]
    DEFAULT_CLASSIFY_THRESHOLD = 0.85

    # Detection
    DETECTION_CONFIDENCE = 0.5
    MAX_DETECTIONS = 10
    DEFAULT_CONCEPT = "fursuiter head"

    # Background isolation
    DEFAULT_BACKGROUND_MODE = "solid"
    DEFAULT_BACKGROUND_COLOR = (128, 128, 128)
    DEFAULT_BLUR_RADIUS = 25

    # Image processing
    PATCH_SIZE = 14
    TARGET_IMAGE_SIZE = 630
    SAM3_IMAGE_SIZE = 644

    # Search
    MAX_EXAMPLES = 10
    DEFAULT_TOP_K = 5
    DEFAULT_MIN_CONFIDENCE = 0.6  # 60% minimum confidence for displaying results
    HNSW_M = 32
    HNSW_EF_CONSTRUCTION = 200
    HNSW_EF_SEARCH = 50

    # Batch processing
    DEFAULT_BATCH_SIZE = 16

    @staticmethod
    def get_device() -> str:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap >= (6, 0):
                return "cuda"
        if torch.backends.mps.is_available():
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['FAISS_OPT_LEVEL'] = ''
            return "mps"
        return "cpu"

    @classmethod
    def get_segmentor_device(cls) -> str:
        """Get device for SAM3 segmentor. Falls back to CPU if not enough VRAM."""
        device = cls.get_device()
        if device == "cuda":
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_vram_gb < cls.SAM3_VRAM_REQUIRED_GB:
                print(f"GPU has {total_vram_gb:.1f}GB VRAM, SAM3 needs ~{cls.SAM3_VRAM_REQUIRED_GB}GB. Using CPU for segmentation.")
                return "cpu"
        return device

    @classmethod
    def get_absolute_path(cls, relative_path: str) -> str:
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(cls.BASE_DIR, relative_path)
