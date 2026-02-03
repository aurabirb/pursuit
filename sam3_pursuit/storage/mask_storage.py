"""Storage for segmentation masks."""

import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config


def _normalize_concept(concept: str) -> str:
    """Normalize concept string for use in path (replace non-alphanumeric with _)."""
    return re.sub(r'[^a-zA-Z0-9]', '_', concept).strip('_') or "default"


class MaskStorage:
    """Handles saving and loading segmentation masks."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(Config.MASKS_DIR)

    def _get_mask_dir(self, source: str, model: str, concept: str) -> Path:
        """Get directory for masks: {base}/{source}/{model}/{concept}/"""
        return self.base_dir / (source or "unknown") / model / _normalize_concept(concept)

    def save_mask(
        self,
        mask: np.ndarray,
        name: str,
        source: str,
        model: str,
        concept: str,
    ) -> str:
        """Save a segmentation mask as PNG.

        Args:
            mask: Binary mask array (H, W) with values 0-255 or 0-1
            name: Base name for the mask file (without extension)
            source: Ingestion source (e.g., "tgbot", "furtrack")
            model: Segmentor model name (e.g., "sam3")
            concept: Segmentation concept (e.g., "fursuiter head")

        Returns:
            Path to the saved mask file
        """
        target_dir = self._get_mask_dir(source, model, concept)
        target_dir.mkdir(parents=True, exist_ok=True)

        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        path = target_dir / f"{name}.png"
        Image.fromarray(mask, mode="L").save(path, optimize=True)
        return str(path)

    def load_mask(self, name: str, source: str, model: str, concept: str) -> Optional[np.ndarray]:
        """Load a segmentation mask from file.

        Returns:
            Binary mask array (H, W) with values 0-255, or None if not found
        """
        path = self._get_mask_dir(source, model, concept) / f"{name}.png"
        if not path.exists():
            return None
        return np.array(Image.open(path).convert("L"))

    def get_mask_path(self, name: str, source: str, model: str, concept: str) -> Path:
        """Get the full path for a mask file."""
        return self._get_mask_dir(source, model, concept) / f"{name}.png"

    def mask_exists(self, name: str, source: str, model: str, concept: str) -> bool:
        """Check if a mask file exists."""
        return self.get_mask_path(name, source, model, concept).exists()

    def find_masks_for_post(self, post_id: str, source: str, model: str, concept: str) -> list[Path]:
        """Find all segment masks for a post_id ({post_id}_seg_*.png)."""
        mask_dir = self._get_mask_dir(source, model, concept)
        if not mask_dir.exists():
            return []
        return sorted(mask_dir.glob(f"{post_id}_seg_*.png"), key=lambda p: int(p.stem.split("_seg_")[-1]))

    def load_masks_for_post(self, post_id: str, source: str, model: str, concept: str) -> list[tuple[int, np.ndarray]]:
        """Load all segment masks for a post_id. Returns list of (segment_index, mask_array)."""
        results = []
        for path in self.find_masks_for_post(post_id, source, model, concept):
            seg_idx = int(path.stem.split("_seg_")[-1])
            results.append((seg_idx, np.array(Image.open(path).convert("L"))))
        return results
