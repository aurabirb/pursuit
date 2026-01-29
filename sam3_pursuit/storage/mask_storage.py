"""Storage for segmentation masks."""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config


class MaskStorage:
    """Handles saving and loading segmentation masks."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(Config.MASKS_DIR)
        self.ingest_dir = self.base_dir / "ingest"
        self.search_dir = self.base_dir / "search"

    def save_mask(
        self,
        mask: np.ndarray,
        name: str,
        search: bool = False,
    ) -> str:
        """Save a segmentation mask as PNG.

        Args:
            mask: Binary mask array (H, W) with values 0-255 or 0-1
            name: Base name for the mask file (without extension)
            search: If True, save to search dir; otherwise ingest dir

        Returns:
            Path to the saved mask file (relative to base_dir)
        """
        target_dir = self.search_dir if search else self.ingest_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Normalize mask to 0-255 range
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # Save as PNG (lossless, efficient for binary masks)
        path = target_dir / f"{name}.png"
        Image.fromarray(mask, mode="L").save(path, optimize=True)

        # Return relative path for storage in database
        return str(path.relative_to(self.base_dir.parent))

    def load_mask(self, mask_path: str) -> Optional[np.ndarray]:
        """Load a segmentation mask from file.

        Args:
            mask_path: Path to the mask file (relative or absolute)

        Returns:
            Binary mask array (H, W) with values 0-255, or None if not found
        """
        # Handle both relative and absolute paths
        path = Path(mask_path)
        if not path.is_absolute():
            path = self.base_dir.parent / mask_path

        if not path.exists():
            return None

        mask = np.array(Image.open(path).convert("L"))
        return mask

    def get_mask_path(self, name: str, search: bool = False) -> Path:
        """Get the full path for a mask file.

        Args:
            name: Base name for the mask file (without extension)
            search: If True, use search dir; otherwise ingest dir

        Returns:
            Full path to the mask file
        """
        target_dir = self.search_dir if search else self.ingest_dir
        return target_dir / f"{name}.png"

    def mask_exists(self, name: str, search: bool = False) -> bool:
        """Check if a mask file exists.

        Args:
            name: Base name for the mask file (without extension)
            search: If True, check search dir; otherwise ingest dir

        Returns:
            True if the mask file exists
        """
        return self.get_mask_path(name, search).exists()
