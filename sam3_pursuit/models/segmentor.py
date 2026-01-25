"""SAM3-based fursuit segmentation using text prompts."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config


@dataclass
class SegmentationResult:
    crop: Image.Image
    mask: np.ndarray
    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float


class FursuitSegmentor:
    """SAM3 segmentation using text prompts like "fursuiter"."""

    DEFAULT_CONCEPT = "fursuiter"

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
        confidence_threshold: float = Config.DETECTION_CONFIDENCE,
        max_detections: int = Config.MAX_DETECTIONS
    ):
        self.device = device or Config.get_device()
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.model_name = model_name or Config.SAM3_MODEL
        self.predictor = self._load_model()

    def _load_model(self):
        from ultralytics.models.sam.predict import SAM3SemanticPredictor

        model_path = f"{self.model_name}.pt"
        print(f"Loading SAM3 model: {model_path} on {self.device}")

        overrides = dict(
            conf=self.confidence_threshold,
            task="segment",
            mode="predict",
            model=model_path,
            device=self.device,
            verbose=False,
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)
        print("SAM3 loaded - text prompts enabled")
        return predictor

    def segment(self, image: Image.Image, concept: str = DEFAULT_CONCEPT) -> list[SegmentationResult]:
        """Segment image using SAM3 text prompt."""
        image_np = np.array(image.convert("RGB"))
        self.predictor.set_image(image_np)
        results = self.predictor(text=[concept])
        return self._process_results(image, results)

    def _process_results(self, image: Image.Image, results) -> list[SegmentationResult]:
        segmentation_results = []

        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes

            for i, mask in enumerate(masks):
                if i >= self.max_detections:
                    break

                confidence = 1.0
                if boxes is not None and boxes.conf is not None and len(boxes.conf) > i:
                    confidence = float(boxes.conf[i])

                if confidence < self.confidence_threshold:
                    continue

                bbox = self._mask_to_bbox(mask)
                if bbox is None:
                    continue

                crop = self._create_crop(image, bbox)
                segmentation_results.append(SegmentationResult(
                    crop=crop,
                    mask=mask.astype(np.uint8),
                    bbox=bbox,
                    confidence=confidence
                ))

        # Fallback: return full image if no segments found
        if not segmentation_results:
            w, h = image.size
            segmentation_results.append(SegmentationResult(
                crop=image.copy(),
                mask=np.ones((h, w), dtype=np.uint8),
                bbox=(0, 0, w, h),
                confidence=1.0
            ))

        return segmentation_results

    def _mask_to_bbox(self, mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))

    def _create_crop(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
        x, y, w, h = bbox
        return image.crop((x, y, x + w, y + h))
