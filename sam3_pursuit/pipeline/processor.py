from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config
from sam3_pursuit.models.segmentor import FursuitSegmentor, FullImageSegmentor, SegmentationResult
from sam3_pursuit.models.embedder import FursuitEmbedder
from sam3_pursuit.models.preprocessor import BackgroundIsolator, IsolationConfig


@dataclass
class ProcessingResult:
    segmentation: SegmentationResult
    embedding: np.ndarray
    isolated_crop: Optional[Image.Image] = None
    segmentor_model: str = "unknown"
    segmentor_concept: Optional[str] = None


class ProcessingPipeline:
    def __init__(
        self,
        device: Optional[str] = None,
        embedder_model_name: str = Config.DINOV2_MODEL,
        isolation_config: Optional[IsolationConfig] = None,
        segmentor_model_name: Optional[str] = None,
        segmentor_concept: Optional[str] = None,
    ):
        self.device = device or Config.get_device()
        self.segmentor = FursuitSegmentor(device=self.device, concept=segmentor_concept) if segmentor_model_name else FullImageSegmentor()
        self.segmentor_concept = segmentor_concept
        self.embedder = FursuitEmbedder(device=self.device, model_name=embedder_model_name)
        self.isolator = BackgroundIsolator(isolation_config)

    def _resize_to_patch_multiple(self, image: Image.Image, target_size: int = 630) -> Image.Image:
        w, h = image.size
        if w >= h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        new_w = max(Config.PATCH_SIZE, (new_w // Config.PATCH_SIZE) * Config.PATCH_SIZE)
        new_h = max(Config.PATCH_SIZE, (new_h // Config.PATCH_SIZE) * Config.PATCH_SIZE)
        print(f"Resizing image from ({w}, {h}) to ({new_w}, {new_h})")
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def process(self, image: Image.Image) -> list[ProcessingResult]:
        segmentations = self.segmentor.segment(image)
        # sort by confidence descending
        segmentations.sort(key=lambda s: s.confidence, reverse=True)
        segmentations = segmentations[:Config.MAX_DETECTIONS]
        results = []
        for seg in segmentations:
            isolated = self.isolator.isolate(seg.crop, seg.crop_mask)
            isolated_crop = self._resize_to_patch_multiple(isolated)
            embedding = self.embedder.embed(isolated_crop)
            results.append(ProcessingResult(
                segmentation=seg,
                embedding=embedding,
                isolated_crop=isolated_crop,
                segmentor_model=seg.segmentor,
                segmentor_concept=self.segmentor_concept,
            ))
        return results

    def embed_only(self, image: Image.Image) -> np.ndarray:
        return self.embedder.embed(image)

    def process_with_masks(self, image: Image.Image, masks: list[tuple[int, np.ndarray]]) -> list[ProcessingResult]:
        """Process image using pre-existing masks (skips segmentation)."""
        results = []
        for seg_idx, mask in masks:
            seg = SegmentationResult.from_mask(image, mask, segmentor=self.segmentor_model_name)
            if seg is None:
                continue
            isolated = self.isolator.isolate(seg.crop, seg.crop_mask)
            isolated_crop = self._resize_to_patch_multiple(isolated)
            embedding = self.embedder.embed(isolated_crop)
            results.append(ProcessingResult(
                segmentation=seg, embedding=embedding, isolated_crop=isolated_crop,
                segmentor_model=self.segmentor_model_name, segmentor_concept=self.segmentor_concept))
        return results

    @property
    def segmentor_model_name(self) -> str:
        return self.segmentor.model_name

    @property
    def embedder_model_name(self) -> str:
        return self.embedder.model_name

    @property
    def isolation_config(self) -> IsolationConfig:
        return self.isolator.config
