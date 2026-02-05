"""Processing pipeline: segmentation, isolation, embedding."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config
from sam3_pursuit.models.embedder import FursuitEmbedder
from sam3_pursuit.models.preprocessor import BackgroundIsolator, IsolationConfig
from sam3_pursuit.models.segmentor import (
    FullImageSegmentor,
    SAM3FursuitSegmentor,
    SegmentationResult,
)
from sam3_pursuit.storage.mask_storage import MaskStorage


@dataclass
class CacheKey:
    post_id: str
    source: str


@dataclass
class ProcessingResult:
    segmentation: SegmentationResult
    embedding: np.ndarray
    isolated_crop: Optional[Image.Image] = None
    segmentor_model: str = "unknown"
    segmentor_concept: Optional[str] = None
    mask_reused: bool = False


class ProcessingPipeline:
    """Segment, isolate, and embed an image."""

    def __init__(
        self,
        device: Optional[str] = None,
        isolation_config: Optional[IsolationConfig] = None,
        segmentor_model_name: Optional[str] = "",
        segmentor_concept: Optional[str] = "",
    ):
        self.device = device or Config.get_device()
        segmentor_device = Config.get_segmentor_device()
        if segmentor_model_name == Config.SAM3_MODEL:
            self.segmentor = SAM3FursuitSegmentor(device=segmentor_device, concept=segmentor_concept)
        else:
            self.segmentor = FullImageSegmentor()
        self.segmentor_model_name = self.segmentor.model_name
        self.segmentor_concept = segmentor_concept or ""
        self.embedder_model_name = Config.DINOV2_MODEL
        self.embedder = FursuitEmbedder(device=self.device, model_name=self.embedder_model_name)
        self.isolator = BackgroundIsolator(isolation_config)
        self.isolation_config = self.isolator.config

    def process(self, image: Image.Image, cache_key: Optional[CacheKey] = None) -> list[ProcessingResult]:
        segmentations, mask_reused = self._segment(image, cache_key)
        return self._process_segmentations(segmentations, mask_reused)

    def _segment(self, image: Image.Image, cache_key: Optional[CacheKey] = None) -> tuple[list[SegmentationResult], bool]:
        segmentations = self.segmentor.segment(image)
        return segmentations, False

    def _process_segmentations(self, segmentations: list[SegmentationResult], mask_reused: bool) -> list[ProcessingResult]:
        proc_results = []
        for seg in segmentations:
            isolated = self.isolator.isolate(seg.crop, seg.crop_mask)
            isolated_crop = self._resize_to_patch_multiple(isolated)
            embedding = self.embedder.embed(isolated_crop)
            proc_results.append(ProcessingResult(
                segmentation=seg,
                embedding=embedding,
                isolated_crop=isolated_crop,
                segmentor_model=seg.segmentor,
                segmentor_concept=self.segmentor_concept,
                mask_reused=mask_reused,
            ))
        return proc_results

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
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


class CachedProcessingPipeline(ProcessingPipeline):
    """ProcessingPipeline with mask caching via MaskStorage."""

    def __init__(self, mask_storage: Optional[MaskStorage] = None, **kwargs):
        super().__init__(**kwargs)
        self.mask_storage = mask_storage or MaskStorage()

    def _segment(self, image: Image.Image, cache_key: Optional[CacheKey] = None) -> tuple[list[SegmentationResult], bool]:
        if cache_key is not None:
            segmentations = self._load_segments_for_post(
                cache_key.post_id, cache_key.source,
                self.segmentor_model_name, self.segmentor_concept, image,
            )
            if segmentations:
                return segmentations, True

        segmentations = self.segmentor.segment(image)

        if cache_key is not None:
            try:
                self._save_segments_for_post(
                    cache_key.post_id, cache_key.source,
                    self.segmentor_model_name, self.segmentor_concept, segmentations,
                )
            except Exception as e:
                print(f"Failed to save segments for {cache_key.post_id}: {e}")

        return segmentations, False

    def _load_segments_for_post(self, post_id: str, source: str, model: str, concept: str, image: Image.Image) -> list[SegmentationResult]:
        if self.mask_storage.has_no_segments_marker(post_id, source, model, concept):
            return FullImageSegmentor().segment(image)
        masks = self.mask_storage.load_masks_for_post(post_id, source, model, concept)
        segmentations = [
            SegmentationResult.from_mask(image, mask, segmentor=self.segmentor_model_name) for mask in masks
        ]
        segmentations = [s for s in segmentations if s]
        return segmentations

    def _save_segments_for_post(self, post_id: str, source: str, model: str, concept: str, segmentations: list[SegmentationResult]) -> None:
        masks = [seg.mask for seg in segmentations if seg.mask is not None]
        if masks:
            self.mask_storage.save_masks_for_post(post_id, source, self.segmentor_model_name, self.segmentor_concept, masks)
        else:
            self.mask_storage.save_no_segments_marker(post_id, source, self.segmentor_model_name, self.segmentor_concept)
