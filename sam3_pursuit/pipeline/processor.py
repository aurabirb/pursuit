from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config
from sam3_pursuit.models.segmentor import FursuitSegmentor, SegmentationResult
from sam3_pursuit.models.embedder import FursuitEmbedder
from sam3_pursuit.models.preprocessor import BackgroundIsolator, IsolationConfig


@dataclass
class ProcessingResult:
    segmentation: SegmentationResult
    embedding: np.ndarray
    isolated_crop: Optional[Image.Image] = None
    segmentor_model: str = "unknown"


class ProcessingPipeline:
    def __init__(
        self,
        device: Optional[str] = None,
        sam_model: Optional[str] = None,
        dinov2_model: str = Config.DINOV2_MODEL,
        isolation_config: Optional[IsolationConfig] = None
    ):
        self.device = device or Config.get_device()
        self.segmentor = FursuitSegmentor(device=self.device, model_name=sam_model)
        self.embedder = FursuitEmbedder(device=self.device, model_name=dinov2_model)
        self.isolator = BackgroundIsolator(isolation_config)

    def process(self, image: Image.Image, concept: str = "fursuiter") -> list[ProcessingResult]:
        segmentations = self.segmentor.segment(image, concept=concept)
        results = []
        for seg in segmentations:
            isolated = self.isolator.isolate(seg.crop, seg.crop_mask)
            embedding = self.embedder.embed(isolated)
            results.append(ProcessingResult(
                segmentation=seg,
                embedding=embedding,
                isolated_crop=isolated,
                segmentor_model=self.segmentor.model_name
            ))
        return results

    def process_full_image(self, image: Image.Image) -> ProcessingResult:
        w, h = image.size
        full_mask = np.ones((h, w), dtype=np.uint8)
        segmentation = SegmentationResult(
            crop=image.copy(),
            mask=full_mask,
            crop_mask=full_mask,
            bbox=(0, 0, w, h),
            confidence=1.0
        )
        isolated = self.isolator.isolate(image, full_mask)
        embedding = self.embedder.embed(isolated)
        return ProcessingResult(
            segmentation=segmentation,
            embedding=embedding,
            isolated_crop=isolated,
            segmentor_model=self.segmentor.model_name
        )

    def embed_only(self, image: Image.Image) -> np.ndarray:
        return self.embedder.embed(image)

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        return self.embedder.embed_batch(images)

    def process_batch(self, images: list[Image.Image], concept: str = "fursuiter") -> list[list[ProcessingResult]]:
        all_segs: list[list[SegmentationResult]] = []
        for image in images:
            all_segs.append(self.segmentor.segment(image, concept=concept))

        all_isolated: list[Image.Image] = []
        crop_map: list[tuple[int, int]] = []
        for img_idx, segs in enumerate(all_segs):
            for seg_idx, seg in enumerate(segs):
                isolated = self.isolator.isolate(seg.crop, seg.crop_mask)
                all_isolated.append(isolated)
                crop_map.append((img_idx, seg_idx))

        all_embeddings = self.embedder.embed_batch(all_isolated) if all_isolated else np.array([])

        results: list[list[ProcessingResult]] = [[] for _ in images]
        for i, (img_idx, seg_idx) in enumerate(crop_map):
            results[img_idx].append(ProcessingResult(
                segmentation=all_segs[img_idx][seg_idx],
                embedding=all_embeddings[i],
                isolated_crop=all_isolated[i],
                segmentor_model=self.segmentor.model_name
            ))
        return results

    def segment_batch(self, images: list[Image.Image], concept: str = "fursuiter") -> list[list[SegmentationResult]]:
        return [self.segmentor.segment(img, concept=concept) for img in images]

    def isolate_and_embed(self, all_segs: list[list[SegmentationResult]], num_workers: int = 4) -> list[list[ProcessingResult]]:
        crops_and_masks = []
        crop_map: list[tuple[int, int]] = []
        for img_idx, segs in enumerate(all_segs):
            for seg_idx, seg in enumerate(segs):
                crops_and_masks.append((seg.crop, seg.crop_mask))
                crop_map.append((img_idx, seg_idx))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            all_isolated = list(executor.map(lambda x: self.isolator.isolate(x[0], x[1]), crops_and_masks))

        all_embeddings = self.embedder.embed_batch(all_isolated) if all_isolated else np.array([])

        results: list[list[ProcessingResult]] = [[] for _ in all_segs]
        for i, (img_idx, seg_idx) in enumerate(crop_map):
            results[img_idx].append(ProcessingResult(
                segmentation=all_segs[img_idx][seg_idx],
                embedding=all_embeddings[i],
                isolated_crop=all_isolated[i],
                segmentor_model=self.segmentor.model_name
            ))
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
