"""Main API for fursuit character identification."""

import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

from sam3_pursuit.config import Config
from sam3_pursuit.pipeline.processor import ProcessingPipeline
from sam3_pursuit.storage.database import Database, Detection
from sam3_pursuit.storage.vector_index import VectorIndex


@dataclass
class IdentificationResult:
    character_name: Optional[str]
    confidence: float
    distance: float
    post_id: str
    bbox: tuple[int, int, int, int]
    segmentor_model: str = "unknown"


class SAM3FursuitIdentifier:
    """Main API for fursuit character identification."""

    def __init__(
        self,
        db_path: str = Config.DB_PATH,
        index_path: str = Config.INDEX_PATH,
        device: Optional[str] = None
    ):
        self.device = device or Config.get_device()
        print(f"Initializing SAM3FursuitIdentifier on {self.device}")

        self.db = Database(db_path)
        self.index = VectorIndex(index_path)
        self.pipeline = ProcessingPipeline(device=self.device)

        print(f"Identifier ready. Index: {self.index.size} embeddings")

    def identify(
        self,
        image: Image.Image,
        top_k: int = Config.DEFAULT_TOP_K,
        use_segmentation: bool = False
    ) -> list[IdentificationResult]:
        """Identify fursuit character(s) in an image."""
        if self.index.size == 0:
            print("Warning: Index is empty")
            return []

        if use_segmentation:
            results = self.pipeline.process(image)
            all_matches = []
            for proc_result in results:
                matches = self._search_embedding(proc_result.embedding, top_k)
                all_matches.extend(matches)
            all_matches.sort(key=lambda x: x.confidence, reverse=True)
            return all_matches[:top_k]
        else:
            embedding = self.pipeline.embed_only(image)
            return self._search_embedding(embedding, top_k)

    def _search_embedding(self, embedding: np.ndarray, top_k: int) -> list[IdentificationResult]:
        distances, indices = self.index.search(embedding, top_k * 2)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            detection = self.db.get_detection_by_embedding_id(int(idx))
            if detection is None:
                continue

            # Distance to confidence: [0, 2] -> [1, 0]
            confidence = max(0.0, 1.0 - distance / 2.0)

            results.append(IdentificationResult(
                character_name=detection.character_name,
                confidence=confidence,
                distance=float(distance),
                post_id=detection.post_id,
                bbox=(detection.bbox_x, detection.bbox_y,
                      detection.bbox_width, detection.bbox_height),
                segmentor_model=detection.segmentor_model
            ))

        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:top_k]

    def add_images(
        self,
        character_names: list[str],
        image_paths: list[str],
        batch_size: int = Config.DEFAULT_BATCH_SIZE,
        use_segmentation: bool = True,
        concept: str = Config.DEFAULT_CONCEPT,
        save_crops: bool = False,
        source_url: Optional[str] = None,
    ) -> int:
        """Add images for characters to the index."""
        added_count = 0

        assert len(character_names) == len(image_paths)

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_char_names = character_names[i:i + batch_size]
            batch_images = []
            batch_post_ids = []
            batch_filenames = []

            for img_path in batch_paths:
                try:
                    image = self._load_image(img_path)
                    batch_images.append(image)
                    batch_post_ids.append(self._extract_post_id(img_path))
                    batch_filenames.append(os.path.basename(img_path))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

            if not batch_images:
                continue

            if use_segmentation:
                for image, post_id, filename, character_name in zip(batch_images, batch_post_ids, batch_filenames, batch_char_names):
                    proc_results = self.pipeline.process(image, concept=concept)
                    for proc_result in proc_results:
                        self._add_single_embedding(
                            embedding=proc_result.embedding,
                            post_id=post_id,
                            character_name=character_name,
                            bbox=proc_result.segmentation.bbox,
                            confidence=proc_result.segmentation.confidence,
                            segmentor_model=proc_result.segmentor_model,
                            source_filename=filename,
                            source_url=source_url,
                            is_cropped=True,
                            segmentation_concept=concept,
                            crop_image=proc_result.segmentation.crop if save_crops else None,
                        )
                        added_count += 1
            else:
                embeddings = self.pipeline.embed_batch(batch_images)

                for embedding, post_id, image, filename in zip(
                    embeddings, batch_post_ids, batch_images, batch_filenames
                ):
                    w, h = image.size
                    self._add_single_embedding(
                        embedding=embedding,
                        post_id=post_id,
                        character_name=character_name,
                        bbox=(0, 0, w, h),
                        confidence=1.0,
                        source_filename=filename,
                        source_url=source_url,
                        is_cropped=False,
                        segmentation_concept=None,
                    )
                    added_count += 1

            print(f"Added {added_count} images...")

        self.index.save()
        return added_count

    def _add_single_embedding(
        self,
        embedding: np.ndarray,
        post_id: str,
        character_name: str,
        bbox: tuple[int, int, int, int],
        confidence: float,
        segmentor_model: Optional[str] = None,
        source_filename: Optional[str] = None,
        source_url: Optional[str] = None,
        is_cropped: bool = False,
        segmentation_concept: Optional[str] = None,
        preprocessing_info: Optional[dict] = None,
        crop_image: Optional[Image.Image] = None,
    ):
        embedding_id = self.index.add(embedding.reshape(1, -1))

        if segmentor_model is None:
            segmentor_model = self.pipeline.segmentor_model_name

        crop_path = None
        if crop_image is not None:
            crops_dir = Path(Config.CROPS_DIR)
            crops_dir.mkdir(exist_ok=True)
            crop_path = str(crops_dir / f"{post_id}_{embedding_id}.jpg")
            crop_image.convert("RGB").save(crop_path, quality=90)

        detection = Detection(
            id=None,
            post_id=post_id,
            character_name=character_name,
            embedding_id=embedding_id,
            bbox_x=bbox[0],
            bbox_y=bbox[1],
            bbox_width=bbox[2],
            bbox_height=bbox[3],
            confidence=confidence,
            segmentor_model=segmentor_model,
            source_filename=source_filename,
            source_url=source_url,
            is_cropped=is_cropped,
            segmentation_concept=segmentation_concept,
            preprocessing_info=json.dumps(preprocessing_info) if preprocessing_info else None,
            crop_path=crop_path,
        )
        self.db.add_detection(detection)
        print(f'Saved {character_name} at {bbox} confidence {confidence}')

    def _load_image(self, img_path: str) -> Image.Image:
        if img_path.startswith(('http://', 'https://')):
            response = requests.get(img_path, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            if not Path(img_path).exists():
                raise FileNotFoundError()
            img = Image.open(img_path)
        if img is None:
            raise ValueError()
        return img

    def _extract_post_id(self, img_path: str) -> str:
        basename = os.path.basename(img_path)
        return os.path.splitext(basename)[0]

    def get_stats(self) -> dict:
        db_stats = self.db.get_stats()
        db_stats["index_size"] = self.index.size
        return db_stats
