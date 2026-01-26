import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

from sam3_pursuit.config import Config
from sam3_pursuit.models.preprocessor import IsolationConfig
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


@dataclass
class SegmentResults:
    segment_index: int
    segment_bbox: tuple[int, int, int, int]
    segment_confidence: float
    matches: list[IdentificationResult]


class SAM3FursuitIdentifier:
    def __init__(
        self,
        db_path: str = Config.DB_PATH,
        index_path: str = Config.INDEX_PATH,
        device: Optional[str] = None,
        isolation_config: Optional[IsolationConfig] = None
    ):
        self.device = device or Config.get_device()
        self.db = Database(db_path)
        self.index = VectorIndex(index_path)
        self.pipeline = ProcessingPipeline(device=self.device, isolation_config=isolation_config)

    def _build_preprocessing_info(self) -> str:
        parts = []
        iso = self.pipeline.isolation_config
        mode_map = {"solid": "s", "blur": "b", "none": "n"}
        parts.append(f"bg:{mode_map.get(iso.mode, 'n')}")
        if iso.mode == "solid":
            r, g, b = iso.background_color
            parts.append(f"bgc:{r:02x}{g:02x}{b:02x}")
        if iso.mode == "blur":
            parts.append(f"br:{iso.blur_radius}")
        emb = self.pipeline.embedder_model_name
        if "dinov2-base" in emb:
            emb = "dv2b"
        elif "dinov2-large" in emb:
            emb = "dv2l"
        elif "dinov2-giant" in emb:
            emb = "dv2g"
        else:
            emb = emb.split("/")[-1][:8]
        parts.append(f"emb:{emb}")
        parts.append(f"idx:{self.index.index_type}")
        return "|".join(parts)

    def identify(
        self,
        image: Image.Image,
        top_k: int = Config.DEFAULT_TOP_K,
        use_segmentation: bool = False,
        save_crops: bool = False,
        crop_prefix: str = "query",
    ) -> list[IdentificationResult] | list[SegmentResults]:
        if self.index.size == 0:
            return []

        if use_segmentation:
            proc_results = self.pipeline.process(image)
            segment_results = []
            for i, proc_result in enumerate(proc_results):
                if save_crops and proc_result.isolated_crop:
                    self._save_debug_crop(proc_result.isolated_crop, f"{crop_prefix}_{i}")
                matches = self._search_embedding(proc_result.embedding, top_k)
                segment_results.append(SegmentResults(
                    segment_index=i,
                    segment_bbox=proc_result.segmentation.bbox,
                    segment_confidence=proc_result.segmentation.confidence,
                    matches=matches,
                ))
            return segment_results
        else:
            if save_crops:
                from sam3_pursuit.models.embedder import _resize_to_patch_multiple
                resized = _resize_to_patch_multiple(image.convert("RGB"))
                self._save_debug_crop(resized, f"{crop_prefix}_full")
            embedding = self.pipeline.embed_only(image)
            return self._search_embedding(embedding, top_k)

    def _save_debug_crop(self, image: Image.Image, name: str, search: bool = True):
        crops_dir = Path(Config.CROPS_SEARCH_DIR if search else Config.CROPS_INGEST_DIR)
        crops_dir.mkdir(parents=True, exist_ok=True)
        path = crops_dir / f"{name}.jpg"
        image.convert("RGB").save(path, quality=90)

    def _search_embedding(self, embedding: np.ndarray, top_k: int) -> list[IdentificationResult]:
        distances, indices = self.index.search(embedding, top_k * 2)
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            detection = self.db.get_detection_by_embedding_id(int(idx))
            if detection is None:
                continue
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
        num_workers: int = 4,
    ) -> int:
        assert len(character_names) == len(image_paths)
        if use_segmentation:
            return self._add_images_with_segmentation(
                character_names, image_paths, concept, save_crops, source_url, num_workers
            )
        return self._add_images_batched(
            character_names, image_paths, batch_size, save_crops, source_url, num_workers
        )

    def _load_image_task(self, args: tuple) -> tuple:
        idx, img_path = args
        try:
            return (idx, self._load_image(img_path), None)
        except Exception as e:
            return (idx, None, str(e))

    def _load_chunk(self, indices: list[int], image_paths: list[str], num_workers: int) -> dict:
        loaded = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            tasks = [(i, image_paths[i]) for i in indices]
            for idx, image, error in executor.map(self._load_image_task, tasks):
                if not error:
                    loaded[idx] = image
        return loaded

    def _add_images_with_segmentation(
        self,
        character_names: list[str],
        image_paths: list[str],
        concept: str,
        save_crops: bool,
        source_url: Optional[str],
        num_workers: int,
    ) -> int:
        from sam3_pursuit.storage.database import get_git_version

        added_count = 0
        preprocessing_info = self._build_preprocessing_info()
        git_version = get_git_version()

        post_ids = [self._extract_post_id(p) for p in image_paths]
        posts_to_process = self.db.get_posts_needing_update(post_ids, git_version, preprocessing_info)
        filtered_indices = [i for i, pid in enumerate(post_ids) if pid in posts_to_process]
        if not filtered_indices:
            return 0

        total = len(filtered_indices)
        chunk_size = max(16, num_workers * 2)

        chunks = [filtered_indices[i:i + chunk_size] for i in range(0, total, chunk_size)]

        # Load and segment first chunk
        loaded = self._load_chunk(chunks[0], image_paths, num_workers)
        valid_indices = [i for i in chunks[0] if i in loaded]
        images = [loaded[i] for i in valid_indices]
        current_segs = self.pipeline.segment_batch(images, concept=concept) if images else []
        current_data = (valid_indices, current_segs)

        for chunk_idx in range(len(chunks)):
            valid_indices, all_segs = current_data

            # Start loading + segmenting next chunk in background
            if chunk_idx + 1 < len(chunks):
                next_loaded = self._load_chunk(chunks[chunk_idx + 1], image_paths, num_workers)
                next_valid = [i for i in chunks[chunk_idx + 1] if i in next_loaded]
                next_images = [next_loaded[i] for i in next_valid]

            # Process current chunk (isolation + embedding) while next is being prepared
            if valid_indices and all_segs:
                batch_results = self.pipeline.isolate_and_embed(all_segs, num_workers=num_workers)

                for batch_idx, idx in enumerate(valid_indices):
                    character_name = character_names[idx]
                    img_path = image_paths[idx]
                    post_id = self._extract_post_id(img_path)
                    filename = os.path.basename(img_path)

                    for proc_result in batch_results[batch_idx]:
                        crop_to_save = proc_result.isolated_crop if save_crops else None
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
                            preprocessing_info=preprocessing_info,
                            crop_image=crop_to_save,
                        )
                        added_count += 1

            # Segment next batch (this is the slow part)
            if chunk_idx + 1 < len(chunks):
                next_segs = self.pipeline.segment_batch(next_images, concept=concept) if next_images else []
                current_data = (next_valid, next_segs)

            processed = min((chunk_idx + 1) * chunk_size, total)
            print(f"Processed {processed}/{total} images, {added_count} embeddings")

        self.index.save()
        return added_count

    def _add_images_batched(
        self,
        character_names: list[str],
        image_paths: list[str],
        batch_size: int,
        save_crops: bool,
        source_url: Optional[str],
        num_workers: int,
    ) -> int:
        from sam3_pursuit.storage.database import get_git_version

        added_count = 0
        preprocessing_info = self._build_preprocessing_info()
        git_version = get_git_version()

        post_ids = [self._extract_post_id(p) for p in image_paths]
        posts_to_process = self.db.get_posts_needing_update(post_ids, git_version, preprocessing_info)
        filtered_indices = [i for i, pid in enumerate(post_ids) if pid in posts_to_process]
        if not filtered_indices:
            return 0

        total = len(filtered_indices)

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_indices = filtered_indices[batch_start:batch_end]

            loaded = {}
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                tasks = [(i, image_paths[i]) for i in batch_indices]
                for idx, image, error in executor.map(self._load_image_task, tasks):
                    if not error:
                        loaded[idx] = image

            valid_indices = [i for i in batch_indices if i in loaded]
            if not valid_indices:
                continue

            images = [loaded[i] for i in valid_indices]
            embeddings = self.pipeline.embed_batch(images)

            for i, idx in enumerate(valid_indices):
                image = loaded[idx]
                character_name = character_names[idx]
                img_path = image_paths[idx]
                post_id = self._extract_post_id(img_path)
                filename = os.path.basename(img_path)
                w, h = image.size

                self._add_single_embedding(
                    embedding=embeddings[i],
                    post_id=post_id,
                    character_name=character_name,
                    bbox=(0, 0, w, h),
                    confidence=1.0,
                    source_filename=filename,
                    source_url=source_url,
                    is_cropped=False,
                    segmentation_concept=None,
                    preprocessing_info=preprocessing_info,
                )
                added_count += 1

            print(f"Processed {min(batch_end, total)}/{total} images")

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
        preprocessing_info: Optional[str] = None,
        crop_image: Optional[Image.Image] = None,
    ):
        embedding_id = self.index.add(embedding.reshape(1, -1))
        if segmentor_model is None:
            segmentor_model = self.pipeline.segmentor_model_name

        crop_path = None
        if crop_image is not None:
            crops_dir = Path(Config.CROPS_INGEST_DIR)
            crops_dir.mkdir(parents=True, exist_ok=True)
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
            preprocessing_info=preprocessing_info,
            crop_path=crop_path,
        )
        self.db.add_detection(detection)

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
