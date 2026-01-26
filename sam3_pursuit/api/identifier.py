"""Main API for fursuit character identification."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Results for a single detected segment."""
    segment_index: int
    segment_bbox: tuple[int, int, int, int]
    segment_confidence: float
    matches: list[IdentificationResult]


class SAM3FursuitIdentifier:
    """Main API for fursuit character identification."""

    def __init__(
        self,
        db_path: str = Config.DB_PATH,
        index_path: str = Config.INDEX_PATH,
        device: Optional[str] = None,
        isolation_config: Optional[IsolationConfig] = None
    ):
        self.device = device or Config.get_device()
        print(f"Initializing SAM3FursuitIdentifier on {self.device}")

        self.db = Database(db_path)
        self.index = VectorIndex(index_path)
        self.pipeline = ProcessingPipeline(device=self.device, isolation_config=isolation_config)

        print(f"Identifier ready. Index: {self.index.size} embeddings")

    def _build_preprocessing_info(self) -> str:
        """Build compact preprocessing metadata string.

        Format: pipe-separated key:value pairs
        Keys: bg (background mode), bgc (color hex), br (blur radius),
              emb (embedder), idx (index type)
        """
        parts = []
        iso = self.pipeline.isolation_config

        # Background mode: s=solid, b=blur, n=none
        mode_map = {"solid": "s", "blur": "b", "none": "n"}
        parts.append(f"bg:{mode_map.get(iso.mode, 'n')}")

        # Background color (only for solid mode, as hex without #)
        if iso.mode == "solid":
            r, g, b = iso.background_color
            parts.append(f"bgc:{r:02x}{g:02x}{b:02x}")

        # Blur radius (only for blur mode)
        if iso.mode == "blur":
            parts.append(f"br:{iso.blur_radius}")

        # Embedder model (shortened)
        emb = self.pipeline.embedder_model_name
        if "dinov2-base" in emb:
            emb = "dv2b"
        elif "dinov2-large" in emb:
            emb = "dv2l"
        elif "dinov2-giant" in emb:
            emb = "dv2g"
        else:
            emb = emb.split("/")[-1][:8]  # Last part, max 8 chars
        parts.append(f"emb:{emb}")

        # Index type
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
        """Identify fursuit character(s) in an image.

        Args:
            image: Input image
            top_k: Number of results to return per segment
            use_segmentation: Whether to use SAM3 segmentation
            save_crops: Whether to save preprocessed crops for debugging
            crop_prefix: Prefix for saved crop filenames

        Returns:
            When use_segmentation=False: List of IdentificationResult
            When use_segmentation=True: List of SegmentResults (one per segment)
        """
        if self.index.size == 0:
            print("Warning: Index is empty")
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
            # For non-segmented, save the resized input that goes to embedder
            if save_crops:
                from sam3_pursuit.models.embedder import _resize_to_patch_multiple
                resized = _resize_to_patch_multiple(image.convert("RGB"))
                self._save_debug_crop(resized, f"{crop_prefix}_full")
            embedding = self.pipeline.embed_only(image)
            return self._search_embedding(embedding, top_k)

    def _save_debug_crop(self, image: Image.Image, name: str, search: bool = True):
        """Save a debug crop image.

        Args:
            image: Image to save
            name: Filename (without extension)
            search: If True, save to search dir; if False, save to ingest dir
        """
        crops_dir = Path(Config.CROPS_SEARCH_DIR if search else Config.CROPS_INGEST_DIR)
        crops_dir.mkdir(parents=True, exist_ok=True)
        path = crops_dir / f"{name}.jpg"
        image.convert("RGB").save(path, quality=90)
        print(f"Saved crop: {path}")

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
        num_workers: int = 4,
    ) -> int:
        """Add images for characters to the index.

        Args:
            character_names: List of character names for each image
            image_paths: List of image paths
            batch_size: Batch size for embedding (used when use_segmentation=False)
            use_segmentation: Whether to use SAM3 segmentation
            concept: SAM3 concept for segmentation
            save_crops: Whether to save crop images
            source_url: Optional source URL
            num_workers: Number of threads for parallel image loading

        Returns:
            Number of embeddings added
        """
        assert len(character_names) == len(image_paths)

        if use_segmentation:
            return self._add_images_with_segmentation(
                character_names, image_paths, concept, save_crops, source_url, num_workers
            )
        else:
            return self._add_images_batched(
                character_names, image_paths, batch_size, save_crops, source_url, num_workers
            )

    def _load_image_task(self, args: tuple) -> tuple:
        """Load a single image (for thread pool). Returns (index, image, error)."""
        idx, img_path = args
        try:
            image = self._load_image(img_path)
            return (idx, image, None)
        except Exception as e:
            return (idx, None, str(e))

    def _add_images_with_segmentation(
        self,
        character_names: list[str],
        image_paths: list[str],
        concept: str,
        save_crops: bool,
        source_url: Optional[str],
        num_workers: int,
    ) -> int:
        """Add images with segmentation (parallel loading, sequential processing)."""
        from sam3_pursuit.storage.database import get_git_version

        added_count = 0
        preprocessing_info = self._build_preprocessing_info()
        git_version = get_git_version()

        # Filter out images already processed with current git version
        post_ids = [self._extract_post_id(p) for p in image_paths]
        posts_to_process = self.db.get_posts_needing_update(post_ids, git_version)

        # Build filtered lists
        filtered_indices = [i for i, pid in enumerate(post_ids) if pid in posts_to_process]
        if not filtered_indices:
            print("All images already processed with current git version")
            return 0

        skipped = len(image_paths) - len(filtered_indices)
        if skipped > 0:
            print(f"Skipping {skipped} images already processed with git version {git_version}")

        total = len(filtered_indices)

        # Process in chunks to limit memory usage
        chunk_size = num_workers * 2

        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_indices = filtered_indices[chunk_start:chunk_end]

            # Parallel load images in this chunk
            loaded = {}
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                tasks = [(i, image_paths[i]) for i in chunk_indices]
                for idx, image, error in executor.map(self._load_image_task, tasks):
                    if error:
                        print(f"Error loading {image_paths[idx]}: {error}")
                    else:
                        loaded[idx] = image

            # Process loaded images sequentially (GPU operations)
            for idx in chunk_indices:
                if idx not in loaded:
                    continue

                image = loaded[idx]
                character_name = character_names[idx]
                img_path = image_paths[idx]
                post_id = self._extract_post_id(img_path)
                filename = os.path.basename(img_path)

                proc_results = self.pipeline.process(image, concept=concept)
                for proc_result in proc_results:
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

                # Free memory
                del loaded[idx]

            print(f"Processed {min(chunk_end, total)}/{total} images, {added_count} embeddings")

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
        """Add images without segmentation (batched embedding)."""
        from sam3_pursuit.storage.database import get_git_version

        added_count = 0
        preprocessing_info = self._build_preprocessing_info()
        git_version = get_git_version()

        # Filter out images already processed with current git version
        post_ids = [self._extract_post_id(p) for p in image_paths]
        posts_to_process = self.db.get_posts_needing_update(post_ids, git_version)

        # Build filtered list of indices
        filtered_indices = [i for i, pid in enumerate(post_ids) if pid in posts_to_process]
        if not filtered_indices:
            print("All images already processed with current git version")
            return 0

        skipped = len(image_paths) - len(filtered_indices)
        if skipped > 0:
            print(f"Skipping {skipped} images already processed with git version {git_version}")

        total = len(filtered_indices)

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_indices = filtered_indices[batch_start:batch_end]

            # Parallel load images
            loaded = {}
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                tasks = [(i, image_paths[i]) for i in batch_indices]
                for idx, image, error in executor.map(self._load_image_task, tasks):
                    if error:
                        print(f"Error loading {image_paths[idx]}: {error}")
                    else:
                        loaded[idx] = image

            # Collect valid images for batched embedding
            valid_indices = [i for i in batch_indices if i in loaded]
            if not valid_indices:
                continue

            images = [loaded[i] for i in valid_indices]

            # Batch embed
            embeddings = self.pipeline.embed_batch(images)

            # Add to index
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
