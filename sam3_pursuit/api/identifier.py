"""Main public API for fursuit identification."""

import json
import os
import sqlite3
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
    """Result of character identification."""
    character_name: Optional[str]
    confidence: float
    distance: float
    post_id: str
    bbox: tuple[int, int, int, int]  # x, y, width, height
    segmentor_model: str = "unknown"  # Which segmentor indexed this result


class SAM3FursuitIdentifier:
    """Main API for fursuit character identification.

    Provides methods to:
    - Identify characters in new images
    - Add new character images to the index
    - Process unindexed images from the download database
    """

    def __init__(
        self,
        db_path: str = Config.DB_PATH,
        index_path: str = Config.INDEX_PATH,
        device: Optional[str] = None
    ):
        """Initialize the identifier.

        Args:
            db_path: Path to SQLite database.
            index_path: Path to FAISS index.
            device: Device for inference. Auto-detected if None.
        """
        self.device = device or Config.get_device()

        print(f"Initializing SAM3FursuitIdentifier on {self.device}")

        # Initialize storage
        self.db = Database(db_path)
        self.index = VectorIndex(index_path)

        # Initialize pipeline
        self.pipeline = ProcessingPipeline(device=self.device)

        print(f"Identifier ready. Index contains {self.index.size} embeddings")

    def identify(
        self,
        image: Image.Image,
        top_k: int = Config.DEFAULT_TOP_K,
        use_segmentation: bool = False
    ) -> list[IdentificationResult]:
        """Identify fursuit character(s) in an image.

        Args:
            image: PIL Image to identify.
            top_k: Number of top matches to return.
            use_segmentation: Whether to segment the image first.

        Returns:
            List of IdentificationResult objects sorted by confidence.
        """
        if self.index.size == 0:
            print("Warning: Index is empty. Add images first.")
            return []

        if use_segmentation:
            # Process with segmentation
            results = self.pipeline.process(image)
            all_matches = []

            for proc_result in results:
                matches = self._search_embedding(proc_result.embedding, top_k)
                all_matches.extend(matches)

            # Sort by confidence and deduplicate
            all_matches.sort(key=lambda x: x.confidence, reverse=True)
            return all_matches[:top_k]
        else:
            # Process full image
            embedding = self.pipeline.embed_only(image)
            return self._search_embedding(embedding, top_k)

    def _search_embedding(
        self,
        embedding: np.ndarray,
        top_k: int
    ) -> list[IdentificationResult]:
        """Search for matches given an embedding.

        Args:
            embedding: Query embedding.
            top_k: Number of results.

        Returns:
            List of IdentificationResult objects.
        """
        distances, indices = self.index.search(embedding, top_k * 2)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            detection = self.db.get_detection_by_embedding_id(int(idx))
            if detection is None:
                continue

            # Convert distance to confidence (L2 normalized vectors)
            # Distance range is [0, 2] for normalized vectors
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
        character_name: str,
        image_paths: list[str],
        batch_size: int = Config.DEFAULT_BATCH_SIZE,
        use_segmentation: bool = False,
        concept: str = Config.DEFAULT_CONCEPT,
        save_crops: bool = False,
        source_url: Optional[str] = None,
    ) -> int:
        """Add images for a character to the index.

        Args:
            character_name: Name of the character.
            image_paths: List of image paths or URLs.
            batch_size: Batch size for processing.
            use_segmentation: Whether to segment images (for multi-character images).
            concept: SAM3 text prompt concept (default: "fursuiter").
            save_crops: Whether to save crop images for debugging.
            source_url: Optional source URL for tracking origin.

        Returns:
            Number of images successfully added.
        """
        added_count = 0

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_post_ids = []
            batch_filenames = []

            # Load images
            for img_path in batch_paths:
                try:
                    image = self._load_image(img_path)
                    if image is None:
                        continue
                    batch_images.append(image)
                    batch_post_ids.append(self._extract_post_id(img_path))
                    batch_filenames.append(os.path.basename(img_path))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

            if not batch_images:
                continue

            if use_segmentation:
                # Process each image with segmentation
                for image, post_id, filename in zip(batch_images, batch_post_ids, batch_filenames):
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
                # Batch process without segmentation
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

            print(f"Added {added_count} images for {character_name}")

        # Save index after processing
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
        """Add a single embedding to the index and database."""
        # Add to FAISS index
        embedding_id = self.index.add(embedding.reshape(1, -1))

        # Use current pipeline's segmentor model if not specified
        if segmentor_model is None:
            segmentor_model = self.pipeline.segmentor_model_name

        # Save crop image if provided
        crop_path = None
        if crop_image is not None:
            crops_dir = Path(Config.CROPS_DIR)
            crops_dir.mkdir(exist_ok=True)
            crop_path = str(crops_dir / f"{post_id}_{embedding_id}.jpg")
            crop_image.convert("RGB").save(crop_path, quality=90)

        # Add to database
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

    def process_unindexed(
        self,
        images_dir: str = Config.IMAGES_DIR,
        batch_size: int = Config.DEFAULT_BATCH_SIZE
    ) -> int:
        """Process unindexed images from the download database.

        Reads character info from furtrack.db and indexes images
        that haven't been processed yet.

        Args:
            images_dir: Directory containing downloaded images.
            batch_size: Batch size for processing.

        Returns:
            Number of images processed.
        """
        old_db_path = Config.OLD_DB_PATH

        if not os.path.exists(old_db_path):
            print(f"Old database not found at {old_db_path}")
            return 0

        # Get records from old database
        conn = sqlite3.connect(old_db_path)
        c = conn.cursor()

        c.execute("""
            SELECT post_id, char, url, raw
            FROM furtrack
            WHERE char != '' AND url != ''
        """)

        records = c.fetchall()
        conn.close()

        print(f"Found {len(records)} records in old database")

        # Group by character
        character_images: dict[str, list[str]] = {}

        for post_id, char_name, url, raw in records:
            # Skip if already indexed
            if self.db.has_post(post_id):
                continue

            # Check if image exists
            img_path = os.path.join(images_dir, f"{post_id}.jpg")
            if not os.path.exists(img_path):
                continue

            if char_name not in character_images:
                character_images[char_name] = []
            character_images[char_name].append(img_path)

        # Process each character
        total_added = 0
        for char_name, img_paths in character_images.items():
            print(f"Processing {len(img_paths)} images for {char_name}")
            added = self.add_images(char_name, img_paths, batch_size)
            total_added += added

        print(f"Processed {total_added} total images")
        return total_added

    def _load_image(self, img_path: str) -> Optional[Image.Image]:
        """Load image from path or URL.

        Args:
            img_path: Local path or URL.

        Returns:
            PIL Image or None if loading fails.
        """
        try:
            if img_path.startswith(('http://', 'https://')):
                response = requests.get(img_path, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            else:
                return Image.open(img_path)
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            return None

    def _extract_post_id(self, img_path: str) -> str:
        """Extract post ID from image path.

        Args:
            img_path: Image path or URL.

        Returns:
            Post ID string.
        """
        # Get filename without extension
        basename = os.path.basename(img_path)
        return os.path.splitext(basename)[0]

    def get_stats(self) -> dict:
        """Get system statistics.

        Returns:
            Dictionary with system stats.
        """
        db_stats = self.db.get_stats()
        db_stats["index_size"] = self.index.size
        return db_stats
