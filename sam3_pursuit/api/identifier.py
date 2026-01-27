import os
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
        isolation_config: Optional[IsolationConfig] = None,
        segmentor_model_name: Optional[str] = "",
        segmentor_concept: Optional[str] = "",
    ):
        self.device = device or Config.get_device()
        self.db = Database(db_path)
        self.index = VectorIndex(index_path)
        self.pipeline = ProcessingPipeline(
            device=self.device,
            isolation_config=isolation_config,
            segmentor_model_name=segmentor_model_name,
            segmentor_concept=segmentor_concept
        )

    def _build_preprocessing_info(self) -> str:
        parts = []
        seg = self.pipeline.segmentor_model_name or ""
        parts.append(f"seg:{seg}")
        if seg:
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
        save_crops: bool = False,
        crop_prefix: str = "query",
    ) -> list[SegmentResults]:
        if self.index.size == 0:
            print("Warning: Index is empty, no matches possible")
            return []

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
    

    def _save_debug_crop(self, image: Image.Image, name: str, search: bool = True):
        crops_dir = Path(Config.CROPS_SEARCH_DIR if search else Config.CROPS_INGEST_DIR)
        crops_dir.mkdir(parents=True, exist_ok=True)
        path = crops_dir / f"{name}.jpg"
        print(f"Saving debug crop to {path}")
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
        save_crops: bool = False,
        source_url: Optional[str] = None,
        add_full_image: bool = True,
    ) -> int:
        assert len(character_names) == len(image_paths)

        from sam3_pursuit.storage.database import get_git_version

        added_count = 0
        preprocessing_info = self._build_preprocessing_info()
        git_version = get_git_version()

        post_ids = [self._extract_post_id(p) for p in image_paths]
        posts_to_process = self.db.get_posts_needing_update(post_ids, git_version, preprocessing_info)
        print(f"Processing {len(posts_to_process)} new/updated posts out of {len(post_ids)} total")
        filtered_indices = [i for i, pid in enumerate(post_ids) if pid in posts_to_process]
        if not filtered_indices:
            return 0

        total = len(filtered_indices)
        save_interval = 50  # Save index every N images

        for i, idx in enumerate(filtered_indices):
            character_name = character_names[idx]
            img_path = image_paths[idx]
            post_id = self._extract_post_id(img_path)
            filename = os.path.basename(img_path)

            # Load image
            try:
                image = self._load_image(img_path)
            except Exception as e:
                print(f"[{i+1}/{total}] Failed to load {filename}: {e}")
                continue

            w, h = image.size

            # Add full image embedding (no segmentation/isolation)
            if add_full_image:
                resized_full = self.pipeline._resize_to_patch_multiple(image)
                full_embedding = self.pipeline.embed_only(resized_full)
                self._add_single_embedding(
                    embedding=full_embedding,
                    post_id=post_id,
                    character_name=character_name,
                    bbox=(0, 0, w, h),
                    confidence=1.0,
                    segmentor_model="full",
                    source_filename=filename,
                    source_url=source_url,
                    is_cropped=False,
                    segmentation_concept=None,
                    preprocessing_info=preprocessing_info,
                )
                added_count += 1

            # Process: segment -> isolate -> embed
            proc_results = self.pipeline.process(image)

            if not proc_results:
                print(f"[{i+1}/{total}] {character_name}: no segments found (full image added)")
                continue

            for proc_result in proc_results:
                if save_crops and proc_result.isolated_crop:
                    self._save_debug_crop(proc_result.isolated_crop, f"{post_id}_seg_{proc_results.index(proc_result)}", search=False)
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
                    segmentation_concept=proc_result.segmentor_concept,
                    preprocessing_info=preprocessing_info,
                )
                added_count += 1

            segment_count = len(proc_results)
            full_msg = "+full" if add_full_image else ""
            print(f"[{i+1}/{total}] {character_name}: {segment_count} segments{full_msg}, {added_count} total")

            # Periodically save index with backup
            if (i + 1) % save_interval == 0:
                self.index.save(backup=True)
                print(f"  Index saved ({self.index.size} embeddings)")

        self.index.save(backup=True)
        print(f"Final index saved ({self.index.size} embeddings)")
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
    ):
        embedding_id = self.index.add(embedding.reshape(1, -1))
        if segmentor_model is None:
            segmentor_model = self.pipeline.segmentor_model_name

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
