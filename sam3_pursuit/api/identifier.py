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
from sam3_pursuit.storage.mask_storage import MaskStorage
from sam3_pursuit.storage.vector_index import VectorIndex


@dataclass
class IdentificationResult:
    character_name: Optional[str]
    confidence: float
    distance: float
    post_id: str
    bbox: tuple[int, int, int, int]
    segmentor_model: str = "unknown"
    source: Optional[str] = None


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
        self.mask_storage = MaskStorage()
        self._sync_index_and_db()
        self.pipeline = ProcessingPipeline(
            device=self.device,
            isolation_config=isolation_config,
            segmentor_model_name=segmentor_model_name,
            segmentor_concept=segmentor_concept
        )

    def _sync_index_and_db(self):
        """Ensure FAISS index and database are in sync (crash recovery)."""
        max_valid_id = self.index.size - 1
        max_db_id = self.db.get_next_embedding_id() - 1
        if max_db_id > max_valid_id:
            deleted = self.db.delete_orphaned_detections(max_valid_id)
            if deleted > 0:
                print(f"Sync: deleted {deleted} orphaned detections (embedding_id > {max_valid_id})")

    def _short_embedder_name(self) -> str:
        emb = self.pipeline.embedder_model_name
        if "dinov2-base" in emb:
            return "dv2b"
        elif "dinov2-large" in emb:
            return "dv2l"
        elif "dinov2-giant" in emb:
            return "dv2g"
        return emb.split("/")[-1][:8]

    def _build_preprocessing_info(self) -> str:
        """Build fingerprint for segmented crops."""
        iso = self.pipeline.isolation_config
        mode_map = {"solid": "s", "blur": "b", "none": "n"}
        parts = [
            "v2",
            f"seg:{self.pipeline.segmentor_model_name}",
            f"con:{(self.pipeline.segmentor_concept or '').replace('|', '.')}",
            f"bg:{mode_map.get(iso.mode, 'n')}",
        ]
        if iso.mode == "solid":
            r, g, b = iso.background_color
            parts.append(f"bgc:{r:02x}{g:02x}{b:02x}")
        elif iso.mode == "blur":
            parts.append(f"br:{iso.blur_radius}")
        parts += [f"emb:{self._short_embedder_name()}", f"tsz:{Config.TARGET_IMAGE_SIZE}"]
        return "|".join(parts)

    def _build_full_preprocessing_info(self) -> str:
        return f"v2|seg:full|emb:{self._short_embedder_name()}|tsz:{Config.TARGET_IMAGE_SIZE}"

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
                self._save_debug_crop(proc_result.isolated_crop, f"{crop_prefix}_{i}", source="search")
            matches = self._search_embedding(proc_result.embedding, top_k)
            segment_results.append(SegmentResults(
                segment_index=i,
                segment_bbox=proc_result.segmentation.bbox,
                segment_confidence=proc_result.segmentation.confidence,
                matches=matches,
            ))
        return segment_results
    

    def _save_debug_crop(
        self,
        image: Image.Image,
        name: str,
        source: Optional[str] = None,
    ) -> None:
        """Save crop image for debugging (optional, use --save-crops)."""
        crops_dir = Path(Config.CROPS_INGEST_DIR) / (source or "unknown")
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crops_dir / f"{name}.jpg"
        image.convert("RGB").save(crop_path, quality=90)

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
                segmentor_model=detection.segmentor_model,
                source=detection.source,
            ))
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:top_k]

    def add_images(
        self,
        character_names: list[str],
        image_paths: list[str],
        save_crops: bool = False,
        source: Optional[str] = None,
        uploaded_by: Optional[str] = None,
        add_full_image: bool = True,
        batch_size: int = 100,
    ) -> int:
        assert len(character_names) == len(image_paths)

        character_names = [name.lower().replace(" ", "_") for name in character_names]

        full_preproc = self._build_full_preprocessing_info() if add_full_image else None
        seg_preproc = self._build_preprocessing_info()

        post_ids = [self._extract_post_id(p) for p in image_paths]
        posts_need_full = self.db.get_posts_needing_update(post_ids, full_preproc, source) if add_full_image else set()
        posts_need_seg = self.db.get_posts_needing_update(post_ids, seg_preproc, source)
        posts_to_process = posts_need_full | posts_need_seg

        print(f"Processing {len(posts_to_process)} posts ({len(posts_need_full)} need full, {len(posts_need_seg)} need seg)")
        filtered_indices = [i for i, pid in enumerate(post_ids) if pid in posts_to_process]
        if not filtered_indices:
            return 0

        total = len(filtered_indices)
        added_count = 0
        pending_embeddings: list[np.ndarray] = []
        pending_detections: list[Detection] = []

        def make_detection(post_id, character_name, bbox, confidence, segmentor_model, filename, preproc_info):
            return Detection(
                id=None, post_id=post_id, character_name=character_name, embedding_id=-1,
                bbox_x=bbox[0], bbox_y=bbox[1], bbox_width=bbox[2], bbox_height=bbox[3],
                confidence=confidence, segmentor_model=segmentor_model,
                source=source, uploaded_by=uploaded_by, source_filename=filename,
                preprocessing_info=preproc_info,
            )

        def flush_batch():
            """Commit DB then save FAISS. If interrupted, _sync_index_and_db cleans up orphans."""
            nonlocal added_count
            if not pending_embeddings:
                return
            start_id = self.index.add(np.vstack(pending_embeddings).astype(np.float32))
            for i, detection in enumerate(pending_detections):
                detection.embedding_id = start_id + i
            self.db.add_detections_batch(pending_detections)
            self.index.save(backup=True)
            added_count += len(pending_detections)
            print(f"  Batch saved: {len(pending_detections)} embeddings (index: {self.index.size})")
            pending_embeddings.clear()
            pending_detections.clear()

        for i, idx in enumerate(filtered_indices):
            character_name = character_names[idx]
            img_path = image_paths[idx]
            post_id = self._extract_post_id(img_path)
            filename = os.path.basename(img_path)

            try:
                image = self._load_image(img_path)
            except Exception as e:
                print(f"[{i+1}/{total}] Failed to load {filename}: {e}")
                continue

            w, h = image.size

            if add_full_image and post_id in posts_need_full:
                resized_full = self.pipeline._resize_to_patch_multiple(image)
                pending_embeddings.append(self.pipeline.embed_only(resized_full).reshape(1, -1))
                pending_detections.append(make_detection(
                    post_id, character_name, (0, 0, w, h), 1.0, "full", filename, full_preproc))

            if post_id in posts_need_seg:
                proc_results = self.pipeline.process(image)
                for j, proc_result in enumerate(proc_results):
                    seg_name = f"{post_id}_seg_{j}"
                    # Always save mask for potential reprocessing
                    if proc_result.segmentation.crop_mask is not None:
                        self.mask_storage.save_mask(
                            proc_result.segmentation.crop_mask, seg_name,
                            source=source or "unknown",
                            model=proc_result.segmentor_model,
                            concept=proc_result.segmentor_concept or "")
                    if save_crops and proc_result.isolated_crop:
                        self._save_debug_crop(proc_result.isolated_crop, seg_name, source=source)
                    pending_embeddings.append(proc_result.embedding.reshape(1, -1))
                    pending_detections.append(make_detection(
                        post_id, character_name, proc_result.segmentation.bbox,
                        proc_result.segmentation.confidence, proc_result.segmentor_model,
                        filename, seg_preproc))

                full_msg = "+full" if (add_full_image and post_id in posts_need_full) else ""
                print(f"[{i+1}/{total}] {character_name}: {len(proc_results)} segments{full_msg}")
            else:
                print(f"[{i+1}/{total}] {character_name}: full only")

            if len(pending_embeddings) >= batch_size:
                flush_batch()

        flush_batch()

        print(f"Ingestion complete: {added_count} embeddings added (index: {self.index.size})")
        return added_count

    def _load_image(self, img_path: str) -> Image.Image:
        img_path = str(img_path)
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
