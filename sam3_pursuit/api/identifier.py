import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config, sanitize_path_component
from sam3_pursuit.models.preprocessor import IsolationConfig
from sam3_pursuit.pipeline.processor import CachedProcessingPipeline, CacheKey, SHORT_NAME_TO_CLI
from sam3_pursuit.storage.database import Database
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


def _save_debug_crop(
    image: Image.Image,
    name: str,
    source: Optional[str] = None,
) -> None:
    """Save crop image for debugging (optional, use --save-crops)."""
    crops_dir = Path(Config.CROPS_INGEST_DIR) / sanitize_path_component(source or "unknown")
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crops_dir / f"{sanitize_path_component(name)}.jpg"
    image.convert("RGB").save(crop_path, quality=90)


def discover_datasets(base_dir: str = Config.BASE_DIR) -> list[tuple[str, str]]:
    """Find all (db_path, index_path) pairs in base_dir.

    Matches *.db files that have a corresponding *.index file.
    Ignores backup files (*.backup.*, *.bak*).
    """
    base = Path(base_dir)
    if not base.is_dir():
        return []
    datasets = []
    for db_file in sorted(base.glob("*.db")):
        name = db_file.name
        if ".backup." in name or ".bak" in name:
            continue
        index_file = db_file.with_suffix(".index")
        if index_file.exists() and ".backup." not in index_file.name and ".bak" not in index_file.name:
            datasets.append((str(db_file), str(index_file)))
    return datasets


class FursuitIdentifier:
    """Read-only identification across a dataset."""

    def __init__(
        self,
        db_path: str,
        index_path: str,
        device: Optional[str] = None,
        isolation_config: Optional[IsolationConfig] = None,
        segmentor_model_name: Optional[str] = "",
        segmentor_concept: Optional[str] = "",
        embedder=None,
        preprocessors: Optional[list] = None,
    ):

        if not embedder:
            emb = detect_embedder(db_path=db_path)
            embedder = build_embedder_for_name(short_name = emb, device = device)

        embedding_dim = embedder.embedding_dim if embedder else Config.EMBEDDING_DIM

        self.db = Database(db_path)
        self.index = VectorIndex(index_path, embedding_dim=embedding_dim)

        # Non-cached pipeline for query processing only
        self.pipeline = CachedProcessingPipeline(
            device=device,
            isolation_config=isolation_config,
            segmentor_model_name=segmentor_model_name,
            segmentor_concept=segmentor_concept,
            embedder=embedder,
            preprocessors=preprocessors,
        )

        self.fallback_pipeline = CachedProcessingPipeline(
            device=device,
            isolation_config=isolation_config,
            segmentor_model_name="full",
            segmentor_concept="",
            embedder=embedder,
            preprocessors=preprocessors,
        )

    @property
    def total_index_size(self) -> int:
        return self.index.size

    def identify(
        self,
        image: Image.Image,
        top_k: int = Config.DEFAULT_TOP_K,
        save_crops: bool = False,
        crop_prefix: str = "query",
    ) -> list[SegmentResults]:
        if self.total_index_size == 0:
            print("WARN: Index is empty, no matches possible")
            return []

        # Generate cache key from image content so segmentation masks are
        # reused when the same image is processed by multiple identifiers.
        image_bytes = image.tobytes()
        image_hash = hashlib.md5(image_bytes).hexdigest()
        cache_key = CacheKey(post_id=image_hash, source="query")

        proc_results = self.pipeline.process(image, cache_key=cache_key)
        if not proc_results:
            proc_results = self.fallback_pipeline.process(image, cache_key=cache_key)

        segment_results = []
        for i, proc_result in enumerate(proc_results):
            if save_crops and proc_result.isolated_crop:
                _save_debug_crop(proc_result.isolated_crop, f"{crop_prefix}_{i}", source="search")
            matches = self._search_embedding(proc_result.embedding, top_k)
            segment_results.append(SegmentResults(
                segment_index=i,
                segment_bbox=proc_result.segmentation.bbox,
                segment_confidence=proc_result.segmentation.confidence,
                matches=matches,
            ))
        return segment_results

    def _search_embedding(self, embedding: np.ndarray, top_k: int):
        all_results: list[IdentificationResult] = []

        if self.total_index_size == 0:
            return all_results
        distances, indices = self.index.search(embedding, top_k * 2)
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            detection = self.db.get_detection_by_embedding_id(int(idx))
            if detection is None:
                continue
            confidence = max(0.0, 1.0 - distance / 2.0)
            all_results.append(IdentificationResult(
                character_name=detection.character_name,
                confidence=confidence,
                distance=float(distance),
                post_id=detection.post_id,
                bbox=(detection.bbox_x, detection.bbox_y,
                        detection.bbox_width, detection.bbox_height),
                segmentor_model=detection.segmentor_model,
                source=detection.source,
            ))
        # Deduplicate by character name: keep only the best match per character
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        seen_chars: dict[str, None] = {}
        deduped: list[IdentificationResult] = []
        for r in all_results:
            key = (r.character_name or "").lower()
            if key not in seen_chars:
                seen_chars[key] = None
                deduped.append(r)
        return deduped[:top_k]

    def search_text(self, text: str, top_k: int = Config.DEFAULT_TOP_K) -> list[IdentificationResult]:
        """Search for characters by text description. Requires CLIP or SigLIP embedder."""
        embedder = self.pipeline.embedder
        if not hasattr(embedder, "embed_text"):
            raise ValueError(
                f"Text search requires a CLIP or SigLIP embedder. "
                f"This dataset uses {self.pipeline.get_embedder_short_name()}."
            )
        if self.total_index_size == 0:
            print("Warning: All indexes are empty, no matches possible")
            return []
        embedding = embedder.embed_text(text)
        results = self._search_embedding(embedding, top_k)
        # Use embedder-native confidence if available (e.g. SigLIP sigmoid scaling)
        if hasattr(embedder, "text_confidence"):
            for r in results:
                r.confidence = embedder.text_confidence(r.distance)
            results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def get_stats(self) -> dict:
        stats = self.db.get_stats()
        stats["index_size"] = self.total_index_size
        return stats

    @staticmethod
    def get_combined_stats(stats_list: list[dict]):
        from collections import Counter
        # Multi-dataset: aggregate
        ret = {
            "total_detections": 0,
            "unique_characters": 0,
            "unique_posts": 0,
            "top_characters": 0,
            "segmentor_breakdown": {},
            "preprocessing_breakdown": {},
            "git_version_breakdown": {},
            "source_breakdown": {},
            "index_size": 0,
            "num_datasets": len(stats_list),
        }
        for k, v in ret.items():
            if isinstance(v, dict):
                cnt = Counter(v)
                for stats in stats_list:
                    cnt.update(stats.get(k, {}))
                ret[k] = dict(cnt)
            else:
                ret[k] = sum([stats.get(k, 0) for stats in stats_list])
        return ret


def merge_multi_dataset_results(
    all_results: list[list[SegmentResults]],
    top_k: int = 5,
) -> list[SegmentResults]:
    """Merge results from multiple datasets using reciprocal rank fusion.

    Within each dataset, matches are already deduplicated by character name
    (done in _search_embedding). Cross-dataset merging sums 1/rank scores
    per character, breaking ties by best raw confidence.

    Returns merged SegmentResults with deduplicated, ranked matches.
    """
    if not all_results:
        return []

    # Use the first non-empty result set to determine segment structure
    base = None
    for r in all_results:
        if r:
            base = r
            break
    if base is None:
        return []

    merged = []
    for seg_idx, base_seg in enumerate(base):
        # Collect matches for this segment across all datasets
        # Each dataset's matches are already deduped by character
        per_dataset_matches: list[list[IdentificationResult]] = []
        for dataset_results in all_results:
            if seg_idx < len(dataset_results):
                per_dataset_matches.append(dataset_results[seg_idx].matches)

        # Reciprocal rank fusion: sum 1/rank per character across datasets
        # Also track best raw confidence for tie-breaking
        char_scores: dict[str, float] = {}
        char_best_match: dict[str, IdentificationResult] = {}

        for matches in per_dataset_matches:
            for rank, m in enumerate(matches, 1):
                key = (m.character_name or "").lower()
                char_scores[key] = char_scores.get(key, 0.0) + 1.0 / rank
                if key not in char_best_match or m.confidence > char_best_match[key].confidence:
                    char_best_match[key] = m

        # Sort by RRF score (descending), then by best confidence (descending)
        ranked = sorted(
            char_scores.keys(),
            key=lambda k: (char_scores[k], char_best_match[k].confidence),
            reverse=True,
        )

        merged_matches = sorted(
            [char_best_match[k] for k in ranked[:top_k]],
            key=lambda m: m.confidence,
            reverse=True,
        )
        merged.append(SegmentResults(
            segment_index=base_seg.segment_index,
            segment_bbox=base_seg.segment_bbox,
            segment_confidence=base_seg.segment_confidence,
            matches=merged_matches,
        ))

    return merged


def detect_embedder(db_path: str, default: str = Config.DEFAULT_EMBEDDER):
    """Detects the short name of the embedder from the dataset"""
    emb = Database.read_metadata_lightweight(db_path, Config.METADATA_KEY_EMBEDDER)
    if emb:
        print(f"Auto-detected embedder for {db_path}: {emb} ({SHORT_NAME_TO_CLI.get(emb, emb)})")
        return emb
    print(f"WARN: Could not find embedder for {db_path}, using default: {default}")
    return default


def build_embedder_for_name(short_name: str, device: Optional[str] = None):
    """Instantiate an embedder from its short name (e.g. 'siglip', 'dv2b', 'clip')."""
    from sam3_pursuit.pipeline.processor import SHORT_NAME_TO_CLI

    cli_name = SHORT_NAME_TO_CLI.get(short_name, short_name)
    if cli_name in ("siglip", "google/siglip-base-patch16-224"):
        from sam3_pursuit.models.embedder import SigLIPEmbedder
        return SigLIPEmbedder(device=device)
    if cli_name == "clip":
        from sam3_pursuit.models.embedder import CLIPEmbedder
        return CLIPEmbedder(device=device)
    if cli_name == "dinov2-base":
        from sam3_pursuit.models.embedder import DINOv2Embedder
        return DINOv2Embedder(model_name=Config.DINOV2_MODEL, device=device)
    if cli_name == "dinov2-large":
        from sam3_pursuit.models.embedder import DINOv2Embedder
        return DINOv2Embedder(model_name=Config.DINOV2_LARGE_MODEL, device=device)
    if cli_name == "dinov2-base+colorhist":
        from sam3_pursuit.models.embedder import DINOv2Embedder, ColorHistogramEmbedder
        return ColorHistogramEmbedder(DINOv2Embedder(device=device))
    # Fallback: default SigLIP
    from sam3_pursuit.models.embedder import SigLIPEmbedder
    return SigLIPEmbedder(device=device)
