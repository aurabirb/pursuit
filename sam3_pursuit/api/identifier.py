import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config, sanitize_path_component
from sam3_pursuit.models.preprocessor import IsolationConfig
from sam3_pursuit.pipeline.processor import CachedProcessingPipeline, CacheKey
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
    """Read-only identification across one or more datasets."""

    def __init__(
        self,
        datasets: list[tuple[str, str]],
        device: Optional[str] = None,
        isolation_config: Optional[IsolationConfig] = None,
        segmentor_model_name: Optional[str] = "",
        segmentor_concept: Optional[str] = "",
        embedder=None,
        preprocessors: Optional[list] = None,
    ):
        """
        Args:
            datasets: list of (db_path, index_path) tuples to search across.
        """

        self.stores: list[tuple[Database, VectorIndex]] = []
        embedding_dim = embedder.embedding_dim if embedder else Config.EMBEDDING_DIM
        for db_path, index_path in datasets:
            db = Database(db_path)
            index = VectorIndex(index_path, embedding_dim=embedding_dim)
            self.stores.append((db, index))

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
        return sum(index.size for _, index in self.stores)

    def identify(
        self,
        image: Image.Image,
        top_k: int = Config.DEFAULT_TOP_K,
        save_crops: bool = False,
        crop_prefix: str = "query",
    ) -> list[SegmentResults]:
        if self.total_index_size == 0:
            print("Warning: All indexes are empty, no matches possible")
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

    def _search_embedding(self, embedding: np.ndarray, top_k: int) -> list[IdentificationResult]:
        """Search all stores, merge results, deduplicate by character (best confidence wins)."""
        all_results: list[IdentificationResult] = []

        for db, index in self.stores:
            if index.size == 0:
                continue
            distances, indices = index.search(embedding, top_k * 2)
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                detection = db.get_detection_by_embedding_id(int(idx))
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

        all_results.sort(key=lambda x: x.confidence, reverse=True)
        return all_results[:top_k]

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
        return self._search_embedding(embedding, top_k)

    def get_stats(self) -> dict:
        """Aggregated stats across all stores."""
        if len(self.stores) == 1:
            db, index = self.stores[0]
            stats = db.get_stats()
            stats["index_size"] = index.size
            return stats

        # Multi-dataset: aggregate
        total_detections = 0
        all_characters = set()
        all_posts = set()
        total_index_size = 0

        for db, index in self.stores:
            stats = db.get_stats()
            total_detections += stats["total_detections"]
            total_index_size += index.size
            # Get character names from this db
            all_characters.update(db.get_all_character_names())
            # Get post ids
            conn = db._connect()
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT post_id FROM detections")
            all_posts.update(row[0] for row in cursor.fetchall())

        return {
            "total_detections": total_detections,
            "unique_characters": len(all_characters),
            "unique_posts": len(all_posts),
            "index_size": total_index_size,
            "num_datasets": len(self.stores),
        }


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


def create_identifiers(
    datasets: Optional[list[tuple[str, str]]] = None,
    base_dir: str = Config.BASE_DIR,
    **kwargs,
) -> list["FursuitIdentifier"]:
    """Create one FursuitIdentifier per embedder group.

    If datasets is None, auto-discovers all *.db/*.index pairs in base_dir.
    Groups datasets by stored embedder metadata and creates a separate
    identifier with the correct embedder for each group.

    Additional kwargs are passed to FursuitIdentifier (device, isolation_config, etc.).
    The 'embedder' kwarg is overridden per group.
    """
    from sam3_pursuit.pipeline.processor import DEFAULT_EMBEDDER_SHORT

    if datasets is None:
        datasets = discover_datasets(base_dir)
        if not datasets:
            raise FileNotFoundError(
                f"No datasets found in {base_dir}. "
                "Expected *.db and *.index file pairs."
            )
        names = [Path(db).stem for db, _ in datasets]
        print(f"Auto-discovered {len(datasets)} dataset(s): {', '.join(names)}")

    # Group datasets by embedder metadata
    groups: dict[str, list[tuple[str, str]]] = {}
    for db_path, index_path in datasets:
        emb = Database.read_metadata_lightweight(db_path, Config.METADATA_KEY_EMBEDDER)
        groups.setdefault(emb or DEFAULT_EMBEDDER_SHORT, []).append((db_path, index_path))

    kwargs.pop("embedder", None)  # discard: we build per-group embedders
    identifiers = []
    for embedder_short, group_datasets in groups.items():
        embedder = build_embedder_for_name(embedder_short, device=kwargs.get("device"))
        names = [Path(db).stem for db, _ in group_datasets]
        print(f"Identifier for {embedder_short}: {', '.join(names)}")
        ident = FursuitIdentifier(
            datasets=group_datasets,
            embedder=embedder,
            **kwargs,
        )
        identifiers.append(ident)
    return identifiers
