import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

from sam3_pursuit.config import Config
from sam3_pursuit.storage.database import (
    SOURCES_AVAILABLE, SOURCE_NFC25,
    get_source_url,
)

from sam3_pursuit.pipeline.processor import DEFAULT_EMBEDDER_SHORT, SHORT_NAME_TO_CLI


def _get_dataset_paths(dataset: str) -> tuple[str, str]:
    """Get db and index paths for a dataset name."""
    if dataset == Config.DEFAULT_DATASET:
        return Config.DB_PATH, Config.INDEX_PATH
    base_dir = os.path.dirname(Config.DB_PATH)
    return os.path.join(base_dir, f"{dataset}.db"), os.path.join(base_dir, f"{dataset}.index")


def _get_dataset_dir(dataset: str) -> Path:
    """Get the root image directory for a dataset.

    Non-default datasets: datasets/{dataset}/
    Default dataset: project root (sources are {source}_images/ dirs)
    """
    if dataset == Config.DEFAULT_DATASET:
        return Path(Config.BASE_DIR)
    return Path(Config.BASE_DIR) / "datasets" / dataset


def _get_source_subdirs(dataset_dir: Path, dataset: str) -> list[tuple[str, Path]]:
    """Get (source_name, path) pairs for all source subdirs in a dataset dir.

    Non-default: datasets/{dataset}/{source}/ -> source = dir name
    Default: {source}_images/ at project root -> source = name without _images
    """
    if not dataset_dir.is_dir():
        return []
    results = []
    if dataset == Config.DEFAULT_DATASET:
        for d in sorted(dataset_dir.iterdir()):
            if d.is_dir() and d.name.endswith("_images"):
                source_name = d.name.removesuffix("_images")
                results.append((source_name, d))
    else:
        for d in sorted(dataset_dir.iterdir()):
            if d.is_dir():
                results.append((d.name, d))
    return results


def _copy_dataset_files(
    source_dataset: str,
    target_dataset: str,
    by_source: str | None = None,
    by_character: str | None = None,
    shard_idx: int | None = None,
    shards: int = 1,
) -> int:
    """Copy image files from source dataset dir to target dataset dir.

    Filters by source subdir name and character dir name.
    When shards > 1, filters by hash(image_stem) % shards == shard_idx.
    Returns count of files copied.
    """
    src_dir = _get_dataset_dir(source_dataset)
    tgt_dir = _get_dataset_dir(target_dataset)

    source_subdirs = _get_source_subdirs(src_dir, source_dataset)
    if by_source:
        source_subdirs = [(name, path) for name, path in source_subdirs if name == by_source]

    char_names = None
    if by_character:
        char_names = {c.strip() for c in by_character.split(",")}

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    copied = 0

    for source_name, source_path in source_subdirs:
        # Determine target source subdir name
        if target_dataset == Config.DEFAULT_DATASET:
            tgt_source_dir = tgt_dir / f"{source_name}_images"
        else:
            tgt_source_dir = tgt_dir / source_name

        for char_dir in sorted(source_path.iterdir()):
            if not char_dir.is_dir():
                continue
            if char_names and char_dir.name not in char_names:
                continue

            for img_file in sorted(char_dir.iterdir()):
                if img_file.suffix.lower() not in IMAGE_EXTS:
                    continue
                if shards > 1 and shard_idx is not None:
                    if hash(img_file.stem) % shards != shard_idx:
                        continue

                dest = tgt_source_dir / char_dir.name / img_file.name
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_file, dest)
                copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser(
        prog="pursuit",
        description="Fursuit character recognition using SAM3 and DINOv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  pursuit identify photo.jpg
  pursuit add -c "CharName" -s manual img1.jpg img2.jpg
  pursuit ingest directory --data-dir ./characters/ -s furtrack
  pursuit download furtrack --all
  pursuit stats

  # Validation workflow (--dataset sets output-dir and excludes main dataset)
  pursuit --dataset validation download furtrack -c "CharName" --max-images 3
  pursuit --dataset validation ingest directory -s furtrack
  pursuit evaluate  # --from validation --against {Config.DEFAULT_DATASET}
        """
    )
    parser.add_argument("--dataset", "-d", "-ds", default=Config.DEFAULT_DATASET,
                        help=f"Dataset name (default: {Config.DEFAULT_DATASET}). Sets db/index paths to <name>.db/<name>.index")
    parser.add_argument("--no-segment", "-S", dest="segment", action="store_false", help="Do not use segmentation")
    parser.add_argument("--concept", default=Config.DEFAULT_CONCEPT, help="SAM3 concept")
    parser.add_argument("--background", "-bg", default=Config.DEFAULT_BACKGROUND_MODE,
                        choices=["none", "solid", "blur"],
                        help="Background isolation mode (default: solid)")
    parser.add_argument("--bg-color", default="128,128,128",
                        help="Background color as R,G,B for solid mode (default: 128,128,128)")
    parser.add_argument("--blur-radius", type=int, default=Config.DEFAULT_BLUR_RADIUS,
                        help="Blur radius for blur mode (default: 25)")
    parser.add_argument("--embedder", "-emb",
                        choices=list(SHORT_NAME_TO_CLI.values()),
                        default=Config.DEFAULT_EMBEDDER,
                        help=f"Embedder model (default: {Config.DEFAULT_EMBEDDER})")
    parser.add_argument("--grayscale", "-gray", action="store_true",
                        help="Apply grayscale preprocessing before embedding")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    identify_parser = subparsers.add_parser("identify", help="Identify character in an image")
    identify_parser.add_argument("image", help="Path to image file")
    identify_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    identify_parser.add_argument("--min-confidence", "-m", type=float, default=Config.DEFAULT_MIN_CONFIDENCE,
                                 help="Minimum confidence threshold.")
    identify_parser.add_argument("--save-crops", action="store_true", help="Save preprocessed crops for debugging")

    add_parser = subparsers.add_parser("add", help="Add images for a character")
    add_parser.add_argument("--character", "-c", required=True, help="Character name")
    add_parser.add_argument("--source", "-s", required=True, choices=SOURCES_AVAILABLE,
                           help="Source dataset for provenance")
    add_parser.add_argument("images", nargs="+", help="Image paths")
    add_parser.add_argument("--save-crops", action="store_true", help="Save crop images for debugging")
    add_parser.add_argument("--no-full", dest="add_full_image", action="store_false",
                           help="Don't add full image embedding (only segments)")
    _add_classify_args(add_parser)

    show_parser = subparsers.add_parser("show", help="View database entries")
    show_parser.add_argument("--by-id", type=int, help="Query by detection ID")
    show_parser.add_argument("--by-character", help="Query by character name")
    show_parser.add_argument("--by-post", help="Query by post ID")
    show_parser.add_argument("--json", action="store_true", help="Output as JSON")

    ingest_parser = subparsers.add_parser("ingest", help="Bulk ingest images")
    ingest_parser.add_argument("method", choices=["directory", "nfc25", "barq"],
                               help="Ingestion method")
    ingest_parser.add_argument("--data-dir", "-dd", help="Data directory (default: datasets/<dataset>/<source>)")
    ingest_parser.add_argument("--source", "-s", required=False, choices=SOURCES_AVAILABLE,
                           help="Source dataset for provenance")
    ingest_parser.add_argument("--limit", type=int, help="Limit number of images per character")
    ingest_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    ingest_parser.add_argument("--save-crops", action="store_true", help="Save crop images")
    ingest_parser.add_argument("--no-full", dest="add_full_image", action="store_false",
                               help="Don't add full image embedding (only segments)")
    _add_classify_args(ingest_parser)

    stats_parser = subparsers.add_parser("stats", help="Show system statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    segment_parser = subparsers.add_parser("segment", help="Test segmentation on an image")
    segment_parser.add_argument("images", nargs="+", help="Image paths")
    segment_parser.add_argument("--output-dir", "-o", help="Output directory for crops")
    segment_parser.add_argument("--cache-masks", action="store_true", help="Read and write masks cache")
    segment_parser.add_argument("--source", "-s", required='--cache' in sys.argv, choices=SOURCES_AVAILABLE)

    classify_parser = subparsers.add_parser("classify", help="Classify images as fursuit or not")
    classify_parser.add_argument("path", help="Path to image file or directory")
    classify_parser.add_argument("--threshold", type=float, default=Config.DEFAULT_CLASSIFY_THRESHOLD,
                                 help="Threshold for fursuit classification")
    classify_parser.add_argument("--json", action="store_true", help="Output as JSON")

    download_parser = subparsers.add_parser("download", help="Download images from external sources")
    download_subparsers = download_parser.add_subparsers(dest="source", help="Download source")

    furtrack_parser = download_subparsers.add_parser("furtrack", help="Download from FurTrack")
    furtrack_parser.add_argument("--character", "-c", help="Download specific character")
    furtrack_parser.add_argument("--all", "-a", dest="download_all", action="store_true",
                                  help="Download all characters")
    furtrack_parser.add_argument("--max-images", "-m", type=int, default=2,
                                  help="Max images per character (default: 2)")
    furtrack_parser.add_argument("--output-dir", "-o", default="furtrack_images",
                                  help="Output directory (default: furtrack_images)")
    furtrack_parser.add_argument("--exclude-datasets", "-e", help="Skip post_ids in these datasets (comma-separated)")

    barq_parser = download_subparsers.add_parser("barq", help="Download from Barq (requires BARQ_BEARER_TOKEN)")
    barq_parser.add_argument("--lat", type=float, default=52.378, help="Latitude (default: Amsterdam)")
    barq_parser.add_argument("--lon", type=float, default=4.9, help="Longitude")
    barq_parser.add_argument("--max-pages", type=int, default=100, help="Max pages to fetch")
    barq_parser.add_argument("--all-images", action="store_true", help="Download all images per profile")
    barq_parser.add_argument("--max-age", type=float, help="Skip profiles cached within N days")
    barq_parser.add_argument("--output-dir", "-o", default="barq_images", help="Output directory (default: barq_images)")
    barq_parser.add_argument("--clean", action="store_true", help="Delete existing images below threshold (no download)")
    barq_parser.add_argument("--exclude-datasets", "-e", help="Skip post_ids in these datasets (comma-separated)")
    _add_classify_args(barq_parser, default=True)

    combine_parser = subparsers.add_parser("combine", help="Combine multiple datasets into one")
    combine_parser.add_argument("datasets", nargs="+", help="Source dataset names")
    combine_parser.add_argument("--output", "-o", required=True, help="Target dataset name")

    split_parser = subparsers.add_parser("split", help="Split a dataset by criteria")
    split_parser.add_argument("source_dataset", help="Source dataset name")
    split_parser.add_argument("--output", "-o", required=True, help="Target dataset name")
    split_parser.add_argument("--by-source", help="Filter by ingestion source")
    split_parser.add_argument("--by-character", help="Filter by character name(s), comma-separated")
    split_parser.add_argument("--shards", type=int, default=1, help="Number of shards to split into (default: 1)")

    search_parser = subparsers.add_parser("search", help="Search for characters by text description (CLIP/SigLIP only)")
    search_parser.add_argument("query", help="Text description of the fursuit to search for")
    search_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("--min-confidence", "-m", type=float, default=Config.DEFAULT_MIN_CONFIDENCE,
                               help="Minimum confidence threshold")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate one dataset against another")
    evaluate_parser.add_argument("--from", dest="from_dataset", default="validation",
                                  help="Dataset to evaluate (default: validation)")
    evaluate_parser.add_argument("--against", default=Config.DEFAULT_DATASET,
                                  help=f"Dataset to query against (default: {Config.DEFAULT_DATASET})")
    evaluate_parser.add_argument("--top-k", "-k", type=int, default=5, help="Top-k for accuracy calculation")
    evaluate_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Set db/index paths from --dataset
    args.db, args.index = _get_dataset_paths(args.dataset)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "identify":
        identify_command(args)
    elif args.command == "add":
        add_command(args)
    elif args.command == "show":
        show_command(args)
    elif args.command == "ingest":
        ingest_command(args)
    elif args.command == "stats":
        stats_command(args)
    elif args.command == "segment":
        segment_command(args)
    elif args.command == "classify":
        classify_command(args)
    elif args.command == "download":
        download_command(args)
    elif args.command == "search":
        search_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "combine":
        combine_command(args)
    elif args.command == "split":
        split_command(args)


def _add_classify_args(parser, default=None):
    parser.add_argument("--skip-non-fursuit", action=argparse.BooleanOptionalAction,
                        default=default,
                        help="Skip non-fursuit images using CLIP classifier")
    parser.add_argument("--threshold", type=float, default=Config.DEFAULT_CLASSIFY_THRESHOLD,
                        help="Classification threshold for --skip-non-fursuit")


def _get_isolation_config(args):
    from sam3_pursuit.models.preprocessor import IsolationConfig

    bg_color = Config.DEFAULT_BACKGROUND_COLOR
    if hasattr(args, "bg_color") and args.bg_color:
        try:
            parts = args.bg_color.split(",")
            bg_color = (int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            print(f"Warning: Invalid bg-color '{args.bg_color}', using default")

    mode = args.background if hasattr(args, "background") else Config.DEFAULT_BACKGROUND_MODE
    blur_radius = args.blur_radius if hasattr(args, "blur_radius") else Config.DEFAULT_BLUR_RADIUS

    return IsolationConfig(
        mode=mode,
        background_color=bg_color,
        blur_radius=blur_radius
    )


def _build_embedder(args):
    """Build embedder from CLI args. Returns (embedder, None) or None for default."""
    embedder_name = getattr(args, "embedder", Config.DEFAULT_EMBEDDER)
    if embedder_name == Config.DEFAULT_EMBEDDER:
        return None  # use pipeline default
    if embedder_name == "dinov2-base":
        from sam3_pursuit.models.embedder import DINOv2Embedder
        return DINOv2Embedder(model_name=Config.DINOV2_MODEL)
    if embedder_name == "dinov2-large":
        from sam3_pursuit.models.embedder import DINOv2Embedder
        return DINOv2Embedder(model_name=Config.DINOV2_LARGE_MODEL)
    if embedder_name == "clip":
        from sam3_pursuit.models.embedder import CLIPEmbedder
        return CLIPEmbedder()
    if embedder_name == "siglip":
        from sam3_pursuit.models.embedder import SigLIPEmbedder
        return SigLIPEmbedder()
    if embedder_name == "dinov2-base+colorhist":
        from sam3_pursuit.models.embedder import DINOv2Embedder, ColorHistogramEmbedder
        return ColorHistogramEmbedder(DINOv2Embedder())
    return None


def _build_preprocessors(args):
    """Build preprocessor list from CLI args."""
    preprocessors = []
    if getattr(args, "grayscale", False):
        from sam3_pursuit.models.preprocessor import grayscale_preprocessor
        preprocessors.append(grayscale_preprocessor)
    return preprocessors or None


def _auto_detect_embedder(args):
    """If user didn't explicitly pass --embedder, check DB metadata for stored embedder.

    Uses Database.read_metadata_lightweight to avoid double Database() init.
    """
    from sam3_pursuit.storage.database import Database

    embedder_name = getattr(args, "embedder", Config.DEFAULT_EMBEDDER)
    if embedder_name != Config.DEFAULT_EMBEDDER:
        return  # User explicitly chose an embedder

    db_path = getattr(args, "db", None)
    if not db_path:
        return

    stored = Database.read_metadata_lightweight(db_path, Config.METADATA_KEY_EMBEDDER)
    if stored and stored != DEFAULT_EMBEDDER_SHORT:
        cli_name = SHORT_NAME_TO_CLI.get(stored)
        if cli_name:
            print(f"Auto-detected embedder from dataset: {cli_name} ({stored})")
            args.embedder = cli_name


def _get_ingestor(args):
    from sam3_pursuit.api.identifier import FursuitIngestor
    isolation_config = _get_isolation_config(args)
    segmentor_model_name = Config.SAM3_MODEL if getattr(args, "segment", True) else None
    segmentor_concept = args.concept if hasattr(args, "concept") and args.concept else Config.DEFAULT_CONCEPT
    _auto_detect_embedder(args)
    print(f"Using embedder: {args.get("embedder", Config.DEFAULT_EMBEDDER)}")
    embedder = _build_embedder(args)
    preprocessors = _build_preprocessors(args)

    return FursuitIngestor(
        db_path=args.db,
        index_path=args.index,
        isolation_config=isolation_config,
        segmentor_model_name=segmentor_model_name,
        segmentor_concept=segmentor_concept,
        embedder=embedder,
        preprocessors=preprocessors)


def _get_excluded_post_ids(exclude_datasets: str) -> set[str]:
    """Get all post_ids from the specified datasets."""
    from sam3_pursuit.storage.database import Database

    if not exclude_datasets:
        return set()

    excluded = set()
    for dataset in exclude_datasets.split(","):
        dataset = dataset.strip()
        if not dataset:
            continue
        db_path, _ = _get_dataset_paths(dataset)
        if not os.path.exists(db_path):
            print(f"Warning: Dataset '{dataset}' not found at {db_path}")
            continue
        db = Database(db_path)
        conn = db._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT post_id FROM detections")
        excluded.update(row[0] for row in cursor.fetchall())
        db.close()
        print(f"Excluding {len(excluded)} post_ids from {dataset}")

    return excluded


def identify_command(args):
    """Handle identify command."""

    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    ingestor = _get_ingestor(args)
    image = Image.open(image_path)
    save_crops = getattr(args, "save_crops", False)
    crop_prefix = image_path.stem
    results = ingestor.identify(
        image,
        top_k=args.top_k,
        save_crops=save_crops,
        crop_prefix=crop_prefix,
    )

    if not results:
        print("No matches found.")
        return

    min_confidence = args.min_confidence

    print(f"\nAnalyzed {len(results)} segment(s) in {image_path.name}")
    print("=" * 60)

    for seg_result in results:
        print(f"\nSegment {seg_result.segment_index + 1}")
        print(f"  BBox: {seg_result.segment_bbox}")
        print(f"  Detection confidence: {seg_result.segment_confidence:.2%}")
        print("-" * 60)

        # Filter matches below threshold
        filtered_matches = [m for m in seg_result.matches if m.confidence >= min_confidence]

        if not filtered_matches:
            print(f"  No matches found above {min_confidence:.0%} confidence.")
        else:
            print(f"  Top {len(filtered_matches)} matches (>= {min_confidence:.0%}):")
            for i, match in enumerate(filtered_matches, 1):
                print(f"  {i}. {match.character_name or 'Unknown'}")
                print(f"     Confidence: {match.confidence:.2%}")
                print(f"     Distance: {match.distance:.4f}")
                print(f"     Post ID: {match.post_id}")
                if match.source:
                    print(f"     Source: {match.source}")
                url = get_source_url(match.source, match.post_id)
                if url:
                    print(f"     URL: {url}")


def search_command(args):
    """Handle search command - text-based search (lightweight, no SAM3 needed)."""
    from sam3_pursuit.storage.database import Database

    _auto_detect_embedder(args)
    embedder = _build_embedder(args)

    # Check embedder supports text search
    if embedder is None:
        # Default embedder (DINOv2) - no text support
        stored = Database.read_metadata_lightweight(args.db, Config.METADATA_KEY_EMBEDDER)
        embedder_name = stored or DEFAULT_EMBEDDER_SHORT
        cli_name = SHORT_NAME_TO_CLI.get(embedder_name, embedder_name)
        print(f"Error: Text search requires a CLIP or SigLIP embedder. This dataset uses {cli_name}.")
        sys.exit(1)

    if not hasattr(embedder, "embed_text"):
        print(f"Error: Text search requires a CLIP or SigLIP embedder. The '{args.embedder}' embedder does not support text search.")
        sys.exit(1)

    if not _dataset_has_db(args.dataset):
        print(f"Error: Dataset '{args.dataset}' not found.")
        sys.exit(1)

    db, index = _open_dataset(args.dataset)

    if index.size == 0:
        print("Error: Index is empty, no matches possible.")
        sys.exit(1)

    # Embed text query and search
    embedding = embedder.embed_text(args.query)
    top_k = args.top_k
    min_confidence = args.min_confidence

    distances, indices = index.search(embedding, top_k * 3)

    # Collect results, deduplicate by character (best match per character)
    seen_characters = {}
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        detection = db.get_detection_by_embedding_id(int(idx))
        if detection is None:
            continue
        confidence = max(0.0, 1.0 - distance / 2.0)
        if confidence < min_confidence:
            continue
        char_name = detection.character_name or "unknown"
        if char_name not in seen_characters or confidence > seen_characters[char_name]["confidence"]:
            seen_characters[char_name] = {
                "character_name": char_name,
                "confidence": confidence,
                "distance": float(distance),
                "post_id": detection.post_id,
                "source": detection.source,
                "num_matches": seen_characters.get(char_name, {}).get("num_matches", 0) + 1,
            }
        else:
            seen_characters[char_name]["num_matches"] = seen_characters[char_name].get("num_matches", 0) + 1

    results = sorted(seen_characters.values(), key=lambda x: x["confidence"], reverse=True)[:top_k]

    if args.json:
        print(json.dumps({"query": args.query, "results": results}, indent=2))
    else:
        if not results:
            print(f"No matches found for: \"{args.query}\"")
            return

        print(f"\nSearch results for: \"{args.query}\"")
        print("=" * 60)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['character_name']}")
            print(f"     Confidence: {r['confidence']:.2%}")
            print(f"     Distance: {r['distance']:.4f}")
            print(f"     Best match post: {r['post_id']}")
            if r["source"]:
                url = get_source_url(r["source"], r["post_id"])
                if url:
                    print(f"     URL: {url}")
            if r["num_matches"] > 1:
                print(f"     ({r['num_matches']} matching embeddings)")

    db.close()


def add_command(args):
    """Handle add command."""
    # Verify images exist
    valid_paths = []
    for img_path in args.images:
        if Path(img_path).exists():
            valid_paths.append(img_path)
        else:
            print(f"Warning: Image not found: {img_path}")

    if not valid_paths:
        print("Error: No valid images provided.")
        sys.exit(1)

    if not args.character:
        print("Error: Character name is required.")
        sys.exit(1)

    ingestor = _get_ingestor(args)
    add_full_image = getattr(args, "add_full_image", True)
    if args.source not in SOURCES_AVAILABLE:
        print(f"Error: Invalid source '{args.source}'. Must be one of: {', '.join(SOURCES_AVAILABLE)}")
        sys.exit(1)
    added = ingestor.add_images(
        character_names=[args.character] * len(valid_paths),
        image_paths=valid_paths,
        save_crops=args.save_crops,
        source=args.source,
        add_full_image=add_full_image,
        skip_non_fursuit=args.skip_non_fursuit,
        classify_threshold=args.threshold,
    )

    print(f"\nAdded {added} images for character '{args.character}'")

def show_command(args):
    """Handle show command - view database entries."""
    from sam3_pursuit.storage.database import Database

    db_path = args.db if args.db else Config.DB_PATH
    db = Database(db_path)

    detections = []
    if args.by_id:
        det = db.get_detection_by_id(args.by_id)
        if det:
            detections = [det]
    elif args.by_character:
        detections = db.get_detections_by_character(args.by_character)
    elif args.by_post:
        detections = db.get_detections_by_post_id(args.by_post)
    else:
        print("Error: Specify one of --by-id, --by-character, or --by-post")
        sys.exit(1)

    if not detections:
        print("No detections found.")
        return

    if args.json:
        output = [
            {
                "id": d.id,
                "post_id": d.post_id,
                "character_name": d.character_name,
                "embedding_id": d.embedding_id,
                "bbox": [d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height],
                "confidence": d.confidence,
                "segmentor_model": d.segmentor_model,
                "source": d.source,
                "uploaded_by": d.uploaded_by,
                "source_filename": d.source_filename,
                "url": get_source_url(d.source, d.post_id),
                "created_at": str(d.created_at) if d.created_at else None,
            }
            for d in detections
        ]
        print(json.dumps(output, indent=2))
    else:
        print(f"\nFound {len(detections)} detection(s):\n")
        for d in detections:
            print(f"  ID: {d.id}")
            print(f"  Character: {d.character_name or 'Unknown'}")
            print(f"  Post ID: {d.post_id}")
            print(f"  Embedding ID: {d.embedding_id}")
            print(f"  BBox: ({d.bbox_x}, {d.bbox_y}, {d.bbox_width}x{d.bbox_height})")
            print(f"  Confidence: {d.confidence:.2%}")
            print(f"  Segmentor: {d.segmentor_model}")
            if d.source:
                print(f"  Source: {d.source}")
            if d.uploaded_by:
                print(f"  Uploaded by: {d.uploaded_by}")
            if d.source_filename:
                print(f"  Filename: {d.source_filename}")
            url = get_source_url(d.source, d.post_id)
            if url:
                print(f"  URL: {url}")
            if d.preprocessing_info:
                print(f"  Preprocessing: {d.preprocessing_info}")
            if d.git_version:
                print(f"  Git version: {d.git_version}")
            if d.created_at:
                print(f"  Created: {d.created_at}")
            print()


def ingest_command(args):
    """Handle ingest command - bulk import images."""
    # Set default data-dir based on dataset and source if not provided
    if not args.data_dir:
        if args.source:
            args.data_dir = f"datasets/{args.dataset}/{args.source}"
        else:
            print("Error: --data-dir is required (or specify --source for default path)")
            sys.exit(1)

    if args.method == "directory":
        if not args.source:
            print("Error: --source is required for directory ingestion.")
            sys.exit(1)
        ingest_from_directory(args)
    elif args.method == "nfc25":
        ingest_from_nfc25(args)
    elif args.method == "barq":
        ingest_from_barq(args)


def ingest_from_directory(args):
    """Ingest images from directory structure: data_dir/character_name/*.jpg"""

    ingestor = _get_ingestor(args)
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Determine source for provenance
    if args.source not in SOURCES_AVAILABLE:
        print(f"Error: Invalid source '{args.source}'. Must be one of: {', '.join(SOURCES_AVAILABLE)}")
        sys.exit(1)
    source = args.source

    def get_images():
        for char_dir in sorted(data_dir.iterdir()):
            if not char_dir.is_dir():
                continue

            character_name = char_dir.name
            images = list(char_dir.glob("*.jpg")) + list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpeg"))

            if args.limit:
                images = images[:args.limit]

            print(f"Ingesting {len(images)} images for {character_name}")
            for img in images:
                yield (character_name, img)

    names, images = zip(*get_images())
    add_full_image = getattr(args, "add_full_image", True)
    added = ingestor.add_images(
        character_names=list(names),
        image_paths=[str(p) for p in images],
        save_crops=args.save_crops,
        source=source,
        add_full_image=add_full_image,
        skip_non_fursuit=args.skip_non_fursuit,
        classify_threshold=args.threshold,
    )

    print(f"\nTotal: Added {added} images")


def ingest_from_nfc25(args):
    """Ingest images from NFC25 dataset."""
    ingestor = _get_ingestor(args)

    data_dir = Path(args.data_dir)
    json_path = data_dir / "nfc25-fursuit-list.json"
    images_dir = data_dir / "fursuit_images"

    if not json_path.exists():
        print(f"Error: NFC25 JSON not found: {json_path}")
        sys.exit(1)

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    with open(json_path) as f:
        fursuit_list = json.load(f)['FursuitList']

    print(f"Found {len(fursuit_list)} fursuits in NFC25 dataset")

    total_added = 0

    char_names = []
    img_paths = []
    for fursuit in fursuit_list:
        char_names.append(fursuit.get("NickName", ""))
        img_filename = str(fursuit.get("ImageUrl")).split("/")[-1]
        img_paths.append(str(images_dir / img_filename))
        if args.limit and total_added >= args.limit:
            break

    add_full_image = getattr(args, "add_full_image", True)
    added = ingestor.add_images(
        character_names=char_names,
        image_paths=img_paths,
        save_crops=args.save_crops,
        source=SOURCE_NFC25,
        add_full_image=add_full_image,
        skip_non_fursuit=args.skip_non_fursuit,
        classify_threshold=args.threshold,
    )
    total_added += added

    print(f"\nTotal: Added {total_added} images")


def ingest_from_barq(args):
    """Ingest images from Barq download directory.

    Folder structure: {profile_id}.{character_name}/{image_uuid}.jpg
    Image UUIDs are used as post_ids (extracted from filename automatically).
    Profile metadata can be looked up from barq_cache.db if needed.
    """
    from sam3_pursuit.storage.database import SOURCE_BARQ

    ingestor = _get_ingestor(args)
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Names that are placeholders from Barq's API (not real character names)
    placeholder_names = {"likes only", "liked only", "private", "mutuals only"}

    def _resolve_barq_name(profile_id: str, fallback: str) -> str:
        """Try to resolve a better name from barq cache for placeholder names."""
        try:
            from sam3_pursuit.tools.download_barq import get_cached_profile, get_folder_name
            cached = get_cached_profile(profile_id)
            if cached:
                # Re-resolve using updated logic
                resolved = get_folder_name(cached).split(".", 1)[1]
                if resolved.lower().strip() not in placeholder_names:
                    return resolved
        except Exception:
            pass
        return profile_id  # Fall back to profile ID

    def get_images():
        # Iterate directories matching pattern: {profile_id}.{name}
        for char_dir in sorted(data_dir.iterdir()):
            if not char_dir.is_dir():
                continue

            dir_name = char_dir.name
            # Parse profile_id.character_name format
            if "." not in dir_name:
                continue

            character_name = dir_name.split(".", 1)[1]

            # Handle placeholder names from Barq API
            if character_name.lower().strip() in placeholder_names:
                profile_id = dir_name.split(".", 1)[0]
                old_name = character_name
                character_name = _resolve_barq_name(profile_id, character_name)
                print(f"Resolved placeholder name '{old_name}' -> '{character_name}'")

            images = list(char_dir.glob("*.jpg")) + list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpeg"))

            if args.limit:
                images = images[:args.limit]

            if images:
                print(f"Ingesting {len(images)} images for {character_name}")

            for img in images:
                yield (character_name, img)

    names, images = zip(*get_images())
    add_full_image = getattr(args, "add_full_image", True)
    added = ingestor.add_images(
        character_names=list(names),
        image_paths=[str(p) for p in images],
        save_crops=args.save_crops,
        source=SOURCE_BARQ,
        add_full_image=add_full_image,
        skip_non_fursuit=args.skip_non_fursuit,
        classify_threshold=args.threshold,
    )

    print(f"\nTotal: Added {added} images from Barq")


def stats_command(args):
    """Handle stats command."""
    ingestor = _get_ingestor(args)
    stats = ingestor.get_stats()

    if hasattr(args, "json") and args.json:
        print(json.dumps(stats, indent=2, default=str))
    else:
        print(f"\nPursuit Statistics ({args.dataset})")
        print("=" * 50)
        print(f"Total detections: {stats['total_detections']}")
        print(f"Unique characters: {stats['unique_characters']}")
        print(f"Unique posts: {stats['unique_posts']}")
        print(f"Index size: {stats['index_size']}")

        if stats.get('segmentor_breakdown'):
            print("\nSegmentor breakdown:")
            for model, count in stats['segmentor_breakdown'].items():
                print(f"  {model}: {count}")

        if stats.get('preprocessing_breakdown'):
            print("\nPreprocessing configs:")
            for config, count in stats['preprocessing_breakdown'].items():
                print(f"  {config}: {count}")

        if stats.get('git_version_breakdown'):
            print("\nGit versions:")
            for version, count in stats['git_version_breakdown'].items():
                print(f"  {version or 'unknown'}: {count}")

        if stats.get('source_breakdown'):
            print("\nIngestion sources:")
            for source, count in stats['source_breakdown'].items():
                print(f"  {source or 'unknown'}: {count}")

        if stats['top_characters']:
            print("\nTop characters:")
            for name, count in stats['top_characters']:
                print(f"  {name}: {count} images")


def segment_command(args):
    from sam3_pursuit.pipeline.processor import CacheKey, CachedProcessingPipeline
    from sam3_pursuit.api.identifier import FursuitIngestor
    from glob import glob

    concept = getattr(args, "concept", None) or Config.DEFAULT_CONCEPT
    source = args.source or "unknown"
    pipeline = CachedProcessingPipeline(segmentor_concept=concept, segmentor_model_name=Config.SAM3_MODEL)
    images = sum([glob(f"{p}/**", recursive=True) if Path(p).is_dir() else [p] for p in args.images], [])
    images = [p for p in images if not Path(p).is_dir()]
    errors = []
    for n, image_path in enumerate(images):
        progress = f"[{n+1}/{len(images)}]"
        image_path = Path(image_path)
        print(f"{progress} Segmenting {image_path}")
        if not image_path.exists():
            print(f"{progress} Error: Image not found: {image_path}")
            sys.exit(1)

        post_id = FursuitIngestor._extract_post_id(str(image_path))
        cache_key = None if not args.cache_masks else CacheKey(post_id, source)
        try:
            image = Image.open(image_path)
            results, mask_reused = pipeline._segment(image, cache_key)
        except Exception as e:
            errors.append(image_path)
            print(f"Error: {e}")

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for i, r in enumerate(results):
                crop_path = output_dir / f"{image_path.stem}_crop_{i}.jpg"
                r.crop.save(crop_path)
                print(f"Saved crop: {crop_path}")

        mask_msg = " (mask reused)" if mask_reused else ""
        print(f"{progress} Found {len(results)} segment(s){mask_msg}")
        for i, r in enumerate(results):
            print(f"  {i+1}: bbox={r.bbox}, conf={r.confidence:.0%}")
    if errors:
        print("Failed images:")
    for err in errors:
        print(err)


def classify_command(args):
    """Handle classify command - classify images as fursuit or not."""
    from sam3_pursuit.models.classifier import ImageClassifier

    target = Path(args.path)
    if not target.exists():
        print(f"Error: Path not found: {target}")
        sys.exit(1)

    classifier = ImageClassifier()
    threshold = args.threshold

    if target.is_file():
        image_paths = [target]
    else:
        image_paths = sorted(
            p for p in target.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not image_paths:
            print(f"No images found in {target}")
            sys.exit(1)

    results = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path)
            scores = classifier.classify(image)
            top_fursuit = max(
                (scores[l] for l in Config.CLASSIFY_FURSUIT_LABELS),
                default=0,
            )
            results.append({
                "file": str(img_path),
                "scores": scores,
                "top_fursuit_score": top_fursuit,
                "is_fursuit": top_fursuit >= threshold,
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            name = Path(r["file"]).name
            status = "FURSUIT" if r["is_fursuit"] else "NOT FURSUIT"
            print(f"\n{name}: {status} (top: {r['top_fursuit_score']:.1%})")
            for label, score in sorted(r["scores"].items(), key=lambda x: -x[1]):
                bar = "#" * int(score * 30)
                print(f"  {label:25s} {score:6.1%} {bar}")


def download_command(args):
    """Handle download command."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

    # When downloading to a non-default dataset, auto-configure paths
    if args.dataset != Config.DEFAULT_DATASET:
        if not args.output_dir or (args.output_dir in ("furtrack_images", "barq_images",)):
            args.output_dir = f"datasets/{args.dataset}/{args.source}"

    excluded_post_ids = _get_excluded_post_ids(getattr(args, "exclude_datasets", ""))

    if args.source == "furtrack":
        from sam3_pursuit.tools import download_furtrack
        if args.output_dir:
            download_furtrack.IMAGES_DIR = args.output_dir
        if excluded_post_ids:
            download_furtrack.EXCLUDED_POST_IDS = excluded_post_ids
        if args.character:
            count = download_furtrack.download_character(args.character, args.max_images)
            print(f"Downloaded {count} images")
        elif args.download_all:
            download_furtrack.download_all_characters(args.max_images)
        else:
            print("Error: Specify --character or --all")
            sys.exit(1)

    elif args.source == "barq":
        import asyncio
        from sam3_pursuit.tools import download_barq
        if args.output_dir:
            download_barq.IMAGES_DIR = args.output_dir
        if excluded_post_ids:
            download_barq.EXCLUDED_POST_IDS = excluded_post_ids

        if args.clean:
            from sam3_pursuit.models.classifier import ImageClassifier
            classifier = ImageClassifier()
            download_barq.clean_images(classifier.fursuit_score, args.threshold)
        else:
            score_fn = None
            if args.skip_non_fursuit:
                from sam3_pursuit.models.classifier import ImageClassifier
                classifier = ImageClassifier()
                score_fn = classifier.fursuit_score
            asyncio.run(download_barq.download_all_profiles(args.lat, args.lon, args.max_pages, args.all_images, args.max_age, score_fn=score_fn, threshold=args.threshold))

    else:
        print("Error: Use 'pursuit download furtrack' or 'pursuit download barq'")
        sys.exit(1)


def evaluate_command(args):
    """Evaluate one dataset against another."""
    import numpy as np

    for ds_name in (args.from_dataset, args.against):
        if not _dataset_has_db(ds_name):
            db_path, _ = _get_dataset_paths(ds_name)
            print(f"Error: Dataset '{ds_name}' not found at {db_path}")
            sys.exit(1)

    from_db, from_index = _open_dataset(args.from_dataset)
    against_db, against_index = _open_dataset(args.against)

    if from_index.size == 0:
        print(f"Error: Dataset '{args.from_dataset}' is empty.")
        sys.exit(1)

    if against_index.size == 0:
        print(f"Error: Dataset '{args.against}' is empty.")
        sys.exit(1)

    if from_index.embedding_dim != against_index.embedding_dim:
        from_emb = from_db.get_metadata(Config.METADATA_KEY_EMBEDDER) or "unknown"
        against_emb = against_db.get_metadata(Config.METADATA_KEY_EMBEDDER) or "unknown"
        print(f"Error: Embedding dimension mismatch: '{args.from_dataset}' is {from_index.embedding_dim}D ({from_emb}), "
              f"'{args.against}' is {against_index.embedding_dim}D ({against_emb}). "
              f"Datasets must use the same embedder.")
        sys.exit(1)

    top_k = args.top_k

    # Get all detections from "from" dataset
    conn = from_db._connect()
    cursor = conn.cursor()
    cursor.execute("SELECT embedding_id, character_name, source, preprocessing_info FROM detections ORDER BY embedding_id")
    from_detections = cursor.fetchall()

    if not from_detections:
        print(f"Error: No detections in '{args.from_dataset}' database.")
        sys.exit(1)

    # Evaluate
    top_1_correct = 0
    top_k_correct = 0
    total = 0
    by_source = {}
    by_preprocessing = {}
    by_character = {}
    confidence_buckets = {i: {"correct": 0, "total": 0} for i in range(10)}

    for emb_id, char_name, source, preproc in from_detections:
        if emb_id >= from_index.size:
            continue

        # Reconstruct embedding
        embedding = np.zeros(from_index.embedding_dim, dtype=np.float32)
        from_index.index.reconstruct(emb_id, embedding)

        # Query against dataset
        distances, indices = against_index.search(embedding.reshape(1, -1), top_k)

        # Get predicted characters
        predictions = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            det = against_db.get_detection_by_embedding_id(int(idx))
            if det:
                confidence = max(0.0, 1.0 - dist / 2.0)
                predictions.append((det.character_name, confidence))

        if not predictions:
            continue

        total += 1
        top_1_pred, top_1_conf = predictions[0]
        is_top_1_correct = top_1_pred == char_name
        is_top_k_correct = any(p[0] == char_name for p in predictions[:top_k])

        if is_top_1_correct:
            top_1_correct += 1
        if is_top_k_correct:
            top_k_correct += 1

        # Confidence calibration
        bucket = min(9, int(top_1_conf * 10))
        confidence_buckets[bucket]["total"] += 1
        if is_top_1_correct:
            confidence_buckets[bucket]["correct"] += 1

        # By source
        src_key = source or "unknown"
        if src_key not in by_source:
            by_source[src_key] = {"correct": 0, "total": 0, "top_k_correct": 0}
        by_source[src_key]["total"] += 1
        if is_top_1_correct:
            by_source[src_key]["correct"] += 1
        if is_top_k_correct:
            by_source[src_key]["top_k_correct"] += 1

        # By preprocessing
        prep_key = preproc or "unknown"
        if prep_key not in by_preprocessing:
            by_preprocessing[prep_key] = {"correct": 0, "total": 0, "top_k_correct": 0}
        by_preprocessing[prep_key]["total"] += 1
        if is_top_1_correct:
            by_preprocessing[prep_key]["correct"] += 1
        if is_top_k_correct:
            by_preprocessing[prep_key]["top_k_correct"] += 1

        # By character
        if char_name not in by_character:
            by_character[char_name] = {"correct": 0, "total": 0, "top_k_correct": 0}
        by_character[char_name]["total"] += 1
        if is_top_1_correct:
            by_character[char_name]["correct"] += 1
        if is_top_k_correct:
            by_character[char_name]["top_k_correct"] += 1

    if total == 0:
        print("Error: No valid samples to evaluate.")
        sys.exit(1)

    top_1_acc = top_1_correct / total
    top_k_acc = top_k_correct / total

    results = {
        "from_dataset": args.from_dataset,
        "against_dataset": args.against,
        "total_samples": total,
        "top_1_accuracy": top_1_acc,
        f"top_{top_k}_accuracy": top_k_acc,
        "k_value": top_k,
        "by_source": {k: {"count": v["total"], "top_1_accuracy": v["correct"] / v["total"], f"top_{top_k}_accuracy": v["top_k_correct"] / v["total"]} for k, v in by_source.items()},
        "by_preprocessing": {k: {"count": v["total"], "top_1_accuracy": v["correct"] / v["total"], f"top_{top_k}_accuracy": v["top_k_correct"] / v["total"]} for k, v in by_preprocessing.items()},
        "by_character": {k: {"count": v["total"], "top_1_accuracy": v["correct"] / v["total"], f"top_{top_k}_accuracy": v["top_k_correct"] / v["total"]} for k, v in by_character.items()},
        "confidence_calibration": [{"range": f"{i*10}-{(i+1)*10}%", "count": v["total"], "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0} for i, v in confidence_buckets.items()],
    }

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nEvaluation: {args.from_dataset} → {args.against}")
        print("=" * 50)
        print(f"Total samples: {total}")
        print(f"Top-1 accuracy: {top_1_acc:.1%}")
        print(f"Top-{top_k} accuracy: {top_k_acc:.1%}")

        if by_source:
            print(f"\nBy Source:")
            for src, metrics in sorted(by_source.items(), key=lambda x: -x[1]["total"]):
                acc = metrics["correct"] / metrics["total"]
                print(f"  {src}: {acc:.1%} (n={metrics['total']})")

        if by_preprocessing:
            print(f"\nBy Preprocessing:")
            for prep, metrics in sorted(by_preprocessing.items(), key=lambda x: -x[1]["total"]):
                acc = metrics["correct"] / metrics["total"]
                print(f"  {prep}: {acc:.1%} (n={metrics['total']})")

        print(f"\nConfidence Calibration:")
        for bucket in results["confidence_calibration"]:
            if bucket["count"] > 0:
                print(f"  {bucket['range']}: {bucket['accuracy']:.1%} accurate (n={bucket['count']})")


def _dataset_has_db(dataset: str) -> bool:
    """Check if a dataset has both a .db and .index file."""
    db_path, index_path = _get_dataset_paths(dataset)
    return os.path.exists(db_path) and os.path.exists(index_path)


def _open_dataset(dataset_name: str):
    """Open a dataset's Database and VectorIndex."""
    from sam3_pursuit.storage.database import Database
    from sam3_pursuit.storage.vector_index import VectorIndex
    db_path, index_path = _get_dataset_paths(dataset_name)
    return Database(db_path), VectorIndex(index_path)


def _create_target_dataset(dataset_name: str, embedding_dim: int, embedder_name: str | None = None):
    """Create/open a target dataset with correct embedding dim and propagate embedder metadata."""
    from sam3_pursuit.storage.database import Database
    from sam3_pursuit.storage.vector_index import VectorIndex
    db_path, index_path = _get_dataset_paths(dataset_name)
    db = Database(db_path)
    index = VectorIndex(index_path, embedding_dim=embedding_dim)
    if embedder_name and db.get_metadata(Config.METADATA_KEY_EMBEDDER) is None:
        db.set_metadata(Config.METADATA_KEY_EMBEDDER, embedder_name)
    return db, index


def _fetch_detections(db, where_clause: str | None = None, params: list | None = None):
    """Fetch detections from a database, optionally filtered."""
    conn = db._connect()
    cursor = conn.cursor()
    query = f"SELECT {db._SELECT_FIELDS} FROM detections"
    if where_clause:
        query += f" WHERE {where_clause}"
    query += " ORDER BY embedding_id"
    cursor.execute(query, params or [])
    return [db._row_to_detection(row) for row in cursor.fetchall()]


def _shard_name(output: str, shard_idx: int, shards: int) -> str:
    """Get the target dataset name for a shard."""
    return output if shards == 1 else f"{output}_{shard_idx}"


def _copy_detections(detections, source_index, target_db, target_index, batch_size=500):
    """Copy detections and their embeddings from source dataset to target dataset.

    Returns (copied, skipped) counts.
    """
    import numpy as np
    from sam3_pursuit.storage.database import Detection

    # Build set of existing keys in target for dedup
    # Include all fields that distinguish unique detections: same post can have
    # multiple characters, and same character can have multiple segments (bboxes)
    conn = target_db._connect()
    cursor = conn.cursor()
    cursor.execute("SELECT post_id, preprocessing_info, source, character_name, segmentor_model, bbox_x, bbox_y, bbox_width, bbox_height FROM detections")
    existing = {row for row in cursor.fetchall()}

    next_emb_id = target_db.get_next_embedding_id()
    copied = 0
    skipped = 0

    batch_detections = []
    batch_embeddings = []

    for det in detections:
        key = (det.post_id, det.preprocessing_info, det.source, det.character_name,
               det.segmentor_model, det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height)
        if key in existing:
            skipped += 1
            continue

        # Reconstruct embedding from source index
        if det.embedding_id >= source_index.size:
            skipped += 1
            continue
        embedding = np.zeros(source_index.embedding_dim, dtype=np.float32)
        source_index.index.reconstruct(det.embedding_id, embedding)

        # Remap embedding_id
        new_det = Detection(
            id=None,
            post_id=det.post_id,
            character_name=det.character_name,
            embedding_id=next_emb_id,
            bbox_x=det.bbox_x,
            bbox_y=det.bbox_y,
            bbox_width=det.bbox_width,
            bbox_height=det.bbox_height,
            confidence=det.confidence,
            segmentor_model=det.segmentor_model,
            source=det.source,
            uploaded_by=det.uploaded_by,
            source_filename=det.source_filename,
            preprocessing_info=det.preprocessing_info,
            git_version=det.git_version,
        )

        batch_detections.append(new_det)
        batch_embeddings.append(embedding)
        existing.add(key)
        next_emb_id += 1

        if len(batch_detections) >= batch_size:
            target_index.add(np.array(batch_embeddings, dtype=np.float32))
            target_db.add_detections_batch(batch_detections)
            copied += len(batch_detections)
            batch_detections = []
            batch_embeddings = []

    # Flush remaining
    if batch_detections:
        target_index.add(np.array(batch_embeddings, dtype=np.float32))
        target_db.add_detections_batch(batch_detections)
        copied += len(batch_detections)

    return copied, skipped


def combine_command(args):
    """Combine multiple datasets into one."""
    # Check output doesn't collide with sources
    if args.output in args.datasets:
        print(f"Error: Output dataset '{args.output}' cannot be one of the source datasets")
        sys.exit(1)

    datasets_with_db = [ds for ds in args.datasets if _dataset_has_db(ds)]

    # Copy DB records + embeddings for datasets that have them
    total_copied = 0
    total_skipped = 0

    if datasets_with_db:
        # Open all sources once (used for both validation and copying)
        sources = {ds: _open_dataset(ds) for ds in datasets_with_db}

        # Validate all source datasets use the same embedder
        source_dims = {ds: idx.embedding_dim for ds, (_, idx) in sources.items()}
        unique_dims = set(source_dims.values())
        if len(unique_dims) > 1:
            detail = ", ".join(f"'{k}': {v}D" for k, v in source_dims.items())
            print(f"Error: Cannot combine datasets with different embedding dimensions: {detail}")
            sys.exit(1)

        source_embedders = {ds: db.get_metadata(Config.METADATA_KEY_EMBEDDER) for ds, (db, _) in sources.items()}
        unique_embedders = set(e for e in source_embedders.values() if e is not None)
        if len(unique_embedders) > 1:
            detail = ", ".join(f"'{k}': {v}" for k, v in source_embedders.items() if v is not None)
            print(f"Error: Cannot combine datasets with different embedders: {detail}")
            sys.exit(1)

        embedding_dim = next(iter(unique_dims))
        source_embedder = next(iter(unique_embedders)) if unique_embedders else None

        target_db, target_index = _create_target_dataset(args.output, embedding_dim, source_embedder)

        for ds_name, (source_db, source_index) in sources.items():
            detections = _fetch_detections(source_db)

            print(f"Copying {len(detections)} detections from '{ds_name}'...")
            copied, skipped = _copy_detections(detections, source_index, target_db, target_index)
            total_copied += copied
            total_skipped += skipped
            print(f"  Copied: {copied}, Skipped (duplicates): {skipped}")

            source_db.close()

        target_index.save()
        target_db.close()

        print(f"Combined {total_copied} detections into '{args.output}' ({total_skipped} duplicates skipped)")

    # Copy image files
    total_files = 0
    for ds_name in args.datasets:
        files_copied = _copy_dataset_files(ds_name, args.output)
        if files_copied:
            print(f"Copied {files_copied} files from '{ds_name}'")
        total_files += files_copied

    if total_files:
        print(f"Copied {total_files} image files total")

    if not datasets_with_db and total_files == 0:
        print("Warning: No databases or image files found in source datasets")

    print("\nDone.")


def split_command(args):
    """Split a dataset by criteria."""
    if not args.by_source and not args.by_character:
        print("Error: At least one filter required (--by-source or --by-character)")
        sys.exit(1)

    shards = args.shards
    if shards < 1:
        print("Error: --shards must be >= 1")
        sys.exit(1)

    has_db = _dataset_has_db(args.source_dataset)

    # Copy DB records + embeddings if DB exists
    if has_db:
        source_db, source_index = _open_dataset(args.source_dataset)
        source_embedder = source_db.get_metadata(Config.METADATA_KEY_EMBEDDER)

        # Build filter query
        conditions = []
        params = []
        if args.by_source:
            conditions.append("source = ?")
            params.append(args.by_source)
        if args.by_character:
            char_names = [c.strip() for c in args.by_character.split(",")]
            placeholders = ",".join("?" * len(char_names))
            conditions.append(f"character_name IN ({placeholders})")
            params.extend(char_names)

        detections = _fetch_detections(source_db, " AND ".join(conditions), params)

        if detections:
            print(f"Found {len(detections)} matching detections")

            # Assign detections to shards
            if shards == 1:
                shard_map = {0: detections}
            else:
                shard_map = {i: [] for i in range(shards)}
                for det in detections:
                    shard_idx = hash(det.post_id) % shards
                    shard_map[shard_idx].append(det)

            for shard_idx in range(shards):
                shard_detections = shard_map[shard_idx]
                target_name = _shard_name(args.output, shard_idx, shards)

                target_db, target_index = _create_target_dataset(
                    target_name, source_index.embedding_dim, source_embedder)

                print(f"Writing {len(shard_detections)} detections to '{target_name}'...")
                copied, skipped = _copy_detections(shard_detections, source_index, target_db, target_index)
                print(f"  Copied: {copied}, Skipped (duplicates): {skipped}")

                target_index.save()
                target_db.close()
        else:
            print("No detections match the specified filters.")

        source_db.close()

    # Copy image files
    total_files = 0
    for shard_idx in range(shards):
        target_name = _shard_name(args.output, shard_idx, shards)
        files_copied = _copy_dataset_files(
            args.source_dataset,
            target_name,
            by_source=args.by_source,
            by_character=args.by_character,
            shard_idx=shard_idx if shards > 1 else None,
            shards=shards,
        )
        if files_copied:
            print(f"Copied {files_copied} files to '{target_name}'")
        total_files += files_copied

    if total_files:
        print(f"Copied {total_files} image files total")

    if not has_db and total_files == 0:
        print("Warning: No database or image files found for source dataset")

    print("\nDone.")


if __name__ == "__main__":
    main()
