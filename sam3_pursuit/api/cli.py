import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from sam3_pursuit.config import Config


def main():
    parser = argparse.ArgumentParser(
        prog="pursuit",
        description="Fursuit character recognition using SAM3 and DINOv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Identify character in an image
  pursuit identify photo.jpg

  # Identify with segmentation (for multi-character images)
  pursuit identify photo.jpg
  pursuit identify photo.jpg --no-segment

  # Add images for a character
  pursuit add --character "CharacterName" image1.jpg image2.jpg

  # View database entries for a character
  pursuit show --by-character "CharacterName"

  # Bulk ingest from a directory
  pursuit ingest --source directory --data-dir ./characters/

  # Show statistics
  pursuit stats
        """
    )
    parser.add_argument("--db", default=Config.DB_PATH, help="Database path")
    parser.add_argument("--index", default=Config.INDEX_PATH, help="Index path")
    parser.add_argument("--no-segment", "-S", dest="segment", action="store_false", help="Do not use segmentation")
    parser.add_argument("--concept", default=Config.DEFAULT_CONCEPT, help="SAM3 concept")
    parser.add_argument("--background", "-bg", default=Config.DEFAULT_BACKGROUND_MODE,
                        choices=["none", "solid", "blur"],
                        help="Background isolation mode (default: solid)")
    parser.add_argument("--bg-color", default="128,128,128",
                        help="Background color as R,G,B for solid mode (default: 128,128,128)")
    parser.add_argument("--blur-radius", type=int, default=Config.DEFAULT_BLUR_RADIUS,
                        help="Blur radius for blur mode (default: 25)")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    identify_parser = subparsers.add_parser("identify", help="Identify character in an image")
    identify_parser.add_argument("image", help="Path to image file")
    identify_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    identify_parser.add_argument("--save-crops", action="store_true", help="Save preprocessed crops for debugging")

    add_parser = subparsers.add_parser("add", help="Add images for a character")
    add_parser.add_argument("--character", "-c", required=True, help="Character name")
    add_parser.add_argument("images", nargs="+", help="Image paths")
    add_parser.add_argument("--save-crops", action="store_true", help="Save crop images for debugging")

    show_parser = subparsers.add_parser("show", help="View database entries")
    show_parser.add_argument("--by-id", type=int, help="Query by detection ID")
    show_parser.add_argument("--by-character", help="Query by character name")
    show_parser.add_argument("--by-post", help="Query by post ID")
    show_parser.add_argument("--json", action="store_true", help="Output as JSON")

    ingest_parser = subparsers.add_parser("ingest", help="Bulk ingest images")
    ingest_parser.add_argument("--source", required=True, choices=["directory", "furtrack", "nfc25"],
                               help="Source type")
    ingest_parser.add_argument("--data-dir", required=True, help="Data directory")
    ingest_parser.add_argument("--limit", type=int, help="Limit number of images per character")
    ingest_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    ingest_parser.add_argument("--save-crops", action="store_true", help="Save crop images")

    stats_parser = subparsers.add_parser("stats", help="Show system statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

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


def _get_identifier(args):
    from sam3_pursuit.api.identifier import SAM3FursuitIdentifier

    db_path = args.db if hasattr(args, "db") and args.db else Config.DB_PATH
    index_path = args.index if hasattr(args, "index") and args.index else Config.INDEX_PATH
    isolation_config = _get_isolation_config(args)
    segmentor_model_name = Config.SAM3_MODEL if getattr(args, "segment", True) else None
    segmentor_concept = args.concept if hasattr(args, "concept") and args.concept else Config.DEFAULT_CONCEPT

    return SAM3FursuitIdentifier(
        db_path=db_path,
        index_path=index_path,
        isolation_config=isolation_config,
        segmentor_model_name=segmentor_model_name,
        segmentor_concept=segmentor_concept)


def identify_command(args):
    """Handle identify command."""

    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    identifier = _get_identifier(args)
    image = Image.open(image_path)
    save_crops = getattr(args, "save_crops", False)
    crop_prefix = image_path.stem
    results = identifier.identify(
        image,
        top_k=args.top_k,
        save_crops=save_crops,
        crop_prefix=crop_prefix,
    )

    if not results:
        print("No matches found.")
        return

    print(f"\nAnalyzed {len(results)} segment(s) in {image_path.name}")
    print("=" * 60)

    for seg_result in results:
        print(f"\nSegment {seg_result.segment_index + 1}")
        print(f"  BBox: {seg_result.segment_bbox}")
        print(f"  Detection confidence: {seg_result.segment_confidence:.2%}")
        print("-" * 60)

        if not seg_result.matches:
            print("  No matches found for this segment.")
        else:
            print(f"  Top {len(seg_result.matches)} matches:")
            for i, match in enumerate(seg_result.matches, 1):
                print(f"  {i}. {match.character_name or 'Unknown'}")
                print(f"     Confidence: {match.confidence:.2%}")
                print(f"     Distance: {match.distance:.4f}")
                print(f"     Post ID: {match.post_id}")


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

    identifier = _get_identifier(args)
    added = identifier.add_images(
        character_names=[args.character] * len(valid_paths),
        image_paths=valid_paths,
        save_crops=args.save_crops
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
                "source_filename": d.source_filename,
                "source_url": d.source_url,
                "is_cropped": d.is_cropped,
                "segmentation_concept": d.segmentation_concept,
                "crop_path": d.crop_path,
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
            if d.source_filename:
                print(f"  Source: {d.source_filename}")
            if d.segmentation_concept:
                print(f"  Concept: {d.segmentation_concept}")
            if d.preprocessing_info:
                print(f"  Preprocessing: {d.preprocessing_info}")
            if d.git_version:
                print(f"  Git version: {d.git_version}")
            if d.crop_path:
                print(f"  Crop: {d.crop_path}")
            if d.created_at:
                print(f"  Created: {d.created_at}")
            print()


def ingest_command(args):
    """Handle ingest command - bulk import images."""

    if args.source == "directory":
        ingest_from_directory(args)
    elif args.source == "furtrack":
        ingest_from_furtrack(args)
    elif args.source == "nfc25":
        ingest_from_nfc25(args)


def ingest_from_directory(args):
    """Ingest images from directory structure: data_dir/character_name/*.jpg"""
    
    identifier = _get_identifier(args)
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    total_added = 0

    for char_dir in sorted(data_dir.iterdir()):
        if not char_dir.is_dir():
            continue

        character_name = char_dir.name
        images = list(char_dir.glob("*.jpg")) + list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpeg"))

        if args.limit:
            images = images[:args.limit]

        if images:
            print(f"Ingesting {len(images)} images for {character_name}")
            added = identifier.add_images(
                character_names=[character_name] * len(images),
                image_paths=[str(p) for p in images],
                save_crops=args.save_crops,
            )
            total_added += added

    print(f"\nTotal: Added {total_added} images")


def ingest_from_furtrack(args):
    """Ingest images from FurTrack download database."""
    import sqlite3

    identifier = _get_identifier(args)

    data_dir = Path(args.data_dir)
    furtrack_db = data_dir / "furtrack.db"
    images_dir = data_dir / "furtrack_images"

    if not furtrack_db.exists():
        print(f"Error: FurTrack database not found: {furtrack_db}")
        sys.exit(1)

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    conn = sqlite3.connect(furtrack_db)
    c = conn.cursor()

    # TODO: only select entries with single character?

    c.execute("""
        SELECT post_id, char, url
        FROM furtrack
        WHERE char != '' AND url != ''
    """)

    records = c.fetchall()
    conn.close()

    print(f"Found {len(records)} records in FurTrack database")

    # Group by character
    character_images: dict[str, list[str]] = {}

    for post_id, char_name, url in records: # TODO: record source_url in add_image
        img_path = images_dir / f"{post_id}.jpg"
        if not img_path.exists():
            continue

        if char_name not in character_images:
            character_images[char_name] = []
        character_images[char_name].append(str(img_path))

    total_added = 0

    for char_name, img_paths in sorted(character_images.items()):
        if args.limit:
            img_paths = img_paths[:args.limit]

        print(f"Ingesting {len(img_paths)} images for {char_name}")
        added = identifier.add_images(
            character_names=[char_name] * len(img_paths),
            image_paths=img_paths,
            save_crops=args.save_crops,
        )
        total_added += added

    print(f"\nTotal: Added {total_added} images")


def ingest_from_nfc25(args):
    """Ingest images from NFC25 dataset."""
    identifier = _get_identifier(args)

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

    added = identifier.add_images(
        character_names=char_names,
        image_paths=img_paths,
        save_crops=args.save_crops,
        source_url=str(fursuit.get("ImageUrl")),
    )
    total_added += added

    print(f"\nTotal: Added {total_added} images")


def stats_command(args):
    """Handle stats command."""
    identifier = _get_identifier(args)
    stats = identifier.get_stats()

    if hasattr(args, "json") and args.json:
        print(json.dumps(stats, indent=2, default=str))
    else:
        print("\nPursuit - Fursuit Recognition System Statistics")
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

        if stats['top_characters']:
            print("\nTop characters:")
            for name, count in stats['top_characters']:
                print(f"  {name}: {count} images")


if __name__ == "__main__":
    main()
