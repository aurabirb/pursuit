"""Command-line interface for the SAM3 fursuit recognition system."""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

from sam3_pursuit.config import Config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pursuit",
        description="Fursuit character recognition using SAM3 and DINOv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Identify character in an image
  pursuit identify photo.jpg

  # Identify with segmentation (for multi-character images)
  pursuit identify photo.jpg --segment

  # Add images for a character
  pursuit add --character "CharacterName" image1.jpg image2.jpg

  # Test segmentation on an image
  pursuit segment photo.jpg --output-dir ./crops/

  # View database entries for a character
  pursuit show --by-character "CharacterName"

  # Bulk ingest from a directory
  pursuit ingest --source directory --data-dir ./characters/

  # Show statistics
  pursuit stats
        """
    )
    parser.add_argument("--db", default=Config.DB_PATH, help="Database pat")
    parser.add_argument("--index", default=Config.INDEX_PATH, help="Index pat")
    parser.add_argument("--no-segment", "-S", dest="segment", action="store_false", help="Do not use segmentation")
    parser.add_argument("--concept", default=Config.DEFAULT_CONCEPT, help="SAM3 concept")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Identify command
    identify_parser = subparsers.add_parser("identify", help="Identify character in an image")
    identify_parser.add_argument("image", help="Path to image file")
    identify_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add images for a character")
    add_parser.add_argument("--character", "-c", required=True, help="Character name")
    add_parser.add_argument("images", nargs="+", help="Image paths")
    add_parser.add_argument("--save-crops", action="store_true", help="Save crop images for debugging")

    # Segment command (NEW)
    segment_parser = subparsers.add_parser("segment", help="Test segmentation on an image")
    segment_parser.add_argument("image", help="Path to image file")
    segment_parser.add_argument("--output-dir", "-o", help="Save crops to directory")
    segment_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Show command (NEW)
    show_parser = subparsers.add_parser("show", help="View database entries")
    show_parser.add_argument("--by-id", type=int, help="Query by detection ID")
    show_parser.add_argument("--by-character", help="Query by character name")
    show_parser.add_argument("--by-post", help="Query by post ID")
    show_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Ingest command (NEW)
    ingest_parser = subparsers.add_parser("ingest", help="Bulk ingest images")
    ingest_parser.add_argument("--source", required=True, choices=["directory", "furtrack", "nfc25"],
                               help="Source type")
    ingest_parser.add_argument("--data-dir", required=True, help="Data directory")
    ingest_parser.add_argument("--limit", type=int, help="Limit number of images per character")
    ingest_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    ingest_parser.add_argument("--save-crops", action="store_true", help="Save crop images")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show system statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handlers
    if args.command == "identify":
        identify_command(args)
    elif args.command == "add":
        add_command(args)
    elif args.command == "segment":
        segment_command(args)
    elif args.command == "show":
        show_command(args)
    elif args.command == "ingest":
        ingest_command(args)
    elif args.command == "stats":
        stats_command(args)


def _get_identifier(args):
    """Create identifier with optional custom paths."""
    from sam3_pursuit.api.identifier import SAM3FursuitIdentifier

    db_path = args.db if hasattr(args, "db") and args.db else Config.DB_PATH
    index_path = args.index if hasattr(args, "index") and args.index else Config.INDEX_PATH

    return SAM3FursuitIdentifier(db_path=db_path, index_path=index_path)


def identify_command(args):
    """Handle identify command."""
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    identifier = _get_identifier(args)
    image = Image.open(image_path)
    results = identifier.identify(image, top_k=args.top_k, use_segmentation=args.segment)

    if not results:
        print("No matches found.")
        return

    print(f"\nTop {len(results)} matches for {image_path.name}:")
    print("-" * 60)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.character_name or 'Unknown'}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Distance: {result.distance:.4f}")
        print(f"   Post ID: {result.post_id}")
        print()


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

    identifier = _get_identifier(args)
    added = identifier.add_images(
        character_names=[args.character] * len(valid_paths),
        image_paths=valid_paths,
        use_segmentation=args.segment,
        concept=args.concept,
        save_crops=args.save_crops,
    )

    print(f"\nAdded {added} images for character '{args.character}'")


def segment_command(args):
    """Handle segment command - test segmentation on an image."""
    from sam3_pursuit.models.segmentor import FursuitSegmentor

    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    segmentor = FursuitSegmentor()
    image = Image.open(image_path)

    results = segmentor.segment(image, concept=args.concept)

    if args.json:
        output = {
            "image": str(image_path),
            "concept": args.concept,
            "segments": [
                {
                    "index": i,
                    "bbox": list(r.bbox),
                    "confidence": r.confidence,
                    "crop_size": list(r.crop.size),
                }
                for i, r in enumerate(results)
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nSegmentation results for {image_path}")
        print(f"Concept: {args.concept}")
        print(f"Found {len(results)} segment(s):\n")

        for i, r in enumerate(results):
            print(f"  [{i}] bbox={r.bbox}, confidence={r.confidence:.2%}, size={r.crop.size}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = image_path.stem

        for i, r in enumerate(results):
            crop_path = output_dir / f"{base_name}_crop_{i}.jpg"
            r.crop.convert("RGB").save(crop_path, quality=90)
            print(f"Saved: {crop_path}")


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
            if d.crop_path:
                print(f"  Crop: {d.crop_path}")
            if d.created_at:
                print(f"  Created: {d.created_at}")
            print()


def ingest_command(args):
    """Handle ingest command - bulk import images."""
    identifier = _get_identifier(args)

    if args.source == "directory":
        ingest_from_directory(identifier, args)
    elif args.source == "furtrack":
        ingest_from_furtrack(identifier, args)
    elif args.source == "nfc25":
        ingest_from_nfc25(identifier, args)


def ingest_from_directory(identifier, args):
    """Ingest images from directory structure: data_dir/character_name/*.jpg"""
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
                batch_size=args.batch_size,
                use_segmentation=args.segment,
                concept=args.concept,
                save_crops=args.save_crops,
            )
            total_added += added

    print(f"\nTotal: Added {total_added} images")


def ingest_from_furtrack(identifier, args):
    """Ingest images from FurTrack download database."""
    import sqlite3

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

    for post_id, char_name, url in records:
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
            batch_size=args.batch_size,
            use_segmentation=args.segment,
            concept=args.concept,
            save_crops=args.save_crops,
        )
        total_added += added

    print(f"\nTotal: Added {total_added} images")


def ingest_from_nfc25(identifier, args):
    """Ingest images from NFC25 dataset."""
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
        # batch_size=1,
        use_segmentation=False,
        concept=args.concept,
        save_crops=args.save_crops,
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
                print(f"  {model}: {count} embeddings")

        if stats['top_characters']:
            print("\nTop characters:")
            for name, count in stats['top_characters']:
                print(f"  {name}: {count} images")


if __name__ == "__main__":
    main()
