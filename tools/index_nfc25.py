#!/usr/bin/env python3
"""Index NFC25 fursuit badge database.

This script indexes the NFC25 fursuit badge photo collection into the
SAM3 recognition system.

The NFC25 dataset contains ~2,305 fursuit badge photos from NordicFuzzCon 2025.

Usage:
    python tools/index_nfc25.py --help
    python tools/index_nfc25.py --data-dir /path/to/nfc25-fursuits
    python tools/index_nfc25.py --data-dir /path/to/nfc25-fursuits --limit 100
"""

import argparse
import json
import os
import time

from PIL import Image


def load_fursuit_list(data_dir: str) -> list[dict]:
    """Load NFC25 fursuit list from JSON."""
    json_path = os.path.join(data_dir, "nfc25-fursuit-list.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Fursuit list not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    return data.get("FursuitList", [])


def get_image_path(fursuit: dict, images_dir: str) -> str | None:
    """Get local image path for a fursuit entry."""
    image_url = fursuit.get("ImageUrl", "")
    if not image_url:
        return None

    filename = image_url.split("/")[-1]
    filepath = os.path.join(images_dir, filename)

    return filepath if os.path.exists(filepath) else None


def index_nfc25(
    data_dir: str,
    db_name: str = "nfc25.db",
    index_name: str = "nfc25.index",
    limit: int | None = None,
    batch_size: int = 50
):
    """Index the NFC25 fursuit database.

    Args:
        data_dir: Path to NFC25 data directory containing JSON and images.
        db_name: Name for the SQLite database file.
        index_name: Name for the FAISS index file.
        limit: Maximum number of images to index (None for all).
        batch_size: Progress update interval.
    """
    from sam3_pursuit import SAM3FursuitIdentifier
    from sam3_pursuit.config import Config

    images_dir = os.path.join(data_dir, "fursuit_images")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Set up paths
    db_path = os.path.join(Config.BASE_DIR, db_name)
    index_path = os.path.join(Config.BASE_DIR, index_name)

    print(f"Data directory: {data_dir}")
    print(f"Database: {db_path}")
    print(f"Index: {index_path}")
    print()

    # Load fursuit list
    print("Loading NFC25 fursuit list...")
    fursuits = load_fursuit_list(data_dir)
    print(f"Total entries: {len(fursuits)}")

    # Filter to available images
    available = []
    for fursuit in fursuits:
        img_path = get_image_path(fursuit, images_dir)
        if img_path:
            available.append((fursuit, img_path))

    print(f"Available images: {len(available)}")

    if limit:
        available = available[:limit]
        print(f"Limited to: {len(available)}")

    # Initialize identifier
    print("\nInitializing identifier...")
    identifier = SAM3FursuitIdentifier(db_path=db_path, index_path=index_path)

    # Check already indexed
    already_indexed = 0
    to_index = []

    for fursuit, img_path in available:
        filename = os.path.basename(img_path)
        post_id = os.path.splitext(filename)[0]

        if identifier.db.has_post(post_id):
            already_indexed += 1
        else:
            to_index.append((fursuit, img_path))

    print(f"Already indexed: {already_indexed}")
    print(f"To index: {len(to_index)}")

    if not to_index:
        print("\nNothing to index!")
        return

    # Index images
    print(f"\nIndexing {len(to_index)} images...")
    start_time = time.time()
    indexed = 0
    errors = 0

    for i, (fursuit, img_path) in enumerate(to_index):
        nickname = fursuit.get("NickName", "Unknown")

        try:
            added = identifier.add_images(
                character_name=nickname,
                image_paths=[img_path]
            )
            indexed += added

            if (i + 1) % batch_size == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(to_index) - i - 1) / rate if rate > 0 else 0
                print(f"Progress: {i+1}/{len(to_index)} ({indexed} indexed, {errors} errors) "
                      f"- {rate:.1f} img/s - ETA: {remaining/60:.1f}min")

        except Exception as e:
            print(f"Error indexing {nickname}: {e}")
            errors += 1

    # Summary
    elapsed = time.time() - start_time
    print()
    print("Indexing complete!")
    print(f"  Indexed: {indexed}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if indexed > 0:
        print(f"  Rate: {indexed/elapsed:.1f} images/second")

    # Final stats
    stats = identifier.get_stats()
    print()
    print("Database stats:")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Unique characters: {stats['unique_characters']}")
    print(f"  Index size: {stats['index_size']}")


def main():
    parser = argparse.ArgumentParser(
        description="Index NFC25 fursuit badge database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Index all images
    python tools/index_nfc25.py --data-dir /media/user/SSD2TB/nfc25-fursuits

    # Index first 100 images (for testing)
    python tools/index_nfc25.py --data-dir /path/to/nfc25 --limit 100

Expected directory structure:
    nfc25-fursuits/
    ├── nfc25-fursuit-list.json
    └── fursuit_images/
        ├── uuid1.png
        ├── uuid2.png
        └── ...
        """
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to NFC25 data directory"
    )
    parser.add_argument(
        "--db-name",
        default="nfc25.db",
        help="Database filename (default: nfc25.db)"
    )
    parser.add_argument(
        "--index-name",
        default="nfc25.index",
        help="Index filename (default: nfc25.index)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to index"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Progress update interval (default: 50)"
    )

    args = parser.parse_args()

    index_nfc25(
        data_dir=args.data_dir,
        db_name=args.db_name,
        index_name=args.index_name,
        limit=args.limit,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
