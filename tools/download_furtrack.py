#!/usr/bin/env python3
"""Download fursuit images from FurTrack.

Downloads images into character subfolders for use with `pursuit ingest --source directory`.

Usage:
    python tools/download_furtrack.py --help
    python tools/download_furtrack.py --download-characters
    python tools/download_furtrack.py --download-character "CharName"
"""

import argparse
import asyncio
import json
from pathlib import Path
import random
import sqlite3

import aiohttp
import requests

# Configuration
MAX_IMAGES_PER_CHAR = 2
CACHE_DB = "furtrack_cache.db"
IMAGES_DIR = "furtrack_images"
MAX_CONCURRENT_DOWNLOADS = 20

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en,ru-RU;q=0.9,ru;q=0.8,nl;q=0.7",
    "cache-control": "no-cache",
    "origin": "https://www.furtrack.com",
    "pragma": "no-cache",
    "referer": "https://www.furtrack.com/",
    "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
}


def get_cache_db() -> sqlite3.Connection:
    """Get connection to cache database, creating table if needed."""
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            url TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
    """)
    return conn


def get_json_cached(url: str) -> dict | None:
    """Fetch JSON from URL, using SQLite cache."""
    conn = get_cache_db()

    # Check cache
    row = conn.execute("SELECT data FROM cache WHERE url = ?", (url,)).fetchone()
    if row:
        conn.close()
        return json.loads(row[0])

    # Fetch from network
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        conn.close()
        return None

    # Store as plain JSON
    conn.execute("INSERT OR REPLACE INTO cache (url, data) VALUES (?, ?)", (url, json.dumps(data)))
    conn.commit()
    conn.close()
    return data


def get_char_tags(tags: list[dict]) -> list[str]:
    """Extract character tags (prefix '1:') from tag list."""
    return [t["tagName"].removeprefix("1:") for t in tags if t.get("tagName", "").startswith("1:")]


def is_valid_post(raw: dict) -> bool:
    """Check if post is valid (single character, has image, not video)."""
    if not raw.get("success"):
        return False

    tags = raw.get("tags", [])
    char_tags = get_char_tags(tags)
    if len(char_tags) != 1:
        return False

    tag_names = [t.get("tagName", "") for t in tags]
    if any(t in tag_names for t in ["tagging_incomplete", "missing_character"]):
        return False

    post = raw.get("post", {})
    if not post.get("postId") or post.get("videoId"):
        return False

    return True


def get_all_characters() -> list[tuple[str, int]]:
    """Get list of all character tags from FurTrack."""
    url = "https://solar.furtrack.com/get/tags/all"
    data = get_json_cached(url)
    if not data:
        return []
    return [
        (t["tagName"].removeprefix("1:"), t.get("tagCount", 0))
        for t in data.get("tags", [])
        if t.get("tagName", "").startswith("1:")
    ]


def get_character_post_ids(char: str) -> list[int]:
    """Get all post IDs for a character from FurTrack."""
    url = f"https://solar.furtrack.com/get/index/1:{char}"
    data = get_json_cached(url)
    if not data:
        return []
    post_ids = [int(p["postId"]) for p in data.get("posts", [])]
    random.shuffle(post_ids)
    return post_ids


def get_post_metadata(post_id: int) -> dict | None:
    """Get post metadata, using cache."""
    url = f"https://solar.furtrack.com/get/p/{post_id}"
    return get_json_cached(url)


def sanitize_folder_name(name: str) -> str:
    """Sanitize character name for use as folder name."""
    # Replace problematic characters
    return name.replace("/", "_").replace("\\", "_").replace(":", "_").replace("\0", "")


def get_existing_images(char_folder: Path) -> set[str]:
    """Get set of post IDs already downloaded for a character."""
    if not char_folder.exists():
        return set()
    return {p.stem for p in char_folder.glob("*.jpg")}


async def download_image(
    session: aiohttp.ClientSession,
    post_id: str,
    char_folder: Path,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Download a single image."""
    url = f"https://orca2.furtrack.com/thumb/{post_id}.jpg"
    dest = char_folder / f"{post_id}.jpg"

    async with semaphore:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    dest.write_bytes(await resp.read())
                    print(f"  Downloaded {post_id}.jpg")
                    return True
                else:
                    print(f"  Failed {post_id}: HTTP {resp.status}")
                    return False
        except Exception as e:
            print(f"  Failed {post_id}: {e}")
            return False


async def download_images_async(post_ids: list[str], char_folder: Path) -> int:
    """Download multiple images in parallel."""
    char_folder.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        tasks = [download_image(session, pid, char_folder, semaphore) for pid in post_ids]
        results = await asyncio.gather(*tasks)
        return sum(results)


def download_character(char: str, max_images: int = MAX_IMAGES_PER_CHAR) -> int:
    """Download images for a single character."""
    safe_name = sanitize_folder_name(char)
    char_folder = Path(IMAGES_DIR) / safe_name

    # Check existing downloads
    existing = get_existing_images(char_folder)
    if len(existing) >= max_images:
        return 0

    # Get post IDs for character
    post_ids = get_character_post_ids(char)
    if not post_ids:
        return 0

    # Filter out already downloaded and collect valid posts
    to_download = []
    for pid in post_ids:
        if str(pid) in existing:
            continue
        if len(existing) + len(to_download) >= max_images:
            break

        # Check if post is valid (uses cache)
        raw = get_post_metadata(pid)
        if raw and is_valid_post(raw):
            to_download.append(str(pid))

    if not to_download:
        return 0

    print(f"Downloading {len(to_download)} images for {char}...")
    return asyncio.run(download_images_async(to_download, char_folder))


def download_all_characters(max_images: int = MAX_IMAGES_PER_CHAR):
    """Download images for all characters."""
    characters = get_all_characters()
    print(f"Found {len(characters)} characters")

    # Sort by popularity (most photos first)
    characters.sort(key=lambda c: c[1], reverse=True)

    total = 0
    for i, (char, count) in enumerate(characters):
        safe_name = sanitize_folder_name(char)
        char_folder = Path(IMAGES_DIR) / safe_name
        existing = len(get_existing_images(char_folder))

        if existing >= max_images:
            continue

        print(f"[{i+1}/{len(characters)}] {char} ({existing}/{max_images} existing)")
        downloaded = download_character(char, max_images)
        total += downloaded

    print(f"\nTotal downloaded: {total}")


def main():
    parser = argparse.ArgumentParser(description="Download FurTrack fursuit images")
    parser.add_argument("--download-characters", action="store_true", help="Download all characters")
    parser.add_argument("--download-character", type=str, help="Download specific character")
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES_PER_CHAR, help="Max images per character")

    args = parser.parse_args()

    if args.download_character:
        count = download_character(args.download_character, args.max_images)
        print(f"Downloaded {count} images for {args.download_character}")
    elif args.download_characters:
        download_all_characters(args.max_images)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
