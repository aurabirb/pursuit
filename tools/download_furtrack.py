#!/usr/bin/env python3
"""Download fursuit images from FurTrack.

This script downloads images from furtrack.com API, filters for single-character
tagged images, and stores metadata in a SQLite database.

Usage:
    python tools/download_furtrack.py --help
    python tools/download_furtrack.py --download-characters
    python tools/download_furtrack.py --index
"""

import argparse
import concurrent.futures
import json
import os
import random
import sqlite3
from typing import Optional

import requests

# Configuration
MAX_CHAR_ENTRIES = 10  # Max images per character to download
DB_PATH = "furtrack.db"
IMAGES_DIR = "furtrack_images"
CACHE_DIR = "furtrack_cache"



def get_url(url: str) -> requests.Response:
    """Make HTTP request with browser-like headers."""
    return requests.get(
        url,
        headers={
            "accept": "application/json, text/plain, */*",
            "accept-language": "en,ru-RU;q=0.9,ru;q=0.8,nl;q=0.7",
            "cache-control": "no-cache",
            "origin": "https://www.furtrack.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://www.furtrack.com/",
            "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
        },
        timeout=30,
    )


def get_json(url: str) -> Optional[dict]:
    """Fetch JSON from URL, or find the result in cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, url.replace("https://", "").replace("/", "_") + ".json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    try:
        r = get_url(url)
        if r.status_code != 200:
            raise Exception(f"HTTP status {r.status_code}")
        ret = r.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

    with open(cache_file, "w") as f:
        json.dump(ret, f, indent=2)

    return ret


def make_requests(urls: list[str], max_concurrent: int = 20, verbose: bool = True, function=get_url):
    """Make concurrent HTTP requests."""
    chunks = [urls[i:i + max_concurrent] for i in range(0, len(urls), max_concurrent)]
    total = 0
    for chunk in chunks:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(function, chunk)
            total += len(chunk)
            if verbose:
                print(f"Requested {total}/{len(urls)} urls")
            yield from results


def get_char_tags(tags: list[dict]) -> list[str]:
    """Extract character tags (prefix '1:') from tag list."""
    return [
        t["tagName"].removeprefix("1:")
        for t in tags
        if t.get("tagName", "").startswith("1:")
    ]


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

    if not raw.get("post", {}).get("postId"):
        return False

    if raw.get("post", {}).get("videoId"):
        return False

    return True


def get_img_url(raw: dict) -> str:
    """Get thumbnail URL for a post."""
    post_id = raw.get("post", {}).get("postId", "")
    return f"https://orca2.furtrack.com/thumb/{post_id}.jpg" if post_id else ""


def get_character(raw: dict) -> str:
    """Get character name from post."""
    char_tags = get_char_tags(raw.get("tags", []))
    return char_tags[0] if len(char_tags) == 1 else ""


# Database functions

def create_database():
    """Create FurTrack database if it doesn't exist."""
    if os.path.exists(DB_PATH):
        print(f"{DB_PATH} already exists")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE furtrack (
            post_id TEXT PRIMARY KEY,
            char TEXT,
            url TEXT,
            raw TEXT,
            date_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print(f"Created {DB_PATH}")


def store_posts(posts: list[dict]):
    """Store post metadata in database."""
    if not posts:
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for p in posts:
        c.execute(
            "INSERT OR REPLACE INTO furtrack (post_id, char, url, raw, date_modified) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (p["post_id"], p["char"], p["url"], json.dumps(p["raw"]))
        )

    conn.commit()
    conn.close()


def get_post(post_id: str) -> dict | None:
    """Get post from database by ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT post_id, char, url, raw FROM furtrack WHERE post_id = ?", (post_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return {"post_id": row[0], "char": row[1], "url": row[2], "raw": json.loads(row[3])}
    return None


def count_character_posts(char: str) -> int:
    """Count posts for a character in database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM furtrack WHERE char = ?", (char,))
    count = c.fetchone()[0]
    conn.close()
    return count


# Download functions

def download_images(url_names: list[tuple[str, str]], folder: str) -> int:
    """Download images to folder."""
    os.makedirs(folder, exist_ok=True)

    existing = set(os.listdir(folder))
    to_download = [(u, n) for u, n in url_names if f"{n}.jpg" not in existing]

    if len(to_download) < len(url_names):
        print(f"Skipping {len(url_names) - len(to_download)} existing images")

    if not to_download:
        return 0

    print(f"Downloading {len(to_download)} images...")
    urls = [u for u, _ in to_download]
    names = [n for _, n in to_download]

    for name, r in zip(names, make_requests(urls, verbose=False)):
        if r.status_code == 200:
            with open(os.path.join(folder, f"{name}.jpg"), "wb") as f:
                f.write(r.content)
    return len(to_download)


def fetch_posts(post_ids: list[int]) -> list[dict]:
    """Fetch post metadata from FurTrack API."""
    urls = [f"https://solar.furtrack.com/get/p/{pid}" for pid in post_ids]
    results = []

    for pid, raw in zip(post_ids, make_requests(urls, function=get_json)):
        try:
            if raw is None:
                continue
            if is_valid_post(raw):
                results.append({
                    "post_id": str(pid),
                    "char": get_character(raw),
                    "url": get_img_url(raw),
                    "raw": raw,
                })
            else:
                results.append({"post_id": str(pid), "char": "", "url": "", "raw": raw})
        except Exception as e:
            print(f"Error fetching post {pid}: {e}")

    return results


def get_character_post_ids(char: str) -> list[int]:
    """Get all post IDs for a character from FurTrack."""
    url = f"https://solar.furtrack.com/get/index/1:{char}"
    data = get_json(url)
    post_ids = [int(p["postId"]) for p in data.get("posts", [])]
    random.shuffle(post_ids)
    return post_ids


def get_all_characters() -> list[tuple[str, int]]:
    """Get list of all character tags from FurTrack."""
    url = "https://solar.furtrack.com/get/tags/all"
    data = get_json(url)
    return [
        (t["tagName"].removeprefix("1:"), t.get("tagCount", 0))
        for t in data.get("tags", [])
        if t.get("tagName", "").startswith("1:")
    ]


def download_character(char: str, max_images: int = MAX_CHAR_ENTRIES, folder=IMAGES_DIR) -> int:
    """Download images for a single character."""
    # Skip if we already have enough
    existing = count_character_posts(char)
    if existing >= max_images:
        return 0

    post_ids = get_character_post_ids(char)

    # Filter to posts not in database
    new_ids = [pid for pid in post_ids if get_post(str(pid)) is None]
    new_ids = new_ids[:max_images - existing]

    if not new_ids:
        return 0

    # Fetch metadata
    posts = fetch_posts(new_ids)
    store_posts(posts)

    # Download images
    valid = [(p["url"], p["post_id"]) for p in posts if p["url"] and p["char"]]
    return download_images(valid, folder)


def download_all_characters(folder: str = IMAGES_DIR):
    """Download images for all characters."""
    posts = get_all_characters()
    print(f"Found {len(posts)} characters")

    # sort to put popular characters first
    posts.sort(key=lambda c: c[1], reverse=True)

    # random.seed(42)
    # random.shuffle(posts)

    total = 0
    for i, (char, _) in enumerate(posts):
        print(f"[{i+1}/{len(posts)}] {char}")
        count = download_character(char, MAX_CHAR_ENTRIES, folder)
        total += count
        if count:
            print(f"  Downloaded {count} images")

    print(f"\nTotal downloaded: {total}")


def index_downloaded_images():
    """Index downloaded images using SAM3 system."""
    from sam3_pursuit import SAM3FursuitIdentifier

    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} not found. Run download first.")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT post_id, char FROM furtrack WHERE char != '' AND url != ''")
    rows = c.fetchall()
    conn.close()

    print(f"Found {len(rows)} posts to index")

    identifier = SAM3FursuitIdentifier()
    indexed = 0

    for post_id, char in rows:
        img_path = os.path.join(IMAGES_DIR, f"{post_id}.jpg")
        if os.path.exists(img_path) and not identifier.db.has_post(post_id):
            try:
                identifier.add_images([char], [img_path])
                indexed += 1
            except Exception as e:
                print(f"Error indexing {post_id}: {e}")

    print(f"Indexed {indexed} images")


def main():
    parser = argparse.ArgumentParser(description="Download FurTrack fursuit images")
    parser.add_argument("--download-characters", action="store_true", help="Download all characters")
    parser.add_argument("--download-character", type=str, help="Download specific character")
    parser.add_argument("--index", action="store_true", help="Index downloaded images with SAM3")
    parser.add_argument("--max-images", type=int, default=MAX_CHAR_ENTRIES, help="Max images per character")

    args = parser.parse_args()

    create_database()

    if args.download_character:
        count = download_character(args.download_character, args.max_images)
        print(f"Downloaded {count} images for {args.download_character}")
    elif args.download_characters:
        download_all_characters()
    elif args.index:
        index_downloaded_images()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
