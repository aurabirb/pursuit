#!/usr/bin/env python3
"""Download fursuit images from Barq.

Requires BARQ_BEARER_TOKEN environment variable (or in .env file).

Usage:
    python tools/download_barq.py --download-all
    python tools/download_barq.py --download-all --max-age 7  # refresh profiles older than 7 days
"""

import argparse
import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()

IMAGES_DIR = "barq_images"
CACHE_DB = "barq_cache.db"
MAX_CONCURRENT_DOWNLOADS = 5
GRAPHQL_URL = "https://api.barq.app/graphql"
REQUEST_DELAY = 0.5
MAX_RETRIES = 5
INITIAL_BACKOFF = 5

PROFILE_SEARCH_QUERY = """
query ProfileSearch($filters: ProfileSearchFiltersInput! = {}, $cursor: String = "", $limit: Int = 30) {
  profileSearch(filters: $filters, cursor: $cursor, limit: $limit, sort: distance) {
    uuid
    id
    displayName
    username
    socialAccounts { id socialNetwork isVerified url displayName value accessPermission }
    primaryImage { uuid }
    images { image { uuid } }
  }
}
"""

_conn: sqlite3.Connection | None = None


def get_cache_db() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    _conn = sqlite3.connect(CACHE_DB)
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
    """)
    _conn.commit()
    return _conn


def get_cached_profile(profile_id: str) -> dict | None:
    row = get_cache_db().execute(
        "SELECT data FROM profiles WHERE id = ?", (profile_id,)
    ).fetchone()
    return json.loads(row[0]) if row else None


def get_profile_age_days(profile_id: str) -> float | None:
    row = get_cache_db().execute(
        "SELECT updated_at FROM profiles WHERE id = ?", (profile_id,)
    ).fetchone()
    if not row:
        return None
    return (time.time() - row[0]) / 86400


def save_profile(profile: dict):
    conn = get_cache_db()
    conn.execute(
        "INSERT OR REPLACE INTO profiles (id, data, updated_at) VALUES (?, ?, ?)",
        (profile["id"], json.dumps(profile), int(time.time()))
    )
    conn.commit()


def get_headers() -> dict:
    token = os.environ.get("BARQ_BEARER_TOKEN")
    if not token:
        raise ValueError("BARQ_BEARER_TOKEN environment variable required")
    return {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "origin": "https://app.barq.app",
    }


def get_folder_name(profile: dict) -> str:
    """Get folder name as {id}.{readable_name}."""
    pid = profile.get("id") or profile.get("uuid")

    # Try username first
    name = profile.get("username")

    # Try social accounts: prefer twitter, then telegram
    if not name:
        socials = profile.get("socialAccounts") or []
        # Filter out private/mutuals-only
        valid = [s for s in socials if not _is_private_social(s)]
        # Sort by preference
        priority = {"twitter": 0, "telegram": 1, "furAffinity": 2}
        valid.sort(key=lambda s: priority.get(s.get("socialNetwork"), 99))
        if valid:
            name = valid[0].get("value", "").lstrip("@")

    if not name:
        name = profile.get("displayName") or profile.get("uuid", "unknown")

    # Sanitize
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_").replace("\0", "").replace("|", "_")
    return f"{pid}.{name}"


def _is_private_social(social: dict) -> bool:
    val = (social.get("value") or "").lower()
    return val in ("private", "@private") or val.endswith("mutuals only")


def get_existing_images(char_folder: Path) -> set[str]:
    if not char_folder.exists():
        return set()
    return {p.stem for p in char_folder.glob("*.jpg")}


async def fetch_with_backoff(session: aiohttp.ClientSession, method: str, url: str, **kwargs) -> aiohttp.ClientResponse | None:
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            resp = await session.request(method, url, **kwargs)
            if resp.status == 200:
                return resp
            if resp.status >= 500 or resp.status == 429:
                print(f"  HTTP {resp.status}, retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            return resp
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"  Request failed: {e}, retrying in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff *= 2
    return None


async def fetch_profiles(session: aiohttp.ClientSession, lat: float, lon: float, cursor: str = "", limit: int = 60) -> tuple[list[dict], str]:
    payload = {
        "operationName": "ProfileSearch",
        "variables": {
            "filters": {"location": {"latitude": lat, "longitude": lon, "type": "distance"}},
            "cursor": cursor,
            "limit": limit,
        },
        "query": PROFILE_SEARCH_QUERY,
    }
    resp = await fetch_with_backoff(session, "POST", GRAPHQL_URL, json=payload)
    if not resp or resp.status != 200:
        print(f"API error: {resp.status if resp else 'no response'}")
        return [], ""
    data = await resp.json()
    profiles = data.get("data", {}).get("profileSearch", [])
    next_cursor = str(int(cursor or "0") + limit) if profiles else ""
    return profiles, next_cursor


async def download_image(session: aiohttp.ClientSession, image_uuid: str, dest: Path, semaphore: asyncio.Semaphore) -> bool:
    url = f"https://assets.barq.app/image/{image_uuid}.jpeg?width=512"
    async with semaphore:
        resp = await fetch_with_backoff(session, "GET", url)
        if resp and resp.status == 200:
            dest.write_bytes(await resp.read())
            return True
        print(f"  Failed {image_uuid}: {resp.status if resp else 'no response'}")
        return False


async def download_all_profiles(lat: float, lon: float, max_pages: int = 100, all_images: bool = False, max_age: float | None = None, classify_fn=None):
    """Download profile images from Barq."""
    headers = get_headers()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async with aiohttp.ClientSession(headers=headers) as session:
        cursor = ""
        total = 0
        skipped = 0
        filtered = 0

        for page in range(max_pages):
            if page > 0:
                await asyncio.sleep(REQUEST_DELAY)

            profiles, cursor = await fetch_profiles(session, lat, lon, cursor)
            if not profiles:
                break

            print(f"Page {page + 1}: {len(profiles)} profiles")

            for p in profiles:
                try:
                    pid = p.get("id")
                    if not pid:
                        continue

                    # Check cache age
                    age = get_profile_age_days(pid)
                    if age is not None and max_age and age < max_age:
                        skipped += 1
                        continue

                    # Save profile to cache
                    save_profile(p)

                    # Get image UUIDs
                    if all_images:
                        img_uuids = [((e or {}).get("image") or {}).get("uuid") for e in p.get("images") or []]
                    else:
                        primary = p.get("primaryImage") or {}
                        img_uuids = [primary.get("uuid")] if primary.get("uuid") else []

                    img_uuids = [u for u in img_uuids if u]
                    if not img_uuids:
                        print(f"  {get_folder_name(p)}: no images")
                        continue

                    folder_name = get_folder_name(p)
                    char_folder = Path(IMAGES_DIR) / folder_name
                    existing = get_existing_images(char_folder)
                    new_uuids = [u for u in img_uuids if u not in existing]

                    if not new_uuids:
                        print(f"  {folder_name}: up to date ({len(existing)} images)")
                        continue

                    for img_uuid in new_uuids:
                        char_folder.mkdir(parents=True, exist_ok=True)
                        dest = char_folder / f"{img_uuid}.jpg"

                        if await download_image(session, img_uuid, dest, semaphore):
                            if classify_fn:
                                from PIL import Image
                                try:
                                    img = Image.open(dest)
                                    if not classify_fn(img):
                                        dest.unlink()
                                        print(f"  {folder_name}: {img_uuid}.jpg (filtered: not fursuit)")
                                        filtered += 1
                                        continue
                                except Exception:
                                    pass
                            print(f"  {folder_name}: {img_uuid}.jpg")
                            total += 1
                            existing.add(img_uuid)
                except Exception as e:
                    print(f"  Error processing profile {p.get('id', '?')}: {e}")

            if not cursor:
                break

        filter_msg = f", filtered (not fursuit): {filtered}" if filtered else ""
        print(f"\nTotal downloaded: {total}, skipped (cached): {skipped}{filter_msg}")


def main():
    parser = argparse.ArgumentParser(description="Download Barq profile images")
    parser.add_argument("--download-all", action="store_true", help="Download all profiles")
    parser.add_argument("--lat", type=float, default=52.378, help="Latitude for search center")
    parser.add_argument("--lon", type=float, default=4.9, help="Longitude for search center")
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages to fetch")
    parser.add_argument("--all-images", action="store_true", help="Download all images per profile (not just primary)")
    parser.add_argument("--max-age", type=float, help="Skip profiles cached within this many days")
    parser.add_argument("--skip-non-fursuit", action="store_true", help="Filter out non-fursuit images using CLIP")
    args = parser.parse_args()

    if args.download_all:
        classify_fn = None
        if args.skip_non_fursuit:
            from sam3_pursuit.models.classifier import ImageClassifier
            classifier = ImageClassifier()
            classify_fn = classifier.is_fursuit
        asyncio.run(download_all_profiles(args.lat, args.lon, args.max_pages, args.all_images, args.max_age, classify_fn=classify_fn))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
