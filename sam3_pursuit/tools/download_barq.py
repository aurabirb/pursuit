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
EXCLUDED_POST_IDS: set[str] = set()  # Set by CLI for --exclude-datasets
MAX_CONCURRENT_DOWNLOADS = 5
GRAPHQL_URL = "https://api.barq.app/graphql"
REQUEST_DELAY = 0.5
MAX_RETRIES = 5
INITIAL_BACKOFF = 5

PROFILE_SEARCH_QUERY_MINIMAL = """
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

PROFILE_SEARCH_QUERY_FULL = """
query ProfileSearch($filters: ProfileSearchFiltersInput! = {}, $cursor: String = "", $limit: Int = 30) {
  profileSearch(filters: $filters, cursor: $cursor, limit: $limit, sort: distance) {
    id
    uuid
    displayName
    username
    roles
    age
    dateOfBirth
    ...ProfilePrimaryImagesFragment
    ...ProfileHeaderImagesFragment
    privacySettings {
    ...PrivacySettingsFragment
    __typename
    }
    location {
    ...ProfileLocationFragment
    precision
    __typename
    }
    images {
    ...ProfileImageFragment
    __typename
    }
    bio {
    ...ProfileBioFragment
    __typename
    }
    socialAccounts {
    ...ProfileSocialAccountFragment
    __typename
    }
    bioAd {
    ...ProfileBioAdFragment
    __typename
    }
    kinks(type: all) {
    ...ProfileKinkFragment
    __typename
    }
    sonas {
    ...SonaFragment
    __typename
    }
    __typename
  }
}

fragment ProfilePrimaryImagesFragment on Profile {
  ...ProfileSafePrimaryImageFragment
  ...ProfileExplicitPrimaryImageFragment
  __typename
}

fragment ProfileSafePrimaryImageFragment on Profile {
  primaryImage {
    ...UploadedImageFragment
    __typename
  }
  __typename
}

fragment UploadedImageFragment on UploadedImage {
  uuid
  contentRating
  width
  height
  blurHash
  __typename
}

fragment ProfileExplicitPrimaryImageFragment on Profile {
  primaryImageAd {
    ...UploadedImageFragment
    __typename
  }
  __typename
}

fragment ProfileHeaderImagesFragment on Profile {
  ...ProfileSafeHeaderImageFragment
  ...ProfileExplicitHeaderImageFragment
  __typename
}

fragment ProfileSafeHeaderImageFragment on Profile {
  headerImage {
    ...UploadedImageFragment
    __typename
  }
  __typename
}

fragment ProfileExplicitHeaderImageFragment on Profile {
  headerImageAd {
    ...UploadedImageFragment
    __typename
  }
  __typename
}

fragment PrivacySettingsFragment on PrivacySettings {
  startChat
  viewKinks
  viewAge
  viewAd
  viewProfile
  showLastOnline
  __typename
}

fragment ProfileLocationFragment on ProfileLocation {
  type
  homePlace {
    ...PlaceFragment
    __typename
  }
  place {
    ...PlaceFragment
    __typename
  }
  __typename
}

fragment PlaceFragment on Place {
  id
  place
  region
  country
  countryCode
  longitude
  latitude
  __typename
}

fragment ProfileImageFragment on ProfileImage {
  id
  image {
    ...UploadedImageFragment
    __typename
  }
  accessPermission
  isAd
  __typename
}

fragment ProfileBioFragment on ProfileBio {
  biography
  genders
  languages
  relationshipStatus
  sexualOrientation
  interests
  hobbies {
    ...InterestFragment
    __typename
  }
  __typename
}

fragment InterestFragment on Interest {
  interest
  __typename
}

fragment ProfileSocialAccountFragment on ProfileSocialAccount {
  id
  socialNetwork
  isVerified
  url
  displayName
  value
  accessPermission
  __typename
}

fragment ProfileBioAdFragment on ProfileBioAd {
  biography
  sexPositions
  behaviour
  safeSex
  canHost
  __typename
}

fragment ProfileKinkFragment on ProfileKink {
  pleasureGive
  pleasureReceive
  kink {
    ...KinkFragment
    __typename
  }
  __typename
}

fragment KinkFragment on Kink {
  id
  displayName
  categoryName
  isVerified
  isSinglePlayer
  __typename
}

fragment SonaFragment on Sona {
  id
  displayName
  hasFursuit
  species {
    ...SpeciesFragment
    __typename
  }
  images {
    ...SonaImageFragment
    __typename
  }
  __typename
}

fragment SpeciesFragment on Species {
  id
  displayName
  __typename
}

fragment SonaImageFragment on ProfileImage {
  id
  image {
    ...UploadedImageFragment
    __typename
  }
  isAd
  __typename
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
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS filtered_images (
            image_uuid TEXT PRIMARY KEY,
            filtered_at INTEGER NOT NULL,
            max_score REAL
        )
    """)
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS failed_images (
            image_uuid TEXT PRIMARY KEY,
            failed_at INTEGER NOT NULL,
            status_code INTEGER
        )
    """)
    try:
        _conn.execute("ALTER TABLE filtered_images ADD COLUMN max_score REAL")
    except sqlite3.OperationalError:
        pass
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


def save_filtered_image(image_uuid: str, max_score: float):
    conn = get_cache_db()
    conn.execute(
        "INSERT OR REPLACE INTO filtered_images (image_uuid, filtered_at, max_score) VALUES (?, ?, ?)",
        (image_uuid, int(time.time()), max_score)
    )
    conn.commit()


def get_filtered_images(threshold: float) -> set[str]:
    rows = get_cache_db().execute(
        "SELECT image_uuid FROM filtered_images WHERE max_score IS NULL OR max_score < ?",
        (threshold,)
    ).fetchall()
    return {row[0] for row in rows}


def get_all_classified_images() -> dict[str, float]:
    """Get all image UUIDs with their scores."""
    rows = get_cache_db().execute(
        "SELECT image_uuid, max_score FROM filtered_images WHERE max_score IS NOT NULL"
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def save_failed_image(image_uuid: str, status_code: int):
    conn = get_cache_db()
    conn.execute(
        "INSERT OR IGNORE INTO failed_images (image_uuid, failed_at, status_code) VALUES (?, ?, ?)",
        (image_uuid, int(time.time()), status_code)
    )
    conn.commit()


def get_failed_images() -> set[str]:
    rows = get_cache_db().execute("SELECT image_uuid FROM failed_images").fetchall()
    return {row[0] for row in rows}


def clean_empty_dirs(images_dir: Path | str | None = None) -> int:
    """Remove empty character directories. Returns count of removed directories."""
    images_dir = Path(images_dir or IMAGES_DIR)
    if not images_dir.exists():
        return 0

    removed = 0
    for char_folder in list(images_dir.iterdir()):
        if char_folder.is_dir() and not any(char_folder.iterdir()):
            char_folder.rmdir()
            removed += 1
    return removed


def clean_images(score_fn, threshold: float):
    """Delete existing images that score below the threshold.

    Uses cached scores from database when available, only runs classifier on new images.
    """
    from PIL import Image

    images_dir = Path(IMAGES_DIR)
    if not images_dir.exists():
        print(f"No images directory: {images_dir}")
        return

    # Get cached scores from database
    cached_scores = get_all_classified_images()
    print(f"Found {len(cached_scores)} cached scores in database")

    # Get all images on disk
    all_images = list(images_dir.glob("*/*.jpg"))
    print(f"Found {len(all_images)} images on disk ({images_dir})")

    if not all_images:
        print("Nothing to clean")
        return

    deleted_cached = 0
    deleted_new = 0
    kept = 0
    to_classify = []

    # First pass: use cached scores
    print("Checking cached scores...")
    for img_path in all_images:
        img_uuid = img_path.stem
        if img_uuid in cached_scores:
            score = cached_scores[img_uuid]
            if score < threshold:
                img_path.unlink()
                print(f"  {img_path.parent.name}: {img_uuid}.jpg (deleted from cache: {score:.0%} < {threshold:.0%})")
                deleted_cached += 1
            else:
                kept += 1
        else:
            to_classify.append(img_path)

    # Second pass: classify new images
    if to_classify:
        print(f"\nClassifying {len(to_classify)} new images...")
        for i, img_path in enumerate(to_classify, 1):
            img_uuid = img_path.stem
            char_folder = img_path.parent

            try:
                img = Image.open(img_path)
                score = score_fn(img)
                save_filtered_image(img_uuid, score)
                if score < threshold:
                    img_path.unlink()
                    print(f"  [{i}/{len(to_classify)}] {char_folder.name}: {img_uuid}.jpg (deleted: {score:.0%} < {threshold:.0%})")
                    deleted_new += 1
                else:
                    print(f"  [{i}/{len(to_classify)}] {char_folder.name}: {img_uuid}.jpg (kept: {score:.0%})")
                    kept += 1
            except Exception as e:
                print(f"  [{i}/{len(to_classify)}] Error processing {img_path}: {e}")

    # Clean up empty directories
    removed_dirs = clean_empty_dirs(images_dir)

    # Summary
    total_deleted = deleted_cached + deleted_new
    cache_msg = f" ({deleted_cached} from cache, {deleted_new} newly classified)" if deleted_cached and deleted_new else ""
    empty_msg = f", removed {removed_dirs} empty directories" if removed_dirs else ""
    print(f"\nDeleted: {total_deleted}{cache_msg}, kept: {kept}{empty_msg}")


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


_PLACEHOLDER_NAMES = {"likes only", "liked only", "private", "mutuals only"}


def _is_placeholder_name(name: str) -> bool:
    """Check if a name is a placeholder rather than a real character name."""
    return name.lower().strip() in _PLACEHOLDER_NAMES


USE_SONA_NAMES = False  # Use fursuit character (sona) names instead of display names


def get_folder_name(profile: dict) -> str:
    """Get folder name as {id}.{readable_name}."""
    pid = profile.get("id") or profile.get("uuid")

    # Try username first
    name = profile.get("username")
    if name and _is_placeholder_name(name):
        name = None

    # Try social accounts: prefer twitter, then telegram
    if not name:
        socials = profile.get("socialAccounts") or []
        # Filter out private/mutuals-only/liked-only
        valid = [s for s in socials if not _is_private_social(s)]
        # Sort by preference
        priority = {"twitter": 0, "telegram": 1, "furAffinity": 2}
        valid.sort(key=lambda s: priority.get(s.get("socialNetwork"), 99))
        if valid:
            val = valid[0].get("value", "").lstrip("@")
            if val and not _is_placeholder_name(val):
                name = val

    # Optionally try sona name (character/fursona name from profile)
    if not name and USE_SONA_NAMES:
        sonas = profile.get("sonas") or []
        for sona in sonas:
            sona_name = sona.get("displayName")
            if sona_name and not _is_placeholder_name(sona_name):
                name = sona_name
                break

    # Fall back to displayName or UUID
    if not name:
        display = profile.get("displayName")
        if display and not _is_placeholder_name(display):
            name = display
        else:
            name = profile.get("uuid", "unknown")

    # Sanitize
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_").replace("\0", "").replace("|", "_")
    return f"{pid}.{name}"


def _is_private_social(social: dict) -> bool:
    val = (social.get("value") or "").lower().strip()
    if val in ("private", "@private", "likes only", "@likes only", "liked only", "@liked only"):
        return True
    if val.endswith("mutuals only") or val.endswith("likes only") or val.endswith("liked only"):
        return True
    return False


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


async def fetch_profiles(session: aiohttp.ClientSession, lat: float, lon: float, cursor: str = "", limit: int = 100, full_profile: bool = True) -> tuple[list[dict], str]:
    payload = {
        "operationName": "ProfileSearch",
        "variables": {
            "filters": {"location": {"latitude": lat, "longitude": lon, "type": "distance"}},
            "cursor": cursor,
            "limit": limit,
        },
        "query": PROFILE_SEARCH_QUERY_FULL if full_profile else PROFILE_SEARCH_QUERY_MINIMAL,
    }
    resp = await fetch_with_backoff(session, "POST", GRAPHQL_URL, json=payload)
    if not resp or resp.status != 200:
        print(f"API error: {resp.status if resp else 'no response'}")
        return [], ""
    data = await resp.json()
    profiles = data.get("data", {}).get("profileSearch", [])
    next_cursor = str(int(cursor or "0") + limit) if profiles else ""
    return profiles, next_cursor


async def download_image(session: aiohttp.ClientSession, image_uuid: str, dest: Path, semaphore: asyncio.Semaphore) -> tuple[bool, int | None]:
    """Download image. Returns (success, status_code)."""
    url = f"https://assets.barq.app/image/{image_uuid}.jpeg?width=512"
    async with semaphore:
        resp = await fetch_with_backoff(session, "GET", url)
        if resp and resp.status == 200:
            dest.write_bytes(await resp.read())
            return True, 200
        status = resp.status if resp else None
        print(f"  Failed {image_uuid}: {status or 'no response'}")
        return False, status


def score_and_filter_image(dest: Path, img_uuid: str, score_fn, threshold: float, filtered_uuids: set[str]) -> tuple[bool, float | None]:
    """Score an image and delete if below threshold. Returns (kept, score)."""
    from PIL import Image
    try:
        img = Image.open(dest)
        score = score_fn(img)
        save_filtered_image(img_uuid, score)
        if score < threshold:
            dest.unlink()
            filtered_uuids.add(img_uuid)
            return False, score
        return True, score
    except Exception:
        return True, None  # Keep on error


async def download_all_profiles(lat: float, lon: float, max_pages: int = 100, all_images: bool = False, max_age: float | None = None, score_fn=None, threshold: float = 0.85, full_profile: bool = True):
    """Download profile images from Barq."""
    headers = get_headers()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    filtered_uuids = get_filtered_images(threshold) if score_fn else set()
    cached_scores = get_all_classified_images() if score_fn else {}  # uuid -> score
    failed_uuids = get_failed_images()

    async with aiohttp.ClientSession(headers=headers) as session:
        cursor = ""
        total = 0
        skipped = 0
        filtered = 0
        restored = 0

        for page in range(max_pages):
            if page > 0:
                await asyncio.sleep(REQUEST_DELAY)

            profiles, cursor = await fetch_profiles(session, lat, lon, cursor, full_profile=full_profile)
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

                    # Classify existing images that haven't been scored yet
                    if score_fn:
                        for img_uuid in img_uuids:
                            if img_uuid in existing and img_uuid not in cached_scores:
                                dest = char_folder / f"{img_uuid}.jpg"
                                kept, score = score_and_filter_image(dest, img_uuid, score_fn, threshold, filtered_uuids)
                                if score is not None:
                                    cached_scores[img_uuid] = score
                                if not kept:
                                    existing.discard(img_uuid)
                                    print(f"  {folder_name}: {img_uuid}.jpg (filtered existing: {score:.0%} < {threshold:.0%})")
                                    filtered += 1

                    new_uuids = [u for u in img_uuids if u not in existing and u not in filtered_uuids and u not in failed_uuids and u not in EXCLUDED_POST_IDS]

                    if not new_uuids:
                        print(f"  {folder_name}: up to date ({len(existing)} images)")
                        continue

                    for img_uuid in new_uuids:
                        dest = char_folder / f"{img_uuid}.jpg"

                        # Create directory only when we're about to save a file
                        char_folder.mkdir(parents=True, exist_ok=True)
                        success, status = await download_image(session, img_uuid, dest, semaphore)
                        if not success:
                            if status == 404:
                                save_failed_image(img_uuid, status)
                                failed_uuids.add(img_uuid)
                            continue

                        if score_fn:
                            # Check if we have a cached score that passes threshold (restoring previously filtered image)
                            if img_uuid in cached_scores:
                                cached_score = cached_scores[img_uuid]
                                print(f"  {folder_name}: {img_uuid}.jpg (restored, cached score: {cached_score:.0%})")
                                restored += 1
                            else:
                                kept, score = score_and_filter_image(dest, img_uuid, score_fn, threshold, filtered_uuids)
                                if score is not None:
                                    cached_scores[img_uuid] = score
                                if not kept:
                                    print(f"  {folder_name}: {img_uuid}.jpg (filtered: {score:.0%} < {threshold:.0%})")
                                    filtered += 1
                                    continue
                                print(f"  {folder_name}: {img_uuid}.jpg (score: {score:.0%})")
                        else:
                            print(f"  {folder_name}: {img_uuid}.jpg")
                        total += 1
                        existing.add(img_uuid)
                except Exception as e:
                    print(f"  Error processing profile {p.get('id', '?')}: {e}")

            if not cursor:
                break

        # Clean up empty directories left by filtering
        removed_dirs = clean_empty_dirs()
        restored_msg = f", restored: {restored}" if restored else ""
        filter_msg = f", filtered (not fursuit): {filtered}" if filtered else ""
        empty_msg = f", removed {removed_dirs} empty directories" if removed_dirs else ""
        print(f"\nTotal downloaded: {total}, skipped (cached): {skipped}{restored_msg}{filter_msg}{empty_msg}")


def main():
    parser = argparse.ArgumentParser(description="Download Barq profile images")
    parser.add_argument("--download-all", action="store_true", help="Download all profiles")
    parser.add_argument("--lat", type=float, default=52.378, help="Latitude for search center")
    parser.add_argument("--lon", type=float, default=4.9, help="Longitude for search center")
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages to fetch")
    parser.add_argument("--all-images", action="store_true", help="Download all images per profile (not just primary)")
    parser.add_argument("--minimal", action="store_true", help="Only download image metadata, not full profile data")
    parser.add_argument("--max-age", type=float, help="Skip profiles cached within this many days")
    parser.add_argument("--skip-non-fursuit", action="store_true", help="Filter out non-fursuit images using CLIP")
    args = parser.parse_args()

    if args.download_all:
        score_fn = None
        if args.skip_non_fursuit:
            from sam3_pursuit.models.classifier import ImageClassifier
            classifier = ImageClassifier()
            score_fn = classifier.fursuit_score
        asyncio.run(download_all_profiles(args.lat, args.lon, args.max_pages, args.all_images, args.max_age, score_fn=score_fn, full_profile=not args.minimal))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
