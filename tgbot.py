"""Telegram bot for fursuit character identification using SAM3 system."""

import asyncio
import html
import os
import random
import re
import sys
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

from aiohttp import web
from dotenv import load_dotenv
from PIL import Image
from telegram import ReactionTypeEmoji, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import aitool
import webserver

load_dotenv()

from sam3_pursuit import FursuitIdentifier, FursuitIngestor, Config
from sam3_pursuit.api.identifier import discover_datasets, merge_multi_dataset_results
from sam3_pursuit.api.annotator import annotate_image
from telegram import InputMediaPhoto

from sam3_pursuit.storage.database import Database, SOURCE_TGBOT, get_git_version, get_source_url, get_source_image_url
from typing import Optional

# Pattern to match "character:Name" in caption
CHARACTER_PATTERN = re.compile(r"character:(\S+)", re.IGNORECASE)
# Pattern to match "/furscan CharName" in caption
FURSCAN_CAPTION_PATTERN = re.compile(r"^/furscan\s+(.+)", re.IGNORECASE)

# Global instances (lazy loaded)
_identifiers: Optional[list[FursuitIdentifier]] = None
_ingestor: Optional[FursuitIngestor] = None
_barq_image_to_profile: Optional[dict[str, str]] = None
_barq_nsfw_images: Optional[set[str]] = None
_tgbot_post_to_uploader: Optional[dict[str, str]] = None


def _load_barq_cache():
    """Load barq cache: image→profile URL mapping and NSFW image set."""
    global _barq_image_to_profile, _barq_nsfw_images
    _barq_image_to_profile = {}
    _barq_nsfw_images = set()
    cache_path = Path(Config.BASE_DIR) / "barq_cache.db"
    if not cache_path.exists():
        return
    import json, sqlite3
    try:
        conn = sqlite3.connect(str(cache_path))
        for row in conn.execute("SELECT id, data FROM profiles"):
            data = json.loads(row[1])
            username = data.get("username")
            # Collect all images with their content ratings
            images = []
            if data.get("primaryImage"):
                images.append(data["primaryImage"])
            for img in data.get("images", []):
                if img.get("image"):
                    images.append(img["image"])
            for img in images:
                uid = img.get("uuid")
                if not uid:
                    continue
                rating = img.get("contentRating", "safe")
                if rating in ("explicit", "hard"):
                    _barq_nsfw_images.add(uid)
                if username:
                    _barq_image_to_profile[uid] = f"https://barq.app/@{username}"
        conn.close()
        print(f"Barq cache: {len(_barq_image_to_profile)} image→profile mappings, {len(_barq_nsfw_images)} NSFW images")
    except Exception as e:
        print(f"Warning: could not load barq_cache.db: {e}")


def _load_tgbot_cache():
    """Load tgbot cache: post_id→uploaded_by URL mapping from all datasets."""
    global _tgbot_post_to_uploader
    _tgbot_post_to_uploader = {}
    try:
        identifiers = get_identifiers()
        for ident in identifiers:
            conn = ident.db._connect()
            rows = conn.execute(
                "SELECT DISTINCT post_id, uploaded_by FROM detections WHERE source = ? AND uploaded_by IS NOT NULL",
                (SOURCE_TGBOT,)
            ).fetchall()
            for post_id, uploaded_by in rows:
                if post_id not in _tgbot_post_to_uploader:
                    _tgbot_post_to_uploader[post_id] = uploaded_by
        print(f"Tgbot cache: {len(_tgbot_post_to_uploader)} post→uploader mappings")
    except Exception as e:
        print(f"Warning: could not load tgbot cache: {e}")


def _get_tgbot_uploader_url(post_id: str) -> Optional[str]:
    """Look up a tgbot uploader URL for a post_id."""
    if _tgbot_post_to_uploader is None:
        _load_tgbot_cache()
    return _tgbot_post_to_uploader.get(post_id)


def _register_tgbot_uploader(post_id: str, uploaded_by: str):
    """Register a new tgbot post→uploader mapping in the cache."""
    if _tgbot_post_to_uploader is None:
        _load_tgbot_cache()
    _tgbot_post_to_uploader[post_id] = uploaded_by


def _get_page_url(source: Optional[str], post_id: str) -> Optional[str]:
    """Get a page URL for a detection, preferring barq profile links and tgbot sender links."""
    if source == "barq":
        profile_url = _get_barq_profile_url(post_id)
        if profile_url:
            return profile_url
    if source == SOURCE_TGBOT:
        uploader_url = _get_tgbot_uploader_url(post_id)
        if uploader_url:
            return uploader_url
    return get_source_url(source, post_id)


def _is_barq_nsfw(image_uuid: str) -> bool:
    """Check if a barq image is marked as explicit/hard."""
    if _barq_nsfw_images is None:
        _load_barq_cache()
    return image_uuid in _barq_nsfw_images


def _get_barq_profile_url(image_uuid: str) -> Optional[str]:
    """Look up a barq profile URL for an image UUID using barq_cache.db."""
    if _barq_image_to_profile is None:
        _load_barq_cache()
    return _barq_image_to_profile.get(image_uuid)


def get_identifiers():
    global _identifiers
    if _identifiers:
        return _identifiers
    base_dir = Config.BASE_DIR
    datasets = discover_datasets(base_dir)
    if not datasets:
        raise FileNotFoundError(
            f"No datasets found in {base_dir}. "
            "Expected *.db and *.index file pairs."
        )
    names = [Path(db_path).stem for db_path, _ in datasets]
    print(f"Auto-discovered {len(datasets)} dataset(s): {', '.join(names)}")
    _identifiers = [
        FursuitIdentifier(
            db_path=db_path,
            index_path=index_path,
            segmentor_model_name=Config.SAM3_MODEL,
            segmentor_concept=Config.DEFAULT_CONCEPT,
        )
        for db_path, index_path in datasets
    ]
    return _identifiers


def get_ingestor() -> FursuitIngestor:
    """Get or create the ingestor instance (writes to DB)."""
    global _ingestor
    if _ingestor is None:
        _ingestor = FursuitIngestor(segmentor_model_name=Config.SAM3_MODEL, segmentor_concept=Config.DEFAULT_CONCEPT)
    return _ingestor


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages.

    If caption contains "character:Name", adds the image to the database.
    Otherwise, identifies the character in the image.
    """
    if not update.message:
        print("Invalid message", file=sys.stderr)
        return

    attachment = update.message.effective_attachment
    if not attachment:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Please send a photo."
        )
        return

    caption = update.message.caption or ""
    print(f"Received photo with caption: {caption}")

    # Check if this is an add request via /furscan caption
    furscan_match = FURSCAN_CAPTION_PATTERN.search(caption)
    if furscan_match:
        character_name = re.sub(r"^this\s+is\s+", "", furscan_match.group(1), flags=re.IGNORECASE).strip().lstrip("@")
        if character_name:
            await add_photo(update, context, character_name)
            return

    # Check if this is an add request via character:Name caption
    match = CHARACTER_PATTERN.search(caption)
    if match:
        await add_photo(update, context, match.group(1))
    else:
        await identify_photo(update, context)

async def download_tg_file(new_file):
    tg_dir = f"datasets/{Config.DEFAULT_DATASET}/tg_download"
    os.makedirs(tg_dir, exist_ok=True)
    temp_path = Path(tg_dir) / f"{new_file.file_id}.jpg"
    print(f"Downloading into {temp_path}")
    with open(temp_path, 'wb') as f:
        bs = await new_file.download_as_bytearray()
        f.write(bs)
        f.flush()
    return temp_path

def make_tgbot_post_id(chat_id: int, msg_id: int, file_id: str) -> str:
    """Create a unique post_id from telegram message identifiers."""
    return f"{chat_id}_{msg_id}_{file_id}"


async def _react(message, emoji: str):
    """Set a reaction on a message. Best-effort, silently ignores errors."""
    try:
        await message.set_reaction([ReactionTypeEmoji(emoji=emoji)])
    except Exception:
        pass


def _get_sender_url(user) -> Optional[str]:
    """Get a t.me/ URL for a Telegram user, or user ID as fallback."""
    if user and user.username:
        return f"https://t.me/{user.username}"
    if user:
        return str(user.id)
    return None


async def add_photo(update: Update, context: ContextTypes.DEFAULT_TYPE, character_name: str,
                    photo_message=None):
    """Add a photo to the database for a character.

    Args:
        photo_message: If provided, use this message's photo instead of update.message.
    """
    msg = photo_message or update.message
    await _react(update.message, "🗿")
    attachment = msg.effective_attachment
    new_file = await attachment[-1].get_file()
    user = update.effective_user
    uploaded_by = _get_sender_url(user)
    post_id = make_tgbot_post_id(update.effective_chat.id, msg.message_id, new_file.file_id)
    try:
        temp_path = await download_tg_file(new_file)
        # Rename temp file to use post_id so identifier extracts it correctly
        post_id_path = temp_path.parent / f"{post_id}.jpg"
        temp_path.rename(post_id_path)
        ingestor = get_ingestor()
        added = await asyncio.to_thread(
            ingestor.add_images,
            character_names=[character_name],
            image_paths=[str(post_id_path)],
            source=SOURCE_TGBOT,
            uploaded_by=uploaded_by,
            add_full_image=True,
        )

        # Clean up temp file
        # os.unlink(temp_path)

        if added > 0:
            if uploaded_by:
                _register_tgbot_uploader(post_id, uploaded_by)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Added {added} image(s) for character '{character_name}'."
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Failed to add image for '{character_name}'. No segments detected."
            )

    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Error adding image. Please try again."
        )


async def identify_and_send(context: ContextTypes.DEFAULT_TYPE, chat_id: int,
                           photo_attachment, reply_to_message_id: int = None,
                           react_message=None):
    """Download photo, identify characters, and send annotated result."""
    if react_message:
        await _react(react_message, "👀")
    new_file = await photo_attachment[-1].get_file()
    temp_path = await download_tg_file(new_file)
    image = Image.open(temp_path)

    identifiers = get_identifiers()

    def _run_identify():
        all_results = [ident.identify(image, top_k=Config.DEFAULT_TOP_K) for ident in identifiers]
        return merge_multi_dataset_results(all_results, top_k=Config.DEFAULT_TOP_K)

    results = await asyncio.to_thread(_run_identify)

    reply_kwargs = {"chat_id": chat_id}
    if reply_to_message_id:
        reply_kwargs["reply_to_message_id"] = reply_to_message_id

    if not results:
        await context.bot.send_message(**reply_kwargs, text="No matching characters found.")
        return

    min_confidence = Config.DEFAULT_MIN_CONFIDENCE
    lines = []
    for i, result in enumerate(results, 1):
        filtered = [m for m in result.matches if m.confidence >= min_confidence]
        if not filtered:
            continue
        lines.append(f"Segment {i}:")
        for n, m in enumerate(filtered):
            url = _get_page_url(m.source, m.post_id)
            img_url = get_source_image_url(m.source, m.post_id)
            name = html.escape(m.character_name or 'Unknown')
            pct = f"{m.confidence*100:.1f}%"
            name_part = f"<a href=\"{url}\">{name}</a>" if url else name
            pct_part = f"<a href=\"{img_url}\">{pct}</a>" if img_url else pct
            lines.append(f"  {n+1}. {name_part} ({pct_part})")

    if not lines:
        await context.bot.send_message(**reply_kwargs, text=f"No matches above {min_confidence:.0%} confidence.")
        return

    watermark_text = f"Pursuit {get_git_version()}"
    annotated = await asyncio.to_thread(annotate_image, image, results, min_confidence, watermark_text)
    with NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        annotated.save(f, format="JPEG", quality=90)
        temp_annotated_path = f.name

    msg = "\n".join(lines)
    print(msg)

    with open(temp_annotated_path, 'rb') as photo_file:
        await context.bot.send_photo(**reply_kwargs, photo=photo_file)
    await context.bot.send_message(**reply_kwargs, text=msg, parse_mode="HTML",
                                    disable_web_page_preview=True)
    os.unlink(temp_annotated_path)


async def identify_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Identify characters in a directly sent photo."""
    try:
        await identify_and_send(context, update.effective_chat.id, update.message.effective_attachment,
                               react_message=update.message)
    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Error identifying photo. Please try again.")


async def whodis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /whodis command - identify a photo being replied to."""
    if not update.message or not update.message.reply_to_message:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Reply to a photo with /whodis to identify it."
        )
        return

    reply_to = update.message.reply_to_message
    if not reply_to.photo:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="The message you replied to doesn't contain a photo."
        )
        return

    try:
        await identify_and_send(context, update.effective_chat.id, reply_to.photo, reply_to.message_id,
                               react_message=update.message)
    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Error identifying photo. Please try again."
        )


async def furscan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /furscan command.

    With args (e.g. /furscan foxona or /furscan this is @aurabirb):
        Add the photo to the database with the given character name.
        Photo can be in the replied-to message or as caption on a photo.
    Without args:
        Identify the photo being replied to.
    """
    if not update.message:
        return

    character_name = " ".join(context.args) if context.args else None

    if not character_name:
        # No name given - fall back to identify behavior
        await whodis(update, context)
        return

    # Strip leading @ and "this is" prefix for convenience
    character_name = re.sub(r"^this\s+is\s+", "", character_name, flags=re.IGNORECASE).strip()
    character_name = character_name.lstrip("@")

    if not character_name:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Please provide a character name. Example: /furscan foxona"
        )
        return

    # Find the photo: either in the message itself (caption on photo) or in the replied-to message
    photo_message = None
    if update.message.photo:
        photo_message = update.message
    elif update.message.reply_to_message and update.message.reply_to_message.photo:
        photo_message = update.message.reply_to_message
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Reply to a photo with /furscan CharacterName, or send a photo with /furscan CharacterName as caption."
        )
        return

    await add_photo(update, context, character_name, photo_message=photo_message)


async def reply_to_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Identify characters when replying to a photo with a bot mention."""
    user = update.effective_user
    username = f"@{user.username}" if user and user.username else (str(user.id) if user else "unknown")
    chat_id = update.effective_chat.id if update.effective_chat else "unknown"
    msg_text = update.message.text if update.message else None

    print(f"reply_to_photo: user={username} chat={chat_id} text={msg_text!r}", file=sys.stderr)

    if not update.message or not update.message.reply_to_message:
        print(f"  -> no message or reply_to_message", file=sys.stderr)
        return
    reply_to = update.message.reply_to_message
    if not reply_to.photo:
        print(f"  -> reply_to has no photo (has: text={bool(reply_to.text)}, caption={bool(reply_to.caption)}, document={bool(reply_to.document)})", file=sys.stderr)
        return

    text = (update.message.text or "").lower()
    bot_username = (await context.bot.get_me()).username.lower()
    print(f"  -> looking for @{bot_username} in {text!r}", file=sys.stderr)
    if f"@{bot_username}" not in text:
        print(f"  -> mention not found", file=sys.stderr)
        return

    print(f"  -> proceeding to identify photo", file=sys.stderr)

    try:
        await identify_and_send(context, update.effective_chat.id, reply_to.photo, reply_to.message_id,
                               react_message=update.message)
    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id, reply_to_message_id=reply_to.message_id, text="Error identifying photo. Please try again."
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Send me a photo to identify fursuit characters.\n\n"
             "To add: reply to a photo with /furscan CharacterName\n"
             "To add: send photo with caption /furscan CharacterName\n"
             "To add: send photo with caption character:Name\n"
             "To identify in groups: reply to a photo with /whodis or /furscan\n"
             "To search by description: /search blue fox with white markings\n"
             "To show a character: /show CharacterName"
    )


async def show(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /show and /search commands - show example images of a character."""
    if not context.args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Usage: /show CharacterName"
        )
        return

    query = " ".join(context.args)
    try:
        identifiers = get_identifiers()
        # Collect character names from all databases across all identifiers
        all_names: set[str] = set()
        all_dbs: list[Database] = []
        for ident in identifiers:
            all_dbs.append(ident.db)
            all_names.update(ident.db.get_all_character_names())

        # Try exact match first (case-insensitive)
        name_lower = {n.lower(): n for n in all_names}
        matched_names = []
        if query.lower() in name_lower:
            matched_names = [name_lower[query.lower()]]
        else:
            # Fuzzy match using difflib
            from difflib import get_close_matches
            # Match against lowercased names, map back to originals
            close = get_close_matches(query.lower(), name_lower.keys(), n=Config.DEFAULT_TOP_K, cutoff=0.5)
            matched_names = [name_lower[c] for c in close]

        if not matched_names:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"No characters found matching '{query}'."
            )
            return

        # Gather detections from all matched names across all databases
        detections = []
        for name in matched_names:
            for db in all_dbs:
                detections.extend(db.get_detections_by_character(name))

        if len(matched_names) > 1:
            names_list = ", ".join(matched_names)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Matched characters: {names_list}"
            )

        # Collect unique image URLs (one per post), skip NSFW barq images
        seen_posts = set()
        media = []
        for det in detections:
            if det.post_id in seen_posts:
                continue
            if det.source == "barq" and _is_barq_nsfw(det.post_id):
                continue
            url = get_source_image_url(det.source, det.post_id)
            if not url:
                continue
            seen_posts.add(det.post_id)
            page_url = _get_page_url(det.source, det.post_id)
            safe_name = html.escape(det.character_name or "Unknown")
            safe_source = html.escape(det.source or "unknown")
            caption = f"{safe_name} ({safe_source})"
            if page_url:
                caption = f"<a href=\"{page_url}\">{safe_name}</a> ({safe_source})"
            media.append(InputMediaPhoto(media=url, caption=caption, parse_mode="HTML"))

        if not media:
            names_list = ", ".join(matched_names)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"No linkable images found for {names_list} (only manual/tgbot sources)."
            )
            return

        # Pick up to 5 random photos
        if len(media) > 5:
            media = random.sample(media, 5)

        # Try media group first, fall back to individual photos
        try:
            if len(media) >= 2:
                await context.bot.send_media_group(
                    chat_id=update.effective_chat.id, media=media)
            else:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=media[0].media, caption=media[0].caption,
                    parse_mode=media[0].parse_mode)
        except Exception:
            for item in media:
                try:
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=item.media, caption=item.caption,
                        parse_mode=item.parse_mode)
                except Exception:
                    continue

    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error looking up character: {e}"
        )


async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /search command - text-based fursuit search using CLIP/SigLIP embeddings."""
    if not context.args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Usage: /search blue fox with white markings"
        )
        return

    query = " ".join(context.args)
    try:
        identifiers = get_identifiers()
        # Only search identifiers whose embedder supports text
        text_identifiers = [
            ident for ident in identifiers
            if hasattr(ident.pipeline.embedder, "embed_text")
        ]
        if not text_identifiers:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Text search is not available. No datasets use a text-capable embedder (CLIP/SigLIP)."
            )
            return

        # Merge text search results from all text-capable identifiers
        def _run_search():
            results = []
            for ident in text_identifiers:
                results.extend(ident.search_text(query, top_k=10))
            return results
        all_results = await asyncio.to_thread(_run_search)

        if not all_results:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"No matches found for '{query}'."
            )
            return

        # Deduplicate by character, keeping best match
        seen = {}
        for r in all_results:
            name = r.character_name or "unknown"
            if name not in seen or r.confidence > seen[name].confidence:
                seen[name] = r
        top_matches = sorted(seen.values(), key=lambda x: x.confidence, reverse=True)[:5]
        print(f'Found top matches: {top_matches}')

        lines = [f"Search results for '<b>{html.escape(query)}</b>':"]
        for i, m in enumerate(top_matches, 1):
            name = html.escape(m.character_name or "Unknown")
            url = _get_page_url(m.source, m.post_id)
            if url:
                lines.append(f"  {i}. <a href=\"{url}\">{name}</a> ({m.confidence*100:.1f}%)")
            else:
                lines.append(f"  {i}. {name} ({m.confidence*100:.1f}%)")

        # Send one example image per top match character
        media = []
        for m in top_matches:
            # Find one linkable image for this character
            for ident in identifiers:
                found = False
                for det in ident.db.get_detections_by_character(m.character_name):
                    if det.source == "barq" and _is_barq_nsfw(det.post_id):
                        continue
                    img_url = get_source_image_url(det.source, det.post_id)
                    if not img_url:
                        continue
                    page_url = _get_page_url(det.source, det.post_id)
                    safe_name = html.escape(det.character_name or "Unknown")
                    safe_source = html.escape(det.source or "unknown")
                    caption = f"{safe_name} ({safe_source})"
                    if page_url:
                        caption = f"<a href=\"{page_url}\">{safe_name}</a> ({safe_source})"
                    media.append(InputMediaPhoto(media=img_url, caption=caption, parse_mode="HTML"))
                    found = True
                    break
                if found:
                    break

        msg = "\n".join(lines)
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=msg, parse_mode="HTML",
            disable_web_page_preview=True)

        # Try media group first, fall back to individual photos
        if media:
            try:
                if len(media) >= 2:
                    await context.bot.send_media_group(
                        chat_id=update.effective_chat.id, media=media)
                else:
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=media[0].media, caption=media[0].caption,
                        parse_mode=media[0].parse_mode)
            except Exception:
                for item in media:
                    try:
                        await context.bot.send_photo(
                            chat_id=update.effective_chat.id,
                            photo=item.media, caption=item.caption,
                            parse_mode=item.parse_mode)
                    except Exception:
                        continue
    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error performing search: {e}"
        )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats command."""
    try:
        identifiers = get_identifiers()
        all_characters = set()
        all_posts = set()
        stats_list = [ident.get_stats() for ident in identifiers]
        combined_stats = FursuitIdentifier.get_combined_stats(stats_list)
        for ident in identifiers:
            # Collect character names and post IDs from each db
            all_characters.update(ident.db.get_all_character_names())
            all_posts.update(ident.db.get_all_post_ids())
        combined_stats = {
            **combined_stats,
            "unique_characters": len(all_characters),
            "unique_posts": len(all_posts),
        }
        import json
        msg = json.dumps(combined_stats, indent=2, ensure_ascii=False, default=str)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error getting stats: {e}"
        )


async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /restart command - restart the bot process."""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Restarting bot..."
    )

    # Stop the application gracefully
    context.application.stop_running()

    # Replace current process with a new instance
    os.execv(sys.executable, [sys.executable] + sys.argv)


async def debug_all_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug handler to log all incoming messages."""
    msg = update.message
    if not msg:
        return
    user = update.effective_user
    username = f"@{user.username}" if user and user.username else (str(user.id) if user else "unknown")
    print(f"DEBUG incoming: user={username} chat={update.effective_chat.id}", file=sys.stderr)
    print(f"  text={msg.text!r} caption={msg.caption!r}", file=sys.stderr)
    print(f"  photo={bool(msg.photo)} reply_to={bool(msg.reply_to_message)}", file=sys.stderr)
    if msg.reply_to_message:
        print(f"  reply_to.photo={bool(msg.reply_to_message.photo)}", file=sys.stderr)


def build_application(token: str):
    """Create a Telegram application with all handlers."""
    app = ApplicationBuilder().token(token).concurrent_updates(True).build()
    # Debug handler - logs all messages (group=-1 runs before other handlers)
    app.add_handler(MessageHandler(filters.ALL, debug_all_messages), group=-1)
    app.add_handler(MessageHandler((~filters.COMMAND) & filters.PHOTO, photo))
    app.add_handler(MessageHandler((~filters.COMMAND) & filters.TEXT & filters.REPLY, reply_to_photo))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("show", show))
    app.add_handler(CommandHandler("search", search))
    app.add_handler(CommandHandler("whodis", whodis))
    app.add_handler(CommandHandler("furscan", furscan))
    app.add_handler(CommandHandler("aitool", aitool.handle_aitool))
    app.add_handler(CommandHandler("restart", restart))
    app.add_handler(CommandHandler("commit", aitool.handle_commit))
    return app


async def run_bot_and_web():
    """Run multiple Telegram bots and web server concurrently."""
    tokens_str = os.environ.get("TG_BOT_TOKENS", os.environ.get("TG_BOT_TOKEN", ""))
    tokens = [t.strip() for t in tokens_str.split(",") if t.strip()]

    if not tokens:
        print("Error: TG_BOT_TOKEN or TG_BOT_TOKENS not set", file=sys.stderr)
        sys.exit(1)

    if not webserver.STATIC_DIR.exists():
        webserver.STATIC_DIR.mkdir(parents=True)

    applications = [build_application(token) for token in tokens]

    web_app = webserver.create_app()
    web_runner = web.AppRunner(web_app)

    print(f"Starting {len(applications)} bot(s) and web server...")

    await webserver.start_server(web_runner)

    for app in applications:
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        me = await app.bot.get_me()
        print(f"  @{me.username} running")

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        for app in applications:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
        await web_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(run_bot_and_web())
