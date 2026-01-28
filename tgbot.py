"""Telegram bot for fursuit character identification using SAM3 system."""

import os
import re
import sys
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from PIL import Image
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from sam3_pursuit import SAM3FursuitIdentifier, Config

# Pattern to match "character:Name" in caption
CHARACTER_PATTERN = re.compile(r"character:(\S+)", re.IGNORECASE)

# Global identifier instance (lazy loaded)
_identifier = None


def get_identifier() -> SAM3FursuitIdentifier:
    """Get or create the identifier instance."""
    global _identifier
    if _identifier is None:
        _identifier = SAM3FursuitIdentifier(segmentor_model_name=Config.SAM3_MODEL, segmentor_concept=Config.DEFAULT_CONCEPT)
    return _identifier


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

    # Check if this is an add request
    match = CHARACTER_PATTERN.search(caption)
    if match:
        await add_photo(update, context, match.group(1))
    else:
        await identify_photo(update, context)

async def download_tg_file(new_file):
    from pathlib import Path
    os.makedirs("tg_download", exist_ok=True)
    temp_path = Path("tg_download") / f"{new_file.file_id}.jpg"
    print(f"Downloading into {temp_path}")
    with open(temp_path, 'wb') as f:
        bs = await new_file.download_as_bytearray()
        f.write(bs)
        f.flush()
    return temp_path

async def add_photo(update: Update, context: ContextTypes.DEFAULT_TYPE, character_name: str):
    """Add a photo to the database for a character."""
    attachment = update.message.effective_attachment
    new_file = await attachment[-1].get_file()
    try:
        temp_path = await download_tg_file(new_file)
        identifier = get_identifier()
        added = identifier.add_images(
            character_names=[character_name],
            image_paths=[temp_path],
            source_url=f"tg://bot_upload/chat/{update.effective_chat.id}/msg/{update.message.message_id}/file/{new_file.file_id}",
            add_full_image=True,
        )

        # Clean up temp file
        # os.unlink(temp_path)

        if added > 0:
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
        print(f"Error adding image: {e}", file=sys.stderr)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error adding image: {e}"
        )


async def identify_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Identify the character in a photo."""
    attachment = update.message.effective_attachment
    new_file = await attachment[-1].get_file()

    try:
        temp_path = await download_tg_file(new_file)

        # Load image and identify
        image = Image.open(temp_path)
        identifier = get_identifier()
        results = identifier.identify(image, top_k=5)

        if not results:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="No matching characters found."
            )
            return

        # Format response, filtering by minimum confidence
        min_confidence = Config.DEFAULT_MIN_CONFIDENCE
        lines = []
        for i, result in enumerate(results, 1):
            filtered_matches = [m for m in result.matches if m.confidence >= min_confidence]
            if not filtered_matches:
                continue
            lines.append(f"Characters at segment {i}:")
            for n, m in enumerate(filtered_matches):
                name = m.character_name or "Unknown"
                confidence = m.confidence * 100
                lines.append(f"{n+1}. {name} ({confidence:.1f}%)")

        if not lines:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"No matches found above {min_confidence:.0%} confidence."
            )
            return

        msg = "\n".join(lines)
        print(msg)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error processing image: {e}"
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="I'm a bot that identifies fursuiters in pictures.\n\n"
             "Send me a photo and I'll try to identify the character!\n\n"
             "To add a new character to the database, send a photo with "
             "the caption: character:Name"
    )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats command."""
    try:
        identifier = get_identifier()
        stats = identifier.get_stats()
        import yaml
        msg = yaml.safe_dump(stats)

        # msg = (
        #     f"Database Statistics:\n"
        #     f"- Total detections: {stats['total_detections']}\n"
        #     f"- Unique characters: {stats['unique_characters']}\n"
        #     f"- Unique posts: {stats['unique_posts']}\n"
        #     f"- Index size: {stats['index_size']}"
        # )
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

    except Exception as e:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error getting stats: {e}"
        )


if __name__ == "__main__":
    load_dotenv()
    token = os.environ.get("TG_BOT_TOKEN", "")

    if not token:
        print("Error: TG_BOT_TOKEN not set in environment", file=sys.stderr)
        sys.exit(1)

    application = ApplicationBuilder().token(token).build()

    # Add handlers
    application.add_handler(MessageHandler((~filters.COMMAND) & filters.PHOTO, photo))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats))

    print("Bot running...")
    application.run_polling()
