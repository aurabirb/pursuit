"""Telegram bot for fursuit character identification using SAM3 system."""

import asyncio
import os
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from aiohttp import web
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

import aitool
import webserver

load_dotenv()

from sam3_pursuit import FursuitIngestor, Config
from sam3_pursuit.api.annotator import annotate_image
from sam3_pursuit.storage.database import SOURCE_TGBOT, get_git_version, get_source_url

# Pattern to match "character:Name" in caption
CHARACTER_PATTERN = re.compile(r"character:(\S+)", re.IGNORECASE)

# Global ingestor instance (lazy loaded)
_ingestor = None


def get_ingestor() -> FursuitIngestor:
    """Get or create the ingestor instance."""
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

    # Check if this is an add request
    match = CHARACTER_PATTERN.search(caption)
    if match:
        await add_photo(update, context, match.group(1))
    else:
        await identify_photo(update, context)

async def download_tg_file(new_file):
    os.makedirs("tg_download", exist_ok=True)
    temp_path = Path("tg_download") / f"{new_file.file_id}.jpg"
    print(f"Downloading into {temp_path}")
    with open(temp_path, 'wb') as f:
        bs = await new_file.download_as_bytearray()
        f.write(bs)
        f.flush()
    return temp_path

def make_tgbot_post_id(chat_id: int, msg_id: int, file_id: str) -> str:
    """Create a unique post_id from telegram message identifiers."""
    return f"{chat_id}_{msg_id}_{file_id}"


async def add_photo(update: Update, context: ContextTypes.DEFAULT_TYPE, character_name: str):
    """Add a photo to the database for a character."""
    attachment = update.message.effective_attachment
    new_file = await attachment[-1].get_file()
    user = update.effective_user
    uploaded_by = f"@{user.username}" if user and user.username else (str(user.id) if user else None)
    post_id = make_tgbot_post_id(update.effective_chat.id, update.message.message_id, new_file.file_id)
    try:
        temp_path = await download_tg_file(new_file)
        # Rename temp file to use post_id so identifier extracts it correctly
        post_id_path = temp_path.parent / f"{post_id}.jpg"
        temp_path.rename(post_id_path)
        identifier = get_ingestor()
        added = identifier.add_images(
            character_names=[character_name],
            image_paths=[str(post_id_path)],
            source=SOURCE_TGBOT,
            uploaded_by=uploaded_by,
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


async def identify_and_send(context: ContextTypes.DEFAULT_TYPE, chat_id: int,
                           photo_attachment, reply_to_message_id: int = None):
    """Download photo, identify characters, and send annotated result."""
    new_file = await photo_attachment[-1].get_file()
    temp_path = await download_tg_file(new_file)
    image = Image.open(temp_path)
    results = get_ingestor().identify(image, top_k=5)

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
            url = get_source_url(m.source, m.post_id)
            name = m.character_name or 'Unknown'
            if url:
                lines.append(f"  {n+1}. <a href=\"{url}\">{name}</a> ({m.confidence*100:.1f}%)")
            else:
                lines.append(f"  {n+1}. {name} ({m.confidence*100:.1f}%)")

    if not lines:
        await context.bot.send_message(**reply_kwargs, text=f"No matches above {min_confidence:.0%} confidence.")
        return

    watermark_text = f"Pursuit {get_git_version()}"
    annotated = annotate_image(image, results, min_confidence, watermark_text)
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
        await identify_and_send(context, update.effective_chat.id, update.message.effective_attachment)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Error: {e}")


async def whodis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /whodis and /furspy commands - identify a photo being replied to."""
    if not update.message or not update.message.reply_to_message:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Reply to a photo with /whodis or /furspy to identify it."
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
        await identify_and_send(context, update.effective_chat.id, reply_to.photo, reply_to.message_id)
    except Exception as e:
        print(f"Error in whodis: {e}", file=sys.stderr)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error: {e}"
        )


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
        await identify_and_send(context, update.effective_chat.id, reply_to.photo, reply_to.message_id)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        await context.bot.send_message(
            chat_id=update.effective_chat.id, reply_to_message_id=reply_to.message_id, text=f"Error: {e}"
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Send me a photo to identify fursuit characters.\n\n"
             "To add: send photo with caption character:Name\n"
             "To identify in groups: reply to a photo with /whodis or /furspy"
    )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats command."""
    try:
        identifier = get_ingestor()
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
    app = ApplicationBuilder().token(token).build()
    # Debug handler - logs all messages (group=-1 runs before other handlers)
    app.add_handler(MessageHandler(filters.ALL, debug_all_messages), group=-1)
    app.add_handler(MessageHandler((~filters.COMMAND) & filters.PHOTO, photo))
    app.add_handler(MessageHandler((~filters.COMMAND) & filters.TEXT & filters.REPLY, reply_to_photo))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("whodis", whodis))
    app.add_handler(CommandHandler("furspy", whodis))
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
