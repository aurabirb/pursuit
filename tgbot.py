"""Telegram bot for fursuit character identification using SAM3 system."""

import asyncio
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

load_dotenv()
# AI tool configuration (from environment variables with baked-in defaults)
AITOOL_WORK_DIR = os.environ.get("AITOOL_WORK_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pursuit")))
AITOOL_BINARY = os.environ.get("AITOOL_BINARY", "claude")
AITOOL_ARGS = os.environ.get("AITOOL_ARGS", "--allowedTools 'Bash(git:*) Edit Write Read Glob Grep' -p")
AITOOL_TIMEOUT = int(os.environ.get("AITOOL_TIMEOUT", "600"))  # 10 minutes default
AITOOL_UPDATE_INTERVAL = float(os.environ.get("AITOOL_UPDATE_INTERVAL", "5.0"))  # seconds between updates
AITOOL_ALLOWED_USERS = os.environ.get("AITOOL_ALLOWED_USERS", "")  # comma-separated list of telegram user IDs or usernames

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


def is_user_authorized(update: Update) -> bool:
    """Check if the user is authorized to use /aitool."""
    if not AITOOL_ALLOWED_USERS:
        return False  # No users configured = no access

    allowed = [u.strip().lower() for u in AITOOL_ALLOWED_USERS.split(",") if u.strip()]
    user = update.effective_user
    if not user:
        return False

    # Check by user ID or username
    user_id_str = str(user.id)
    username = (user.username or "").lower()
    print(f"username: {username} id: {user_id_str}")

    return user_id_str in allowed or username in allowed


async def aitool(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /aitool command - run AI tool with prompt and stream output.

    Usage:
        /aitool <prompt>       - Continue previous conversation with prompt
        /aitool new <prompt>   - Start a new conversation with prompt
    """
    chat_id = update.effective_chat.id

    # Check authorization
    if not is_user_authorized(update):
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ You are not authorized to use this command."
        )
        return

    if not update.message or not context.args:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Usage:\n  /aitool <prompt> - continue conversation\n  /aitool new <prompt> - start new conversation\n\nExample: /aitool fix the bug in main.py"
        )
        return

    # Check for 'new' subcommand
    args = list(context.args)
    use_continue = True
    if args and args[0].lower() == "new":
        use_continue = False
        args = args[1:]

    if not args:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Please provide a prompt after 'new'."
        )
        return

    prompt = " ".join(args)

    # Send initial status
    mode = "continuing" if use_continue else "new conversation"
    status_msg = await context.bot.send_message(
        chat_id=chat_id,
        text=f"🔧 Running: {AITOOL_BINARY} ({mode})\nPrompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n\n⏳ Starting..."
    )

    # Build command: claude --allowedTools '...' -p [-c] "prompt"
    import shlex
    cmd_parts = [AITOOL_BINARY] + shlex.split(AITOOL_ARGS)
    if use_continue:
        cmd_parts.append("-c")
    cmd_parts.append(prompt)

    work_dir = AITOOL_WORK_DIR if AITOOL_WORK_DIR else None

    output_buffer = []
    last_update_time = 0
    process = None

    try:
        # Start the process
        print(f"Starting prompt: {cmd_parts}")
        process = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=work_dir,
        )

        async def read_output():
            """Read output from process and update buffer."""
            nonlocal last_update_time
            while True:
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=1.0
                    )
                    if not line:
                        break
                    decoded = line.decode('utf-8', errors='replace').rstrip()
                    if decoded:
                        output_buffer.append(decoded)
                except asyncio.TimeoutError:
                    continue

        async def send_updates():
            """Periodically send output updates to user."""
            nonlocal last_update_time
            last_sent_len = 0
            while process.returncode is None:
                await asyncio.sleep(AITOOL_UPDATE_INTERVAL)
                if len(output_buffer) > last_sent_len:
                    # Get new lines since last update
                    new_lines = output_buffer[last_sent_len:]
                    last_sent_len = len(output_buffer)

                    # Truncate if too long for Telegram (4096 char limit)
                    text = "\n".join(new_lines)
                    if len(text) > 3900:
                        text = text[-3900:]
                        text = "...(truncated)\n" + text

                    try:
                        await context.bot.send_message(chat_id=chat_id, text=f"📤 Output:\n```\n{text}\n```", parse_mode="Markdown")
                    except Exception as e:
                        # Fallback without markdown if it fails
                        try:
                            await context.bot.send_message(chat_id=chat_id, text=f"📤 Output:\n{text}")
                        except Exception:
                            pass

        # Run both tasks with timeout
        try:
            read_task = asyncio.create_task(read_output())
            update_task = asyncio.create_task(send_updates())

            await asyncio.wait_for(read_task, timeout=AITOOL_TIMEOUT)
            update_task.cancel()

        except asyncio.TimeoutError:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Timeout after {AITOOL_TIMEOUT}s - terminating process..."
            )
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()

        # Wait for process to complete
        await process.wait()
        return_code = process.returncode

        # Send final output
        if output_buffer:
            final_output = "\n".join(output_buffer[-50:])  # Last 50 lines
            if len(output_buffer) > 50:
                final_output = f"...(showing last 50 of {len(output_buffer)} lines)\n" + final_output
            if len(final_output) > 3900:
                final_output = final_output[-3900:]

            await context.bot.send_message(
                chat_id=chat_id,
                text=f"✅ Completed (exit code: {return_code})\n\n📄 Final output:\n```\n{final_output}\n```",
                parse_mode="Markdown"
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"✅ Completed (exit code: {return_code})\n\n(no output)"
            )

    except FileNotFoundError:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Error: Binary '{AITOOL_BINARY}' not found. Make sure it's installed and in PATH."
        )
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Error: {e}"
        )
    finally:
        if process and process.returncode is None:
            try:
                process.kill()
            except Exception:
                pass

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


async def commit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /commit command - git add, commit and push changes."""
    import subprocess

    # Get commit message from command arguments, or use default
    commit_msg = " ".join(context.args) if context.args else "Update from bot"

    try:
        # Get the working directory (where tgbot.py is located)
        working_dir = os.path.dirname(os.path.abspath(__file__)) or "."

        # Git add all changes
        result = subprocess.run(
            ["git", "add", "-A"],
            cwd=working_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"git add failed: {result.stderr}"
            )
            return

        # Git commit
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=working_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Nothing to commit."
                )
                return
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"git commit failed: {result.stderr}"
            )
            return

        # Git push
        result = subprocess.run(
            ["git", "push"],
            cwd=working_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"git push failed: {result.stderr}"
            )
            return

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Changes committed and pushed.\nMessage: {commit_msg}"
        )

    except Exception as e:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error: {e}"
        )


if __name__ == "__main__":
    token = os.environ.get("TG_BOT_TOKEN", "")

    if not token:
        print("Error: TG_BOT_TOKEN not set in environment", file=sys.stderr)
        sys.exit(1)

    application = ApplicationBuilder().token(token).build()

    # Add handlers
    application.add_handler(MessageHandler((~filters.COMMAND) & filters.PHOTO, photo))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("aitool", aitool))
    application.add_handler(CommandHandler("restart", restart))
    application.add_handler(CommandHandler("commit", commit))

    print("Bot running...")
    print(f"AI tool config: binary={AITOOL_BINARY}, timeout={AITOOL_TIMEOUT}s")
    print(f"  args: {AITOOL_ARGS}")
    print(f"  work_dir: {AITOOL_WORK_DIR}")
    print(f"  allowed_users: {AITOOL_ALLOWED_USERS or '(none - /aitool disabled)'}")
    application.run_polling()
