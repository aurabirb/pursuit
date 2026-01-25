# Pursuit - Fursuit Character Recognition

Fursuit character recognition system using computer vision. Identifies fursuit characters from photos by matching against a database of known characters.

## Current Status

**Working:** SAM3 + DINOv2 pipeline on CUDA

The system uses SAM3 text prompts with `"fursuiter"` for automatic detection of all fursuits in an image.

## Quick Start

```bash
# Setup
uv venv .venv && source .venv/bin/activate
uv pip install -e .

# CLI
python -m sam3_pursuit.api.cli identify photo.jpg
python -m sam3_pursuit.api.cli add -c "CharName" img1.jpg img2.jpg
python -m sam3_pursuit.api.cli stats

# Index NFC25 database
python scripts/index_nfc25.py

# Telegram bot
TG_BOT_TOKEN=xxx python tgbot.py

# Tests
python -m pytest tests/
python tests/test_nfc25.py embedding
python tests/test_nfc25.py search
```

## Architecture

```
Image → SAM3 (segment by "fursuiter") → DINOv2 (embed) → FAISS (search) → Results
```

```
sam3_pursuit/
├── models/
│   ├── segmentor.py      SAM3 segmentation with text prompts
│   └── embedder.py       DINOv2 768D embeddings
├── storage/
│   ├── database.py       SQLite detection metadata
│   └── vector_index.py   FAISS HNSW index
├── pipeline/
│   └── processor.py      Segmentation + embedding pipeline
└── api/
    ├── identifier.py     Main API: SAM3FursuitIdentifier
    └── cli.py            Command-line interface

scripts/
└── index_nfc25.py        Index NFC25 fursuit database

download.py               FurTrack scraper
tgbot.py                  Telegram bot
```

## Key APIs

```python
from sam3_pursuit import SAM3FursuitIdentifier

identifier = SAM3FursuitIdentifier()

# Identify character in image
results = identifier.identify(image, top_k=5)

# Add images for a character
identifier.add_images("CharacterName", ["img1.jpg", "img2.jpg"])

# Segment (default concept: "fursuiter")
segmentor.segment(image)
segmentor.segment(image, concept="person")  # custom concept
```

## Data Sources

- **FurTrack**: `download.py` scrapes furtrack.com API
- **NFC25**: 2,305 fursuit badge photos in `/media/user/SSD2TB/nfc25-fursuits/`

## Storage

| File | Contents |
|------|----------|
| `furtrack_sam3.db` | Detection metadata (SQLite) |
| `faiss_sam3.index` | 768D embeddings (FAISS HNSW) |
| `nfc25.db` / `nfc25.index` | NFC25 database |

## Config

Key settings in `sam3_pursuit/config.py`:

```python
SAM3_MODEL = "sam3"
DINOV2_MODEL = "facebook/dinov2-base"
EMBEDDING_DIM = 768
DEFAULT_CONCEPT = "fursuiter"  # SAM3 text prompt
```

## Environment

- `TG_BOT_TOKEN` - Telegram bot token
- `HF_TOKEN` - HuggingFace token (for SAM3)

Device auto-selection: CUDA > MPS > CPU

## References

- [SAM3 Paper](https://arxiv.org/abs/2511.16719) - Segment Anything with Concepts
- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [DINOv2](https://github.com/facebookresearch/dinov2) - Self-supervised vision
