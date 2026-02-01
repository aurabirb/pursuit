# Pursuit - Fursuit Character Recognition

Fursuit character recognition system using SAM3 + DINOv2. Identifies fursuit characters from photos by matching against a database of known characters.

## How It Works

```
Image → SAM3 (detect "fursuiter head") → Background Isolation → DINOv2 (768D embedding) → FAISS (similarity search) → Results
```

1. **SAM3** segments all fursuiters in the image using the text prompt `"fursuiter head"`
2. **Background Isolation** replaces the background with a solid color or blur to reduce noise
3. **DINOv2** generates a 768-dimensional embedding for each isolated fursuiter crop
4. **FAISS** finds the most similar embeddings in the database
5. Results are returned with character names and confidence scores

## Requirements

- Python 3.10+
- CUDA GPU (recommended) or CPU
- ~4GB disk space for SAM3 model
- HuggingFace account with SAM3 access

## Installation

### 1. Clone and setup environment

```bash
git clone https://github.com/aurabirb/pursuit.git
cd pursuit

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install package
uv pip install -e .
```

### 2. Get SAM3 access (required)

SAM3 requires HuggingFace authentication:

1. Create account at https://huggingface.co
2. Request access at https://huggingface.co/facebook/sam3
3. Wait for approval (usually hours)
4. Create a token at https://huggingface.co/settings/tokens
5. Login locally:

```bash
pip install huggingface_hub
hf auth login
```

### 3. Download SAM3 model (~3.5GB)

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam3', 'sam3.pt', local_dir='.')"
```

Or manually download `sam3.pt` from https://huggingface.co/facebook/sam3 and place in project root.

### 4. Verify installation

```bash
python -c "
from sam3_pursuit.models.segmentor import FursuitSegmentor
s = FursuitSegmentor()
print('SAM3 ready!')
"
```

Expected output:
```
Loading SAM3 model: sam3.pt on cuda
SAM3 loaded successfully - text prompts enabled!
SAM3 ready!
```

## Quick Start

### Identify a character

```bash
pursuit identify photo.jpg
pursuit identify photo.jpg --segment --concept "mascot"
```

### Add images for a new character

```bash
pursuit add -c "CharacterName" -s manual img1.jpg img2.jpg img3.jpg
pursuit add -c "CharacterName" -s furtrack img1.jpg img2.jpg --save-crops
```

### Test segmentation on an image

```bash
pursuit segment photo.jpg
pursuit segment photo.jpg --output-dir ./crops/ --json
```

### Download images from FurTrack

```bash
pursuit download furtrack --character "CharacterName"
pursuit download furtrack --all --max-images 5
```

### View database entries

```bash
pursuit show --by-character "CharacterName"
pursuit show --by-id 42 --json
pursuit show --by-post "uuid-here"
```

### Bulk ingest images

```bash
# From directory structure: data_dir/character_name/*.jpg
pursuit ingest directory --data-dir ./characters/ --source manual

# From directory, sourced from furtrack
pursuit ingest directory --data-dir ./furtrack_chars/ --source furtrack

# From NFC25 dataset
pursuit ingest nfc25 --data-dir ./nfc25-fursuits/
```

### View database statistics

```bash
pursuit stats
pursuit stats --json
```

### Run Telegram bot

```bash
# Single bot
export TG_BOT_TOKEN="your_bot_token"
python tgbot.py

# Multiple bots (comma-separated tokens, shared database)
export TG_BOT_TOKENS="token1,token2,token3"
python tgbot.py
```

## Python API

```python
from sam3_pursuit import SAM3FursuitIdentifier
from sam3_pursuit.models.preprocessor import IsolationConfig
from PIL import Image

# Initialize with default settings (loads SAM3 + DINOv2)
identifier = SAM3FursuitIdentifier()

# Or customize background isolation
isolation_config = IsolationConfig(
    mode="solid",                    # "solid", "blur", or "none"
    background_color=(128, 128, 128),  # Gray background
    blur_radius=25                   # For blur mode
)
identifier = SAM3FursuitIdentifier(isolation_config=isolation_config)

# Identify character in image
image = Image.open("photo.jpg")
results = identifier.identify(image, top_k=5)

for segment in results:
    print(f"Segment {segment.segment_index} at {segment.segment_bbox}:")
    for match in segment.matches:
        print(f"  {match.character_name}: {match.confidence:.1%}")

# Add images for characters
identifier.add_images(["MyCharacter", "Zygote"], ["img1.jpg", "img2.jpg"])

# Get statistics
stats = identifier.get_stats()
print(f"Database contains {stats['unique_characters']} characters")
```

### Using the segmentor directly

```python
from sam3_pursuit.models.segmentor import FursuitSegmentor
from PIL import Image

segmentor = FursuitSegmentor()
image = Image.open("photo.jpg")

# Segment with default concept ("fursuiter")
results = segmentor.segment(image)

for r in results:
    print(f"Found: bbox={r.bbox}, confidence={r.confidence:.2f}")
    r.crop.save(f"crop_{r.bbox[0]}.jpg")
    # Also available: r.mask, r.crop_mask for background isolation
```

### Using background isolation

```python
from sam3_pursuit.models.preprocessor import BackgroundIsolator, IsolationConfig
from PIL import Image
import numpy as np

# Configure isolation
config = IsolationConfig(mode="solid", background_color=(128, 128, 128))
isolator = BackgroundIsolator(config)

# Isolate foreground from background using a mask
crop = Image.open("crop.jpg")
mask = np.ones((crop.height, crop.width), dtype=np.uint8)  # Binary mask
isolated = isolator.isolate(crop, mask)
```

### Using mask storage

```python
from sam3_pursuit.storage.mask_storage import MaskStorage
import numpy as np

# Initialize mask storage (uses default pursuit_masks/ directory)
mask_storage = MaskStorage()

# Save a segmentation mask
mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
mask_path = mask_storage.save_mask(mask, "character_001", search=False)
print(f"Mask saved to: {mask_path}")

# Load a mask back
loaded_mask = mask_storage.load_mask(mask_path)

# Check if a mask exists
exists = mask_storage.mask_exists("character_001", search=False)
```

## Building a Database

### Option 1: Add images manually

```bash
# Add images for individual characters
pursuit add -c "CharName1" -s manual char1_*.jpg
pursuit add -c "CharName2" -s manual char2_*.jpg

# With segmentation (for multi-character photos)
pursuit add -c "CharName1" -s manual photo.jpg
```

### Option 2: Bulk ingest from directory

Organize images as `characters/CharacterName/*.jpg`:

```bash
pursuit ingest directory --data-dir ./characters/ --source manual
```

### Option 3: Download from FurTrack

```bash
pursuit download furtrack --all --max-images 5
pursuit ingest directory --data-dir ./furtrack_images/ -s furtrack
```

### Option 4: Index NFC25 database

If you have the NFC25 fursuit badge dataset:

```bash
pursuit ingest nfc25 --data-dir /path/to/nfc25-fursuits
```

Expected directory structure:
```
nfc25-fursuits/
├── nfc25-fursuit-list.json
└── fursuit_images/
    ├── uuid1.png
    ├── uuid2.png
    └── ...
```

## Project Structure

```
pursuit/
├── sam3_pursuit/           # Main package
│   ├── api/
│   │   ├── cli.py          # Command-line interface
│   │   └── identifier.py   # Main API: SAM3FursuitIdentifier
│   ├── models/
│   │   ├── segmentor.py    # SAM3 segmentation (FursuitSegmentor, FullImageSegmentor)
│   │   ├── embedder.py     # DINOv2 embeddings (FursuitEmbedder)
│   │   └── preprocessor.py # Background isolation (BackgroundIsolator, IsolationConfig)
│   ├── pipeline/
│   │   └── processor.py    # Segmentation + isolation + embedding pipeline
│   ├── storage/
│   │   ├── database.py     # SQLite metadata storage
│   │   ├── vector_index.py # FAISS vector index
│   │   └── mask_storage.py # Segmentation mask storage
│   └── config.py           # Configuration
├── tools/                  # Data collection & debugging
│   ├── download_furtrack.py    # FurTrack image downloader
│   ├── index_nfc25.py          # NFC25 database indexer
│   └── test_segmentation.py    # Segmentation testing & visualization
├── tests/                  # Unit tests
│   └── test_identifier.py
├── tgbot.py               # Telegram bot
├── sam3.pt                # SAM3 model weights (download separately)
├── pyproject.toml         # Package definition
└── CLAUDE.md              # This file
```

## Storage Files

| File | Description |
|------|-------------|
| `sam3.pt` | SAM3 model weights (~3.5GB) |
| `pursuit.db` | SQLite database with detection metadata (default) |
| `pursuit.index` | FAISS index with embeddings (default) |
| `pursuit_crops/` | Saved crop images for debugging (when using `--save-crops`) |
| `pursuit_masks/` | Saved segmentation masks (when using `--save-crops`) |

These files are gitignored. Use `--db` and `--index` flags to use custom paths.

## Configuration

Key settings in `sam3_pursuit/config.py`:

```python
# File paths
DEFAULT_DB_NAME = "pursuit.db"         # Default database file
DEFAULT_INDEX_NAME = "pursuit.index"   # Default index file
DEFAULT_MASKS_DIR = "pursuit_masks"    # Segmentation masks directory

# Models
SAM3_MODEL = "sam3"                    # Model name
DINOV2_MODEL = "facebook/dinov2-base"  # Embedding model
EMBEDDING_DIM = 768                    # DINOv2 output dimension

# Detection
DEFAULT_CONCEPT = "fursuiter head"     # SAM3 text prompt
DETECTION_CONFIDENCE = 0.5             # Minimum confidence threshold
MAX_DETECTIONS = 10                    # Max segments per image

# Background isolation
DEFAULT_BACKGROUND_MODE = "solid"      # "solid", "blur", or "none"
DEFAULT_BACKGROUND_COLOR = (128, 128, 128)  # Gray background
DEFAULT_BLUR_RADIUS = 25               # Blur radius for "blur" mode

# Image processing
PATCH_SIZE = 14                        # DINOv2 patch size
TARGET_IMAGE_SIZE = 630                # Resize target (multiple of PATCH_SIZE)
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TG_BOT_TOKEN` | Telegram bot token (single bot) | For bot only |
| `TG_BOT_TOKENS` | Comma-separated tokens (multiple bots) | Alternative to above |
| `HF_TOKEN` | HuggingFace token | For SAM3 download |

## Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Test SAM3 segmentation on an image
pursuit segment photo.jpg --output-dir ./debug/

# Test with different concept
pursuit segment photo.jpg --concept "mascot" --json

# View database entries for debugging
pursuit show --by-character "CharName" --json
```

## Troubleshooting

### "FileNotFoundError: sam3.pt"

Download the SAM3 model:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam3', 'sam3.pt', local_dir='.')"
```

### "Access denied" when downloading SAM3

1. Ensure you've requested access at https://huggingface.co/facebook/sam3
2. Wait for approval email
3. Run `huggingface-cli login` and enter your token

### "CUDA out of memory"

SAM3 requires ~4GB VRAM. Options:
- Use a smaller batch size
- Process images sequentially
- Use CPU (slower): set `device="cpu"` in config

### Segmentation finds no fursuits

- Try different text prompts: `"mascot"`, `"costume"`, `"character"`
- Lower the confidence threshold in config
- Check if the image is clear and well-lit

## Device Support

Automatic device selection: CUDA → MPS → CPU

Force specific device:
```python
identifier = SAM3FursuitIdentifier(device="cuda")  # or "cpu", "mps"
```

## References

- [SAM3 Paper](https://arxiv.org/abs/2511.16719) - Segment Anything with Concepts
- [SAM3 on HuggingFace](https://huggingface.co/facebook/sam3)
- [Ultralytics SAM3 Docs](https://docs.ultralytics.com/models/sam-3/)
- [DINOv2](https://github.com/facebookresearch/dinov2) - Self-supervised vision

## License

MIT
