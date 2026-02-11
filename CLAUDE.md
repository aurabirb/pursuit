# Pursuit - Fursuit Character Recognition

Fursuit character recognition system using SAM3 + SigLIP. Identifies fursuit characters from photos by matching against a database of known characters.

## How It Works

```
Image → SAM3 (detect "fursuiter head") → Background Isolation → SigLIP (embedding) → FAISS (similarity search) → Results
```

1. **SAM3** segments all fursuiters in the image using the text prompt `"fursuiter head"`
2. **Background Isolation** replaces the background with a solid color or blur to reduce noise
3. **SigLIP** generates an embedding for each isolated fursuiter crop
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
from sam3_pursuit.models.segmentor import SAM3FursuitSegmentor
s = SAM3FursuitSegmentor()
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

### Search by text description

```bash
pursuit search "blue fox with white markings"
pursuit search "red wolf" --top-k 10 --json
```

Note: Text search requires a dataset built with a CLIP or SigLIP embedder (SigLIP is the default). DINOv2 datasets do not support text search.

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

### Download images from Barq

```bash
export BARQ_BEARER_TOKEN="your_token"
pursuit download barq --lat 52.378 --lon 4.9
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

### Working with multiple datasets

Use `--dataset` (`-d` or `-ds`) to work with different datasets. Each dataset has its own `.db` and `.index` files.

```bash
# Add to default dataset (pursuit.db)
pursuit add -c "CharName" -s manual img1.jpg

# Add to a different dataset (validation.db)
pursuit -ds validation add -c "CharName" -s manual img1.jpg

# View stats for a specific dataset
pursuit -ds validation stats
```

### Validation workflow

Build a validation set and evaluate model accuracy:

```bash
# 1. Download validation images (auto-excludes main dataset, outputs to datasets/validation/furtrack/)
pursuit -ds validation download furtrack -c "CharName" -m 5
pursuit -ds validation download furtrack --all -m 2

# 2. Ingest into validation dataset (auto data-dir: datasets/validation/furtrack/)
pursuit -ds validation ingest directory -s furtrack

# 3. Evaluate validation set against main dataset
pursuit evaluate
pursuit evaluate --from validation --against pursuit --top-k 5
pursuit evaluate --json
```

The evaluate command outputs:
- Top-1 and top-k accuracy
- Breakdown by source, preprocessing config, and character
- Confidence calibration (accuracy per confidence bucket)

### Download with exclusions

Skip images already in specified datasets:

```bash
# Download only images not in the main dataset
pursuit download furtrack --all -e pursuit

# Download only images not in any existing dataset
pursuit download barq -e pursuit,validation
```

### Combine datasets

Merge multiple datasets into a single target dataset (non-destructive, source datasets unchanged):

```bash
# Combine two datasets into one
pursuit combine pursuit validation --output merged

# Combine three datasets
pursuit combine pursuit validation test --output all_data
```

Duplicates (same `post_id` + `preprocessing_info` + `source`) are automatically skipped.

### Split a dataset

Extract a subset of a dataset by criteria (non-destructive, source dataset unchanged):

```bash
# Extract only furtrack entries
pursuit split pursuit --output furtrack_only --by-source furtrack

# Extract specific characters
pursuit split pursuit --output subset --by-character "CharA,CharB"

# Split into shards (creates barq_split_0, barq_split_1)
pursuit split pursuit --output barq_split --by-source barq --shards 2

# Combine filters
pursuit split pursuit --output ft_chars --by-source furtrack --by-character "CharA"
```

At least one filter (`--by-source` or `--by-character`) is required. Sharding uses `hash(post_id) % shards` for deterministic assignment.

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
from sam3_pursuit import FursuitIdentifier, FursuitIngestor, create_identifiers
from sam3_pursuit.models.preprocessor import IsolationConfig
from sam3_pursuit.config import Config
from PIL import Image

# --- Identification (read-only, supports multiple datasets) ---

# Single dataset (datasets is required)
identifier = FursuitIdentifier(
    datasets=[(Config.DB_PATH, Config.INDEX_PATH)],
)

# Multiple datasets with same embedder (searches all, merges results)
identifier = FursuitIdentifier(
    datasets=[("pursuit.db", "pursuit.index"), ("validation.db", "validation.index")],
)

# Auto-discover datasets and group by embedder (one identifier per embedder)
# This is the recommended way when datasets may use different embedders.
identifiers = create_identifiers()  # discovers *.db/*.index in Config.BASE_DIR

# Identify across all identifiers (segmentation is cached after the first)
image = Image.open("photo.jpg")
all_results = [ident.identify(image, top_k=5) for ident in identifiers]

# Merge results per segment
results = all_results[0] if all_results else []
for other in all_results[1:]:
    for seg, other_seg in zip(results, other):
        seg.matches.extend(other_seg.matches)
        seg.matches.sort(key=lambda x: x.confidence, reverse=True)
        seg.matches = seg.matches[:5]

for segment in results:
    print(f"Segment {segment.segment_index} at {segment.segment_bbox}:")
    for match in segment.matches:
        print(f"  {match.character_name}: {match.confidence:.1%}")

# Search by text (only works on identifiers with CLIP/SigLIP embedder)
text_identifiers = [i for i in identifiers if hasattr(i.pipeline.embedder, "embed_text")]
for ident in text_identifiers:
    results = ident.search_text("blue fox with white markings", top_k=5)
    for match in results:
        print(f"  {match.character_name}: {match.confidence:.1%}")

# Get statistics (aggregated across all datasets in an identifier)
stats = identifier.get_stats()
print(f"Database contains {stats['unique_characters']} characters")

# --- Ingestion (writes to a single dataset) ---

ingestor = FursuitIngestor()

# Or customize background isolation
isolation_config = IsolationConfig(
    mode="solid",                    # "solid", "blur", or "none"
    background_color=(128, 128, 128),  # Gray background
    blur_radius=25                   # For blur mode
)
ingestor = FursuitIngestor(isolation_config=isolation_config)

# Add images for characters
ingestor.add_images(["MyCharacter", "Zygote"], ["img1.jpg", "img2.jpg"])
```

### Using the segmentor directly

```python
from sam3_pursuit.models.segmentor import SAM3FursuitSegmentor
from PIL import Image

segmentor = SAM3FursuitSegmentor()
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
pursuit ingest directory --data-dir ./datasets/pursuit/furtrack/ -s furtrack
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
│   │   ├── identifier.py   # FursuitIdentifier (read-only, multi-dataset search), create_identifiers (multi-embedder factory)
│   │   └── ingestor.py     # FursuitIngestor (ingestion, single dataset)
│   ├── models/
│   │   ├── segmentor.py    # SAM3 segmentation (SAM3FursuitSegmentor, FullImageSegmentor)
│   │   ├── embedder.py     # Embedders: SigLIP (default), DINOv2, CLIP (DINOv2Embedder, SigLIPEmbedder, CLIPEmbedder)
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
| `pursuit.db` | SQLite database with detection metadata (default dataset) |
| `pursuit.index` | FAISS index with embeddings (default dataset) |
| `<name>.db` | Database for custom dataset (e.g., `validation.db`) |
| `<name>.index` | Index for custom dataset (e.g., `validation.index`) |
| `pursuit_crops/` | Saved crop images for debugging (when using `--save-crops`) |
| `pursuit_masks/` | Saved segmentation masks (when using `--save-crops`) |
| `datasets/<dataset>/<source>/` | Default download/ingest directory for non-default datasets |

These files are gitignored. Use `--dataset` (`-ds`) to switch between datasets.

## Configuration

Key settings in `sam3_pursuit/config.py`:

```python
# Dataset name (change this to rename all default files)
DEFAULT_DATASET = "pursuit"            # Used for db/index/crops/masks naming

# File paths (derived from DEFAULT_DATASET)
DEFAULT_DB_NAME = f"{DEFAULT_DATASET}.db"
DEFAULT_INDEX_NAME = f"{DEFAULT_DATASET}.index"
DEFAULT_CROPS_DIR = f"{DEFAULT_DATASET}_crops"
DEFAULT_MASKS_DIR = f"{DEFAULT_DATASET}_masks"

# Models
SAM3_MODEL = "sam3"                    # Model name
DEFAULT_EMBEDDER = "siglip"            # Default embedder (SigLIP)
SIGLIP_MODEL = "google/siglip-base-patch16-224"
EMBEDDING_DIM = 768                    # Embedding output dimension

# Detection
DEFAULT_CONCEPT = "fursuiter head"     # SAM3 text prompt
DETECTION_CONFIDENCE = 0.5             # Minimum confidence threshold
MAX_DETECTIONS = 10                    # Max segments per image

# Background isolation
DEFAULT_BACKGROUND_MODE = "solid"      # "solid", "blur", or "none"
DEFAULT_BACKGROUND_COLOR = (128, 128, 128)  # Gray background
DEFAULT_BLUR_RADIUS = 25               # Blur radius for "blur" mode

# Image processing
TARGET_IMAGE_SIZE = 630                # Resize target
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TG_BOT_TOKEN` | Telegram bot token (single bot) | For bot only |
| `TG_BOT_TOKENS` | Comma-separated tokens (multiple bots) | Alternative to above |
| `HF_TOKEN` | HuggingFace token | For SAM3 download |
| `BARQ_BEARER_TOKEN` | Barq API bearer token | For barq download |

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
ingestor = FursuitIngestor(device="cuda")  # or "cpu", "mps"
```

## References

- [SAM3 Paper](https://arxiv.org/abs/2511.16719) - Segment Anything with Concepts
- [SAM3 on HuggingFace](https://huggingface.co/facebook/sam3)
- [Ultralytics SAM3 Docs](https://docs.ultralytics.com/models/sam-3/)
- [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) - Default embedder
- [DINOv2](https://github.com/facebookresearch/dinov2) - Alternative embedder (no text search)

## License

MIT
