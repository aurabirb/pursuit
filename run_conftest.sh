#!/usr/bin/env bash
# Pipeline Variant Comparison for Character Confusion Testing
#
# Tests 7 pipeline variants on 9 characters that are being confused:
#   lindley, squiggles_(folf), wulf_(sharkitty), blueberry_the_wolf,
#   nori_(gatordog), smores_(dutch_angel_dragon), ozzle, selichat, chidsh
#
# Prerequisites:
#   - pursuit installed: uv pip install -e .
#   - SAM3 model (sam3.pt) in project root
#   - furtrack_images/ directory with images for the 9 characters
#
# Usage:
#   chmod +x run_conftest.sh
#   ./run_conftest.sh

set -euo pipefail

CHARACTERS="lindley,squiggles_(folf),wulf_(sharkitty),blueberry_the_wolf,nori_(gatordog),smores_(dutch_angel_dragon),ozzle,selichat,chidsh"

# ── Step 1: Create symlinked directory with only the 9 target characters ──
echo "=== Creating conftest_images/ with target characters ==="
mkdir -p conftest_images
for char in lindley "squiggles_(folf)" "wulf_(sharkitty)" blueberry_the_wolf "nori_(gatordog)" "smores_(dutch_angel_dragon)" ozzle selichat chidsh; do
    if [ ! -d "furtrack_images/$char" ]; then
        echo "ERROR: furtrack_images/$char not found!"
        exit 1
    fi
    ln -sfn "$(pwd)/furtrack_images/$char" "conftest_images/$char"
done
echo "Created conftest_images/ with $(ls conftest_images | wc -l) characters"

# ── Step 2: Ingest baseline (DINOv2-base, solid gray) ──
# This is the slow one — SAM3 runs segmentation and caches masks for all subsequent variants.
echo ""
echo "=== [1/7] Baseline: DINOv2-base + solid gray ==="
pursuit -ds conftest_baseline ingest directory --data-dir ./conftest_images/ -s furtrack --save-crops

# ── Step 3: Ingest CLIP variant ──
echo ""
echo "=== [2/7] CLIP ViT-B/32 ==="
pursuit -ds conftest_clip --embedder clip ingest directory --data-dir ./conftest_images/ -s furtrack --save-crops

# ── Step 4: Ingest grayscale (B&W) variant ──
echo ""
echo "=== [3/7] DINOv2-base + grayscale ==="
pursuit -ds conftest_bw --grayscale ingest directory --data-dir ./conftest_images/ -s furtrack --save-crops

# ── Step 5: Ingest blur background variant ──
echo ""
echo "=== [4/7] DINOv2-base + blur background ==="
pursuit -ds conftest_blur --background blur ingest directory --data-dir ./conftest_images/ -s furtrack --save-crops

# ── Step 6: Ingest DINOv2-large variant ──
echo ""
echo "=== [5/7] DINOv2-large ==="
pursuit -ds conftest_dv2l --embedder dinov2-large ingest directory --data-dir ./conftest_images/ -s furtrack --save-crops

# ── Step 7: Ingest SigLIP variant ──
echo ""
echo "=== [6/7] SigLIP ==="
pursuit -ds conftest_siglip --embedder siglip ingest directory --data-dir ./conftest_images/ -s furtrack --save-crops

# ── Step 8: Ingest DINOv2-base + color histogram variant ──
echo ""
echo "=== [7/7] DINOv2-base + color histogram ==="
pursuit -ds conftest_chist --embedder dinov2-base+colorhist ingest directory --data-dir ./conftest_images/ -s furtrack --save-crops

# ── Step 9: Show stats for all variants ──
echo ""
echo "=========================================="
echo "=== Dataset stats ==="
echo "=========================================="
for ds in conftest_baseline conftest_clip conftest_bw conftest_blur conftest_dv2l conftest_siglip conftest_chist; do
    echo ""
    echo "--- $ds ---"
    pursuit -ds "$ds" stats
done

# ── Step 10: Evaluate each variant (leave-one-out within the dataset) ──
echo ""
echo "=========================================="
echo "=== Evaluation results ==="
echo "=========================================="

VARIANTS="conftest_baseline conftest_clip conftest_bw conftest_blur conftest_dv2l conftest_siglip conftest_chist"
RESULTS_DIR="conftest_results"
mkdir -p "$RESULTS_DIR"

for ds in $VARIANTS; do
    echo ""
    echo "--- Evaluating $ds ---"
    pursuit evaluate --from "$ds" --against "$ds" --json > "$RESULTS_DIR/${ds}.json" 2>&1 || true
    # Also print human-readable
    pursuit evaluate --from "$ds" --against "$ds" 2>&1 | tee "$RESULTS_DIR/${ds}.txt"
done

echo ""
echo "=========================================="
echo "=== Done! Results saved to $RESULTS_DIR/ ==="
echo "=========================================="
echo ""
echo "Variants tested:"
echo "  conftest_baseline  - DINOv2-base + solid gray (768d)"
echo "  conftest_clip      - CLIP ViT-B/32 (512d)"
echo "  conftest_bw        - DINOv2-base + grayscale (768d)"
echo "  conftest_blur      - DINOv2-base + blur background (768d)"
echo "  conftest_dv2l      - DINOv2-large (1024d)"
echo "  conftest_siglip    - SigLIP-base (768d)"
echo "  conftest_chist     - DINOv2-base + color histogram (832d)"
