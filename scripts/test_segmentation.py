"""Test and visualize SAM3 segmentation on sample images."""

import json
import os
import random

from PIL import Image
import numpy as np

# NFC25 paths
NFC25_DIR = "/media/user/SSD2TB/nfc25-fursuits"
NFC25_JSON = os.path.join(NFC25_DIR, "nfc25-fursuit-list.json")
NFC25_IMAGES = os.path.join(NFC25_DIR, "fursuit_images")


def get_sample_images(n: int = 5, seed: int = 42) -> list[tuple[str, dict]]:
    """Get n random sample images from NFC25."""
    with open(NFC25_JSON) as f:
        data = json.load(f)

    random.seed(seed)
    samples = random.sample(data["FursuitList"], n * 2)  # Get extra in case some don't exist

    result = []
    for fursuit in samples:
        image_url = fursuit.get("ImageUrl", "")
        if not image_url:
            continue

        filename = image_url.split("/")[-1]
        filepath = os.path.join(NFC25_IMAGES, filename)

        if os.path.exists(filepath):
            info = {
                "nickname": fursuit.get("NickName"),
                "species": fursuit.get("Species"),
            }
            result.append((filepath, info))

            if len(result) >= n:
                break

    return result


def test_segmentation():
    """Test SAM3 segmentation on sample images."""
    from sam3_pursuit.models.segmentor import FursuitSegmentor

    print("Initializing segmentor...")
    segmentor = FursuitSegmentor()

    print(f"\nModel: {segmentor.model_name}")
    print(f"Text prompts supported: {segmentor.supports_text_prompts}")
    print()

    # Get sample images
    samples = get_sample_images(5)
    print(f"Testing on {len(samples)} images:\n")

    for filepath, info in samples:
        print(f"Image: {info['nickname']} ({info['species']})")
        print(f"  Path: {os.path.basename(filepath)}")

        image = Image.open(filepath)
        print(f"  Size: {image.size}")

        results = segmentor.segment(image)
        print(f"  Segmentation: {len(results)} segments")
        for i, r in enumerate(results):
            print(f"    [{i}] bbox={r.bbox}, conf={r.confidence:.2f}, crop_size={r.crop.size}")

        print()


def test_full_pipeline():
    """Test the full processing pipeline."""
    from sam3_pursuit.pipeline.processor import ProcessingPipeline

    print("Initializing pipeline...")
    pipeline = ProcessingPipeline()

    print(f"\nText prompts supported: {pipeline.supports_text_prompts}")
    print()

    # Get sample images
    samples = get_sample_images(3)
    print(f"Testing pipeline on {len(samples)} images:\n")

    for filepath, info in samples:
        print(f"Image: {info['nickname']} ({info['species']})")

        image = Image.open(filepath)

        # Process with segmentation
        results = pipeline.process_fursuits(image)
        print(f"  Processed: {len(results)} results")
        for i, r in enumerate(results):
            print(f"    [{i}] embedding_shape={r.embedding.shape}, bbox={r.segmentation.bbox}")

        # Process full image (no segmentation)
        result = pipeline.process_full_image(image)
        print(f"  Full image: embedding_shape={result.embedding.shape}")

        print()


def visualize_segmentation(output_dir: str = "/tmp/seg_test"):
    """Visualize segmentation results by saving crops."""
    from sam3_pursuit.models.segmentor import FursuitSegmentor

    os.makedirs(output_dir, exist_ok=True)

    print("Initializing segmentor...")
    segmentor = FursuitSegmentor()

    samples = get_sample_images(3)
    print(f"\nSaving segmentation visualizations to {output_dir}\n")

    for filepath, info in samples:
        print(f"Processing: {info['nickname']}")

        image = Image.open(filepath)
        results = segmentor.segment(image)

        # Save original (convert RGBA to RGB for JPEG)
        base_name = info['nickname'].replace(" ", "_").replace("/", "_")[:20]
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(os.path.join(output_dir, f"{base_name}_original.jpg"))

        # Save crops
        for i, r in enumerate(results):
            crop_path = os.path.join(output_dir, f"{base_name}_crop_{i}.jpg")
            crop = r.crop.convert('RGB') if r.crop.mode == 'RGBA' else r.crop
            crop.save(crop_path)
            print(f"  Saved crop {i}: {r.crop.size}")

    print(f"\nVisualization complete! Check {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_segmentation.py [segment|pipeline|visualize]")
        print()
        print("Tests:")
        print("  segment   - Test SAM3 segmentor")
        print("  pipeline  - Test full processing pipeline")
        print("  visualize - Save segmentation crops to /tmp/seg_test")
        sys.exit(1)

    test_name = sys.argv[1]

    if test_name == "segment":
        test_segmentation()
    elif test_name == "pipeline":
        test_full_pipeline()
    elif test_name == "visualize":
        visualize_segmentation()
    else:
        print(f"Unknown test: {test_name}")
        sys.exit(1)
