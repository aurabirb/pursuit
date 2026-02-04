"""Benchmark: measure per-image ingestion time when masks are pre-computed.

Usage:
    python -m pytest tests/test_ingest_timing.py -v -s
    python tests/test_ingest_timing.py
"""

import os
import shutil
import tempfile
import time
import unittest

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config
from sam3_pursuit.storage.mask_storage import MaskStorage


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(TESTS_DIR, "blazi_wolf.1.jpg")


def _generate_realistic_mask(image: Image.Image, seg_index: int = 0) -> np.ndarray:
    """Generate a plausible segmentation mask (centered ellipse)."""
    w, h = image.size
    y, x = np.ogrid[:h, :w]
    # Offset each segment slightly so they don't overlap
    cx = w // 2 + seg_index * (w // 6)
    cy = h // 2
    rx, ry = w // 4, h // 3
    mask = ((x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2 <= 1).astype(np.uint8) * 255
    return mask


@unittest.skipIf(not os.path.exists(TEST_IMAGE), "Test image not found")
class TestIngestTiming(unittest.TestCase):
    """Benchmark ingestion speed with pre-computed masks."""

    NUM_IMAGES = 10

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, "bench.db")
        cls.index_path = os.path.join(cls.tmpdir, "bench.index")
        cls.mask_dir = os.path.join(cls.tmpdir, "masks")

        # Prepare test images (copies of the same file with unique post_ids)
        cls.image_dir = os.path.join(cls.tmpdir, "images")
        os.makedirs(cls.image_dir)

        cls.post_ids = []
        cls.image_paths = []
        source_img = Image.open(TEST_IMAGE)

        mask_storage = MaskStorage(base_dir=cls.mask_dir)

        for i in range(cls.NUM_IMAGES):
            post_id = f"bench_{i:04d}"
            img_path = os.path.join(cls.image_dir, f"{post_id}.jpg")
            source_img.save(img_path, quality=90)
            cls.post_ids.append(post_id)
            cls.image_paths.append(img_path)

            # Pre-save a mask (simulating masks copied from GPU server)
            mask = _generate_realistic_mask(source_img)
            mask_storage.save_mask(mask, f"{post_id}_seg_0", "barq", Config.SAM3_MODEL, Config.DEFAULT_CONCEPT)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def test_ingest_with_masks_timing(self):
        """Time full ingestion pipeline with pre-computed masks."""
        from sam3_pursuit.api.identifier import SAM3FursuitIdentifier

        # --- Phase 1: Identifier initialization ---
        t0 = time.perf_counter()
        identifier = SAM3FursuitIdentifier(
            db_path=self.db_path,
            index_path=self.index_path,
            segmentor_model_name=Config.SAM3_MODEL,
            segmentor_concept=Config.DEFAULT_CONCEPT,
        )
        # Point mask storage at our pre-computed masks
        identifier.mask_storage = MaskStorage(base_dir=self.mask_dir)
        t_init = time.perf_counter() - t0

        # --- Phase 2: Ingestion ---
        character_names = ["bench_char"] * self.NUM_IMAGES

        t1 = time.perf_counter()
        added = identifier.add_images(
            character_names=character_names,
            image_paths=self.image_paths,
            source="barq",
            add_full_image=True,
        )
        t_ingest = time.perf_counter() - t1

        t_total = t_init + t_ingest
        per_image = t_ingest / self.NUM_IMAGES if self.NUM_IMAGES else 0

        print(f"\n{'=' * 60}")
        print(f"INGESTION BENCHMARK ({self.NUM_IMAGES} images, masks pre-computed)")
        print(f"{'=' * 60}")
        print(f"  Device:              {Config.get_device()}")
        print(f"  Identifier init:     {t_init:.3f}s  (DINOv2 load, no SAM3)")
        print(f"  Ingestion total:     {t_ingest:.3f}s")
        print(f"  Per image:           {per_image:.3f}s  ({per_image * 1000:.0f}ms)")
        print(f"  Images added:        {added}")
        print(f"  Total wall time:     {t_total:.3f}s")
        print(f"{'=' * 60}")

        self.assertGreater(added, 0, "Should have added embeddings")

    def test_per_step_breakdown(self):
        """Time individual steps: mask load, isolation, embedding, db."""
        from sam3_pursuit.models.embedder import FursuitEmbedder
        from sam3_pursuit.models.preprocessor import BackgroundIsolator, IsolationConfig
        from sam3_pursuit.models.segmentor import SegmentationResult
        from sam3_pursuit.pipeline.processor import ProcessingPipeline

        mask_storage = MaskStorage(base_dir=self.mask_dir)
        image = Image.open(TEST_IMAGE)

        # Init models once
        t0 = time.perf_counter()
        pipeline = ProcessingPipeline(
            segmentor_model_name=Config.SAM3_MODEL,
            segmentor_concept=Config.DEFAULT_CONCEPT,
        )
        t_pipeline_init = time.perf_counter() - t0

        # Benchmark individual steps
        t_load_mask = 0
        t_from_mask = 0
        t_isolate = 0
        t_resize = 0
        t_embed = 0
        n = self.NUM_IMAGES

        for i in range(n):
            post_id = self.post_ids[i]

            t = time.perf_counter()
            masks = mask_storage.load_masks_for_post(post_id, "barq", Config.SAM3_MODEL, Config.DEFAULT_CONCEPT)
            t_load_mask += time.perf_counter() - t

            for seg_idx, mask in masks:
                t = time.perf_counter()
                seg = SegmentationResult.from_mask(image, mask, segmentor=Config.SAM3_MODEL)
                t_from_mask += time.perf_counter() - t

                t = time.perf_counter()
                isolated = pipeline.isolator.isolate(seg.crop, seg.crop_mask)
                t_isolate += time.perf_counter() - t

                t = time.perf_counter()
                resized = pipeline._resize_to_patch_multiple(isolated)
                t_resize += time.perf_counter() - t

                t = time.perf_counter()
                embedding = pipeline.embedder.embed(resized)
                t_embed += time.perf_counter() - t

        print(f"\n{'=' * 60}")
        print(f"PER-STEP BREAKDOWN ({n} images, 1 mask each)")
        print(f"{'=' * 60}")
        print(f"  Pipeline init:       {t_pipeline_init:.3f}s  (DINOv2 only, SAM3 deferred)")
        print(f"  Mask load:           {t_load_mask:.3f}s  ({t_load_mask / n * 1000:.1f}ms/img)")
        print(f"  Crop from mask:      {t_from_mask:.3f}s  ({t_from_mask / n * 1000:.1f}ms/img)")
        print(f"  Background isolate:  {t_isolate:.3f}s  ({t_isolate / n * 1000:.1f}ms/img)")
        print(f"  Resize:              {t_resize:.3f}s  ({t_resize / n * 1000:.1f}ms/img)")
        print(f"  DINOv2 embed:        {t_embed:.3f}s  ({t_embed / n * 1000:.1f}ms/img)")
        t_total_steps = t_load_mask + t_from_mask + t_isolate + t_resize + t_embed
        print(f"  ─────────────────────────────────")
        print(f"  Total per-image:     {t_total_steps / n * 1000:.0f}ms")
        print(f"  DINOv2 share:        {t_embed / t_total_steps * 100:.0f}%")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    unittest.main()
