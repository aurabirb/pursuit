"""Basic tests for the SAM3 fursuit recognition system."""

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config
from sam3_pursuit.storage.database import Database, Detection
from sam3_pursuit.storage.vector_index import VectorIndex


class TestDatabase(unittest.TestCase):
    """Tests for the database module."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_file.close()
        self.db = Database(self.temp_file.name)

    def tearDown(self):
        """Clean up temporary database."""
        os.unlink(self.temp_file.name)

    def test_add_detection(self):
        """Test adding a detection record."""
        detection = Detection(
            id=None,
            post_id="12345",
            character_name="TestChar",
            embedding_id=0,
            bbox_x=10,
            bbox_y=20,
            bbox_width=100,
            bbox_height=100,
            confidence=0.95
        )

        row_id = self.db.add_detection(detection)
        self.assertIsNotNone(row_id)

    def test_get_detection_by_embedding_id(self):
        """Test retrieving a detection by embedding ID."""
        detection = Detection(
            id=None,
            post_id="12345",
            character_name="TestChar",
            embedding_id=42,
            bbox_x=10,
            bbox_y=20,
            bbox_width=100,
            bbox_height=100,
            confidence=0.95
        )

        self.db.add_detection(detection)
        retrieved = self.db.get_detection_by_embedding_id(42)
        if retrieved is None:
            self.fail("Failed to retrieve detection by embedding ID")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.post_id, "12345")
        self.assertEqual(retrieved.character_name, "TestChar")

    def test_get_stats(self):
        """Test getting database statistics."""
        # Add some detections
        for i in range(5):
            detection = Detection(
                id=None,
                post_id=f"post_{i}",
                character_name="TestChar",
                embedding_id=i,
                bbox_x=0,
                bbox_y=0,
                bbox_width=100,
                bbox_height=100,
                confidence=0.9
            )
            self.db.add_detection(detection)

        stats = self.db.get_stats()
        self.assertEqual(stats["total_detections"], 5)
        self.assertEqual(stats["unique_characters"], 1)
        self.assertEqual(stats["unique_posts"], 5)


class TestMultipleSegments(unittest.TestCase):
    """Tests for multiple segments from the same image."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_file.close()
        self.db = Database(self.temp_file.name)

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_multiple_segments_no_collision(self):
        """Multiple segments from same image should not collide."""
        from sam3_pursuit.storage.database import SOURCE_TGBOT

        post_id = "test_image_001"
        source = SOURCE_TGBOT
        preproc = "v2|seg:sam3|con:fursuiter head|bg:s|bgc:808080|emb:dv2b|tsz:630"

        # Add 3 segments from same image (simulating 3 detected fursuiters)
        for i in range(3):
            detection = Detection(
                id=None,
                post_id=post_id,
                character_name="test_char",
                embedding_id=i,
                bbox_x=i * 100, bbox_y=0, bbox_width=100, bbox_height=100,
                confidence=0.95,
                segmentor_model="sam3",
                source=source,
                preprocessing_info=preproc,
            )
            self.db.add_detection(detection)

        # All 3 should be stored
        detections = self.db.get_detections_by_post_id(post_id)
        self.assertEqual(len(detections), 3)

    def test_deduplication_with_source(self):
        """Posts already processed should not need update."""
        from sam3_pursuit.storage.database import SOURCE_TGBOT, SOURCE_NFC25

        post_id = "test_image_002"
        preproc = "v2|seg:sam3|con:fursuiter|bg:s|bgc:808080|emb:dv2b|tsz:630"

        # Add detection with tgbot source
        self.db.add_detection(Detection(
            id=None, post_id=post_id, character_name="char",
            embedding_id=100, bbox_x=0, bbox_y=0, bbox_width=100, bbox_height=100,
            confidence=0.9, source=SOURCE_TGBOT, preprocessing_info=preproc,
        ))

        # Same source+post+preproc should not need update
        needs = self.db.get_posts_needing_update([post_id], preproc, SOURCE_TGBOT)
        self.assertEqual(needs, set())

        # Different source should need update (post_id collision handled)
        needs = self.db.get_posts_needing_update([post_id], preproc, SOURCE_NFC25)
        self.assertEqual(needs, {post_id})


class TestVectorIndex(unittest.TestCase):
    """Tests for the vector index module."""

    def setUp(self):
        """Create a temporary index for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".index")
        self.temp_file.close()
        os.unlink(self.temp_file.name)  # Remove so VectorIndex creates new
        self.index = VectorIndex(
            index_path=self.temp_file.name,
            embedding_dim=768
        )

    def tearDown(self):
        """Clean up temporary index."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_add_and_search(self):
        """Test adding embeddings and searching."""
        # Create random embeddings
        embeddings = np.random.randn(10, 768).astype(np.float32)
        # L2 normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add to index
        start_id = self.index.add(embeddings)
        self.assertEqual(start_id, 0)
        self.assertEqual(self.index.size, 10)

        # Search with first embedding
        query = embeddings[0]
        distances, indices = self.index.search(query, top_k=3)

        # First result should be the same embedding (distance ~0)
        self.assertEqual(indices[0][0], 0)
        self.assertLess(distances[0][0], 0.01)

    def test_save_and_load(self):
        """Test saving and loading index."""
        embeddings = np.random.randn(5, 768).astype(np.float32)
        self.index.add(embeddings)
        self.index.save()

        # Load in new instance
        new_index = VectorIndex(
            index_path=self.temp_file.name,
            embedding_dim=768
        )
        self.assertEqual(new_index.size, 5)


class TestConfig(unittest.TestCase):
    """Tests for the config module."""

    def test_get_device(self):
        """Test device selection."""
        device = Config.get_device()
        self.assertIn(device, ["cuda", "mps", "cpu"])

    def test_paths(self):
        """Test path configurations."""
        self.assertTrue(Config.DB_PATH.endswith(".db"))
        self.assertTrue(Config.INDEX_PATH.endswith(".index"))


class TestClassifier(unittest.TestCase):
    """Tests for the CLIP image classifier."""

    TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
    DRAWING_IMAGE = os.path.join(TESTS_DIR, "barq_drawing.jpg")
    FURSUIT_IMAGE = os.path.join(TESTS_DIR, "blazi_wolf.1.jpg")

    @classmethod
    def setUpClass(cls):
        from sam3_pursuit.models.classifier import ImageClassifier
        cls.classifier = ImageClassifier()

    def test_drawing_not_fursuit(self):
        """A cartoon drawing should not be classified as a fursuit."""
        image = Image.open(self.DRAWING_IMAGE)
        scores = self.classifier.classify(image)
        self.assertFalse(
            self.classifier.is_fursuit(image),
            f"Drawing should not be classified as fursuit, scores: {scores}",
        )

    def test_fursuit_photo_is_fursuit(self):
        """A real fursuit photo should be classified as a fursuit."""
        image = Image.open(self.FURSUIT_IMAGE)
        scores = self.classifier.classify(image)
        self.assertTrue(
            self.classifier.is_fursuit(image),
            f"Fursuit photo should be classified as fursuit, scores: {scores}",
        )

    def test_classify_returns_all_labels(self):
        """classify() should return scores for all labels."""
        image = Image.open(self.DRAWING_IMAGE)
        scores = self.classifier.classify(image)
        self.assertEqual(set(scores.keys()), set(Config.CLASSIFY_LABELS))
        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestIdentification(unittest.TestCase):
    """Integration tests for character identification.

    These tests require a populated database (e.g., from nfc25 ingest).
    Tests are skipped if the database doesn't exist or is empty.
    """

    TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
    BLAZI_IMAGES = [
        os.path.join(TESTS_DIR, "blazi_wolf.1.jpg"),
        os.path.join(TESTS_DIR, "blazi_wolf.2.jpg"),
        os.path.join(TESTS_DIR, "blazi_wolf.3.jpg"),
    ]
    MULTI_FUR_IMAGE = os.path.join(TESTS_DIR, "3furs.jpg")

    @classmethod
    def setUpClass(cls):
        """Check if database exists and has data."""
        if not os.path.exists(Config.DB_PATH) or not os.path.exists(Config.INDEX_PATH):
            cls.skip_reason = "Database or index not found"
            return

        from sam3_pursuit.storage.vector_index import VectorIndex
        index = VectorIndex(Config.INDEX_PATH)
        if index.size == 0:
            cls.skip_reason = "Index is empty"
            return

        cls.skip_reason = None

        # Load identifier once for all tests
        from sam3_pursuit.api.identifier import SAM3FursuitIdentifier
        cls.identifier = SAM3FursuitIdentifier(segmentor_model_name=Config.SAM3_MODEL, segmentor_concept=Config.DEFAULT_CONCEPT)

    def setUp(self):
        """Skip test if database not available."""
        if getattr(self.__class__, 'skip_reason', None):
            self.skipTest(self.__class__.skip_reason)

    def test_identify_with_segmentation(self):
        """Test identifying a character using segmentation on full photo."""
        image = Image.open(self.BLAZI_IMAGES[0])

        # Use segmentation since test images are full photos
        results = self.identifier.identify(image, top_k=10)

        self.assertGreater(len(results), 0, "Should return at least one result")

        # Print results for debugging
        print(f"\nResults for {os.path.basename(self.BLAZI_IMAGES[0])}:")
        for r in results:
            print(f"\n========\nSegment with {len(r.matches)} matches:")
            for i, m in enumerate(r.matches):
                print(f"  {i+1}. {m.character_name}: {m.confidence:.2%}")

    def test_segmentation_detects_multiple_fursuiters(self):
        """Test that segmentation finds multiple fursuiters in 3furs.jpg."""
        from sam3_pursuit.models.segmentor import FursuitSegmentor

        image = Image.open(self.MULTI_FUR_IMAGE)
        segmentor = FursuitSegmentor()

        results = segmentor.segment(image)

        print(f"\nSegmentation found {len(results)} fursuiter(s) in 3furs.jpg")
        for i, r in enumerate(results):
            print(f"  Segment {i}: bbox={r.bbox}, conf={r.confidence:.2%}")

        # Should detect at least 2 fursuiters (ideally 3)
        self.assertGreaterEqual(
            len(results), 2,
            f"Should detect at least 2 fursuiters, got {len(results)}"
        )

        # Each result should have a valid crop
        for i, result in enumerate(results):
            self.assertIsNotNone(result.crop, f"Segment {i} should have a crop")
            self.assertGreater(result.crop.size[0], 0)
            self.assertGreater(result.crop.size[1], 0)

    def test_identify_multi_fursuit_returns_results(self):
        """Test that identification on 3furs.jpg returns multiple results."""
        image = Image.open(self.MULTI_FUR_IMAGE)

        # Use segmentation to detect multiple characters
        results = self.identifier.identify(image, top_k=15)

        self.assertGreater(len(results), 0, "Should detect at least one character")

        # Print all detected characters
        print(f"\nIdentification results for 3furs.jpg:")
        seen = set()
        for r in results:
            print(f"\n========\nSegment with {len(r.matches)} matches:")
            for m in r.matches:
                if m.character_name and m.character_name not in seen:
                    print(f"  {m.character_name}: {m.confidence:.2%}")
                    seen.add(m.character_name)

    def test_add_and_identify_character(self):
        """Test adding images then identifying - the core use case."""
        # Create a temporary database for this test
        with tempfile.TemporaryDirectory() as tmpdir:
            from sam3_pursuit.api.identifier import SAM3FursuitIdentifier

            db_path = os.path.join(tmpdir, "test.db")
            index_path = os.path.join(tmpdir, "test.index")

            identifier = SAM3FursuitIdentifier(db_path=db_path, index_path=index_path)

            # Add first two Blazi images
            added = identifier.add_images(
                character_names=["Blazi", "Blazi"],
                image_paths=self.BLAZI_IMAGES[:2]
            )
            self.assertGreater(added, 0, "Should add at least one embedding")

            # Now identify using the third image
            image = Image.open(self.BLAZI_IMAGES[2])
            results = identifier.identify(image, top_k=5)

            self.assertGreater(len(results), 0, "Should return results")

            # Blazi should be the top result since we just added them
            top_name = results[0].matches[0].character_name or ""
            self.assertEqual(
                top_name.lower(), "blazi",
                f"Blazi should be top result, got: {top_name}"
            )


class TestBarqIngestion(unittest.TestCase):
    """Tests for Barq image ingestion."""

    TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_IMAGE = os.path.join(TESTS_DIR, "blazi_wolf.1.jpg")

    def test_get_source_url_barq(self):
        """Test that barq source URL links to the image."""
        from sam3_pursuit.storage.database import get_source_url, SOURCE_BARQ

        image_uuid = "abc123-def456-ghi789"
        url = get_source_url(SOURCE_BARQ, image_uuid)

        self.assertEqual(url, f"https://assets.barq.app/image/{image_uuid}.jpeg")

    def test_barq_directory_structure_parsing(self):
        """Test that barq ingestion correctly parses folder structure."""
        import shutil
        from sam3_pursuit.storage.database import SOURCE_BARQ

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create barq-style directory structure: {profile_id}.{name}/{image_uuid}.jpg
            data_dir = os.path.join(tmpdir, "barq_images")
            os.makedirs(data_dir)

            # Create test folders with images
            char1_dir = os.path.join(data_dir, "12345.TestCharacter")
            char2_dir = os.path.join(data_dir, "67890.AnotherChar")
            os.makedirs(char1_dir)
            os.makedirs(char2_dir)

            # Copy test image with UUID-style names
            img1_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            img2_uuid = "11111111-2222-3333-4444-555555555555"
            shutil.copy(self.TEST_IMAGE, os.path.join(char1_dir, f"{img1_uuid}.jpg"))
            shutil.copy(self.TEST_IMAGE, os.path.join(char2_dir, f"{img2_uuid}.jpg"))

            # Create temp db and index
            db_path = os.path.join(tmpdir, "test.db")
            index_path = os.path.join(tmpdir, "test.index")

            from sam3_pursuit.api.identifier import SAM3FursuitIdentifier
            identifier = SAM3FursuitIdentifier(db_path=db_path, index_path=index_path)

            # Run barq ingestion manually (simulating CLI)
            from pathlib import Path
            from sam3_pursuit.storage.database import Database

            data_path = Path(data_dir)
            char_names = []
            img_paths = []

            for char_dir in sorted(data_path.iterdir()):
                if not char_dir.is_dir() or "." not in char_dir.name:
                    continue
                character_name = char_dir.name.split(".", 1)[1]
                for img in char_dir.glob("*.jpg"):
                    char_names.append(character_name)
                    img_paths.append(str(img))

            added = identifier.add_images(
                character_names=char_names,
                image_paths=img_paths,
                source=SOURCE_BARQ,
            )

            self.assertGreater(added, 0, "Should add at least one embedding")

            # Verify detections use image UUID as post_id
            db = Database(db_path)
            det1 = db.get_detections_by_post_id(img1_uuid)
            det2 = db.get_detections_by_post_id(img2_uuid)

            self.assertGreater(len(det1), 0, f"Should find detection for {img1_uuid}")
            self.assertGreater(len(det2), 0, f"Should find detection for {img2_uuid}")

            # Verify character names are correct
            self.assertEqual(det1[0].character_name, "testcharacter")
            self.assertEqual(det2[0].character_name, "anotherchar")

            # Verify source is barq
            self.assertEqual(det1[0].source, SOURCE_BARQ)
            self.assertEqual(det2[0].source, SOURCE_BARQ)

    def test_barq_folder_without_dot_is_skipped(self):
        """Test that folders without profile_id.name format are skipped."""
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "barq_images")
            os.makedirs(data_dir)

            # Create a folder without the dot separator (should be skipped)
            invalid_dir = os.path.join(data_dir, "InvalidFolderName")
            os.makedirs(invalid_dir)
            shutil.copy(self.TEST_IMAGE, os.path.join(invalid_dir, "test.jpg"))

            # Create a valid folder
            valid_dir = os.path.join(data_dir, "123.ValidChar")
            os.makedirs(valid_dir)
            shutil.copy(self.TEST_IMAGE, os.path.join(valid_dir, "valid-uuid.jpg"))

            db_path = os.path.join(tmpdir, "test.db")
            index_path = os.path.join(tmpdir, "test.index")

            from sam3_pursuit.api.identifier import SAM3FursuitIdentifier
            from sam3_pursuit.storage.database import SOURCE_BARQ, Database
            from pathlib import Path

            identifier = SAM3FursuitIdentifier(db_path=db_path, index_path=index_path)
            data_path = Path(data_dir)

            char_names = []
            img_paths = []

            for char_dir in sorted(data_path.iterdir()):
                if not char_dir.is_dir() or "." not in char_dir.name:
                    continue
                character_name = char_dir.name.split(".", 1)[1]
                for img in char_dir.glob("*.jpg"):
                    char_names.append(character_name)
                    img_paths.append(str(img))

            # Should only find 1 image (from valid folder)
            self.assertEqual(len(img_paths), 1)
            self.assertIn("ValidChar", char_names[0])

            added = identifier.add_images(
                character_names=char_names,
                image_paths=img_paths,
                source=SOURCE_BARQ,
            )

            db = Database(db_path)
            # Only valid-uuid should exist
            valid_det = db.get_detections_by_post_id("valid-uuid")
            self.assertGreater(len(valid_det), 0)

            # test.jpg from invalid folder should not exist
            invalid_det = db.get_detections_by_post_id("test")
            self.assertEqual(len(invalid_det), 0)


class TestMaskReuse(unittest.TestCase):
    """Tests for mask storage and reuse functionality."""

    TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_IMAGE = os.path.join(TESTS_DIR, "blazi_wolf.1.jpg")

    def test_mask_save_and_load_roundtrip(self):
        """Test that saved masks can be loaded back identically."""
        from sam3_pursuit.storage.mask_storage import MaskStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MaskStorage(base_dir=tmpdir)
            original_mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255

            storage.save_mask(original_mask, "test_seg_0", "test_source", "sam3", "fursuiter head")
            loaded_mask = storage.load_mask("test_seg_0", "test_source", "sam3", "fursuiter head")

            self.assertIsNotNone(loaded_mask)
            np.testing.assert_array_equal(original_mask, loaded_mask)

    def test_find_masks_for_post(self):
        """Test finding all masks for a post_id."""
        from sam3_pursuit.storage.mask_storage import MaskStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MaskStorage(base_dir=tmpdir)

            # Save 3 masks for the same post
            for i in range(3):
                mask = np.ones((50, 50), dtype=np.uint8) * 255
                storage.save_mask(mask, f"post123_seg_{i}", "barq", "sam3", "fursuiter head")

            masks = storage.find_masks_for_post("post123", "barq", "sam3", "fursuiter head")
            self.assertEqual(len(masks), 3)

            # Should be sorted by segment index
            self.assertTrue(masks[0].name.endswith("_seg_0.png"))
            self.assertTrue(masks[1].name.endswith("_seg_1.png"))
            self.assertTrue(masks[2].name.endswith("_seg_2.png"))

    def test_load_masks_for_post(self):
        """Test loading all masks for a post_id."""
        from sam3_pursuit.storage.mask_storage import MaskStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MaskStorage(base_dir=tmpdir)

            # Save masks with different content
            for i in range(2):
                mask = np.ones((50, 50), dtype=np.uint8) * (i + 1) * 100
                storage.save_mask(mask, f"post456_seg_{i}", "barq", "sam3", "fursuiter head")

            loaded = storage.load_masks_for_post("post456", "barq", "sam3", "fursuiter head")
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0][0], 0)  # segment index
            self.assertEqual(loaded[1][0], 1)

    @unittest.skipIf(not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "blazi_wolf.1.jpg")), "Test image not found")
    def test_process_with_masks_matches_fresh_processing(self):
        """Test that processing with saved masks produces identical embeddings to fresh SAM3."""
        from sam3_pursuit.pipeline.processor import ProcessingPipeline
        from sam3_pursuit.storage.mask_storage import MaskStorage

        image = Image.open(self.TEST_IMAGE)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ProcessingPipeline(segmentor_model_name="sam3", segmentor_concept="fursuiter head")
            storage = MaskStorage(base_dir=tmpdir)

            # Run fresh processing with SAM3
            fresh_results = pipeline.process(image)
            self.assertGreater(len(fresh_results), 0, "Should detect at least one segment")

            # Save full masks (not crop_mask) for bbox computation
            for i, result in enumerate(fresh_results):
                storage.save_mask(
                    result.segmentation.mask,
                    f"test_seg_{i}",
                    "test", "sam3", "fursuiter head"
                )

            # Load masks and reprocess
            loaded_masks = storage.load_masks_for_post("test", "test", "sam3", "fursuiter head")
            reused_results = pipeline.process_with_masks(image, loaded_masks)

            # Compare results
            self.assertEqual(len(fresh_results), len(reused_results), "Should have same number of segments")

            for i, (fresh, reused) in enumerate(zip(fresh_results, reused_results)):
                # Embeddings should be identical
                np.testing.assert_array_almost_equal(
                    fresh.embedding, reused.embedding, decimal=5,
                    err_msg=f"Segment {i} embeddings don't match"
                )
                # Bboxes should match
                self.assertEqual(fresh.segmentation.bbox, reused.segmentation.bbox, f"Segment {i} bboxes don't match")


if __name__ == "__main__":
    unittest.main()
