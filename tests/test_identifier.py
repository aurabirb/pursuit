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
        cls.identifier = SAM3FursuitIdentifier()

    def setUp(self):
        """Skip test if database not available."""
        if getattr(self.__class__, 'skip_reason', None):
            self.skipTest(self.__class__.skip_reason)

    def test_identify_with_segmentation(self):
        """Test identifying a character using segmentation on full photo."""
        image = Image.open(self.BLAZI_IMAGES[0])

        # Use segmentation since test images are full photos
        results = self.identifier.identify(image, top_k=10, use_segmentation=True)

        self.assertGreater(len(results), 0, "Should return at least one result")

        # Print results for debugging
        print(f"\nResults for {os.path.basename(self.BLAZI_IMAGES[0])}:")
        for i, r in enumerate(results[:5]):
            print(f"  {i+1}. {r.character_name}: {r.confidence:.2%}")

    def test_segmentation_detects_multiple_fursuiters(self):
        """Test that segmentation finds multiple fursuiters in 3furs.jpg."""
        from sam3_pursuit.models.segmentor import FursuitSegmentor

        image = Image.open(self.MULTI_FUR_IMAGE)
        segmentor = FursuitSegmentor()

        results = segmentor.segment(image, concept="fursuiter")

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
        results = self.identifier.identify(image, top_k=15, use_segmentation=True)

        self.assertGreater(len(results), 0, "Should detect at least one character")

        # Print all detected characters
        print(f"\nIdentification results for 3furs.jpg:")
        seen = set()
        for r in results:
            if r.character_name and r.character_name not in seen:
                print(f"  {r.character_name}: {r.confidence:.2%}")
                seen.add(r.character_name)

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
                image_paths=self.BLAZI_IMAGES[:2],
                use_segmentation=True,
                concept="fursuiter",
            )
            self.assertGreater(added, 0, "Should add at least one embedding")

            # Now identify using the third image
            image = Image.open(self.BLAZI_IMAGES[2])
            results = identifier.identify(image, top_k=5, use_segmentation=True)

            self.assertGreater(len(results), 0, "Should return results")

            # Blazi should be the top result since we just added them
            top_name = results[0].character_name
            self.assertEqual(
                top_name.lower(), "blazi",
                f"Blazi should be top result, got: {top_name}"
            )


if __name__ == "__main__":
    unittest.main()
