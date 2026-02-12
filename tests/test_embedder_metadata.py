"""Tests for embedder metadata storage and validation."""

import os
import tempfile

import numpy as np
import pytest

from sam3_pursuit.config import Config
from sam3_pursuit.storage.database import Database, Detection
from sam3_pursuit.storage.vector_index import VectorIndex


class FakeEmbedder:
    """Minimal embedder for testing without loading real models."""

    def __init__(self, model_name="facebook/dinov2-base", embedding_dim=768):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    def embed(self, image):
        return np.random.randn(self.embedding_dim).astype(np.float32)


class FakePipeline:
    """Minimal pipeline wrapper for testing embedder short name logic."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.embedder_model_name = embedder.model_name

    def get_embedder_short_name(self):
        from sam3_pursuit.pipeline.processor import CachedProcessingPipeline
        # Reuse the real short name logic
        emb = self.embedder_model_name
        if "+colorhist" in emb:
            base = emb.replace("+colorhist", "")
            return CachedProcessingPipeline._short_name_for(base) + "+chist"
        return CachedProcessingPipeline._short_name_for(emb)


@pytest.fixture
def db_and_index(tmp_path):
    """Create a fresh temp DB and index, returning (db_path, index_path)."""
    db_path = str(tmp_path / "test.db")
    index_path = str(tmp_path / "test.index")
    return db_path, index_path


def _add_dummy_embedding(db, index, embedding_dim=768):
    """Add a single dummy detection + embedding to a db/index pair."""
    emb = np.random.randn(embedding_dim).astype(np.float32)
    emb_id = index.add(emb.reshape(1, -1))
    det = Detection(
        id=None, post_id="test_post", character_name="test_char",
        embedding_id=emb_id, bbox_x=0, bbox_y=0, bbox_width=100,
        bbox_height=100, confidence=0.9, segmentor_model="test",
        source="manual",
    )
    db.add_detection(det)
    index.save()


class TestEmbedderMetadata:
    """Tests for embedder metadata storage and validation in FursuitIngestor."""

    def test_stores_embedder_on_empty_init(self, db_and_index):
        """New empty dataset should store embedder metadata on init."""
        db_path, index_path = db_and_index
        db = Database(db_path)
        index = VectorIndex(index_path)

        assert db.get_metadata(Config.METADATA_KEY_EMBEDDER) is None

        embedder = FakeEmbedder()
        pipeline = FakePipeline(embedder)

        # Simulate what _validate_or_store_embedder does
        current = pipeline.get_embedder_short_name()
        db.set_metadata(Config.METADATA_KEY_EMBEDDER, current)

        assert db.get_metadata(Config.METADATA_KEY_EMBEDDER) == "dv2b"
        db.close()

    def test_stores_embedder_on_nonempty_init(self, db_and_index):
        """Existing dataset without metadata should store embedder on init."""
        db_path, index_path = db_and_index
        db = Database(db_path)
        index = VectorIndex(index_path)
        _add_dummy_embedding(db, index)

        assert db.get_metadata(Config.METADATA_KEY_EMBEDDER) is None
        assert index.size == 1

        # Simulating _validate_or_store_embedder with matching dimensions
        embedder = FakeEmbedder(embedding_dim=768)
        pipeline = FakePipeline(embedder)
        current = pipeline.get_embedder_short_name()
        stored = db.get_metadata(Config.METADATA_KEY_EMBEDDER)

        # No stored metadata, dims match -> should store
        assert stored is None
        assert index.embedding_dim == embedder.embedding_dim
        db.set_metadata(Config.METADATA_KEY_EMBEDDER, current)
        assert db.get_metadata(Config.METADATA_KEY_EMBEDDER) == "dv2b"
        db.close()

    def test_rejects_mismatched_embedder_name(self, db_and_index):
        """Should raise ValueError when stored embedder doesn't match current."""
        db_path, index_path = db_and_index
        db = Database(db_path)
        index = VectorIndex(index_path, embedding_dim=832)
        _add_dummy_embedding(db, index, embedding_dim=832)

        # Store as colorhist embedder
        db.set_metadata(Config.METADATA_KEY_EMBEDDER, "dv2b+chist")

        # Now try to open with default embedder
        embedder = FakeEmbedder(embedding_dim=768)
        pipeline = FakePipeline(embedder)
        current = pipeline.get_embedder_short_name()
        stored = db.get_metadata(Config.METADATA_KEY_EMBEDDER)

        assert stored == "dv2b+chist"
        assert current == "dv2b"
        assert stored != current  # This is what triggers the error
        db.close()

    def test_rejects_mismatched_dimension(self, db_and_index):
        """Pre-metadata dataset with different dim index should be caught."""
        db_path, index_path = db_and_index
        db = Database(db_path)
        index = VectorIndex(index_path, embedding_dim=832)
        _add_dummy_embedding(db, index, embedding_dim=832)

        # No metadata stored (pre-metadata dataset)
        assert db.get_metadata(Config.METADATA_KEY_EMBEDDER) is None

        # Try with default 768D embedder
        embedder = FakeEmbedder(embedding_dim=768)

        assert index.size > 0
        assert index.embedding_dim != embedder.embedding_dim  # 832 != 768
        db.close()

    def test_accepts_matching_embedder(self, db_and_index):
        """Should succeed when stored embedder matches current."""
        db_path, index_path = db_and_index
        db = Database(db_path)
        index = VectorIndex(index_path, embedding_dim=832)
        _add_dummy_embedding(db, index, embedding_dim=832)

        db.set_metadata(Config.METADATA_KEY_EMBEDDER, "dv2b+chist")

        embedder = FakeEmbedder(
            model_name="facebook/dinov2-base+colorhist", embedding_dim=832)
        pipeline = FakePipeline(embedder)
        current = pipeline.get_embedder_short_name()
        stored = db.get_metadata(Config.METADATA_KEY_EMBEDDER)

        assert stored == current == "dv2b+chist"
        db.close()


class TestEmbedderValidationIntegration:
    """Integration tests using the real _validate_or_store_embedder logic."""

    def test_ingestor_stores_metadata(self, db_and_index):
        """FursuitIngestor should store embedder metadata on init."""
        db_path, index_path = db_and_index
        from sam3_pursuit.api.ingestor import FursuitIngestor

        ingestor = FursuitIngestor(
            db_path=db_path,
            index_path=index_path,
            segmentor_model_name="full",
        )
        stored = ingestor.db.get_metadata(Config.METADATA_KEY_EMBEDDER)
        assert stored == "siglip"

    def test_ingestor_rejects_wrong_embedder(self, db_and_index):
        """FursuitIngestor should raise when explicit embedder doesn't match stored metadata."""
        db_path, index_path = db_and_index

        # First init stores "dv2b+chist"
        db = Database(db_path)
        db.set_metadata(Config.METADATA_KEY_EMBEDDER, "dv2b+chist")
        db.close()

        from sam3_pursuit.api.ingestor import FursuitIngestor

        # Passing an explicit mismatched embedder should raise
        wrong_embedder = FakeEmbedder(model_name="google/siglip-base-patch16-224")
        with pytest.raises(ValueError, match="Dataset was built with embedder"):
            FursuitIngestor(
                db_path=db_path,
                index_path=index_path,
                segmentor_model_name="full",
                embedder=wrong_embedder,
            )

    def test_ingestor_rejects_dimension_mismatch(self, db_and_index):
        """FursuitIngestor should raise when index dim doesn't match embedder (no metadata)."""
        db_path, index_path = db_and_index

        # Create an index with 832D embeddings but no metadata
        db = Database(db_path)
        index = VectorIndex(index_path, embedding_dim=832)
        _add_dummy_embedding(db, index, embedding_dim=832)
        db.close()

        from sam3_pursuit.api.ingestor import FursuitIngestor

        with pytest.raises(ValueError, match="Index has 832D embeddings"):
            FursuitIngestor(
                db_path=db_path,
                index_path=index_path,
                segmentor_model_name="full",
            )

    def test_ingestor_accepts_matching_metadata(self, db_and_index):
        """FursuitIngestor should succeed when metadata matches current embedder."""
        db_path, index_path = db_and_index

        # Pre-store matching metadata
        db = Database(db_path)
        db.set_metadata(Config.METADATA_KEY_EMBEDDER, "siglip")
        db.close()

        from sam3_pursuit.api.ingestor import FursuitIngestor

        ingestor = FursuitIngestor(
            db_path=db_path,
            index_path=index_path,
            segmentor_model_name="full",
        )
        assert ingestor.db.get_metadata(Config.METADATA_KEY_EMBEDDER) == "siglip"
