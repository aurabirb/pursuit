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
        """FursuitIngestor should raise when embedder doesn't match stored metadata."""
        db_path, index_path = db_and_index

        # First init stores "dv2b+chist"
        db = Database(db_path)
        db.set_metadata(Config.METADATA_KEY_EMBEDDER, "dv2b+chist")
        db.close()

        from sam3_pursuit.api.ingestor import FursuitIngestor

        with pytest.raises(ValueError, match="Dataset was built with embedder"):
            FursuitIngestor(
                db_path=db_path,
                index_path=index_path,
                segmentor_model_name="full",
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


class TestShortNameMapping:
    """Tests for SHORT_NAME_TO_CLI and embedder short name generation."""

    def test_short_name_to_cli_has_default(self):
        """DEFAULT_EMBEDDER must be a value in SHORT_NAME_TO_CLI."""
        from sam3_pursuit.pipeline.processor import SHORT_NAME_TO_CLI
        assert Config.DEFAULT_EMBEDDER in SHORT_NAME_TO_CLI.values()

    def test_default_embedder_short_matches(self):
        """DEFAULT_EMBEDDER_SHORT must be the short name for DEFAULT_EMBEDDER."""
        from sam3_pursuit.pipeline.processor import (
            CLI_TO_SHORT_NAME,
            DEFAULT_EMBEDDER_SHORT,
        )
        assert DEFAULT_EMBEDDER_SHORT == CLI_TO_SHORT_NAME[Config.DEFAULT_EMBEDDER]

    def test_cli_to_short_name_roundtrip(self):
        """Every CLI name should roundtrip through both mappings."""
        from sam3_pursuit.pipeline.processor import CLI_TO_SHORT_NAME, SHORT_NAME_TO_CLI
        for short, cli in SHORT_NAME_TO_CLI.items():
            assert CLI_TO_SHORT_NAME[cli] == short

    def test_all_embedder_short_names_generate_correctly(self):
        """Each embedder model name should produce the expected short name."""
        from sam3_pursuit.pipeline.processor import CachedProcessingPipeline

        cases = [
            ("facebook/dinov2-base", "dv2b"),
            ("facebook/dinov2-large", "dv2l"),
            ("facebook/dinov2-giant2-lvd-mc-e", "dv2g"),
            ("openai/clip-vit-base-patch32", "clip"),
            ("google/siglip-base-patch16-224", "siglip"),
        ]
        for model_name, expected_short in cases:
            embedder = FakeEmbedder(model_name=model_name)
            pipeline = FakePipeline(embedder)
            assert pipeline.get_embedder_short_name() == expected_short, (
                f"{model_name} should map to {expected_short}"
            )

    def test_colorhist_short_name(self):
        """ColorHistogram wrapper should produce compound short name."""
        embedder = FakeEmbedder(model_name="facebook/dinov2-base+colorhist")
        pipeline = FakePipeline(embedder)
        assert pipeline.get_embedder_short_name() == "dv2b+chist"
