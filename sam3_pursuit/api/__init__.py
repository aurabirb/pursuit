"""API components for the SAM3 fursuit recognition system."""

from sam3_pursuit.api.annotator import annotate_image
from sam3_pursuit.api.identifier import FursuitIdentifier, IdentificationResult, detect_embedder, build_embedder_for_name
from sam3_pursuit.api.ingestor import FursuitIngestor

__all__ = ["FursuitIdentifier", "FursuitIngestor", "IdentificationResult", "annotate_image", "detect_embedder", "build_embedder_for_name"]
