"""SAM3 Fursuit Recognition System

A fursuit character recognition system using SAM3 for detection/segmentation
and DINOv2 for embedding generation.
"""

from sam3_pursuit.api.identifier import (
    SAM3FursuitIdentifier,
    IdentificationResult,
    SegmentResults,
)
from sam3_pursuit.config import Config

__version__ = "1.0.0"
__all__ = ["SAM3FursuitIdentifier", "IdentificationResult", "SegmentResults", "Config"]
