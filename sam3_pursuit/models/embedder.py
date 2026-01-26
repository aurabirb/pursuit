"""DINOv2-based embedding generation."""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from sam3_pursuit.config import Config


class FursuitEmbedder:
    """DINOv2 embeddings for visual similarity search."""

    def __init__(self, device: Optional[str] = None, model_name: str = Config.DINOV2_MODEL):
        self.device = device or Config.get_device()
        self.model_name = model_name

        print(f"Loading DINOv2: {model_name} on {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        print(f"DINOv2 loaded. Dim: {self.embedding_dim}")

    def embed(self, image: Image.Image) -> np.ndarray:
        """Generate L2-normalized embedding for an image."""
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().flatten()

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Generate embeddings for a batch of images."""
        if not images:
            return np.array([], dtype=np.float32)

        images = [img.convert("RGB") for img in images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy().astype(np.float32)
