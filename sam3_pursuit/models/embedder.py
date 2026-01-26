from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from sam3_pursuit.config import Config

PATCH_SIZE = 14


def _resize_to_patch_multiple(image: Image.Image, target_size: int = 630) -> Image.Image:
    w, h = image.size
    if w >= h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    new_w = max(PATCH_SIZE, (new_w // PATCH_SIZE) * PATCH_SIZE)
    new_h = max(PATCH_SIZE, (new_h // PATCH_SIZE) * PATCH_SIZE)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


class FursuitEmbedder:
    def __init__(self, device: Optional[str] = None, model_name: str = Config.DINOV2_MODEL):
        self.device = device or Config.get_device()
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size

    def embed(self, image: Image.Image) -> np.ndarray:
        image = _resize_to_patch_multiple(image.convert("RGB"))
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.array([], dtype=np.float32)
        images = [_resize_to_patch_multiple(img.convert("RGB")) for img in images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype(np.float32)
