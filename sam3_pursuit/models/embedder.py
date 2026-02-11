from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from sam3_pursuit.config import Config


class DINOv2Embedder:
    def __init__(self, device: Optional[str] = None, model_name: str = Config.DINOV2_MODEL):
        self.device = device or Config.get_device()
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size

    def embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()


class CLIPEmbedder:
    def __init__(self, device: Optional[str] = None, model_name: str = Config.CLIP_MODEL):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or Config.get_device()
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(
            model_name, revision=Config.CLIP_MODEL_REVISION,
        )
        self.model = CLIPModel.from_pretrained(
            model_name, revision=Config.CLIP_MODEL_REVISION,
        ).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.projection_dim  # 512 for ViT-B/32

    def embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            pooled = vision_outputs.pooler_output
            projected = self.model.visual_projection(pooled)
            features = projected / projected.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().astype(np.float32)

class SigLIPEmbedder:
    def __init__(self, device: Optional[str] = None, model_name: str = Config.SIGLIP_MODEL):
        self.device = device or Config.get_device()
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.vision_config.hidden_size

    def embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.vision_model(**inputs)
            embedding = outputs.pooler_output
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    def embed_text(self, text: str) -> np.ndarray:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = tokenizer([text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().astype(np.float32)


class ColorHistogramEmbedder:
    """Wraps any embedder and appends a normalized HSV color histogram."""

    def __init__(self, base_embedder, bins: int = Config.COLOR_HIST_BINS):
        self.base_embedder = base_embedder
        self.bins = bins
        self.model_name = f"{base_embedder.model_name}+colorhist"
        self.embedding_dim = base_embedder.embedding_dim + bins

    def embed(self, image: Image.Image) -> np.ndarray:
        base_emb = self.base_embedder.embed(image)
        hist = self._compute_hsv_histogram(image)
        combined = np.concatenate([base_emb, hist])
        combined = combined / np.linalg.norm(combined)
        return combined.astype(np.float32)

    def _compute_hsv_histogram(self, image: Image.Image) -> np.ndarray:
        hsv = image.convert("HSV")
        h_channel = np.array(hsv.getchannel("H")).flatten()
        hist, _ = np.histogram(h_channel, bins=self.bins, range=(0, 256))
        hist = hist.astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        return hist
