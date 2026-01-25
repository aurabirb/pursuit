"""FAISS vector index for similarity search."""

import os

import faiss
import numpy as np

from sam3_pursuit.config import Config


class VectorIndex:
    """HNSW-based FAISS index for fast nearest neighbor search."""

    def __init__(
        self,
        index_path: str = Config.INDEX_PATH,
        embedding_dim: int = Config.EMBEDDING_DIM,
        hnsw_m: int = Config.HNSW_M,
        ef_construction: int = Config.HNSW_EF_CONSTRUCTION,
        ef_search: int = Config.HNSW_EF_SEARCH
    ):
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = self._load_or_create_index()

    def _load_or_create_index(self) -> faiss.Index:
        if os.path.exists(self.index_path):
            print(f"Loading index: {self.index_path}")
            index = faiss.read_index(self.index_path)
            print(f"Index loaded: {index.ntotal} vectors")
        else:
            print(f"Creating HNSW index (dim={self.embedding_dim})")
            index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
        return index

    def add(self, embeddings: np.ndarray) -> int:
        """Add embeddings, return starting ID."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = embeddings.astype(np.float32)
        start_id = self.index.ntotal
        self.index.add(embeddings)
        return start_id

    def search(self, query: np.ndarray, top_k: int = Config.DEFAULT_TOP_K) -> tuple[np.ndarray, np.ndarray]:
        """Search for similar embeddings. Returns (distances, indices)."""
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32)
        return self.index.search(query, top_k)

    def save(self):
        print(f"Saving index: {self.index_path}")
        faiss.write_index(self.index, self.index_path)

    @property
    def size(self) -> int:
        return self.index.ntotal

    def reset(self):
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
