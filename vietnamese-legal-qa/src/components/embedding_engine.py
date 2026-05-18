"""Embedding Engine - Vector hóa text chunks."""

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.settings import Settings


class EmbeddingEngine:
    """Vector hóa text sử dụng PhoBERT hoặc multilingual-e5."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._models = {}

    def load_model(self, model_key: str) -> None:
        """
        Load embedding model.
        
        Args:
            model_key: 'phobert' hoặc 'multilingual_e5'
        """
        if model_key in self._models:
            return

        model_config = self.settings.get_embedding_model(model_key)
        model_name = model_config.get("name")

        print(f"[EmbeddingEngine] Loading model: {model_name}...")
        self._models[model_key] = SentenceTransformer(model_name)
        print(f"[EmbeddingEngine] Model loaded: {model_key}")

    def embed_chunks(self, texts: List[str], model_key: str = "multilingual_e5") -> np.ndarray:
        """
        Vector hóa danh sách text chunks.
        
        Args:
            texts: Danh sách text cần embed
            model_key: Key của model ('phobert' hoặc 'multilingual_e5')
            
        Returns:
            Numpy array of embeddings [n_texts, dimension]
        """
        self.load_model(model_key)
        model = self._models[model_key]

        # For E5 models, prepend "passage: " for documents
        if "e5" in model_key:
            texts = [f"passage: {t}" for t in texts]

        print(f"[EmbeddingEngine] Embedding {len(texts)} chunks with {model_key}...")
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )

        return embeddings

    def embed_query(self, query: str, model_key: str = "multilingual_e5") -> np.ndarray:
        """
        Vector hóa câu query cho retrieval.
        
        Args:
            query: Câu query
            model_key: Key của model
            
        Returns:
            Query embedding vector [dimension]
        """
        self.load_model(model_key)
        model = self._models[model_key]

        # For E5 models, prepend "query: " for queries
        if "e5" in model_key:
            query = f"query: {query}"

        embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )

        return embedding[0]

    def get_available_models(self) -> List[str]:
        """Get list of available embedding model keys."""
        return list(self.settings.embedding.get("models", {}).keys())

    def get_dimension(self, model_key: str) -> int:
        """Get embedding dimension for a model."""
        model_config = self.settings.get_embedding_model(model_key)
        return model_config.get("dimension", 768)
