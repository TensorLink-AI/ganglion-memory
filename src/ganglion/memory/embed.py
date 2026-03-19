"""Embedding protocol and implementations for semantic similarity.

Replaces Jaccard bag-of-words matching with vector embeddings.
Default: sentence-transformers (all-MiniLM-L6-v2, ~80MB).
Fallback: None (Jaccard similarity used when no embedder available).
"""

from __future__ import annotations

import logging
import math
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding text into vectors."""

    async def embed(self, text: str) -> list[float]:
        """Embed a single string into a vector."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings. Default: sequential calls to embed()."""
        ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SentenceTransformerEmbedder:
    """Local embedding using sentence-transformers.

    Loads the model lazily on first use. Thread-safe via asyncio.to_thread.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def _embed_sync(self, text: str) -> list[float]:
        model = self._get_model()
        return model.encode(text, show_progress_bar=False).tolist()

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        return [v.tolist() for v in model.encode(texts, show_progress_bar=False)]

    async def embed(self, text: str) -> list[float]:
        import asyncio
        return await asyncio.to_thread(self._embed_sync, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_batch_sync, texts)


class CallableEmbedder:
    """Wrap any async callable as an Embedder.

    Usage:
        embedder = CallableEmbedder(my_api_embed_function)
    """

    def __init__(self, fn):
        self._fn = fn

    async def embed(self, text: str) -> list[float]:
        return await self._fn(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self._fn(t) for t in texts]


_default_embedder: Embedder | None = None


def get_embedder() -> Embedder | None:
    """Get the global embedder, creating SentenceTransformerEmbedder if possible."""
    global _default_embedder
    if _default_embedder is not None:
        return _default_embedder
    try:
        _default_embedder = SentenceTransformerEmbedder()
        # Test that it can load
        _default_embedder._get_model()
        return _default_embedder
    except (ImportError, Exception) as e:
        logger.debug("No embedding model available: %s", e)
        return None


def set_embedder(embedder: Embedder | None) -> None:
    """Set the global embedder. Pass None to disable embeddings."""
    global _default_embedder
    _default_embedder = embedder


def reset_embedder() -> None:
    """Reset global embedder to None (for testing)."""
    global _default_embedder
    _default_embedder = None
