"""Ganglion Memory v3 — biological memory with counterfactual evaluation."""

from ganglion.memory.agent import MemoryAgent, between_runs, result_to_observation
from ganglion.memory.backends.base import MemoryBackend
from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.embed import (
    CallableEmbedder,
    Embedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    get_embedder,
    set_embedder,
)
from ganglion.memory.loop import MemoryLoop
from ganglion.memory.types import Belief, Delta, Observation, Valence
from ganglion.memory.wrap import memory

__all__ = [
    # Core
    "MemoryLoop",
    "Observation",
    "Belief",
    "Delta",
    "Valence",
    # Embeddings
    "Embedder",
    "SentenceTransformerEmbedder",
    "CallableEmbedder",
    "cosine_similarity",
    "get_embedder",
    "set_embedder",
    # Agent integration
    "MemoryAgent",
    "between_runs",
    "result_to_observation",
    # Backends
    "MemoryBackend",
    "SqliteMemoryBackend",
    # One-line wrapper
    "memory",
]
