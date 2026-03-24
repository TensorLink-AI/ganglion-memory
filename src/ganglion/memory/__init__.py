"""Ganglion Memory v4 — dumb memory, smart retrieval.

One type (Experience) replaces three (Observation, Belief, Delta).
No biological metaphors. No counterfactual evaluation. No confidence
scoring. Just store, retrieve, compress, and optional refine tools.
"""

from ganglion.memory.agent import Agent
from ganglion.memory.backends import Backend
from ganglion.memory.backends.sqlite import SqliteBackend
from ganglion.memory.core import Memory
from ganglion.memory.embed import (
    CallableEmbedder,
    Embedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    get_embedder,
    reset_embedder,
    set_embedder,
)
from ganglion.memory.refine import (
    forget_experience,
    merge_experiences,
    rewrite_experience,
    split_experience,
)
from ganglion.memory.types import Experience
from ganglion.memory.wrap import memory

__all__ = [
    # Core
    "Memory",
    "Experience",
    # Embeddings
    "Embedder",
    "SentenceTransformerEmbedder",
    "CallableEmbedder",
    "cosine_similarity",
    "get_embedder",
    "set_embedder",
    "reset_embedder",
    # Agent integration
    "Agent",
    # Backends
    "Backend",
    "SqliteBackend",
    # Refine tools (ReMem-style)
    "merge_experiences",
    "split_experience",
    "rewrite_experience",
    "forget_experience",
    # One-line wrapper
    "memory",
]
