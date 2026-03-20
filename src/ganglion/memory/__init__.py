"""Ganglion Memory — biological memory primitives for agents.

v2: Embedding-based similarity, LLM reflection, query-aware retrieval,
    cross-belief synthesis.
"""

from ganglion.memory.agent import MemoryAgent, between_runs, result_to_observation
from ganglion.memory.backends.base import MemoryBackend
from ganglion.memory.backends.federated import FederatedMemoryBackend, PeerDiscovery
from ganglion.memory.backends.json_file import JsonMemoryBackend
from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.cortex import spread_activation, temporal_neighbors
from ganglion.memory.embed import (
    CallableEmbedder,
    Embedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    get_embedder,
    set_embedder,
)
from ganglion.memory.loop import MemoryLoop
from ganglion.memory.reflect import reflect, reflect_experience
from ganglion.memory.similarity import jaccard_similarity, tokenize
from ganglion.memory.synthesize import synthesize
from ganglion.memory.types import Belief, Delta, Experience, Observation, Valence
from ganglion.memory.wrap import memory

__all__ = [
    # Core
    "MemoryLoop",
    "Observation",
    "Belief",
    "Delta",
    "Valence",
    "Experience",
    # Similarity
    "jaccard_similarity",
    "tokenize",
    "cosine_similarity",
    # Embeddings
    "Embedder",
    "SentenceTransformerEmbedder",
    "CallableEmbedder",
    "get_embedder",
    "set_embedder",
    # Cortex — advanced retrieval
    "spread_activation",
    "temporal_neighbors",
    # Intelligence layer
    "reflect",
    "reflect_experience",
    "synthesize",
    # Agent integration
    "MemoryAgent",
    "between_runs",
    "result_to_observation",
    # Backends
    "MemoryBackend",
    "SqliteMemoryBackend",
    "JsonMemoryBackend",
    "FederatedMemoryBackend",
    "PeerDiscovery",
    # One-line wrapper
    "memory",
]
