"""Ganglion Memory — biological memory primitives for agents."""

from ganglion.memory.agent import MemoryAgent, between_runs, result_to_observation
from ganglion.memory.backends.base import MemoryBackend
from ganglion.memory.backends.federated import FederatedMemoryBackend, PeerDiscovery
from ganglion.memory.backends.json_file import JsonMemoryBackend
from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.cortex import spread_activation, temporal_neighbors
from ganglion.memory.loop import MemoryLoop
from ganglion.memory.similarity import jaccard_similarity, tokenize
from ganglion.memory.types import Belief, Delta, Observation, Valence
from ganglion.memory.wrap import memory

__all__ = [
    # Core
    "MemoryLoop",
    "Observation",
    "Belief",
    "Delta",
    "Valence",
    # Similarity
    "jaccard_similarity",
    "tokenize",
    # Cortex — advanced retrieval
    "spread_activation",
    "temporal_neighbors",
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
