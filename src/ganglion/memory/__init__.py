"""Ganglion Memory — biological memory primitives for agents."""

from ganglion.memory.agent import MemoryAgent, between_runs, result_to_observation
from ganglion.memory.backends.base import MemoryBackend
from ganglion.memory.backends.federated import FederatedMemoryBackend, PeerDiscovery
from ganglion.memory.backends.json_file import JsonMemoryBackend
from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.cortex import (
    assimilate_with_biology,
    between_runs_with_biology,
    compute_salience,
    consolidate,
    context_with_associations,
    inhibit_competitors,
    spread_activation,
    temporal_neighbors,
)
from ganglion.memory.loop import MemoryLoop
from ganglion.memory.similarity import jaccard_similarity, tokenize
from ganglion.memory.types import Belief, Delta, Observation, Valence

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
    # Cortex — biological extensions
    "spread_activation",
    "consolidate",
    "compute_salience",
    "inhibit_competitors",
    "temporal_neighbors",
    "assimilate_with_biology",
    "context_with_associations",
    "between_runs_with_biology",
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
]
