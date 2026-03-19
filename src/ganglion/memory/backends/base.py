"""Backend protocol for memory storage.

One protocol. Six methods. The embedding field on Belief is stored and
returned transparently — backends that support it store embedding blobs,
others ignore it.
"""

from __future__ import annotations

from typing import Protocol

from ganglion.memory.types import Belief, Observation, Valence


class MemoryBackend(Protocol):
    """The only contract backends must satisfy.

    find_similar uses cosine similarity on embeddings when available,
    falling back to Jaccard token similarity.
    """

    async def find_similar(
        self,
        observation: Observation,
        threshold: float = 0.75,
        embedding: list[float] | None = None,
    ) -> Belief | None:
        """Find the existing belief most similar to this observation.

        If embedding is provided, uses cosine similarity.
        Falls back to Jaccard on description text.
        """
        ...

    async def store(self, belief: Belief) -> None:
        """Persist a new belief."""
        ...

    async def update(self, belief: Belief) -> None:
        """Update an existing belief in place."""
        ...

    async def remove(self, belief: Belief) -> None:
        """Delete a belief."""
        ...

    async def query(
        self,
        capability: str | None = None,
        valence: Valence | None = None,
        entities: tuple[str, ...] = (),
        exclude_source: str | None = None,
        tags: tuple[str, ...] = (),
        min_strength: float = 0.0,
        limit: int = 20,
    ) -> list[Belief]:
        """Retrieve beliefs matching the given filters."""
        ...

    async def all_beliefs(self) -> list[Belief]:
        """Return every belief. Used only for eviction scans."""
        ...
