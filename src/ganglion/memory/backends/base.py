"""Backend protocol for memory storage.

One protocol. Six methods. Replaces the old KnowledgeBackend (11 methods)
and PeerDiscovery (2 methods) with a single contract.
"""

from __future__ import annotations

from typing import Protocol

from ganglion.memory.types import Belief, Observation, Valence


class MemoryBackend(Protocol):
    """The only contract backends must satisfy.

    find_similar is the only method that needs domain-specific thought.
    Everything else is mechanical CRUD.
    """

    async def find_similar(
        self,
        observation: Observation,
        threshold: float = 0.85,
    ) -> Belief | None:
        """Find the existing belief most similar to this observation."""
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
