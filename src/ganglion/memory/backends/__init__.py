"""Backend protocol for memory storage.

One protocol. Backends store and retrieve Experience objects.
No find_similar heuristics — that's the Memory layer's job.

    class Backend(Protocol):
        store, update, delete, get,
        search_by_embedding, query, all, count
"""

from __future__ import annotations

from typing import Any, Protocol

from ganglion.memory.types import Experience


class Backend(Protocol):
    """The contract all backends must satisfy.

    Backends are dumb storage. They don't decide what's similar or
    what should be confirmed — they just persist and retrieve.
    """

    async def store(self, experience: Experience) -> int:
        """Persist a new experience. Returns the assigned ID."""
        ...

    async def update(self, experience: Experience) -> None:
        """Update an existing experience in place."""
        ...

    async def delete(self, experience_id: int) -> None:
        """Delete an experience by ID. No-op if not found."""
        ...

    async def get(self, experience_id: int) -> Experience | None:
        """Get a single experience by ID. Returns None if not found."""
        ...

    async def search_by_embedding(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.3,
        tags: tuple[str, ...] = (),
    ) -> list[tuple[Experience, float]]:
        """Find experiences similar to the embedding vector.

        Returns (experience, similarity_score) pairs sorted by
        score descending. Only returns pairs above threshold.
        """
        ...

    async def query(
        self,
        tags: tuple[str, ...] = (),
        source: str | None = None,
        limit: int = 20,
    ) -> list[Experience]:
        """Retrieve experiences matching filters, ordered by updated_at desc."""
        ...

    async def all(self) -> list[Experience]:
        """Return every experience. Used for compression scans."""
        ...

    async def count(self) -> int:
        """Return total number of stored experiences."""
        ...
