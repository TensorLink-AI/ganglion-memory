"""Memory — store, retrieve, compress. That's it.

    mem = Memory(backend=SqliteBackend("memory.db"))

    exp = await mem.add("batch_size=64 works well", tags=("mining",))
    results = await mem.search("what batch size?", tags=("mining",))
    await mem.confirm(exp.id)
    await mem.compress(tags=("mining",))

No Hebbian strengthening. No lateral inhibition. No crisis detection.
No confidence scoring. Just store, search, confirm/contradict, compress.

Let the LLM decide what matters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

from ganglion.memory.backends import Backend
from ganglion.memory.embed import Embedder, cosine_similarity
from ganglion.memory.types import Experience

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Dumb memory, smart retrieval.

    Stores experiences with optional embeddings. Retrieves by semantic
    similarity or tag filters. Compresses clusters of similar experiences
    into merged summaries (optionally using an LLM synthesizer).

    Attributes:
        backend:              Storage backend (SqliteBackend, etc.).
        embedder:             Optional embedder for semantic search.
        similarity_threshold: Minimum cosine similarity for search results.
        dedup_threshold:      Above this similarity, add() confirms existing
                              instead of creating a new experience.
        max_results:          Default limit for search results.
    """

    backend: Backend
    embedder: Embedder | None = None

    # Retrieval tuning
    similarity_threshold: float = 0.3
    dedup_threshold: float = 0.85
    max_results: int = 20

    # -- Embedding helper ----------------------------------------------------

    async def _embed(self, text: str) -> list[float] | None:
        """Embed text using the configured embedder, or return None."""
        if self.embedder is None:
            return None
        try:
            return await self.embedder.embed(text)
        except Exception as e:
            logger.debug("Embedding failed: %s", e)
            return None

    # -- Write operations ----------------------------------------------------

    async def add(
        self,
        content: str,
        *,
        tags: tuple[str, ...] = (),
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Experience:
        """Store a new experience.

        If an existing experience with very similar content (above
        dedup_threshold) exists in the same tag space, confirms it
        instead of creating a duplicate.

        Returns the created or confirmed Experience.
        """
        embedding = await self._embed(content)

        # Check for near-duplicate via embedding
        if embedding is not None:
            existing = await self.backend.search_by_embedding(
                embedding, limit=1, threshold=self.dedup_threshold, tags=tags,
            )
            if existing:
                exp, _score = existing[0]
                exp.confirmation_count += 1
                exp.updated_at = datetime.now(UTC)
                # Merge tags (union, sorted for determinism)
                merged_tags = set(exp.tags) | set(tags)
                exp.tags = tuple(sorted(merged_tags))
                exp.embedding = embedding
                await self.backend.update(exp)
                return exp

        # No duplicate — create new
        now = datetime.now(UTC)
        exp = Experience(
            content=content,
            tags=tags,
            source=source,
            created_at=now,
            updated_at=now,
            confirmation_count=1,
            contradiction_count=0,
            embedding=embedding,
            metadata=metadata,
        )
        await self.backend.store(exp)
        return exp

    async def get(self, experience_id: int) -> Experience | None:
        """Get a single experience by ID."""
        return await self.backend.get(experience_id)

    async def confirm(self, experience_id: int) -> Experience:
        """Increment confirmation count on an experience.

        Raises ValueError if the experience doesn't exist.
        """
        exp = await self.backend.get(experience_id)
        if exp is None:
            raise ValueError(f"Experience {experience_id} not found")
        exp.confirmation_count += 1
        exp.updated_at = datetime.now(UTC)
        await self.backend.update(exp)
        return exp

    async def contradict(self, experience_id: int) -> Experience:
        """Increment contradiction count on an experience.

        Raises ValueError if the experience doesn't exist.
        """
        exp = await self.backend.get(experience_id)
        if exp is None:
            raise ValueError(f"Experience {experience_id} not found")
        exp.contradiction_count += 1
        exp.updated_at = datetime.now(UTC)
        await self.backend.update(exp)
        return exp

    async def delete(self, experience_id: int) -> None:
        """Remove an experience by ID."""
        await self.backend.delete(experience_id)

    # -- Read operations -----------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        tags: tuple[str, ...] = (),
        limit: int | None = None,
        threshold: float | None = None,
    ) -> list[Experience]:
        """Search by semantic similarity.

        When an embedder is available, embeds the query and searches by
        cosine similarity. Falls back to tag-based query otherwise.

        Returns experiences sorted by relevance (most relevant first).
        """
        limit = limit or self.max_results
        threshold = threshold or self.similarity_threshold

        embedding = await self._embed(query)
        if embedding is not None:
            results = await self.backend.search_by_embedding(
                embedding, limit=limit, threshold=threshold, tags=tags,
            )
            return [exp for exp, _score in results]

        # Fallback: tag-based query (no embedding available)
        return await self.backend.query(tags=tags, limit=limit)

    async def all(self) -> list[Experience]:
        """Return all stored experiences."""
        return await self.backend.all()

    async def count(self) -> int:
        """Total number of stored experiences."""
        return await self.backend.count()

    # -- Compression ---------------------------------------------------------

    async def compress(
        self,
        *,
        tags: tuple[str, ...] = (),
        min_cluster: int = 3,
        threshold: float = 0.7,
        synthesizer: Callable[[list[Experience]], Awaitable[str]] | None = None,
    ) -> list[Experience]:
        """Merge clusters of similar experiences.

        Finds groups of experiences with embedding similarity above threshold,
        merges each group into a single experience. Optionally uses an LLM
        synthesizer to generate the merged content; without one, keeps the
        content of the highest-scored experience in the cluster.

        Args:
            tags:        Only consider experiences with these tags.
            min_cluster: Minimum cluster size to trigger merging.
            threshold:   Cosine similarity threshold for clustering.
            synthesizer: Optional async function that takes a list of
                         experiences and returns synthesized content.

        Returns:
            List of newly created merged experiences.
        """
        if tags:
            experiences = await self.backend.query(tags=tags, limit=10000)
        else:
            experiences = await self.backend.all()

        if len(experiences) < min_cluster:
            return []

        clusters = self._find_clusters(experiences, threshold, min_cluster)

        merged: list[Experience] = []
        for cluster in clusters:
            new_exp = await self._merge_cluster(cluster, synthesizer)
            if new_exp is not None:
                merged.append(new_exp)

        return merged

    def _find_clusters(
        self,
        experiences: list[Experience],
        threshold: float,
        min_size: int,
    ) -> list[list[Experience]]:
        """Group experiences by embedding similarity (greedy seed-anchored)."""
        assigned: set[int | None] = set()
        clusters: list[list[Experience]] = []

        for exp in experiences:
            if exp.id in assigned or exp.embedding is None:
                continue
            cluster = [exp]
            assigned.add(exp.id)

            for other in experiences:
                if other.id in assigned or other.embedding is None:
                    continue
                sim = cosine_similarity(exp.embedding, other.embedding)
                if sim >= threshold:
                    cluster.append(other)
                    assigned.add(other.id)

            if len(cluster) >= min_size:
                clusters.append(cluster)

        return clusters

    async def _merge_cluster(
        self,
        cluster: list[Experience],
        synthesizer: Callable[[list[Experience]], Awaitable[str]] | None = None,
    ) -> Experience | None:
        """Merge a cluster into one experience, deleting the originals."""
        if not cluster:
            return None

        # Generate merged content
        if synthesizer:
            content = await synthesizer(cluster)
        else:
            # Keep the best content (highest net score)
            cluster.sort(key=lambda e: e.net_score, reverse=True)
            content = cluster[0].content

        # Aggregate counts and tags
        total_confirms = sum(e.confirmation_count for e in cluster)
        total_contradicts = sum(e.contradiction_count for e in cluster)
        all_tags = sorted({t for e in cluster for t in e.tags} | {"compressed"})
        source = cluster[0].source
        earliest = min(e.created_at for e in cluster)

        # Delete originals
        for exp in cluster:
            if exp.id is not None:
                await self.backend.delete(exp.id)

        # Store merged
        embedding = await self._embed(content)
        now = datetime.now(UTC)
        merged = Experience(
            content=content,
            tags=tuple(all_tags),
            source=source,
            created_at=earliest,
            updated_at=now,
            confirmation_count=total_confirms,
            contradiction_count=total_contradicts,
            embedding=embedding,
        )
        await self.backend.store(merged)
        return merged
