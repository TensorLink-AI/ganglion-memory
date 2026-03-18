"""JSON file backend for memory storage.

One file (beliefs.json) replaces patterns.json + antipatterns.json +
agent_designs.json. Good for development, single-bot deployments,
and as the building block for federated peer discovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ganglion.memory.types import Belief, Observation, Valence

logger = logging.getLogger(__name__)


class JsonMemoryBackend:
    """Single-file JSON storage for beliefs."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path = self.directory / "beliefs.json"
        self._next_id = 1
        self._cache: list[dict[str, Any]] | None = None

    def _load_sync(self) -> list[dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        if self._path.exists():
            try:
                data: list[dict[str, Any]] = json.loads(self._path.read_text())
                if data:
                    max_id = max(d.get("id", 0) for d in data)
                    self._next_id = max(self._next_id, max_id + 1)
                self._cache = data
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load beliefs: %s", e)
        self._cache = []
        return self._cache

    def _save_sync(self, data: list[dict[str, Any]]) -> None:
        self._cache = data
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def invalidate_cache(self) -> None:
        """Force reload from disk on next access. Useful for federation."""
        self._cache = None

    # -- Write operations --------------------------------------------------

    def _store_sync(self, belief: Belief) -> int:
        data = self._load_sync()
        bid = self._next_id
        self._next_id += 1
        belief.id = bid
        data.append(belief.to_dict())
        self._save_sync(data)
        return bid

    async def store(self, belief: Belief) -> None:
        belief.id = await asyncio.to_thread(self._store_sync, belief)

    def _update_sync(self, belief: Belief) -> None:
        if belief.id is None:
            raise ValueError("Cannot update a belief without an id")
        data = self._load_sync()
        for i, d in enumerate(data):
            if d.get("id") == belief.id:
                data[i] = belief.to_dict()
                self._save_sync(data)
                return
        raise ValueError(f"Belief id={belief.id} not found")

    async def update(self, belief: Belief) -> None:
        await asyncio.to_thread(self._update_sync, belief)

    def _remove_sync(self, belief: Belief) -> None:
        if belief.id is None:
            return
        data = self._load_sync()
        data = [d for d in data if d.get("id") != belief.id]
        self._save_sync(data)

    async def remove(self, belief: Belief) -> None:
        await asyncio.to_thread(self._remove_sync, belief)

    # -- Read operations ---------------------------------------------------

    def _find_similar_sync(
        self,
        observation: Observation,
        threshold: float = 0.85,
    ) -> Belief | None:
        from ganglion.memory.similarity import jaccard_similarity

        data = self._load_sync()
        best_match: Belief | None = None
        best_score = 0.0

        for d in data:
            if d.get("capability") != observation.capability:
                continue
            if d.get("superseded_by"):
                continue
            score = jaccard_similarity(observation.description, d.get("description", ""))
            if score >= threshold and score > best_score:
                best_score = score
                best_match = Belief.from_dict(d)

        return best_match

    async def find_similar(
        self,
        observation: Observation,
        threshold: float = 0.85,
    ) -> Belief | None:
        return await asyncio.to_thread(self._find_similar_sync, observation, threshold)

    def _query_sync(
        self,
        capability: str | None = None,
        valence: Valence | None = None,
        entities: tuple[str, ...] = (),
        exclude_source: str | None = None,
        tags: tuple[str, ...] = (),
        min_strength: float = 0.0,
        limit: int = 20,
    ) -> list[Belief]:
        data = self._load_sync()
        beliefs = [Belief.from_dict(d) for d in data]

        if capability:
            beliefs = [b for b in beliefs if b.capability == capability]
        if valence:
            beliefs = [b for b in beliefs if b.valence == valence]
        if entities:
            beliefs = [b for b in beliefs if any(e in b.entities for e in entities)]
        if exclude_source:
            beliefs = [b for b in beliefs if b.source != exclude_source]
        if tags:
            beliefs = [b for b in beliefs if any(t in b.tags for t in tags)]
        if min_strength > 0:
            beliefs = [b for b in beliefs if b.strength >= min_strength]

        beliefs.sort(key=lambda b: b.last_confirmed, reverse=True)
        return beliefs[:limit]

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
        return await asyncio.to_thread(
            self._query_sync, capability, valence, entities,
            exclude_source, tags, min_strength, limit,
        )

    def _all_beliefs_sync(self) -> list[Belief]:
        return [Belief.from_dict(d) for d in self._load_sync()]

    async def all_beliefs(self) -> list[Belief]:
        return await asyncio.to_thread(self._all_beliefs_sync)
