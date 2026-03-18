"""JSON file backend for memory storage.

One file (beliefs.json) replaces patterns.json + antipatterns.json +
agent_designs.json. Good for development, single-bot deployments,
and as the building block for federated peer discovery.
"""

from __future__ import annotations

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

    def _load(self) -> list[dict[str, Any]]:
        if self._path.exists():
            try:
                data: list[dict[str, Any]] = json.loads(self._path.read_text())
                # Track max id for auto-increment
                if data:
                    max_id = max(d.get("id", 0) for d in data)
                    self._next_id = max(self._next_id, max_id + 1)
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load beliefs: %s", e)
        return []

    def _save(self, data: list[dict[str, Any]]) -> None:
        self._path.write_text(json.dumps(data, indent=2, default=str))

    # -- Write operations --------------------------------------------------

    async def store(self, belief: Belief) -> None:
        data = self._load()
        belief.id = self._next_id
        self._next_id += 1
        data.append(belief.to_dict())
        self._save(data)

    async def update(self, belief: Belief) -> None:
        if belief.id is None:
            raise ValueError("Cannot update a belief without an id")
        data = self._load()
        for i, d in enumerate(data):
            if d.get("id") == belief.id:
                data[i] = belief.to_dict()
                self._save(data)
                return
        raise ValueError(f"Belief id={belief.id} not found")

    async def remove(self, belief: Belief) -> None:
        if belief.id is None:
            return
        data = self._load()
        data = [d for d in data if d.get("id") != belief.id]
        self._save(data)

    # -- Read operations ---------------------------------------------------

    async def find_similar(
        self,
        observation: Observation,
        threshold: float = 0.85,
    ) -> Belief | None:
        """Exact match on (capability, description prefix)."""
        data = self._load()
        desc_prefix = observation.description[:200]
        for d in data:
            if (
                d.get("capability") == observation.capability
                and d.get("description", "").startswith(desc_prefix)
            ):
                return Belief.from_dict(d)
        return None

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
        data = self._load()
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

    async def all_beliefs(self) -> list[Belief]:
        return [Belief.from_dict(d) for d in self._load()]
