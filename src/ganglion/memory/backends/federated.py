"""Federated memory backend — write local, read from all peers.

Same concept as before: each bot writes to its own directory,
reads from everyone's. Now simpler because there's only one
type to merge instead of three.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ganglion.memory.backends.json_file import JsonMemoryBackend
from ganglion.memory.types import Belief, Observation, Valence

logger = logging.getLogger(__name__)


class FederatedMemoryBackend:
    """Write locally, read from all peers."""

    def __init__(self, local: JsonMemoryBackend, peers: PeerDiscovery):
        self.local = local
        self.peers = peers

    async def find_similar(
        self,
        observation: Observation,
        threshold: float = 0.75,
        embedding: list[float] | None = None,
    ) -> Belief | None:
        return await self.local.find_similar(observation, threshold, embedding)

    async def store(self, belief: Belief) -> None:
        await self.local.store(belief)

    async def update(self, belief: Belief) -> None:
        await self.local.update(belief)

    async def remove(self, belief: Belief) -> None:
        await self.local.remove(belief)

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
        local_results = await self.local.query(
            capability=capability, valence=valence, entities=entities,
            exclude_source=exclude_source, tags=tags,
            min_strength=min_strength, limit=limit,
        )
        try:
            peer_results = await self.peers.query_all(
                capability=capability, valence=valence, entities=entities,
                exclude_source=exclude_source, tags=tags,
                min_strength=min_strength, limit=limit,
            )
        except Exception as e:
            logger.warning("Failed to query peers: %s", e)
            peer_results = []

        merged = local_results + peer_results
        merged.sort(key=lambda b: b.strength, reverse=True)
        return merged[:limit]

    async def all_beliefs(self) -> list[Belief]:
        return await self.local.all_beliefs()


class PeerDiscovery:
    """Find and read from peer knowledge stores on the filesystem."""

    def __init__(self, base_dir: str | Path, local_bot_id: str):
        self.base_dir = Path(base_dir)
        self.local_bot_id = local_bot_id

    def _discover_peers(self) -> list[JsonMemoryBackend]:
        if not self.base_dir.is_dir():
            return []
        return [
            JsonMemoryBackend(child)
            for child in sorted(self.base_dir.iterdir())
            if child.is_dir() and child.name != self.local_bot_id
        ]

    async def query_all(
        self,
        capability: str | None = None,
        valence: Valence | None = None,
        entities: tuple[str, ...] = (),
        exclude_source: str | None = None,
        tags: tuple[str, ...] = (),
        min_strength: float = 0.0,
        limit: int = 20,
    ) -> list[Belief]:
        results: list[Belief] = []
        for peer in self._discover_peers():
            peer.invalidate_cache()
            try:
                results.extend(await peer.query(
                    capability=capability, valence=valence, entities=entities,
                    exclude_source=exclude_source, tags=tags,
                    min_strength=min_strength, limit=limit,
                ))
            except Exception as e:
                logger.warning("Failed to read from peer %s: %s", peer.directory, e)
        return results
