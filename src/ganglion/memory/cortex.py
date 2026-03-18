"""
cortex.py — Advanced retrieval capabilities.

Two query functions that compose on top of the existing MemoryLoop:

1. ASSOCIATIVE RETRIEVAL  — spread activation through shared entities/tags
2. TEMPORAL CONTEXT       — beliefs remember their neighbors via time window
"""

from __future__ import annotations

from datetime import timedelta

from ganglion.memory.backends.base import MemoryBackend
from ganglion.memory.types import Belief


# =====================================================================
# 1. ASSOCIATIVE RETRIEVAL — spreading activation
# =====================================================================

async def spread_activation(
    seed: Belief,
    backend: MemoryBackend,
    max_hops: int = 1,
    limit: int = 10,
) -> list[Belief]:
    """Retrieve beliefs associated with the seed through shared entities/tags.

    Returns beliefs ranked by activation strength across ALL hops,
    not just the final hop. The seed itself is excluded.
    """
    seen_ids: set[int | None] = {seed.id}
    scores: dict[int | None, float] = {}
    belief_map: dict[int | None, Belief] = {}
    frontier = [seed]
    decay = 1.0

    for _hop in range(max_hops + 1):
        frontier = list({b.id: b for b in frontier}.values())
        next_frontier: list[Belief] = []

        all_entities: set[str] = set()
        all_tags: set[str] = set()
        for source in frontier:
            all_entities.update(source.entities)
            all_tags.update(source.tags)

        neighbors: list[Belief] = []
        for entity in all_entities:
            neighbors.extend(await backend.query(entities=(entity,), limit=20))
        for tag in all_tags:
            neighbors.extend(await backend.query(tags=(tag,), limit=20))

        for neighbor in neighbors:
            if neighbor.id == seed.id:
                continue

            overlap = (
                len(all_entities & set(neighbor.entities))
                + len(all_tags & set(neighbor.tags))
            )
            score = overlap * neighbor.strength * decay

            # Always update score (takes max) — even if seen before in this hop
            old = scores.get(neighbor.id, 0.0)
            if score > old:
                scores[neighbor.id] = score
                belief_map[neighbor.id] = neighbor

            # But only expand frontier once per belief
            if neighbor.id not in seen_ids:
                next_frontier.append(neighbor)
                seen_ids.add(neighbor.id)

        frontier = next_frontier
        decay *= 0.5

    ranked_ids = sorted(scores.keys(), key=lambda bid: scores.get(bid, 0), reverse=True)
    result: list[Belief] = []
    for bid in ranked_ids:
        if bid in belief_map:
            result.append(belief_map[bid])
        if len(result) >= limit:
            break
    return result


# =====================================================================
# 2. TEMPORAL CONTEXT — episode linking
# =====================================================================

async def temporal_neighbors(
    belief: Belief,
    backend: MemoryBackend,
    window: timedelta = timedelta(hours=1),
    limit: int = 10,
) -> list[Belief]:
    """Find beliefs active around the same time. Scoped query, not all_beliefs."""
    candidates = await backend.query(
        capability=belief.capability,
        limit=200,
    )

    t = belief.last_confirmed
    window_start = t - window
    window_end = t + window

    neighbors = [
        b for b in candidates
        if b.id != belief.id
        and window_start <= b.last_confirmed <= window_end
    ]
    neighbors.sort(key=lambda b: abs((b.last_confirmed - t).total_seconds()))
    return neighbors[:limit]
