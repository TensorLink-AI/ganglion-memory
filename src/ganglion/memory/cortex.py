"""
cortex.py — Five biological gaps, one module.

Adds to the existing MemoryLoop without replacing it:

1. ASSOCIATIVE RETRIEVAL  — spread activation through shared entities/tags
2. CONSOLIDATION          — merge clusters of similar beliefs into meta-beliefs
3. SALIENCE               — metric surprise gates initial encoding strength
4. INHIBITION             — strengthening one belief weakens its competitors
5. TEMPORAL CONTEXT       — beliefs remember their neighbors via time window
"""

from __future__ import annotations

import logging
import math
from datetime import timedelta

from ganglion.memory.backends.base import MemoryBackend
from ganglion.memory.types import Belief, Delta, Observation, Valence

logger = logging.getLogger(__name__)


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
# 2. CONSOLIDATION — episodic → semantic compression
# =====================================================================

async def consolidate(
    backend: MemoryBackend,
    min_cluster_size: int = 3,
    similarity_threshold: float = 0.5,
) -> int:
    """Merge clusters of similar beliefs into meta-beliefs.

    Returns the number of beliefs removed (replaced by consolidations).
    """
    all_beliefs = await backend.all_beliefs()
    if len(all_beliefs) < min_cluster_size:
        return 0

    groups: dict[tuple[str, Valence], list[Belief]] = {}
    for b in all_beliefs:
        key = (b.capability, b.valence)
        groups.setdefault(key, []).append(b)

    removed = 0
    for group in groups.values():
        if len(group) < min_cluster_size:
            continue
        clusters = _cluster_by_overlap(group, similarity_threshold)
        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue
            merged = _merge_cluster(cluster)
            for old in cluster:
                await backend.remove(old)
                removed += 1
            await backend.store(merged)

    return removed


def _cluster_by_overlap(beliefs: list[Belief], threshold: float) -> list[list[Belief]]:
    """Seed-anchored clustering — no feature expansion, no chaining."""
    assigned: set[int | None] = set()
    clusters: list[list[Belief]] = []

    for b in beliefs:
        if b.id in assigned:
            continue
        cluster = [b]
        assigned.add(b.id)
        seed_features = set(b.entities) | set(b.tags)
        if not seed_features:
            continue

        for other in beliefs:
            if other.id in assigned:
                continue
            other_features = set(other.entities) | set(other.tags)
            if not other_features:
                continue
            jaccard = len(seed_features & other_features) / len(seed_features | other_features)
            if jaccard >= threshold:
                cluster.append(other)
                assigned.add(other.id)

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters


def _merge_cluster(cluster: list[Belief]) -> Belief:
    """Merge a cluster into one consolidated meta-belief."""
    cluster.sort(key=lambda b: b.strength, reverse=True)
    strongest = cluster[0]

    total_confirmations = sum(b.confirmation_count for b in cluster)
    avg_confidence = sum(b.confidence for b in cluster) / len(cluster)
    metric_values = [b.metric_value for b in cluster if b.metric_value is not None]
    avg_metric = sum(metric_values) / len(metric_values) if metric_values else None

    all_entities = sorted({e for b in cluster for e in b.entities})
    all_tags = sorted({t for b in cluster for t in b.tags} | {"consolidated"})

    return Belief(
        capability=strongest.capability,
        description=strongest.description,
        valence=strongest.valence,
        confidence=avg_confidence,
        confirmation_count=total_confirmations,
        entities=tuple(all_entities),
        config=strongest.config,
        metric_name=strongest.metric_name,
        metric_value=avg_metric,
        source=strongest.source,
        first_seen=min(b.first_seen for b in cluster),
        last_confirmed=max(b.last_confirmed for b in cluster),
        tags=tuple(all_tags),
    )


# =====================================================================
# 3. SALIENCE — surprise gates encoding strength
# =====================================================================

async def compute_salience(
    obs: Observation,
    backend: MemoryBackend,
    baseline_confidence: float = 1.0,
    max_boost: float = 3.0,
) -> float:
    """Compute initial encoding strength from metric surprise."""
    if obs.metric_value is None:
        return baseline_confidence

    peers = await backend.query(capability=obs.capability, limit=100)
    peer_metrics = [b.metric_value for b in peers if b.metric_value is not None]
    if len(peer_metrics) < 2:
        return baseline_confidence

    mean = sum(peer_metrics) / len(peer_metrics)
    variance = sum((m - mean) ** 2 for m in peer_metrics) / len(peer_metrics)
    std = math.sqrt(variance) if variance > 0 else 1.0

    z = abs(obs.metric_value - mean) / std
    boost = 1.0 + (max_boost - 1.0) * min(z / 2.0, 1.0)
    return baseline_confidence * boost


# =====================================================================
# 4. INHIBITION — lateral inhibition between competitors
# =====================================================================

async def inhibit_competitors(
    strengthened: Belief,
    backend: MemoryBackend,
    inhibition_rate: float = 0.05,
    confidence_floor: float = 0.2,
) -> int:
    """Weaken beliefs that compete with the strengthened one.

    A competitor is a belief with same capability, overlapping
    entities/tags, but DIFFERENT description (different strategy).
    Same-description beliefs are allies, not competitors.

    Confidence is floored to prevent death by inhibition alone —
    only direct contradiction can kill a belief.
    """
    competitors = await backend.query(
        capability=strengthened.capability,
        limit=50,
    )

    inhibited = 0
    strengthened_features = set(strengthened.entities) | set(strengthened.tags)
    if not strengthened_features:
        return 0

    for comp in competitors:
        if comp.id == strengthened.id:
            continue
        if comp.description == strengthened.description:
            continue

        comp_features = set(comp.entities) | set(comp.tags)
        if not comp_features:
            continue

        overlap = len(strengthened_features & comp_features) / len(
            strengthened_features | comp_features
        )
        if overlap < 0.2:
            continue

        reduction = inhibition_rate * overlap
        comp.confidence = max(confidence_floor, comp.confidence - reduction)
        await backend.update(comp)
        inhibited += 1

    return inhibited


# =====================================================================
# 5. TEMPORAL CONTEXT — episode linking
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


# =====================================================================
# Wiring into the MemoryLoop
# =====================================================================

async def assimilate_with_biology(
    loop,  # MemoryLoop
    obs: Observation,
) -> Delta | None:
    """Drop-in replacement for loop.assimilate() with salience + inhibition.

    Correctly distinguishes novel observations (apply salience) from
    agreement with existing beliefs (preserve Hebbian strength).
    """
    existing_before = await loop.backend.find_similar(obs)
    is_novel = existing_before is None

    salience = await compute_salience(obs, loop.backend) if is_novel else 1.0

    delta = await loop.assimilate(obs)

    if is_novel and salience > 1.0:
        created = await loop.backend.find_similar(obs)
        if created is not None:
            created.confidence = salience
            await loop.backend.update(created)

    # Only inhibit competitors when the observation reinforced a belief
    # (agreement or novel). Contradiction weakens the existing belief —
    # inhibiting its competitors would punish rivals of the loser.
    if delta is None or delta.delta_type == "metric_shift":
        target = await loop.backend.find_similar(obs)
        if target is not None:
            await inhibit_competitors(target, loop.backend)

    return delta


async def context_with_associations(
    loop,  # MemoryLoop
    capability: str,
    entities: tuple[str, ...] = (),
    exclude_source: str | None = None,
    tags: tuple[str, ...] = (),
    max_entries: int = 10,
) -> str:
    """Enhanced context_for with associative retrieval."""
    base_context = await loop.context_for(
        capability=capability,
        entities=entities,
        exclude_source=exclude_source,
        tags=tags,
        max_entries=max_entries,
    )

    if not base_context:
        return ""

    beliefs = await loop.backend.query(
        capability=capability,
        entities=entities,
        exclude_source=exclude_source,
        tags=tags,
        limit=3,
    )

    if not beliefs:
        return base_context

    associated: list[Belief] = []
    seen_ids = {b.id for b in beliefs}
    for seed in beliefs[:2]:
        neighbors = await spread_activation(seed, loop.backend, max_hops=1, limit=5)
        for n in neighbors:
            if n.id not in seen_ids:
                associated.append(n)
                seen_ids.add(n.id)

    if not associated:
        return base_context

    associated.sort(key=lambda b: b.strength, reverse=True)
    associated = associated[:5]

    lines = ["\n### Related knowledge"]
    for b in associated:
        prefix = "+" if b.is_pattern else "-" if b.is_antipattern else "~"
        metric = f" ({b.metric_name}={b.metric_value})" if b.metric_value else ""
        via = ", ".join(sorted(set(b.entities) | set(b.tags)))[:60]
        lines.append(f"  {prefix} {b.description}{metric} (via {via})")

    return base_context + "\n" + "\n".join(lines)


# =====================================================================
# Enhanced between_runs — wires consolidation into the sleep boundary
# =====================================================================

async def between_runs_with_biology(
    loop,  # MemoryLoop
    consolidation_threshold: int = 50,
) -> tuple[list[Delta], int, int]:
    """Enhanced between_runs: drain → consolidate → forget. One pass each.

    Ordering matters biologically:
    1. Drain deltas (process signals from the active phase)
    2. Consolidate (compress episodes into schemas — the sleep phase)
    3. Forget (evict the weakest — after compression, not before)
    """
    # 1. Drain signals
    deltas = await loop.drain_deltas()
    if deltas:
        logger.info("Memory shifts this run: %d", len(deltas))
        for d in deltas:
            logger.info("  %s", d.summary)

    # 2. Consolidate if store is large enough
    consolidated = 0
    summary = await loop.summary()
    if summary["total_beliefs"] > consolidation_threshold:
        consolidated = await consolidate(loop.backend)
        if consolidated:
            logger.info("Consolidated %d beliefs during sleep phase", consolidated)

    # 3. Forget weakest (one pass, after consolidation)
    forgotten = await loop.forget()
    if forgotten:
        logger.info("Forgot %d weak beliefs", forgotten)

    return deltas, consolidated, forgotten
