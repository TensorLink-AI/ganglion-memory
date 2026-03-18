"""
MemoryLoop — the single primitive.

Replaces KnowledgeStore (record_success, record_failure, record_agent_design,
to_prompt_context, to_foreign_prompt_context, trim) with one method: assimilate().

    for each observation:
        memory.assimilate(observation)

Everything else — strengthening, weakening, contradiction detection,
entity profiling, eviction — happens inside that single call or
on the read path.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import random
from dataclasses import dataclass, field

from ganglion.memory.backends.base import MemoryBackend
from ganglion.memory.types import Belief, Delta, Observation, Valence

logger = logging.getLogger(__name__)


def _stable_merge(existing: tuple[str, ...], new: tuple[str, ...]) -> tuple[str, ...]:
    """Merge tuples preserving order of existing, appending new unseen items."""
    seen = set(existing)
    merged = list(existing)
    for item in new:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return tuple(merged)


@dataclass
class MemoryLoop:
    """One loop. Five contexts. No special cases."""

    backend: MemoryBackend

    # Tuning knobs
    strengthen_rate: float = 0.1
    weaken_rate: float = 0.3
    death_threshold: float = 0.1
    metric_shift_threshold: float = 0.15
    max_beliefs: int = 1000

    # Salience: surprise-gated encoding
    salience: bool = True

    # Lateral inhibition on agreement
    inhibition_rate: float = 0.05
    inhibition_floor: float = 0.2

    # Cross-agent confirmation weighting
    cross_agent_bonus: float = 2.0

    # Exploration pressure in context_for()
    exploration_rate: float = 0.0

    # Consolidation Jaccard threshold for forget()
    consolidation_threshold: float = 0.5

    _pending_deltas: list[Delta] = field(default_factory=list)
    _delta_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ------------------------------------------------------------------
    # Write path: the single primitive
    # ------------------------------------------------------------------

    async def assimilate(self, obs: Observation) -> Delta | None:
        """Observe → Compare → Update.

        The ONLY write path into memory. Returns a Delta if something
        meaningful changed, None otherwise.
        """
        existing = await self.backend.find_similar(obs)

        # Strategy bundling: include run tag when run_id is present
        run_tags = (f"run:{obs.run_id}",) if obs.run_id else ()
        obs_tags = obs.tags + run_tags

        if existing is None:
            # Novel observation — compute salience for initial confidence
            confidence = await self._compute_salience(obs) if self.salience else 1.0
            await self.backend.store(Belief(
                capability=obs.capability,
                description=obs.description,
                valence=obs.valence,
                confidence=confidence,
                confirmation_count=1,
                entities=obs.entities,
                config=obs.config,
                metric_name=obs.metric_name,
                metric_value=obs.metric_value,
                source=obs.source,
                tags=obs_tags,
                first_seen=obs.timestamp,
                last_confirmed=obs.timestamp,
            ))
            return None

        # Agreement: same valence
        if obs.valence == existing.valence:
            delta = self._check_metric_shift(existing, obs)

            if delta is None:
                # Pure agreement — strengthen
                # Cross-agent confirmation is worth more than self-confirmation
                rate = self.strengthen_rate
                if (
                    self.cross_agent_bonus > 1.0
                    and obs.source
                    and existing.source
                    and obs.source != existing.source
                ):
                    rate *= self.cross_agent_bonus

                existing.confidence = min(10.0, existing.confidence + rate)
                existing.confirmation_count += 1
            else:
                # Metric shifted significantly — don't strengthen
                # Optionally weaken proportionally if the shift is negative
                if (obs.metric_value is not None and existing.metric_value is not None
                        and obs.metric_value < existing.metric_value):
                    existing.confidence = max(
                        self.death_threshold,
                        existing.confidence - self.weaken_rate * (delta.magnitude or 0)
                    )

            existing.last_confirmed = obs.timestamp
            if obs.metric_value is not None:
                existing.last_metric_value = existing.metric_value
                existing.metric_value = obs.metric_value
            existing.entities = _stable_merge(existing.entities, obs.entities)
            existing.tags = _stable_merge(existing.tags, obs_tags)
            await self.backend.update(existing)

            # Lateral inhibition — weaken competitors on agreement
            if self.inhibition_rate > 0:
                await self._inhibit_competitors(existing)

            if delta:
                async with self._delta_lock:
                    self._pending_deltas.append(delta)
            return delta

        # Contradiction: different valence
        delta = Delta(
            old_belief=copy.copy(existing),
            new_observation=obs,
            delta_type="contradiction",
        )
        existing.confidence -= self.weaken_rate

        if existing.confidence <= self.death_threshold:
            existing.superseded_by = obs.description
            await self.backend.update(existing)
            await self.backend.store(Belief(
                capability=obs.capability,
                description=obs.description,
                valence=obs.valence,
                confidence=1.0,
                confirmation_count=1,
                entities=obs.entities,
                config=obs.config,
                metric_name=obs.metric_name,
                metric_value=obs.metric_value,
                source=obs.source,
                tags=obs_tags,
                first_seen=obs.timestamp,
                last_confirmed=obs.timestamp,
            ))
        else:
            await self.backend.update(existing)

        async with self._delta_lock:
            self._pending_deltas.append(delta)
        return delta

    def _check_metric_shift(self, belief: Belief, obs: Observation) -> Delta | None:
        if (
            obs.metric_value is None
            or belief.metric_value is None
            or belief.metric_value == 0
        ):
            return None
        shift = abs(obs.metric_value - belief.metric_value) / abs(belief.metric_value)
        if shift >= self.metric_shift_threshold:
            return Delta(
                old_belief=copy.copy(belief),
                new_observation=obs,
                delta_type="metric_shift",
                magnitude=shift,
            )
        return None

    async def _compute_salience(
        self,
        obs: Observation,
        max_boost: float = 3.0,
    ) -> float:
        """Compute initial encoding strength from metric surprise."""
        if obs.metric_value is None:
            return 1.0

        peers = await self.backend.query(capability=obs.capability, limit=100)
        peer_metrics = [b.metric_value for b in peers if b.metric_value is not None]
        if len(peer_metrics) < 2:
            return 1.0

        mean = sum(peer_metrics) / len(peer_metrics)
        variance = sum((m - mean) ** 2 for m in peer_metrics) / len(peer_metrics)
        std = math.sqrt(variance) if variance > 0 else 1.0

        z = abs(obs.metric_value - mean) / std
        boost = 1.0 + (max_boost - 1.0) * min(z / 2.0, 1.0)
        return boost

    async def _inhibit_competitors(self, strengthened: Belief) -> int:
        """Weaken beliefs competing with the strengthened one.

        A competitor has same capability, overlapping entities/tags,
        but DIFFERENT description. Same-description beliefs are allies.
        Confidence floors at self.inhibition_floor.
        """
        strengthened_features = set(strengthened.entities) | set(strengthened.tags)
        if not strengthened_features:
            return 0

        competitors = await self.backend.query(
            capability=strengthened.capability,
            limit=50,
        )

        inhibited = 0
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

            reduction = self.inhibition_rate * overlap
            comp.confidence = max(self.inhibition_floor, comp.confidence - reduction)
            await self.backend.update(comp)
            inhibited += 1

        return inhibited

    async def drain_deltas(self) -> list[Delta]:
        """Collect and clear pending deltas."""
        async with self._delta_lock:
            deltas = self._pending_deltas.copy()
            self._pending_deltas.clear()
            return deltas

    # ------------------------------------------------------------------
    # Read path: prompt context generation
    # ------------------------------------------------------------------

    async def context_for(
        self,
        capability: str,
        entities: tuple[str, ...] = (),
        exclude_source: str | None = None,
        tags: tuple[str, ...] = (),
        max_entries: int = 10,
    ) -> str:
        """Generate prompt-injectable context.

        Consolidation happens HERE, on read. Beliefs are ranked by
        strength so the most confirmed + most recent surface first.
        No separate compaction pass needed.
        """
        beliefs = await self.backend.query(
            capability=capability,
            entities=entities,
            exclude_source=exclude_source,
            tags=tags,
            limit=max_entries * 3 if self.exploration_rate > 0 else max_entries * 2,
        )

        if not beliefs:
            return ""

        beliefs.sort(key=lambda b: b.strength, reverse=True)

        # Exploration: occasionally promote a lower-ranked belief
        if (
            self.exploration_rate > 0
            and len(beliefs) > max_entries
            and random.random() < self.exploration_rate
        ):
            promoted = random.choice(beliefs[max_entries:])
            beliefs[max_entries - 1] = promoted

        beliefs = beliefs[:max_entries]

        patterns = [b for b in beliefs if b.is_pattern]
        antipatterns = [b for b in beliefs if b.is_antipattern]

        lines: list[str] = ["## What we know"]

        if patterns:
            lines.append("\n### What works")
            for b in patterns:
                metric = f" ({b.metric_name}={b.metric_value})" if b.metric_value else ""
                conf = f" [confirmed {b.confirmation_count}x]" if b.confirmation_count > 1 else ""
                lines.append(f"- {b.description}{metric}{conf}")

        if antipatterns:
            lines.append("\n### What fails")
            for b in antipatterns:
                conf = f" [confirmed {b.confirmation_count}x]" if b.confirmation_count > 1 else ""
                lines.append(f"- {b.description}{conf}")

        return "\n".join(lines)

    async def entity_profile(self, entity: str) -> str:
        """Everything we know about one entity.

        Not a separate store — a view over the same beliefs.
        """
        beliefs = await self.backend.query(entities=(entity,), limit=50)
        if not beliefs:
            return f"No knowledge about '{entity}'."

        beliefs.sort(key=lambda b: b.strength, reverse=True)

        lines = [f"## Profile: {entity}"]
        for b in beliefs:
            prefix = "+" if b.is_pattern else "-" if b.is_antipattern else "~"
            metric = f" ({b.metric_name}={b.metric_value})" if b.metric_value else ""
            lines.append(f"  {prefix} {b.description}{metric}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    async def summary(self) -> dict[str, int]:
        """Snapshot for observation tools."""
        all_b = await self.backend.all_beliefs()
        patterns = sum(1 for b in all_b if b.is_pattern)
        antipatterns = sum(1 for b in all_b if b.is_antipattern)
        return {
            "total_beliefs": len(all_b),
            "patterns": patterns,
            "antipatterns": antipatterns,
        }

    # ------------------------------------------------------------------
    # Eviction: consolidation + strength-based forgetting
    # ------------------------------------------------------------------

    async def forget(self) -> int:
        """Consolidate similar beliefs, then remove weakest when over capacity."""
        all_beliefs = await self.backend.all_beliefs()

        # Phase 1: consolidation — compress before evicting
        consolidated = await self._consolidate(all_beliefs)
        if consolidated > 0:
            # Re-read after consolidation changed the store
            all_beliefs = await self.backend.all_beliefs()

        # Phase 2: eviction — remove weakest
        if len(all_beliefs) <= self.max_beliefs:
            return consolidated
        all_beliefs.sort(key=lambda b: b.strength)
        to_remove = all_beliefs[: len(all_beliefs) - self.max_beliefs]
        for b in to_remove:
            await self.backend.remove(b)
        return consolidated + len(to_remove)

    async def _consolidate(
        self,
        all_beliefs: list[Belief],
        min_cluster_size: int = 3,
    ) -> int:
        """Merge clusters of similar beliefs into meta-beliefs."""
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
            clusters = self._cluster_by_overlap(group)
            for cluster in clusters:
                if len(cluster) < min_cluster_size:
                    continue
                merged = _merge_cluster(cluster)
                for old in cluster:
                    await self.backend.remove(old)
                    removed += 1
                await self.backend.store(merged)

        return removed

    def _cluster_by_overlap(self, beliefs: list[Belief]) -> list[list[Belief]]:
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
                jaccard = len(seed_features & other_features) / len(
                    seed_features | other_features
                )
                if jaccard >= self.consolidation_threshold:
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
