"""
MemoryLoop — the single primitive.

    for each observation:
        memory.assimilate(observation)

Everything else — strengthening, weakening, contradiction detection,
entity profiling, eviction — happens inside that single call or
on the read path.

v2: Embedding-based similarity, query-aware context, simplified knobs.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
from dataclasses import dataclass, field

from ganglion.memory.backends.base import MemoryBackend
from ganglion.memory.embed import Embedder, cosine_similarity
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
    """One loop. Five contexts. No special cases.

    Tuning knobs (6 that matter):
        strengthen_rate     — confidence boost on agreement
        weaken_rate         — confidence drop on contradiction
        death_threshold     — confidence below which beliefs die
        max_beliefs         — capacity limit
        consolidation_threshold — cosine/Jaccard threshold for merging
        inhibition_rate     — lateral inhibition strength (0 to disable)
    """

    backend: MemoryBackend
    embedder: Embedder | None = None

    # Core mechanics
    strengthen_rate: float = 0.1
    weaken_rate: float = 0.3
    death_threshold: float = 0.1
    max_beliefs: int = 1000
    consolidation_threshold: float = 0.5
    inhibition_rate: float = 0.05

    # Salience: surprise-gated encoding (internal heuristic)
    salience: bool = True

    # Metric shift detection
    metric_shift_threshold: float = 0.15

    # Deprecated knobs — accepted for backward compatibility, ignored
    inhibition_floor: float = 0.2
    cross_agent_bonus: float = 2.0
    exploration_rate: float = 0.0
    crisis_multiplier: float = 3.0

    _pending_deltas: list[Delta] = field(default_factory=list)
    _delta_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _contradiction_streak: int = 0

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    async def _embed(self, text: str) -> list[float] | None:
        """Embed text using the configured embedder, or None."""
        if self.embedder is None:
            return None
        try:
            return await self.embedder.embed(text)
        except Exception as e:
            logger.debug("Embedding failed: %s", e)
            return None

    async def _find_similar(
        self,
        obs: Observation,
        threshold: float = 0.75,
        embedding: list[float] | None = None,
    ) -> Belief | None:
        """Call backend.find_similar with backward-compatible signature.

        Old backends may not accept the `embedding` kwarg, so we fall
        back to calling without it if we get a TypeError.
        """
        try:
            return await self.backend.find_similar(
                obs, threshold=threshold, embedding=embedding,
            )
        except TypeError:
            # Old backend doesn't accept embedding param
            return await self.backend.find_similar(obs, threshold=threshold)

    # ------------------------------------------------------------------
    # Write path: the single primitive
    # ------------------------------------------------------------------

    async def assimilate(self, obs: Observation) -> Delta | None:
        """Observe -> Compare -> Update.

        The ONLY write path into memory. Returns a Delta if something
        meaningful changed, None otherwise.
        """
        # Compute embedding for the observation
        obs_embedding = await self._embed(obs.description)

        existing = await self._find_similar(
            obs, threshold=0.75, embedding=obs_embedding,
        )

        # Strategy bundling: include run tag when run_id is present
        run_tags = (f"run:{obs.run_id}",) if obs.run_id else ()
        obs_tags = obs.tags + run_tags

        if existing is None:
            # Novel observation — compute salience for initial confidence
            confidence = await self._compute_salience(obs) if self.salience else 1.0
            belief = Belief(
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
                embedding=obs_embedding,
            )
            await self.backend.store(belief)
            return None

        # Agreement: same valence
        if obs.valence == existing.valence:
            self._contradiction_streak = 0
            delta = self._check_metric_shift(existing, obs)

            if delta is None:
                # Pure agreement — strengthen
                existing.confidence = min(10.0, existing.confidence + self.strengthen_rate)
                existing.confirmation_count += 1
            else:
                # Metric shifted significantly — don't strengthen
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
            # Update embedding if we have a new one
            if obs_embedding is not None:
                existing.embedding = obs_embedding
            await self.backend.update(existing)

            # Lateral inhibition — weaken competitors on agreement
            if self.inhibition_rate > 0:
                await self._inhibit_competitors(existing, obs_embedding)

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
        self._contradiction_streak += 1

        # Crisis mode: consecutive contradictions (>=3) increase plasticity
        effective_weaken = self.weaken_rate
        if self._contradiction_streak >= 3:
            effective_weaken *= 3.0

        existing.confidence -= effective_weaken

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
                embedding=obs_embedding,
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

    async def _inhibit_competitors(
        self,
        strengthened: Belief,
        embedding: list[float] | None = None,
    ) -> int:
        """Weaken beliefs competing with the strengthened one.

        Uses embedding similarity when available, falls back to
        entity/tag overlap.
        """
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

            # Compute overlap score
            overlap = 0.0
            if embedding is not None and comp.embedding is not None:
                # Semantic similarity
                overlap = cosine_similarity(embedding, comp.embedding)
                if overlap < 0.3:
                    continue
            else:
                # Fallback: entity/tag overlap
                strengthened_features = set(strengthened.entities) | set(strengthened.tags)
                comp_features = set(comp.entities) | set(comp.tags)
                if not strengthened_features or not comp_features:
                    continue
                overlap = len(strengthened_features & comp_features) / len(
                    strengthened_features | comp_features
                )
                if overlap < 0.2:
                    continue

            reduction = self.inhibition_rate * overlap
            comp.confidence = max(0.2, comp.confidence - reduction)
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
        query: str = "",
    ) -> str:
        """Generate prompt-injectable context.

        When a query is provided and embeddings are available, ranks
        beliefs by relevance to the current input. Otherwise falls
        back to strength-based ranking.
        """
        beliefs = await self.backend.query(
            capability=capability,
            entities=entities,
            exclude_source=exclude_source,
            tags=tags,
            limit=max_entries * 5,
        )

        if not beliefs:
            return ""

        # Query-aware ranking via embeddings
        if query and self.embedder is not None:
            query_embedding = await self._embed(query)
            if query_embedding is not None:
                scored = []
                for b in beliefs:
                    if b.embedding is not None:
                        relevance = cosine_similarity(query_embedding, b.embedding)
                    else:
                        relevance = 0.0
                    # Blend relevance with strength for final score
                    score = relevance * 0.7 + min(b.strength / 10.0, 1.0) * 0.3
                    scored.append((b, score))
                scored.sort(key=lambda x: x[1], reverse=True)
                # Filter by minimum relevance
                beliefs = [b for b, score in scored[:max_entries] if score > 0.4]
                if beliefs:
                    return self._format_context(beliefs, query)

        # Fallback: strength-based ranking
        beliefs.sort(key=lambda b: b.strength, reverse=True)
        beliefs = beliefs[:max_entries]

        return self._format_context(beliefs)

    def _format_context(
        self,
        beliefs: list[Belief],
        query: str = "",
    ) -> str:
        """Format beliefs as structured context."""
        if not beliefs:
            return ""

        patterns = [b for b in beliefs if b.is_pattern]
        antipatterns = [b for b in beliefs if b.is_antipattern]
        neutral = [b for b in beliefs if b.valence == Valence.NEUTRAL]

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

        if neutral:
            lines.append("\n### Observations")
            for b in neutral:
                conf = f" [confirmed {b.confirmation_count}x]" if b.confirmation_count > 1 else ""
                lines.append(f"- {b.description}{conf}")

        # Flag contradictions
        contradictions = self._find_contradictions(beliefs)
        if contradictions:
            lines.append("\n### Contradictions (investigate)")
            for a, b in contradictions:
                lines.append(f"- '{a.description}' vs '{b.description}'")

        return "\n".join(lines)

    def _find_contradictions(self, beliefs: list[Belief]) -> list[tuple[Belief, Belief]]:
        """Find pairs of beliefs that contradict each other."""
        contradictions = []
        for i, a in enumerate(beliefs):
            for b in beliefs[i + 1:]:
                if a.valence != b.valence and a.valence != Valence.NEUTRAL and b.valence != Valence.NEUTRAL:
                    # Check if they're about the same thing
                    if a.embedding is not None and b.embedding is not None:
                        if cosine_similarity(a.embedding, b.embedding) > 0.6:
                            contradictions.append((a, b))
                    elif set(a.entities) & set(b.entities):
                        contradictions.append((a, b))
        return contradictions

    async def entity_profile(self, entity: str) -> str:
        """Everything we know about one entity."""
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
            all_beliefs = await self.backend.all_beliefs()

        # Decay contradiction streak between runs (crisis is transient)
        self._contradiction_streak = max(0, self._contradiction_streak - 1)

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
        """Seed-anchored clustering using embeddings or entity/tag overlap."""
        assigned: set[int | None] = set()
        clusters: list[list[Belief]] = []

        for b in beliefs:
            if b.id in assigned:
                continue
            cluster = [b]
            assigned.add(b.id)

            for other in beliefs:
                if other.id in assigned:
                    continue

                similarity = 0.0
                # Try embedding similarity first
                if b.embedding is not None and other.embedding is not None:
                    similarity = cosine_similarity(b.embedding, other.embedding)
                else:
                    # Fallback to entity/tag Jaccard
                    seed_features = set(b.entities) | set(b.tags)
                    other_features = set(other.entities) | set(other.tags)
                    if seed_features and other_features:
                        similarity = len(seed_features & other_features) / len(
                            seed_features | other_features
                        )

                if similarity >= self.consolidation_threshold:
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
        embedding=strongest.embedding,
    )
