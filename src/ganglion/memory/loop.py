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

import logging
from dataclasses import dataclass, field

from ganglion.memory.backends.base import MemoryBackend
from ganglion.memory.types import Belief, Delta, Observation, Valence

logger = logging.getLogger(__name__)


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

    _pending_deltas: list[Delta] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Write path: the single primitive
    # ------------------------------------------------------------------

    async def assimilate(self, obs: Observation) -> Delta | None:
        """Observe → Compare → Update.

        The ONLY write path into memory. Returns a Delta if something
        meaningful changed, None otherwise.
        """
        existing = await self.backend.find_similar(obs)

        if existing is None:
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
                tags=obs.tags,
                first_seen=obs.timestamp,
                last_confirmed=obs.timestamp,
            ))
            return None

        # Agreement: same valence
        if obs.valence == existing.valence:
            delta = self._check_metric_shift(existing, obs)
            existing.confidence = min(10.0, existing.confidence + self.strengthen_rate)
            existing.confirmation_count += 1
            existing.last_confirmed = obs.timestamp
            if obs.metric_value is not None:
                existing.last_metric_value = existing.metric_value
                existing.metric_value = obs.metric_value
            # Merge any new entities/tags
            existing.entities = tuple(set(existing.entities) | set(obs.entities))
            existing.tags = tuple(set(existing.tags) | set(obs.tags))
            await self.backend.update(existing)
            if delta:
                self._pending_deltas.append(delta)
            return delta

        # Contradiction: different valence
        delta = Delta(
            old_belief=existing,
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
                tags=obs.tags,
                first_seen=obs.timestamp,
                last_confirmed=obs.timestamp,
            ))
        else:
            await self.backend.update(existing)

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
                old_belief=belief,
                new_observation=obs,
                delta_type="metric_shift",
                magnitude=shift,
            )
        return None

    async def drain_deltas(self) -> list[Delta]:
        """Collect and clear pending deltas."""
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
            limit=max_entries * 2,
        )

        if not beliefs:
            return ""

        beliefs.sort(key=lambda b: b.strength, reverse=True)
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
    # Eviction: strength-based forgetting
    # ------------------------------------------------------------------

    async def forget(self) -> int:
        """Remove weakest beliefs when over capacity."""
        all_beliefs = await self.backend.all_beliefs()
        if len(all_beliefs) <= self.max_beliefs:
            return 0
        all_beliefs.sort(key=lambda b: b.strength)
        to_remove = all_beliefs[: len(all_beliefs) - self.max_beliefs]
        for b in to_remove:
            await self.backend.remove(b)
        return len(to_remove)
