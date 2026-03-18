"""
Agent integration for the memory loop.

Three touch points, any agent, any task:

    1. BEFORE acting  → remember() → inject context into prompt
    2. AFTER acting    → learn(result) → feed outcome into memory
    3. BETWEEN runs   → drain_deltas() + forget()

The agent itself doesn't know about memory. It receives a richer
prompt and reports what happened. That's it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ganglion.memory.loop import MemoryLoop
from ganglion.memory.types import Delta, Observation, Valence

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Observation factory: turns any result dict into an Observation
# ------------------------------------------------------------------

def result_to_observation(
    capability: str,
    result: dict[str, Any],
    source: str | None = None,
    run_id: str | None = None,
    entities: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> Observation:
    """Convert a raw agent result into an Observation.

    Convention: result dicts carry these keys:
        success: bool
        description: str
        metric_name: str | None
        metric_value: float | None
        config: dict | None
        entities: list[str]   (merged with the entities arg)
        tags: list[str]
        error: str | None
    """
    succeeded = result.get("success", False)
    description = result.get("description", "")

    if not succeeded and result.get("error"):
        description = f"{description} — {result['error']}"

    all_entities = set(entities) | set(result.get("entities", []))
    all_tags = set(tags) | set(result.get("tags", []))

    return Observation(
        capability=capability,
        description=description[:500],
        valence=Valence.POSITIVE if succeeded else Valence.NEGATIVE,
        entities=tuple(all_entities),
        config=result.get("config"),
        metric_name=result.get("metric_name"),
        metric_value=result.get("metric_value"),
        source=source,
        run_id=run_id,
        tags=tuple(all_tags),
    )


# ------------------------------------------------------------------
# Memory-aware agent wrapper
# ------------------------------------------------------------------

@dataclass
class MemoryAgent:
    """Wraps any agent with memory. Three methods, no subclassing needed.

    Usage:
        agent = MemoryAgent(memory=loop, capability="mining", bot_id="alpha")

        context = await agent.remember()           # inject into prompt
        result = await my_agent.run(context=context)
        delta = await agent.learn(result)           # feed back
    """

    memory: MemoryLoop
    capability: str
    bot_id: str | None = None
    entities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    run_id: str | None = None
    context_limit: int = 10
    include_foreign: bool = True

    async def remember(self) -> str:
        """BEFORE acting. Build prompt context from memory."""
        parts: list[str] = []

        # Own knowledge
        own = await self.memory.context_for(
            capability=self.capability,
            entities=self.entities,
            tags=self.tags,
            max_entries=self.context_limit,
        )
        if own:
            parts.append(own)

        # Foreign knowledge
        if self.include_foreign and self.bot_id:
            foreign = await self.memory.context_for(
                capability=self.capability,
                entities=self.entities,
                tags=self.tags,
                exclude_source=self.bot_id,
                max_entries=self.context_limit // 2,
            )
            if foreign:
                foreign = foreign.replace(
                    "## What we know",
                    "## What other agents report (unvalidated)",
                )
                parts.append(foreign)

        # Entity profiles
        for entity in self.entities:
            profile = await self.memory.entity_profile(entity)
            if not profile.startswith("No knowledge"):
                parts.append(profile)

        return "\n\n".join(parts)

    async def learn(self, result: dict[str, Any]) -> Delta | None:
        """AFTER acting. Feed the outcome into memory."""
        obs = result_to_observation(
            capability=self.capability,
            result=result,
            source=self.bot_id,
            run_id=self.run_id,
            entities=self.entities,
            tags=self.tags,
        )
        return await self.memory.assimilate(obs)


# ------------------------------------------------------------------
# Run boundary helper
# ------------------------------------------------------------------

async def between_runs(memory: MemoryLoop) -> list[Delta]:
    """BETWEEN runs. Drain deltas and forget weak beliefs."""
    deltas = await memory.drain_deltas()
    if deltas:
        logger.info("Memory shifts this run: %d", len(deltas))
        for d in deltas:
            logger.info("  %s", d.summary)
    forgotten = await memory.forget()
    if forgotten:
        logger.info("Forgot %d weak beliefs", forgotten)
    return deltas
