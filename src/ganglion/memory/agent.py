"""
Agent integration for the memory loop.

Three touch points, any agent, any task:

    1. BEFORE acting  -> remember(query=...) -> inject context into prompt
    2. AFTER acting    -> learn(result, input_text=..., output_text=...) -> feed outcome
    3. BETWEEN runs   -> between_runs() -> drain + forget

The agent itself doesn't know about memory. It receives a richer
prompt and reports what happened. That's it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

        context = await agent.remember(query="optimize batch size")
        result = await my_agent.run(context=context)
        delta = await agent.learn(result, input_text="optimize batch size",
                                   output_text=str(result))
    """

    memory: MemoryLoop
    capability: str
    bot_id: str | None = None
    entities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    run_id: str | None = None
    context_limit: int = 10
    include_foreign: bool = True
    _retrieved_beliefs: list = field(default_factory=list)

    async def remember(self, query: str = "") -> str:
        """BEFORE acting. Build prompt context and track what was retrieved."""
        parts: list[str] = []
        self._retrieved_beliefs = []

        own_context, own_beliefs = await self.memory.retrieve_for(
            capability=self.capability,
            query=query,
            entities=self.entities,
            tags=self.tags,
            max_entries=self.context_limit,
        )
        if own_context:
            parts.append(own_context)
            self._retrieved_beliefs.extend(own_beliefs)

        if self.include_foreign and self.bot_id:
            foreign_context, foreign_beliefs = await self.memory.retrieve_for(
                capability=self.capability,
                query=query,
                entities=self.entities,
                tags=self.tags,
                exclude_source=self.bot_id,
                max_entries=self.context_limit // 2,
            )
            if foreign_context:
                foreign_context = foreign_context.replace(
                    "## Relevant experience",
                    "## What other agents report (unvalidated)",
                )
                parts.append(foreign_context)
                self._retrieved_beliefs.extend(foreign_beliefs)

        for entity in self.entities:
            profile = await self.memory.entity_profile(entity)
            if not profile.startswith("No knowledge"):
                parts.append(profile)

        return "\n\n".join(parts)

    async def learn(
        self,
        result: dict[str, Any],
        input_text: str = "",
        output_text: str = "",
    ) -> Delta | None:
        """AFTER acting. Store experience with dependency tracking."""
        obs = result_to_observation(
            capability=self.capability,
            result=result,
            source=self.bot_id,
            run_id=self.run_id,
            entities=self.entities,
            tags=self.tags,
        )

        # Stamp dependency chain: which beliefs were active when this was produced
        retrieved_ids = tuple(b.id for b in self._retrieved_beliefs if b.id is not None)
        config = dict(obs.config) if obs.config else {}
        if retrieved_ids:
            config["produced_with"] = list(retrieved_ids)
        if input_text:
            config["input_text"] = input_text[:500]
        if output_text:
            config["output_text"] = output_text[:500]

        # Rebuild observation with enriched config (Observation is frozen)
        obs = Observation(
            capability=obs.capability,
            description=obs.description,
            valence=obs.valence,
            entities=obs.entities,
            config=config,
            metric_name=obs.metric_name,
            metric_value=obs.metric_value,
            source=obs.source,
            run_id=obs.run_id,
            tags=obs.tags,
            timestamp=obs.timestamp,
        )

        delta = await self.memory.assimilate(obs)
        self._retrieved_beliefs = []
        return delta


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
