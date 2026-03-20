"""
Agent integration for the memory loop.

Three touch points, any agent, any task:

    1. BEFORE acting  -> remember(query=...) -> inject context into prompt
    2. AFTER acting    -> learn(result, input_text=..., output_text=...) -> feed outcome
    3. BETWEEN runs   -> between_runs() -> synthesize + forget

The agent itself doesn't know about memory. It receives a richer
prompt and reports what happened. That's it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ganglion.memory.embed import cosine_similarity
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
    input_text: str = "",
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

    When input_text is provided, it is stored in config["input_text"]
    so that assimilate() can embed on the task input rather than
    the belief description.
    """
    succeeded = result.get("success", False)
    description = result.get("description", "")

    if not succeeded and result.get("error"):
        description = f"{description} — {result['error']}"

    all_entities = set(entities) | set(result.get("entities", []))
    all_tags = set(tags) | set(result.get("tags", []))

    config = result.get("config") or {}
    if input_text:
        config["input_text"] = input_text[:500]

    return Observation(
        capability=capability,
        description=description[:500],
        valence=Valence.POSITIVE if succeeded else Valence.NEGATIVE,
        entities=tuple(all_entities),
        config=config if config else None,
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

    # Confidence-gated injection
    relevance_threshold: float = 0.6

    # Active memory pruning
    _task_count: int = field(default=0, repr=False)
    refine_interval: int = 10

    async def should_inject(self, query: str, threshold: float | None = None) -> bool:
        """Gate memory injection on relevance.

        Only inject when we have beliefs that are highly relevant
        to the current query. Prevents noise injection on tasks
        the model can handle alone.
        """
        if threshold is None:
            threshold = self.relevance_threshold

        if not self.memory.embedder:
            return True  # Can't gate without embeddings, default to inject

        query_embedding = await self.memory._embed(query)
        if query_embedding is None:
            return True

        beliefs = await self.memory.backend.query(
            capability=self.capability, limit=20,
        )

        if not beliefs:
            return False  # No memory yet, skip injection

        # Check if ANY belief is highly relevant
        max_relevance = 0.0
        for b in beliefs:
            if b.embedding is not None:
                sim = cosine_similarity(query_embedding, b.embedding)
                max_relevance = max(max_relevance, sim)

        return max_relevance >= threshold

    async def remember(self, query: str = "") -> str:
        """BEFORE acting. Build prompt context from memory.

        Uses tiered retrieval:
          Tier 1: Procedural strategies (safe to inject broadly)
          Tier 2: Specific experiences (only if highly relevant)
          Tier 3: Known failures (always useful — what NOT to do)
          Tier 4: Foreign knowledge (unvalidated)
          Tier 5: Entity profiles
        """
        parts: list[str] = []

        # Tier 1: Procedural strategies (high-level, always-safe to inject)
        strategies = await self.memory.context_for(
            capability=self.capability,
            query=query,
            tags=("strategy", "synthesized"),
            max_entries=5,
        )
        if strategies:
            parts.append(strategies)

        # Tier 2: Specific experiences (only inject if highly relevant)
        if await self._query_has_similar_experiences(query, threshold=0.7):
            experiences = await self.memory.context_for(
                capability=self.capability,
                query=query,
                entities=self.entities,
                max_entries=self.context_limit,
            )
            if experiences:
                parts.append(experiences)
        elif not parts:
            # Fallback: if no strategies and no high-relevance experiences,
            # use standard context (backward compatible)
            own = await self.memory.context_for(
                capability=self.capability,
                query=query,
                entities=self.entities,
                tags=self.tags,
                max_entries=self.context_limit,
            )
            if own:
                parts.append(own)

        # Tier 3: Known failures (always useful — what NOT to do)
        antipatterns = await self.memory.backend.query(
            capability=self.capability,
            valence=Valence.NEGATIVE,
            limit=3,
        )
        if antipatterns:
            lines = ["## Known pitfalls"]
            for b in antipatterns:
                lines.append(f"- AVOID: {b.description}")
            pitfalls = "\n".join(lines)
            # Only add if not already covered in strategies/experiences
            if pitfalls not in "\n".join(parts):
                parts.append(pitfalls)

        # Tier 4: Foreign knowledge
        if self.include_foreign and self.bot_id:
            foreign = await self.memory.context_for(
                capability=self.capability,
                query=query,
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

        # Tier 5: Entity profiles
        for entity in self.entities:
            profile = await self.memory.entity_profile(entity)
            if not profile.startswith("No knowledge"):
                parts.append(profile)

        return "\n\n".join(parts)

    async def _query_has_similar_experiences(
        self, query: str, threshold: float = 0.7,
    ) -> bool:
        """Check if there are stored experiences similar to the query."""
        if not query or not self.memory.embedder:
            return False

        query_embedding = await self.memory._embed(query)
        if query_embedding is None:
            return False

        beliefs = await self.memory.backend.query(
            capability=self.capability, limit=20,
        )
        for b in beliefs:
            if b.embedding is not None:
                sim = cosine_similarity(query_embedding, b.embedding)
                if sim >= threshold:
                    return True
        return False

    async def learn(
        self,
        result: dict[str, Any],
        input_text: str = "",
        output_text: str = "",
    ) -> Delta | None:
        """AFTER acting. Feed the outcome into memory.

        When input_text is provided, stores it in the observation config
        so that assimilate() embeds on the task input for better retrieval.

        Periodically runs memory refinement to prune unused beliefs.
        """
        obs = result_to_observation(
            capability=self.capability,
            result=result,
            source=self.bot_id,
            run_id=self.run_id,
            entities=self.entities,
            tags=self.tags,
            input_text=input_text,
        )
        delta = await self.memory.assimilate(obs)

        self._task_count += 1
        if self._task_count % self.refine_interval == 0:
            await self._refine_memory()

        return delta

    async def _refine_memory(self) -> int:
        """Active memory curation — ReMem's Refine step.

        Remove beliefs that were stored but never retrieved,
        merge near-duplicates, and promote high-value beliefs.
        """
        all_beliefs = await self.memory.backend.all_beliefs()
        removed = 0

        # Kill beliefs that were stored but never retrieved
        for b in all_beliefs:
            if (b.last_retrieved is None
                    and b.confirmation_count == 1
                    and b.confidence < 1.5):
                await self.memory.backend.remove(b)
                removed += 1

        # Consolidate similar beliefs without waiting for between_runs
        if removed > 0:
            all_beliefs = await self.memory.backend.all_beliefs()
        consolidated = await self.memory._consolidate(
            all_beliefs, min_cluster_size=2,
        )

        total = removed + consolidated
        if total > 0:
            logger.info("Refined memory: removed %d, consolidated %d", removed, consolidated)
        return total


# ------------------------------------------------------------------
# Run boundary helper
# ------------------------------------------------------------------

async def between_runs(
    memory: MemoryLoop,
    llm_client: Any = None,
    model: str = "claude-haiku",
) -> list[Delta]:
    """BETWEEN runs. Synthesize insights, drain deltas, forget weak beliefs.

    With an LLM client: reasons over accumulated beliefs to produce
    meta-insights before forgetting.
    Without: just drains deltas and forgets (same as before).
    """
    deltas = await memory.drain_deltas()
    if deltas:
        logger.info("Memory shifts this run: %d", len(deltas))
        for d in deltas:
            logger.info("  %s", d.summary)

    # Synthesis: reason over beliefs to produce meta-insights
    if llm_client is not None:
        try:
            from ganglion.memory.synthesize import synthesize
            recent_beliefs = await memory.backend.query(limit=50)
            observations = await synthesize(
                beliefs=recent_beliefs,
                deltas=deltas,
                model=model,
                llm_client=llm_client,
                capability=recent_beliefs[0].capability if recent_beliefs else "general",
            )
            for obs in observations:
                await memory.assimilate(obs)
            if observations:
                logger.info("Synthesized %d meta-insights", len(observations))
        except Exception as e:
            logger.warning("Synthesis failed: %s", e)

    forgotten = await memory.forget()
    if forgotten:
        logger.info("Forgot %d weak beliefs", forgotten)
    return deltas
