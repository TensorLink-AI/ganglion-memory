"""Agent integration for Memory — three touch points.

    agent = Agent(memory=mem, capability="mining")

    # 1. BEFORE acting  → retrieve relevant experience
    context = await agent.remember(query="optimize batch size")

    # 2. AFTER acting   → store the outcome
    exp = await agent.learn(result_dict, input_text="...", output_text="...")

    # 3. BETWEEN runs   → compress similar experiences
    count = await agent.between_runs()

The agent itself doesn't know about memory. It receives a richer
prompt and reports what happened. That's it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ganglion.memory.core import Memory
from ganglion.memory.types import Experience

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """Memory-aware agent wrapper. Three methods, no subclassing needed.

    Usage:
        agent = Agent(memory=mem, capability="mining", bot_id="alpha")

        context = await agent.remember(query="optimize batch size")
        result = await my_agent.run(context=context)
        exp = await agent.learn(result_dict)
    """

    memory: Memory
    capability: str = "general"
    bot_id: str | None = None
    tags: tuple[str, ...] = ()
    max_context: int = 10

    async def remember(self, query: str = "") -> str:
        """BEFORE acting. Retrieve relevant experiences as prompt context.

        Returns a formatted string suitable for injection into a system
        prompt, or empty string if nothing relevant found.
        """
        search_tags = (self.capability, *self.tags)

        if query:
            experiences = await self.memory.search(
                query, tags=search_tags, limit=self.max_context,
            )
        else:
            experiences = await self.memory.backend.query(
                tags=search_tags, limit=self.max_context,
            )

        if not experiences:
            return ""

        return self._format_context(experiences)

    async def learn(
        self,
        result: dict[str, Any],
        *,
        input_text: str = "",
        output_text: str = "",
    ) -> Experience:
        """AFTER acting. Store the outcome as an experience.

        Convention: result dicts carry these keys:
            success:      bool
            description:  str
            error:        str | None
            metric_name:  str | None
            metric_value: float | None
            config:       dict | None
            tags:         list[str]
        """
        success = result.get("success", False)
        description = result.get("description", "")

        if not success and result.get("error"):
            description = f"{description} — {result['error']}"

        # Build metadata from result + input/output
        metadata: dict[str, Any] = {}
        if input_text:
            metadata["input_text"] = input_text[:500]
        if output_text:
            metadata["output_text"] = output_text[:500]
        if result.get("config"):
            metadata["config"] = result["config"]
        if result.get("metric_name"):
            metadata["metric_name"] = result["metric_name"]
            metadata["metric_value"] = result.get("metric_value")

        # Build tags: capability + agent tags + outcome + result tags
        tag_set: set[str] = {self.capability}
        tag_set.update(self.tags)
        tag_set.add("success" if success else "failure")
        if result.get("tags"):
            tag_set.update(result["tags"])

        return await self.memory.add(
            content=description[:500],
            tags=tuple(sorted(tag_set)),
            source=self.bot_id or "",
            metadata=metadata or None,
        )

    async def between_runs(self) -> int:
        """BETWEEN runs. Compress similar experiences.

        Returns the number of merged experiences created.
        """
        merged = await self.memory.compress(
            tags=(self.capability,),
            min_cluster=3,
        )
        if merged:
            logger.info("Compressed into %d merged experiences", len(merged))
        return len(merged)

    # -- Formatting ----------------------------------------------------------

    def _format_context(self, experiences: list[Experience]) -> str:
        """Format experiences as prompt-injectable context.

        Uses a simple markdown format:
            + success experiences
            - failure experiences
            ~ neutral experiences

        When input_text/output_text metadata is available, formats as
        few-shot examples (prior task → outcome → lesson).
        """
        lines: list[str] = ["## Relevant experience"]

        for exp in experiences:
            # Determine status marker from tags
            if "success" in exp.tags:
                status = "+"
            elif "failure" in exp.tags:
                status = "-"
            else:
                status = "~"

            # Count annotations
            count_info = ""
            if exp.confirmation_count > 1:
                count_info = f" [confirmed {exp.confirmation_count}x]"
            if exp.contradiction_count > 0:
                count_info += f" [contradicted {exp.contradiction_count}x]"

            # Few-shot format when we have the full experience tuple
            if exp.metadata and exp.metadata.get("input_text"):
                lines.append(f"\n**Prior task**: {exp.metadata['input_text'][:200]}")
                if exp.metadata.get("output_text"):
                    lines.append(
                        f"**Outcome**: {status} {exp.metadata['output_text'][:150]}"
                    )
                lines.append(f"**Lesson**: {exp.content}{count_info}")
            else:
                # Simple format
                metric = ""
                if exp.metadata and exp.metadata.get("metric_name"):
                    name = exp.metadata["metric_name"]
                    value = exp.metadata.get("metric_value")
                    metric = f" ({name}={value})"
                lines.append(f"- {status} {exp.content}{metric}{count_info}")

        return "\n".join(lines)
