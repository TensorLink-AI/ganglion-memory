"""Cross-belief reasoning — synthesis between runs.

Looks at accumulated beliefs and identifies:
    1. PATTERNS: Multiple beliefs suggesting a rule
    2. CONTRADICTIONS: Conflicting beliefs with evidence comparison
    3. GAPS: Important untested variables
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ganglion.memory.types import Belief, Delta, Observation, Valence

logger = logging.getLogger(__name__)

SYNTHESIZE_PROMPT = """You are reviewing an agent's accumulated knowledge.

RECENT BELIEFS (sorted by strength):
{beliefs_text}

RECENT CHANGES:
{deltas_text}

Look at these beliefs and recent changes, then identify:

1. PATTERNS: Multiple beliefs that together suggest a rule
   (e.g., "all three failures used lr > 0.01" -> "lr > 0.01 fails for this task")

2. CONTRADICTIONS: Beliefs that conflict with each other
   (flag which one has more evidence)

3. GAPS: Important variables that haven't been tested
   (e.g., "we've tried batch sizes 16 and 64 but never 32")

For each insight, output a JSON object with:
- description: the synthesized insight (a NEW belief, not a repeat)
- valence: "positive", "negative", or "neutral"
- entities: the key variables involved
- tags: always include "synthesized"

Output a JSON array of observations. Only output insights that are:
- NOVEL: not restatements of existing beliefs
- SUPPORTED: backed by at least 2 existing beliefs
- ACTIONABLE: useful for future decision-making

If no novel insights exist, output an empty array: []

Output ONLY valid JSON (an array)."""


async def synthesize(
    beliefs: list[Belief],
    deltas: list[Delta] | None = None,
    model: str = "claude-haiku",
    llm_client: Any = None,
    capability: str = "general",
) -> list[Observation]:
    """Reason over accumulated beliefs to produce meta-insights.

    Returns new Observations that represent synthesized knowledge.
    """
    if not beliefs or len(beliefs) < 2:
        return []

    if llm_client is None:
        llm_client = _get_default_client()

    if llm_client is None:
        return _simple_synthesize(beliefs)

    beliefs_text = _format_beliefs_for_synthesis(beliefs)
    deltas_text = _format_deltas(deltas or [])
    prompt = SYNTHESIZE_PROMPT.format(
        beliefs_text=beliefs_text,
        deltas_text=deltas_text,
    )

    try:
        result = await _call_llm(llm_client, prompt, model)
        return _parse_synthesis(result, capability)
    except Exception as e:
        logger.warning("LLM synthesis failed, falling back to simple: %s", e)
        return _simple_synthesize(beliefs)


def _format_beliefs_for_synthesis(beliefs: list[Belief]) -> str:
    lines = []
    for b in beliefs[:50]:
        prefix = "+" if b.is_pattern else "-" if b.is_antipattern else "~"
        conf = f"[confirmed {b.confirmation_count}x, confidence={b.confidence:.2f}]"
        entities = f" entities={list(b.entities)}" if b.entities else ""
        metric = f" {b.metric_name}={b.metric_value}" if b.metric_value else ""
        lines.append(f"  {prefix} {b.description} {conf}{metric}{entities}")
    return "\n".join(lines) or "(none)"


def _format_deltas(deltas: list[Delta]) -> str:
    if not deltas:
        return "(none)"
    lines = []
    for d in deltas[:20]:
        lines.append(f"  - {d.summary}")
    return "\n".join(lines)


def _simple_synthesize(beliefs: list[Belief]) -> list[Observation]:
    """Heuristic synthesis without an LLM.

    Groups beliefs by entities and looks for patterns in valence.
    """
    observations: list[Observation] = []

    # Group beliefs by shared entities
    entity_beliefs: dict[str, list[Belief]] = {}
    for b in beliefs:
        for e in b.entities:
            entity_beliefs.setdefault(e, []).append(b)

    for entity, group in entity_beliefs.items():
        if len(group) < 2:
            continue

        positives = [b for b in group if b.is_pattern]
        negatives = [b for b in group if b.is_antipattern]

        # Pattern: entity consistently fails or succeeds
        if len(negatives) >= 2 and len(positives) == 0:
            observations.append(Observation(
                capability=group[0].capability,
                description=f"{entity} consistently fails ({len(negatives)} failures, 0 successes)",
                valence=Valence.NEGATIVE,
                entities=(entity,),
                tags=("synthesized",),
            ))
        elif len(positives) >= 2 and len(negatives) == 0:
            observations.append(Observation(
                capability=group[0].capability,
                description=f"{entity} consistently succeeds ({len(positives)} successes, 0 failures)",
                valence=Valence.POSITIVE,
                entities=(entity,),
                tags=("synthesized",),
            ))
        elif len(positives) >= 1 and len(negatives) >= 1:
            observations.append(Observation(
                capability=group[0].capability,
                description=(
                    f"{entity} has mixed results "
                    f"({len(positives)} successes, {len(negatives)} failures) — "
                    f"investigate conditions"
                ),
                valence=Valence.NEUTRAL,
                entities=(entity,),
                tags=("synthesized",),
            ))

    return observations


async def _call_llm(client: Any, prompt: str, model: str) -> str:
    """Call an LLM. Supports Anthropic and OpenAI client shapes."""
    if hasattr(client, "messages"):
        response = await client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    if hasattr(client, "chat"):
        response = await client.chat.completions.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    if callable(client):
        return await client(prompt)

    raise TypeError(f"Unsupported LLM client type: {type(client).__name__}")


def _parse_synthesis(llm_output: str, capability: str) -> list[Observation]:
    """Parse the LLM's JSON array response into Observations."""
    text = llm_output.strip()
    if "```" in text:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse synthesis JSON")
        return []

    if not isinstance(data, list):
        return []

    observations = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            valence = Valence(item.get("valence", "neutral"))
        except ValueError:
            valence = Valence.NEUTRAL

        tags = list(item.get("tags", []))
        if "synthesized" not in tags:
            tags.append("synthesized")

        observations.append(Observation(
            capability=capability,
            description=item.get("description", "")[:500],
            valence=valence,
            entities=tuple(item.get("entities", [])),
            tags=tuple(tags),
        ))

    return observations


def _get_default_client() -> Any:
    """Try to create a default LLM client from environment."""
    try:
        import anthropic
        return anthropic.AsyncAnthropic()
    except (ImportError, Exception):
        pass
    try:
        import openai
        return openai.AsyncOpenAI()
    except (ImportError, Exception):
        pass
    return None
