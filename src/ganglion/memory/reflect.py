"""LLM-based reflection — replaces the binary success/failure judge.

After each agent call, ask an LLM to evaluate what happened:
    1. Did this response accomplish the task? (valence)
    2. What specific thing worked or failed? (description — a belief)
    3. What entities/variables are involved?

The reflection extracts the *lesson*, not the transcript.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ganglion.memory.types import Belief, Experience, Observation, Valence

logger = logging.getLogger(__name__)

REFLECT_PROMPT = """You are evaluating an AI agent's action to extract a learning signal.

INPUT the agent received:
{input_text}

OUTPUT the agent produced:
{output_text}

EXISTING BELIEFS the agent had:
{beliefs_text}

Analyze this interaction and extract a single learning observation. Output ONLY valid JSON:
{{
    "valence": "positive" | "negative" | "neutral",
    "description": "<a concise belief statement about what worked or failed — NOT the raw output>",
    "entities": ["<key variables, parameter names, tool names, concepts>"],
    "tags": ["<relevant categories>"]
}}

Rules:
- description should be a generalizable lesson, not a transcript (e.g., "Using batch_size=32 prevents OOM on 8GB GPUs" not "The agent said batch_size=32")
- entities should capture the key variables that matter
- If the outcome is ambiguous, use "neutral"
- Keep description under 200 characters"""

REFLECT_PROMPT_V2 = """You are extracting a reusable lesson from an AI agent interaction.

TASK INPUT:
{input_text}

AGENT OUTPUT:
{output_text}

OUTCOME: {outcome}

Extract a lesson that would help solve SIMILAR future tasks. Output ONLY valid JSON:
{{
    "lesson": "<a generalizable strategy or principle, NOT the specific answer>",
    "strategy_tags": ["<reasoning patterns used, e.g. 'elimination', 'decomposition'>"],
    "input_summary": "<1-sentence description of the task type>",
    "output_summary": "<1-sentence description of what the agent did>"
}}

Rules:
- The lesson should be TRANSFERABLE — useful for similar but different problems
- BAD lesson: "The answer to 2x^2-5x+1=0 is x=2.28 or x=0.22"
- GOOD lesson: "For quadratic equations, apply the quadratic formula directly rather than trying to factor first when coefficients are non-trivial"
- strategy_tags should capture the REASONING PATTERN, not the topic"""


async def reflect(
    input_text: str,
    output_text: str,
    existing_beliefs: list[Belief] | None = None,
    capability: str = "general",
    source: str | None = None,
    model: str = "claude-haiku",
    llm_client: Any = None,
) -> Observation:
    """Ask an LLM to evaluate what just happened.

    Returns a structured Observation extracted from the interaction.
    Falls back to simple heuristic if no LLM client available.
    """
    if llm_client is None:
        llm_client = _get_default_client()

    if llm_client is None:
        return _simple_reflect(input_text, output_text, capability, source)

    beliefs_text = _format_beliefs(existing_beliefs or [])
    prompt = REFLECT_PROMPT.format(
        input_text=input_text[:2000],
        output_text=output_text[:2000],
        beliefs_text=beliefs_text,
    )

    try:
        result = await _call_llm(llm_client, prompt, model)
        return _parse_reflection(result, capability, source)
    except Exception as e:
        logger.warning("LLM reflection failed, falling back to simple: %s", e)
        return _simple_reflect(input_text, output_text, capability, source)


def _format_beliefs(beliefs: list[Belief]) -> str:
    if not beliefs:
        return "(none)"
    lines = []
    for b in beliefs[:20]:
        prefix = "+" if b.is_pattern else "-" if b.is_antipattern else "~"
        lines.append(f"  {prefix} {b.description}")
    return "\n".join(lines)


def _simple_reflect(
    input_text: str,
    output_text: str,
    capability: str,
    source: str | None,
) -> Observation:
    """Fallback: extract basic signal without an LLM."""
    # Heuristic: check for error indicators
    output_lower = output_text.lower()
    error_signals = ["error", "failed", "exception", "traceback", "refused", "cannot", "unable"]
    has_error = any(sig in output_lower for sig in error_signals)

    if has_error:
        valence = Valence.NEGATIVE
        description = f"Failed: {output_text[:200]}"
    elif len(output_text.strip()) < 10:
        valence = Valence.NEUTRAL
        description = f"Minimal response to: {input_text[:200]}"
    else:
        valence = Valence.POSITIVE
        description = f"Succeeded: {output_text[:200]}"

    return Observation(
        capability=capability,
        description=description[:500],
        valence=valence,
        source=source,
    )


async def reflect_experience(
    input_text: str,
    output_text: str,
    success: bool,
    capability: str = "general",
    model: str = "claude-haiku",
    llm_client: Any = None,
) -> Experience:
    """Extract a structured Experience from an interaction.

    Uses the v2 reflection prompt to produce transferable lessons
    in the Evo-Memory ExpRAG format.
    Falls back to simple heuristics without an LLM.
    """
    if llm_client is None:
        llm_client = _get_default_client()

    if llm_client is None:
        return _simple_experience(input_text, output_text, success, capability)

    outcome = "SUCCESS" if success else "FAILURE"
    prompt = REFLECT_PROMPT_V2.format(
        input_text=input_text[:2000],
        output_text=output_text[:2000],
        outcome=outcome,
    )

    try:
        result = await _call_llm(llm_client, prompt, model)
        return _parse_experience(result, input_text, output_text, success, capability)
    except Exception as e:
        logger.warning("LLM experience reflection failed, falling back: %s", e)
        return _simple_experience(input_text, output_text, success, capability)


def _simple_experience(
    input_text: str,
    output_text: str,
    success: bool,
    capability: str,
) -> Experience:
    """Fallback: create a basic Experience without an LLM."""
    if success:
        lesson = f"Approach succeeded for: {input_text[:150]}"
    else:
        lesson = f"Approach failed for: {input_text[:150]}"

    return Experience(
        input_summary=input_text[:200],
        output_summary=output_text[:200],
        success=success,
        lesson=lesson[:300],
        capability=capability,
    )


def _parse_experience(
    llm_output: str,
    input_text: str,
    output_text: str,
    success: bool,
    capability: str,
) -> Experience:
    """Parse the LLM's JSON response into an Experience."""
    text = llm_output.strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _simple_experience(input_text, output_text, success, capability)

    return Experience(
        input_summary=data.get("input_summary", input_text[:200])[:200],
        output_summary=data.get("output_summary", output_text[:200])[:200],
        success=success,
        lesson=data.get("lesson", "")[:300],
        strategy_tags=tuple(data.get("strategy_tags", ())),
        capability=capability,
    )


async def _call_llm(client: Any, prompt: str, model: str) -> str:
    """Call an LLM. Supports Anthropic and OpenAI client shapes."""
    # Anthropic client
    if hasattr(client, "messages"):
        response = await client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    # OpenAI client
    if hasattr(client, "chat"):
        response = await client.chat.completions.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    # Callable
    if callable(client):
        return await client(prompt)

    raise TypeError(f"Unsupported LLM client type: {type(client).__name__}")


def _parse_reflection(llm_output: str, capability: str, source: str | None) -> Observation:
    """Parse the LLM's JSON response into an Observation."""
    # Extract JSON from potential markdown fences
    text = llm_output.strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Observation(
            capability=capability,
            description=llm_output[:500],
            valence=Valence.NEUTRAL,
            source=source,
        )

    valence_str = data.get("valence", "neutral")
    try:
        valence = Valence(valence_str)
    except ValueError:
        valence = Valence.NEUTRAL

    return Observation(
        capability=capability,
        description=data.get("description", llm_output[:500])[:500],
        valence=valence,
        entities=tuple(data.get("entities", [])),
        tags=tuple(data.get("tags", [])),
        source=source,
    )


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
