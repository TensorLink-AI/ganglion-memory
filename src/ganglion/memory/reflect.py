"""Structured experience storage — replaces the LLM reflection path.

After each agent call, extract a structured ⟨task, outcome, lesson⟩ tuple.
The counterfactual comparison in wrap.py is the actual learning signal;
this module just structures the experience for few-shot retrieval.
"""

from __future__ import annotations

import logging

from ganglion.memory.types import Observation, Valence

logger = logging.getLogger(__name__)


def _simple_reflect(
    input_text: str,
    output_text: str,
    capability: str,
    source: str | None,
) -> Observation:
    """Store structured ⟨task, outcome, lesson⟩ — not raw output text.

    The description contains a reusable lesson anchored to the task type.
    The config carries the full input/output pair for few-shot retrieval.
    """
    output_lower = output_text.lower()
    error_signals = ["error", "failed", "exception", "traceback", "refused", "cannot", "unable"]
    has_error = any(sig in output_lower for sig in error_signals)

    task_summary = input_text[:300] if input_text else "unknown task"

    if has_error:
        valence = Valence.NEGATIVE
        description = f"Task failed: {task_summary}"
    elif len(output_text.strip()) < 10:
        valence = Valence.NEUTRAL
        description = f"Inconclusive: {task_summary}"
    else:
        valence = Valence.POSITIVE
        description = f"Task succeeded: {task_summary}"

    # Store the full experience tuple in config for few-shot retrieval
    config = {
        "input_text": input_text[:500],
        "output_text": output_text[:500],
    }

    return Observation(
        capability=capability,
        description=description[:500],
        valence=valence,
        source=source,
        config=config,
    )
