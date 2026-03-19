"""One-line wrapper API for ganglion-memory.

Usage:
    from ganglion.memory import memory

    agent = memory(agent)

That's it. The wrapper auto-detects call conventions (OpenAI messages,
Anthropic system+messages, or plain string) and injects memory context
before each call, then evaluates the response after.

v2: Supports reflection modes ("auto", "simple", or callable),
    query-aware context injection, and embeddings.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any, Callable

from ganglion.memory.agent import MemoryAgent
from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.loop import MemoryLoop

logger = logging.getLogger(__name__)

# Module-level singleton so multiple memory() calls share one store per process.
_default_memory: MemoryLoop | None = None


def _get_or_create_memory(
    db_path: str = "memory.db",
    embedder: Any = None,
) -> MemoryLoop:
    """Return the module-level singleton, creating it on first use."""
    global _default_memory
    if _default_memory is None:
        backend = SqliteMemoryBackend(db_path)
        _default_memory = MemoryLoop(backend=backend, embedder=embedder)
    return _default_memory


# ------------------------------------------------------------------
# Response evaluation
# ------------------------------------------------------------------

def _default_judge(response: Any) -> dict[str, Any]:
    """Auto-detect response type and build a result dict.

    Improved heuristics: checks for error indicators rather than
    defaulting everything to success.
    """
    if isinstance(response, dict):
        if "success" in response:
            return response
        desc = str(response)[:500]
        return {"success": True, "description": desc}

    if isinstance(response, str):
        # Check for error indicators
        lower = response.lower()
        has_error = any(s in lower for s in [
            "error", "failed", "exception", "traceback", "refused",
        ])
        return {
            "success": not has_error,
            "description": response[:500],
        }

    # OpenAI ChatCompletion-like
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            text = choice.message.content or ""
            return {"success": True, "description": text[:500]}

    # Anthropic Message-like
    if hasattr(response, "content") and isinstance(response.content, list):
        for block in response.content:
            if hasattr(block, "text"):
                return {"success": True, "description": block.text[:500]}

    # Fallback
    return {"success": True, "description": str(response)[:500]}


def _extract_query(args: tuple, kwargs: dict) -> str:
    """Extract the user's query from call arguments for context retrieval."""
    # OpenAI-style: messages kwarg with user messages
    if "messages" in kwargs:
        messages = kwargs["messages"]
        if messages and isinstance(messages, list):
            # Find the last user message
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content[:500]

    # Anthropic-style: messages kwarg
    if "messages" in kwargs:
        messages = kwargs["messages"]
        if messages and isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content[:500]

    # Plain string: first positional argument
    if args and isinstance(args[0], str):
        return args[0][:500]

    return ""


# ------------------------------------------------------------------
# Prompt injection helpers
# ------------------------------------------------------------------

def _inject_context(args: tuple, kwargs: dict, context: str) -> tuple[tuple, dict]:
    """Inject memory context into the call arguments.

    Detects three conventions:
    1. OpenAI-style: messages=[{role: system, content: ...}, ...]
    2. Anthropic-style: system="...", messages=[...]
    3. Plain string: first positional arg is a string
    """
    if not context:
        return args, kwargs

    # OpenAI-style: messages kwarg with system message
    if "messages" in kwargs:
        messages = kwargs["messages"]
        if messages and isinstance(messages, list):
            if isinstance(messages[0], dict) and messages[0].get("role") == "system":
                injected = list(messages)
                original = injected[0]["content"]
                injected[0] = {
                    **injected[0],
                    "content": f"{original}\n\n{context}",
                }
                kwargs = {**kwargs, "messages": injected}
                return args, kwargs

    # Anthropic-style: system kwarg
    if "system" in kwargs:
        kwargs = {**kwargs, "system": f"{kwargs['system']}\n\n{context}"}
        return args, kwargs

    # Plain string: first positional argument
    if args and isinstance(args[0], str):
        injected = f"{args[0]}\n\n{context}"
        return (injected, *args[1:]), kwargs

    return args, kwargs


# ------------------------------------------------------------------
# The wrapper
# ------------------------------------------------------------------

def memory(
    fn: Callable | None = None,
    *,
    capability: str = "general",
    bot_id: str | None = None,
    db_path: str = "memory.db",
    judge: Callable | None = None,
    reflection: str = "simple",
    reflect_model: str = "claude-haiku",
    embedder: Any = None,
) -> Any:
    """Wrap a callable with biological memory.

    Can be used as a decorator or a function call:

        @memory
        def agent(prompt): ...

        agent = memory(agent)
        agent = memory(agent, capability="mining", bot_id="alpha")

    Args:
        reflection: "auto" (LLM-based), "simple" (heuristic default),
                    or a callable for custom evaluation.
        reflect_model: Model to use for LLM reflection (when reflection="auto").
        embedder: Optional Embedder instance for semantic similarity.
    """
    # Support @memory without parens (fn is the decorated function)
    if fn is None:
        def decorator(f: Callable) -> Callable:
            return memory(
                f, capability=capability, bot_id=bot_id, db_path=db_path,
                judge=judge, reflection=reflection, reflect_model=reflect_model,
                embedder=embedder,
            )
        return decorator

    if not callable(fn):
        raise TypeError(f"memory() expected a callable, got {type(fn).__name__}")

    mem = _get_or_create_memory(db_path, embedder=embedder)
    agent = MemoryAgent(memory=mem, capability=capability, bot_id=bot_id)

    # Determine the judge function
    if judge is not None:
        judge_fn = judge
    elif reflection == "auto":
        judge_fn = None  # Will use reflection
    else:
        judge_fn = _default_judge

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract query for context retrieval
            query = _extract_query(args, kwargs)

            # Inject memory context
            context = await agent.remember(query=query)
            args, kwargs = _inject_context(args, kwargs, context)

            # Call the original
            response = await fn(*args, **kwargs)

            # Judge and learn
            if judge_fn is not None:
                result = judge_fn(response)
            elif reflection == "auto":
                result = await _reflect_response(
                    query, response, mem, capability, reflect_model,
                )
            else:
                result = _default_judge(response)

            await agent.learn(
                result,
                input_text=query,
                output_text=str(response)[:2000],
            )

            return response

        return async_wrapper
    else:
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            loop = _get_event_loop()

            # Extract query for context retrieval
            query = _extract_query(args, kwargs)

            # Inject memory context
            context = loop.run_until_complete(agent.remember(query=query))
            args, kwargs = _inject_context(args, kwargs, context)

            # Call the original
            response = fn(*args, **kwargs)

            # Judge and learn
            if judge_fn is not None:
                result = judge_fn(response)
            else:
                result = _default_judge(response)

            loop.run_until_complete(agent.learn(
                result,
                input_text=query,
                output_text=str(response)[:2000],
            ))

            return response

        return sync_wrapper


async def _reflect_response(
    input_text: str,
    response: Any,
    memory_loop: MemoryLoop,
    capability: str,
    model: str,
) -> dict[str, Any]:
    """Use LLM reflection to evaluate a response."""
    try:
        from ganglion.memory.reflect import reflect
        output_text = str(response)[:2000]
        beliefs = await memory_loop.backend.query(capability=capability, limit=20)
        obs = await reflect(
            input_text=input_text,
            output_text=output_text,
            existing_beliefs=beliefs,
            capability=capability,
            model=model,
        )
        return {
            "success": obs.valence == Valence.POSITIVE,
            "description": obs.description,
            "entities": list(obs.entities),
            "tags": list(obs.tags),
        }
    except Exception as e:
        logger.warning("LLM reflection failed, using default judge: %s", e)
        return _default_judge(response)


# Avoid circular import at module level
from ganglion.memory.types import Valence  # noqa: E402


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop for sync wrapper use."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
