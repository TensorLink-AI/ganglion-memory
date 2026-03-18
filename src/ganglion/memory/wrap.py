"""One-line wrapper API for ganglion-memory.

Usage:
    from ganglion.memory import memory

    agent = memory(agent)

That's it. The wrapper auto-detects call conventions (OpenAI messages,
Anthropic system+messages, or plain string) and injects memory context
before each call, then evaluates the response after.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any, Callable

from ganglion.memory.agent import MemoryAgent, between_runs
from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.loop import MemoryLoop

logger = logging.getLogger(__name__)

# Module-level singleton so multiple memory() calls share one store per process.
_default_memory: MemoryLoop | None = None


def _get_or_create_memory(db_path: str = "memory.db") -> MemoryLoop:
    """Return the module-level singleton, creating it on first use."""
    global _default_memory
    if _default_memory is None:
        backend = SqliteMemoryBackend(db_path)
        _default_memory = MemoryLoop(backend=backend)
    return _default_memory


# ------------------------------------------------------------------
# Default judge — auto-detects common response types
# ------------------------------------------------------------------

def _default_judge(response: Any) -> dict[str, Any]:
    """Auto-detect response type and build a result dict.

    Handles:
    - dict with 'success' key → pass through
    - str → successful observation
    - OpenAI ChatCompletion-like → extract .choices[0].message.content
    - Anthropic Message-like → extract .content[0].text
    - Anything else → str() it
    """
    if isinstance(response, dict):
        if "success" in response:
            return response
        # Dict without success key — treat as successful
        desc = str(response)[:500]
        return {"success": True, "description": desc}

    if isinstance(response, str):
        return {"success": True, "description": response[:500]}

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
) -> Any:
    """Wrap a callable with biological memory.

    Can be used as a decorator or a function call:

        @memory
        def agent(prompt): ...

        agent = memory(agent)
        agent = memory(agent, capability="mining", bot_id="alpha")
    """
    # Support @memory without parens (fn is the decorated function)
    if fn is None:
        # Called with keyword args: memory(capability="mining")
        # Returns a decorator
        def decorator(f: Callable) -> Callable:
            return memory(f, capability=capability, bot_id=bot_id, db_path=db_path, judge=judge)
        return decorator

    # If fn is not callable, it was used as @memory(capability=...) incorrectly
    if not callable(fn):
        raise TypeError(f"memory() expected a callable, got {type(fn).__name__}")

    mem = _get_or_create_memory(db_path)
    agent = MemoryAgent(memory=mem, capability=capability, bot_id=bot_id)
    judge_fn = judge or _default_judge

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Inject memory context
            context = await agent.remember()
            args, kwargs = _inject_context(args, kwargs, context)

            # Call the original
            response = await fn(*args, **kwargs)

            # Judge and learn
            result = judge_fn(response)
            await agent.learn(result)

            return response

        return async_wrapper
    else:
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Inject memory context
            loop = _get_event_loop()
            context = loop.run_until_complete(agent.remember())
            args, kwargs = _inject_context(args, kwargs, context)

            # Call the original
            response = fn(*args, **kwargs)

            # Judge and learn
            result = judge_fn(response)
            loop.run_until_complete(agent.learn(result))

            return response

        return sync_wrapper


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
