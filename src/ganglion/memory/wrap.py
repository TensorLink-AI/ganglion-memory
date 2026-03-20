"""One-line wrapper API for ganglion-memory.

Usage:
    from ganglion.memory import memory

    agent = memory(agent)

That's it. The wrapper auto-detects call conventions (OpenAI messages,
Anthropic system+messages, or plain string) and injects memory context
before each call, then evaluates the response after.

v3: Counterfactual evaluation with rollback — runs with and without
    memory, keeps the better output, trains beliefs from the comparison.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from datetime import UTC, datetime
from typing import Any, Callable

from ganglion.memory.agent import MemoryAgent
from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.embed import cosine_similarity
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
    # OpenAI/Anthropic-style: messages kwarg with user messages
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
# Counterfactual evaluation
# ------------------------------------------------------------------

async def _evaluate_counterfactual(
    query: str,
    response_clean: Any,
    response_mem: Any,
    memory_loop: MemoryLoop,
    model: str,
) -> str:
    """Compare with-memory and without-memory outputs.

    Returns: "memory" | "clean" | "same"
    """
    # First: quick embedding check — are outputs meaningfully different?
    if memory_loop.embedder:
        emb_clean = await memory_loop._embed(str(response_clean)[:500])
        emb_mem = await memory_loop._embed(str(response_mem)[:500])
        if emb_clean and emb_mem:
            similarity = cosine_similarity(emb_clean, emb_mem)
            if similarity > 0.95:
                return "same"  # outputs nearly identical, memory did nothing

    # Outputs differ — ask LLM which is better
    clean_text = str(response_clean)[:1500]
    mem_text = str(response_mem)[:1500]

    prompt = f"""Compare two responses to the same task. One had no prior context (A), the other had prior experience injected (B).

TASK: {query[:1000]}

RESPONSE A (no context):
{clean_text}

RESPONSE B (with experience):
{mem_text}

Which response demonstrates better reasoning and is more likely correct?
Answer ONLY with the single letter "A" or "B"."""

    try:
        client = _get_comparison_client()
        if client is None:
            return "same"  # Can't compare — conservative, treat as no difference

        if hasattr(client, "messages"):
            resp = await client.messages.create(
                model=model, max_tokens=5,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = resp.content[0].text.strip().upper()
        elif hasattr(client, "chat"):
            resp = await client.chat.completions.create(
                model=model, max_tokens=5,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = resp.choices[0].message.content.strip().upper()
        else:
            return "same"

        if "B" in answer:
            return "memory"
        elif "A" in answer:
            return "clean"
        return "same"
    except Exception:
        return "same"


def _get_comparison_client() -> Any:
    """Get an async LLM client for counterfactual comparison."""
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
        reflection: "simple" (heuristic default) or a callable for custom evaluation.
        reflect_model: Model to use for counterfactual LLM comparison.
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
    judge_fn = judge if judge is not None else _default_judge

    def _judge(resp):
        return judge_fn(resp)

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            query = _extract_query(args, kwargs)
            original_args, original_kwargs = args, kwargs

            # Phase 1: run WITHOUT memory
            response_clean = await fn(*args, **kwargs)

            # Phase 2: check if memory has anything relevant
            context = await agent.remember(query=query)

            if not context:
                # No relevant memory — learn from clean response, return it
                result = _judge(response_clean)
                if query and "description" in result:
                    result["description"] = f"{result['description'][:250]} [task: {query[:200]}]"
                await agent.learn(result, input_text=query, output_text=str(response_clean)[:2000])
                return response_clean

            # Phase 3: run WITH memory
            mem_args, mem_kwargs = _inject_context(original_args, original_kwargs, context)
            response_mem = await fn(*mem_args, **mem_kwargs)

            # Phase 4: compare outputs
            memory_helped = await _evaluate_counterfactual(
                query, response_clean, response_mem, mem, reflect_model,
            )

            if memory_helped == "memory":
                # Memory improved the output — strengthen retrieved beliefs
                for belief in agent._retrieved_beliefs:
                    if belief.id is not None:
                        belief.confidence = min(10.0, belief.confidence + 0.1)
                        belief.last_retrieved = datetime.now(UTC)
                        await mem.backend.update(belief)
                response = response_mem
            elif memory_helped == "same":
                # Memory made no difference — mildly weaken (it's wasting context)
                for belief in agent._retrieved_beliefs:
                    if belief.id is not None:
                        belief.confidence = max(0.1, belief.confidence - 0.05)
                        await mem.backend.update(belief)
                response = response_clean
            else:
                # Memory hurt — rollback, weaken aggressively
                for belief in agent._retrieved_beliefs:
                    if belief.id is not None:
                        belief.confidence = max(0.1, belief.confidence - 0.15)
                        await mem.backend.update(belief)
                response = response_clean

            result = _judge(response)
            if query and "description" in result:
                result["description"] = f"{result['description'][:250]} [task: {query[:200]}]"
            await agent.learn(result, input_text=query, output_text=str(response)[:2000])
            return response

        return async_wrapper
    else:
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            loop = _get_event_loop()
            query = _extract_query(args, kwargs)
            original_args, original_kwargs = args, kwargs

            # Phase 1: run WITHOUT memory
            response_clean = fn(*args, **kwargs)

            # Phase 2: check if memory has anything relevant
            context = loop.run_until_complete(agent.remember(query=query))

            if not context:
                result = _judge(response_clean)
                if query and "description" in result:
                    result["description"] = f"{result['description'][:250]} [task: {query[:200]}]"
                loop.run_until_complete(
                    agent.learn(result, input_text=query, output_text=str(response_clean)[:2000])
                )
                return response_clean

            # Phase 3: run WITH memory
            mem_args, mem_kwargs = _inject_context(original_args, original_kwargs, context)
            response_mem = fn(*mem_args, **mem_kwargs)

            # Phase 4: compare outputs
            memory_helped = loop.run_until_complete(
                _evaluate_counterfactual(
                    query, response_clean, response_mem, mem, reflect_model,
                )
            )

            if memory_helped == "memory":
                for belief in agent._retrieved_beliefs:
                    if belief.id is not None:
                        belief.confidence = min(10.0, belief.confidence + 0.1)
                        belief.last_retrieved = datetime.now(UTC)
                        loop.run_until_complete(mem.backend.update(belief))
                response = response_mem
            elif memory_helped == "same":
                for belief in agent._retrieved_beliefs:
                    if belief.id is not None:
                        belief.confidence = max(0.1, belief.confidence - 0.05)
                        loop.run_until_complete(mem.backend.update(belief))
                response = response_clean
            else:
                for belief in agent._retrieved_beliefs:
                    if belief.id is not None:
                        belief.confidence = max(0.1, belief.confidence - 0.15)
                        loop.run_until_complete(mem.backend.update(belief))
                response = response_clean

            result = _judge(response)
            if query and "description" in result:
                result["description"] = f"{result['description'][:250]} [task: {query[:200]}]"
            loop.run_until_complete(
                agent.learn(result, input_text=query, output_text=str(response)[:2000])
            )
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
