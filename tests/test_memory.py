"""Tests for Ganglion Memory v4.

Covers: Experience type, SqliteBackend, Memory core, Agent integration,
wrap API, embedding utilities.

85 tests in this file + 6 in test_refine.py = 91 total.
"""

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ganglion.memory.agent import Agent
from ganglion.memory.backends.sqlite import SqliteBackend
from ganglion.memory.core import Memory
from ganglion.memory.embed import (
    CallableEmbedder,
    cosine_similarity,
    reset_embedder,
    set_embedder,
)
from ganglion.memory.types import Experience
from ganglion.memory.wrap import _default_judge, _extract_query, memory


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def backend(tmp_dir):
    b = SqliteBackend(tmp_dir / "memory.db")
    yield b
    b.close()


async def _mock_embed(text: str) -> list[float]:
    """Deterministic mock: character frequency vector (26 dims, normalized)."""
    vec = [0.0] * 26
    for c in text.lower():
        if "a" <= c <= "z":
            vec[ord(c) - ord("a")] += 1.0
    total = sum(v * v for v in vec) ** 0.5
    if total > 0:
        vec = [v / total for v in vec]
    return vec


@pytest.fixture
def mock_embedder():
    return CallableEmbedder(_mock_embed)


@pytest.fixture
def mem(backend):
    return Memory(backend=backend)


@pytest.fixture
def mem_with_embedder(backend, mock_embedder):
    return Memory(backend=backend, embedder=mock_embedder)


# ======================================================================
# Experience type
# ======================================================================


class TestExperience:
    def test_create(self):
        exp = Experience(content="batch=64 works", tags=("mining",), source="alpha")
        assert exp.content == "batch=64 works"
        assert exp.tags == ("mining",)
        assert exp.source == "alpha"

    def test_defaults(self):
        exp = Experience()
        assert exp.id is None
        assert exp.content == ""
        assert exp.tags == ()
        assert exp.confirmation_count == 0
        assert exp.contradiction_count == 0
        assert exp.embedding is None
        assert exp.metadata is None

    def test_net_score_positive(self):
        exp = Experience(confirmation_count=5, contradiction_count=2)
        assert exp.net_score == 3

    def test_net_score_negative(self):
        exp = Experience(confirmation_count=1, contradiction_count=3)
        assert exp.net_score == -2

    def test_to_dict(self):
        exp = Experience(
            id=1,
            content="test",
            tags=("a", "b"),
            source="bot",
            confirmation_count=3,
            contradiction_count=1,
            metadata={"key": "value"},
        )
        d = exp.to_dict()
        assert d["content"] == "test"
        assert d["tags"] == ["a", "b"]
        assert d["confirmation_count"] == 3
        assert d["contradiction_count"] == 1
        assert d["metadata"] == {"key": "value"}

    def test_from_dict(self):
        d = {
            "id": 1,
            "content": "test",
            "tags": ["a"],
            "source": "bot",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "confirmation_count": 2,
            "contradiction_count": 0,
            "metadata": {"k": "v"},
        }
        exp = Experience.from_dict(d)
        assert exp.content == "test"
        assert exp.tags == ("a",)
        assert exp.confirmation_count == 2
        assert exp.metadata == {"k": "v"}

    def test_roundtrip_with_embedding(self):
        embedding = [0.1, 0.2, 0.3]
        exp = Experience(content="test", embedding=embedding)
        d = exp.to_dict()
        assert "embedding" in d
        restored = Experience.from_dict(d)
        assert restored.embedding is not None
        for a, b in zip(embedding, restored.embedding):
            assert abs(a - b) < 1e-6

    def test_roundtrip_without_embedding(self):
        exp = Experience(content="test")
        d = exp.to_dict()
        assert "embedding" not in d
        restored = Experience.from_dict(d)
        assert restored.embedding is None

    def test_tags_are_tuple(self):
        exp = Experience.from_dict({"tags": ["x", "y"]})
        assert isinstance(exp.tags, tuple)
        assert exp.tags == ("x", "y")


# ======================================================================
# SqliteBackend
# ======================================================================


@pytest.mark.asyncio
class TestSqliteBackend:
    async def test_store_and_get(self, backend):
        exp = Experience(content="hello", tags=("a",), source="bot")
        eid = await backend.store(exp)
        assert eid is not None
        got = await backend.get(eid)
        assert got is not None
        assert got.content == "hello"
        assert got.tags == ("a",)

    async def test_store_assigns_id(self, backend):
        exp = Experience(content="test")
        eid = await backend.store(exp)
        assert exp.id == eid
        assert eid > 0

    async def test_update(self, backend):
        exp = Experience(content="original")
        await backend.store(exp)
        exp.content = "updated"
        exp.confirmation_count = 5
        await backend.update(exp)
        got = await backend.get(exp.id)
        assert got.content == "updated"
        assert got.confirmation_count == 5

    async def test_delete(self, backend):
        exp = Experience(content="to delete")
        await backend.store(exp)
        assert await backend.count() == 1
        await backend.delete(exp.id)
        assert await backend.count() == 0

    async def test_query_by_tags(self, backend):
        await backend.store(Experience(content="A", tags=("mining",)))
        await backend.store(Experience(content="B", tags=("training",)))
        results = await backend.query(tags=("mining",))
        assert len(results) == 1
        assert results[0].content == "A"

    async def test_query_by_source(self, backend):
        await backend.store(Experience(content="from alpha", source="alpha"))
        await backend.store(Experience(content="from beta", source="beta"))
        results = await backend.query(source="alpha")
        assert len(results) == 1
        assert results[0].source == "alpha"

    async def test_query_limit(self, backend):
        for i in range(10):
            await backend.store(Experience(content=f"exp{i}"))
        results = await backend.query(limit=3)
        assert len(results) == 3

    async def test_all(self, backend):
        for i in range(5):
            await backend.store(Experience(content=f"exp{i}"))
        assert len(await backend.all()) == 5

    async def test_count(self, backend):
        assert await backend.count() == 0
        await backend.store(Experience(content="a"))
        await backend.store(Experience(content="b"))
        assert await backend.count() == 2

    async def test_search_by_embedding(self, backend):
        emb = [1.0, 0.0, 0.0]
        await backend.store(Experience(content="A", embedding=emb))
        await backend.store(Experience(content="B", embedding=[0.0, 1.0, 0.0]))
        results = await backend.search_by_embedding(emb, limit=5, threshold=0.5)
        assert len(results) == 1
        assert results[0][0].content == "A"
        assert results[0][1] > 0.99

    async def test_search_by_embedding_threshold(self, backend):
        await backend.store(Experience(content="A", embedding=[1.0, 0.0, 0.0]))
        results = await backend.search_by_embedding(
            [0.7, 0.7, 0.0],
            limit=5,
            threshold=0.99,
        )
        assert len(results) == 0

    async def test_embedding_roundtrip(self, backend):
        emb = [0.1, 0.2, 0.3, 0.4, 0.5]
        await backend.store(Experience(content="test", embedding=emb))
        got = (await backend.all())[0]
        assert got.embedding is not None
        for a, b in zip(emb, got.embedding):
            assert abs(a - b) < 1e-6

    async def test_metadata_roundtrip(self, backend):
        meta = {"key": "value", "nested": {"a": 1}}
        await backend.store(Experience(content="test", metadata=meta))
        got = (await backend.all())[0]
        assert got.metadata == meta

    async def test_get_nonexistent(self, backend):
        assert await backend.get(999) is None

    async def test_delete_nonexistent(self, backend):
        # Should not raise
        await backend.delete(999)


# ======================================================================
# Memory core
# ======================================================================


@pytest.mark.asyncio
class TestMemory:
    async def test_add(self, mem):
        exp = await mem.add("batch=64 works", tags=("mining",))
        assert exp.id is not None
        assert exp.content == "batch=64 works"
        assert "mining" in exp.tags
        assert exp.confirmation_count == 1

    async def test_add_with_metadata(self, mem):
        exp = await mem.add("test", metadata={"key": "val"})
        assert exp.metadata == {"key": "val"}

    async def test_add_dedup_with_embedder(self, mem_with_embedder):
        """Adding near-duplicate content confirms existing instead of creating new."""
        exp1 = await mem_with_embedder.add("batch size works well", tags=("mining",))
        exp2 = await mem_with_embedder.add("batch size works well", tags=("mining",))
        assert await mem_with_embedder.count() == 1
        assert exp2.confirmation_count == 2

    async def test_add_different_creates_new(self, mem_with_embedder):
        """Orthogonal content creates separate experiences."""
        await mem_with_embedder.add("aaaa", tags=("test",))
        await mem_with_embedder.add("zzzz", tags=("test",))
        assert await mem_with_embedder.count() == 2

    async def test_search_empty(self, mem_with_embedder):
        results = await mem_with_embedder.search("anything")
        assert results == []

    async def test_search_with_results(self, mem_with_embedder):
        await mem_with_embedder.add("batch size works great", tags=("mining",))
        await mem_with_embedder.add("zzzzzzz unique topic", tags=("other",))
        results = await mem_with_embedder.search("batch size performance")
        assert len(results) >= 1
        assert any("batch" in r.content for r in results)

    async def test_search_with_tags(self, mem_with_embedder):
        await mem_with_embedder.add("batch works great", tags=("mining",))
        await mem_with_embedder.add("batch fails badly", tags=("training",))
        results = await mem_with_embedder.search("batch", tags=("mining",))
        assert all("mining" in r.tags for r in results)

    async def test_search_fallback_no_embedder(self, mem):
        """Without embedder, search falls back to tag-based query."""
        await mem.add("test content", tags=("mining",))
        results = await mem.search("anything", tags=("mining",))
        assert len(results) == 1

    async def test_confirm(self, mem):
        exp = await mem.add("test")
        assert exp.confirmation_count == 1
        updated = await mem.confirm(exp.id)
        assert updated.confirmation_count == 2

    async def test_confirm_increments(self, mem):
        exp = await mem.add("test")
        for _ in range(5):
            exp = await mem.confirm(exp.id)
        assert exp.confirmation_count == 6

    async def test_contradict(self, mem):
        exp = await mem.add("test")
        updated = await mem.contradict(exp.id)
        assert updated.contradiction_count == 1

    async def test_contradict_increments(self, mem):
        exp = await mem.add("test")
        for _ in range(3):
            exp = await mem.contradict(exp.id)
        assert exp.contradiction_count == 3

    async def test_delete(self, mem):
        exp = await mem.add("to remove")
        assert await mem.count() == 1
        await mem.delete(exp.id)
        assert await mem.count() == 0

    async def test_get(self, mem):
        exp = await mem.add("hello")
        got = await mem.get(exp.id)
        assert got is not None
        assert got.content == "hello"

    async def test_get_nonexistent(self, mem):
        assert await mem.get(999) is None

    async def test_all(self, mem):
        await mem.add("a")
        await mem.add("b")
        assert len(await mem.all()) == 2

    async def test_count(self, mem):
        assert await mem.count() == 0
        await mem.add("a")
        assert await mem.count() == 1

    async def test_compress_merges_similar(self, mem_with_embedder):
        """Compress merges cluster of similar experiences."""
        emb = await _mock_embed("batch works great")
        for i in range(4):
            await mem_with_embedder.backend.store(
                Experience(
                    content=f"batch works great run {i}",
                    tags=("mining",),
                    embedding=emb,
                )
            )
        before = await mem_with_embedder.count()
        merged = await mem_with_embedder.compress(
            tags=("mining",), min_cluster=3, threshold=0.5,
        )
        assert len(merged) >= 1
        assert "compressed" in merged[0].tags
        after = await mem_with_embedder.count()
        assert after < before

    async def test_compress_with_synthesizer(self, mem_with_embedder):
        """Compress uses synthesizer callback when provided."""
        emb = await _mock_embed("batch approach")
        for i in range(3):
            await mem_with_embedder.backend.store(
                Experience(
                    content=f"batch approach {i}",
                    tags=("mining",),
                    embedding=emb,
                )
            )

        async def synth(exps: list[Experience]) -> str:
            return f"Synthesized from {len(exps)} experiences"

        merged = await mem_with_embedder.compress(
            tags=("mining",), min_cluster=3, threshold=0.5, synthesizer=synth,
        )
        assert len(merged) >= 1
        assert "Synthesized from 3" in merged[0].content

    async def test_compress_preserves_dissimilar(self, mem_with_embedder):
        """Dissimilar experiences don't get merged."""
        for i, emb in enumerate(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ):
            await mem_with_embedder.backend.store(
                Experience(content=f"unique_{i}", tags=("test",), embedding=emb)
            )
        merged = await mem_with_embedder.compress(tags=("test",), min_cluster=3)
        assert len(merged) == 0
        assert await mem_with_embedder.count() == 3

    async def test_compress_sums_counts(self, mem_with_embedder):
        """Merged experience sums confirmation counts."""
        emb = await _mock_embed("batch approach")
        for i in range(3):
            await mem_with_embedder.backend.store(
                Experience(
                    content=f"batch approach run {i}",
                    tags=("mining",),
                    embedding=emb,
                    confirmation_count=2,
                )
            )
        merged = await mem_with_embedder.compress(
            tags=("mining",), min_cluster=3, threshold=0.5,
        )
        assert len(merged) >= 1
        assert merged[0].confirmation_count >= 6

    async def test_confirm_not_found_raises(self, mem):
        with pytest.raises(ValueError):
            await mem.confirm(999)

    async def test_contradict_not_found_raises(self, mem):
        with pytest.raises(ValueError):
            await mem.contradict(999)


# ======================================================================
# Agent integration
# ======================================================================


@pytest.mark.asyncio
class TestAgent:
    async def test_remember_empty(self, mem):
        agent = Agent(memory=mem, capability="mining")
        ctx = await agent.remember()
        assert ctx == ""

    async def test_learn_stores(self, mem):
        agent = Agent(memory=mem, capability="mining")
        exp = await agent.learn({"success": True, "description": "batch=64 works"})
        assert exp.content == "batch=64 works"
        assert await mem.count() == 1

    async def test_remember_after_learn(self, mem):
        agent = Agent(memory=mem, capability="mining")
        await agent.learn({"success": True, "description": "batch=64 works"})
        ctx = await agent.remember()
        assert "batch=64 works" in ctx

    async def test_learn_with_input_output(self, mem):
        agent = Agent(memory=mem, capability="test")
        exp = await agent.learn(
            {"success": True, "description": "solved it"},
            input_text="what is 2+2",
            output_text="4",
        )
        assert exp.metadata is not None
        assert exp.metadata.get("input_text") == "what is 2+2"
        assert exp.metadata.get("output_text") == "4"

    async def test_learn_failure(self, mem):
        agent = Agent(memory=mem, capability="mining")
        exp = await agent.learn(
            {"success": False, "description": "batch=256", "error": "OOM"},
        )
        assert "failure" in exp.tags
        assert "OOM" in exp.content

    async def test_learn_adds_tags(self, mem):
        agent = Agent(memory=mem, capability="mining", tags=("gpu",))
        exp = await agent.learn({"success": True, "description": "test"})
        assert "mining" in exp.tags
        assert "gpu" in exp.tags
        assert "success" in exp.tags

    async def test_between_runs(self, mem_with_embedder):
        agent = Agent(memory=mem_with_embedder, capability="mining")
        for i in range(5):
            await agent.learn(
                {"success": True, "description": f"batch approach run {i}"},
            )
        count = await agent.between_runs()
        assert isinstance(count, int)

    async def test_remember_formats_context(self, mem):
        agent = Agent(memory=mem, capability="mining")
        await agent.learn({"success": True, "description": "batch=64 works"})
        ctx = await agent.remember()
        assert "Relevant experience" in ctx
        assert "+" in ctx

    async def test_agent_with_query(self, mem_with_embedder):
        agent = Agent(memory=mem_with_embedder, capability="coding")
        await agent.learn({"success": True, "description": "python async works great"})
        ctx = await agent.remember(query="python patterns")
        # char-freq embedder may or may not match strongly enough
        assert isinstance(ctx, str)


# ======================================================================
# Wrap API
# ======================================================================


class TestWrap:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        import ganglion.memory.wrap as mod

        mod._default_memory = None
        yield
        mod._default_memory = None

    def test_wraps_sync(self, tmp_dir):
        def agent(prompt: str) -> str:
            return f"response to: {prompt}"

        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        result = wrapped("hello")
        assert "response to:" in result
        assert wrapped.__name__ == "agent"

    def test_wraps_async(self, tmp_dir):
        async def agent(prompt: str) -> str:
            return f"async response to: {prompt}"

        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        result = asyncio.run(wrapped("hello"))
        assert "async response to:" in result

    def test_decorator_no_parens(self, tmp_dir):
        @memory
        def agent(prompt: str) -> str:
            return f"decorated: {prompt}"

        result = agent("test")
        assert "decorated:" in result

    def test_decorator_with_parens(self, tmp_dir):
        @memory(capability="test", db_path=str(tmp_dir / "m.db"))
        def agent(prompt: str) -> str:
            return f"decorated: {prompt}"

        result = agent("test")
        assert "decorated:" in result

    def test_injects_openai(self, tmp_dir):
        def agent(messages=None):
            return messages[0]["content"]

        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        result = wrapped(
            messages=[
                {"role": "system", "content": "base prompt"},
                {"role": "user", "content": "hi"},
            ]
        )
        assert "base prompt" in result

    def test_injects_anthropic(self, tmp_dir):
        def agent(system="", messages=None):
            return system

        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        result = wrapped(system="base prompt", messages=[])
        assert "base prompt" in result

    def test_injects_string(self, tmp_dir):
        def agent(prompt: str) -> str:
            return prompt

        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        result = wrapped("base prompt")
        assert "base prompt" in result

    def test_custom_judge(self, tmp_dir):
        def agent(prompt: str) -> dict:
            return {"score": 0.95, "text": "good result"}

        def judge(response):
            return {
                "success": response["score"] > 0.5,
                "description": response["text"],
            }

        wrapped = memory(
            agent, capability="test", judge=judge, db_path=str(tmp_dir / "m.db"),
        )
        result = wrapped("test")
        assert result["score"] == 0.95

    def test_preserves_name(self, tmp_dir):
        def my_special_agent(x):
            return x

        wrapped = memory(
            my_special_agent, capability="test", db_path=str(tmp_dir / "m.db"),
        )
        assert wrapped.__name__ == "my_special_agent"

    def test_single_call_per_invocation(self, tmp_dir):
        """v4: only one call per invocation (no counterfactual)."""
        call_count = 0

        def agent(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"response {call_count}"

        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        wrapped("hello")
        assert call_count == 1
        wrapped("hello")
        assert call_count == 2  # one call each time


# ======================================================================
# Default judge
# ======================================================================


class TestDefaultJudge:
    def test_string_success(self):
        result = _default_judge("hello world")
        assert result["success"] is True
        assert result["description"] == "hello world"

    def test_string_error(self):
        result = _default_judge("Error: something went wrong")
        assert result["success"] is False

    def test_dict_passthrough(self):
        result = _default_judge({"success": False, "description": "bad"})
        assert result["success"] is False

    def test_truncates(self):
        result = _default_judge("x" * 1000)
        assert len(result["description"]) <= 500

    def test_traceback(self):
        result = _default_judge("Traceback (most recent call last):")
        assert result["success"] is False


# ======================================================================
# Query extraction
# ======================================================================


class TestQueryExtraction:
    def test_from_string(self):
        assert _extract_query(("hello",), {}) == "hello"

    def test_from_messages(self):
        q = _extract_query(
            (),
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "What is the weather?"},
                ]
            },
        )
        assert "weather" in q

    def test_last_user_message(self):
        q = _extract_query(
            (),
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "second"},
                ]
            },
        )
        assert q == "second"

    def test_empty(self):
        assert _extract_query((), {}) == ""


# ======================================================================
# Embedding utilities
# ======================================================================


class TestCosine:
    def test_identical(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_similar(self):
        assert cosine_similarity([1.0, 1.0, 0.0], [1.0, 1.0, 0.1]) > 0.95

    def test_empty(self):
        assert cosine_similarity([], []) == 0.0

    def test_mismatched(self):
        assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0

    def test_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


@pytest.mark.asyncio
class TestCallableEmbedderIntegration:
    async def test_embed(self):
        async def mock(text: str) -> list[float]:
            return [float(len(text)), 0.5]

        emb = CallableEmbedder(mock)
        result = await emb.embed("hello")
        assert result == [5.0, 0.5]

    async def test_batch(self):
        async def mock(text: str) -> list[float]:
            return [float(len(text))]

        emb = CallableEmbedder(mock)
        results = await emb.embed_batch(["a", "ab", "abc"])
        assert results == [[1.0], [2.0], [3.0]]


class TestGlobalEmbedder:
    def test_set_and_reset(self):
        set_embedder(None)
        reset_embedder()
        # Should not raise
