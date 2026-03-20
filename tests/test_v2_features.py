"""Tests for v2/v3 features: embedding-based similarity, query-aware context,
structured reflection, counterfactual evaluation, and dependency tracking.
"""

import tempfile
from pathlib import Path

import pytest

from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.embed import CallableEmbedder
from ganglion.memory.loop import MemoryLoop
from ganglion.memory.reflect import _simple_reflect
from ganglion.memory.types import Belief, Observation, Valence
from ganglion.memory.wrap import _default_judge, _extract_query


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# Helper: deterministic mock embedder
async def _mock_embed(text: str) -> list[float]:
    """Simple embedding: character frequency vector (26 dims)."""
    vec = [0.0] * 26
    for c in text.lower():
        if 'a' <= c <= 'z':
            vec[ord(c) - ord('a')] += 1.0
    # Normalize
    total = sum(v * v for v in vec) ** 0.5
    if total > 0:
        vec = [v / total for v in vec]
    return vec


@pytest.fixture
def mock_embedder():
    return CallableEmbedder(_mock_embed)


@pytest.fixture
def backend(tmp_dir):
    b = SqliteMemoryBackend(tmp_dir / "memory.db")
    yield b
    b.close()


@pytest.fixture
def loop_with_embedder(backend, mock_embedder):
    return MemoryLoop(backend=backend, embedder=mock_embedder)


# ======================================================================
# Embedding-based find_similar
# ======================================================================

@pytest.mark.asyncio
class TestEmbeddingFindSimilar:
    async def test_find_similar_with_embedding(self, loop_with_embedder):
        """Embedding-based similarity finds semantically similar beliefs."""
        loop = loop_with_embedder

        await loop.assimilate(Observation(
            capability="train", description="batch size works well",
            valence=Valence.POSITIVE,
        ))

        obs = Observation(
            capability="train", description="batch size well works",
            valence=Valence.POSITIVE,
        )
        embedding = await _mock_embed(obs.description)
        found = await loop.backend.find_similar(obs, threshold=0.5, embedding=embedding)
        assert found is not None

    async def test_different_text_no_match(self, loop_with_embedder):
        """Very different text doesn't match."""
        loop = loop_with_embedder

        await loop.assimilate(Observation(
            capability="train", description="aaaa",
            valence=Valence.POSITIVE,
        ))

        obs = Observation(
            capability="train", description="zzzz",
            valence=Valence.POSITIVE,
        )
        embedding = await _mock_embed(obs.description)
        found = await loop.backend.find_similar(obs, threshold=0.5, embedding=embedding)
        assert found is None

    async def test_embedding_stored_on_belief(self, loop_with_embedder):
        """Beliefs get embeddings when an embedder is configured."""
        loop = loop_with_embedder

        await loop.assimilate(Observation(
            capability="train", description="test embedding storage",
            valence=Valence.POSITIVE,
        ))

        beliefs = await loop.backend.all_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].embedding is not None
        assert len(beliefs[0].embedding) == 26

    async def test_hebbian_strengthening_with_embeddings(self, loop_with_embedder):
        """Repeated similar observations strengthen via embedding match."""
        loop = loop_with_embedder

        for _ in range(3):
            await loop.assimilate(Observation(
                capability="train", description="batch size works",
                valence=Valence.POSITIVE,
            ))

        beliefs = await loop.backend.all_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].confirmation_count == 3


# ======================================================================
# Embedding roundtrip (storage)
# ======================================================================

@pytest.mark.asyncio
class TestEmbeddingStorage:
    async def test_sqlite_embedding_roundtrip(self, tmp_dir):
        """SQLite stores and retrieves embedding blobs correctly."""
        backend = SqliteMemoryBackend(tmp_dir / "test.db")
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        belief = Belief(
            capability="x", description="test",
            valence=Valence.POSITIVE, embedding=embedding,
        )
        await backend.store(belief)

        beliefs = await backend.all_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].embedding is not None
        for a, b in zip(embedding, beliefs[0].embedding):
            assert abs(a - b) < 1e-6
        backend.close()

    async def test_belief_dict_roundtrip_with_embedding(self):
        """Belief.to_dict() / from_dict() preserves embeddings."""
        embedding = [0.1, 0.2, 0.3]
        belief = Belief(
            capability="x", description="test",
            valence=Valence.POSITIVE, embedding=embedding,
        )
        d = belief.to_dict()
        assert "embedding" in d

        restored = Belief.from_dict(d)
        assert restored.embedding is not None
        for a, b in zip(embedding, restored.embedding):
            assert abs(a - b) < 1e-6

    async def test_belief_dict_roundtrip_without_embedding(self):
        """Belief without embedding roundtrips cleanly."""
        belief = Belief(capability="x", description="test", valence=Valence.POSITIVE)
        d = belief.to_dict()
        assert "embedding" not in d

        restored = Belief.from_dict(d)
        assert restored.embedding is None


# ======================================================================
# Query-aware context_for
# ======================================================================

@pytest.mark.asyncio
class TestQueryAwareContext:
    async def test_context_for_with_query(self, loop_with_embedder):
        """Query-aware context retrieves relevant beliefs."""
        loop = loop_with_embedder

        await loop.assimilate(Observation(
            capability="coding", description="python async works great",
            valence=Valence.POSITIVE,
        ))
        await loop.assimilate(Observation(
            capability="coding", description="javascript promises fail",
            valence=Valence.NEGATIVE,
        ))

        ctx = await loop.context_for("coding", query="python async patterns")
        assert "python" in ctx

    async def test_context_for_without_query_falls_back_to_strength(self, loop_with_embedder):
        """Without a query, falls back to strength-based ranking."""
        loop = loop_with_embedder

        for _ in range(5):
            await loop.assimilate(Observation(
                capability="x", description="strong belief",
                valence=Valence.POSITIVE,
            ))
        await loop.assimilate(Observation(
            capability="x", description="weak belief",
            valence=Valence.POSITIVE,
        ))

        ctx = await loop.context_for("x", max_entries=1)
        assert "strong belief" in ctx


# ======================================================================
# Simple reflection (structured experience storage)
# ======================================================================

class TestSimpleReflect:
    def test_error_detection(self):
        obs = _simple_reflect(
            "do something", "Error: connection failed",
            "test", None,
        )
        assert obs.valence == Valence.NEGATIVE
        assert "failed" in obs.description.lower()

    def test_success_detection(self):
        obs = _simple_reflect(
            "do something", "Here is the result you requested with detailed analysis",
            "test", None,
        )
        assert obs.valence == Valence.POSITIVE

    def test_minimal_response_neutral(self):
        obs = _simple_reflect(
            "do something", "ok",
            "test", None,
        )
        assert obs.valence == Valence.NEUTRAL

    def test_stores_experience_tuple(self):
        """_simple_reflect stores input/output in config for few-shot retrieval."""
        obs = _simple_reflect(
            "solve x^2 = 4", "x = 2 or x = -2",
            "math", None,
        )
        assert obs.config is not None
        assert obs.config["input_text"] == "solve x^2 = 4"
        assert obs.config["output_text"] == "x = 2 or x = -2"

    def test_truncates_long_input(self):
        obs = _simple_reflect(
            "x" * 1000, "short response",
            "test", None,
        )
        assert len(obs.config["input_text"]) <= 500


# ======================================================================
# Updated default judge (improved heuristics)
# ======================================================================

class TestImprovedJudge:
    def test_detects_error_strings(self):
        result = _default_judge("Error: something went wrong")
        assert result["success"] is False

    def test_detects_success_strings(self):
        result = _default_judge("Here is your answer")
        assert result["success"] is True

    def test_detects_traceback(self):
        result = _default_judge("Traceback (most recent call last):")
        assert result["success"] is False

    def test_dict_passthrough(self):
        result = _default_judge({"success": False, "description": "bad"})
        assert result["success"] is False


# ======================================================================
# Query extraction from call args
# ======================================================================

class TestQueryExtraction:
    def test_extract_from_string_arg(self):
        query = _extract_query(("hello world",), {})
        assert query == "hello world"

    def test_extract_from_openai_messages(self):
        query = _extract_query((), {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is the weather?"},
            ]
        })
        assert "weather" in query

    def test_extract_last_user_message(self):
        query = _extract_query((), {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second question"},
            ]
        })
        assert query == "second question"

    def test_no_args_returns_empty(self):
        assert _extract_query((), {}) == ""


# ======================================================================
# Consolidation with embeddings
# ======================================================================

@pytest.mark.asyncio
class TestConsolidationWithEmbeddings:
    async def test_consolidation_uses_embeddings(self, tmp_dir, mock_embedder):
        """Consolidation clusters by embedding similarity when available."""
        backend = SqliteMemoryBackend(tmp_dir / "cons.db")
        loop = MemoryLoop(
            backend=backend, embedder=mock_embedder,
            max_beliefs=1, consolidation_threshold=0.5,
        )

        # Create beliefs with similar embeddings (similar text)
        for i in range(3):
            await loop.assimilate(Observation(
                capability="mining", description=f"batch works great run {i}",
                valence=Valence.POSITIVE,
                entities=("batch",), tags=("gpu",),
            ))

        await loop.backend.store(Belief(
            capability="mining", description="batch size great performance",
            valence=Valence.POSITIVE, entities=("batch",), tags=("gpu",),
            embedding=await _mock_embed("batch size great performance"),
        ))
        await loop.backend.store(Belief(
            capability="mining", description="batch approach works nicely",
            valence=Valence.POSITIVE, entities=("batch",), tags=("gpu",),
            embedding=await _mock_embed("batch approach works nicely"),
        ))

        all_before = await loop.backend.all_beliefs()
        await loop.forget()
        all_after = await loop.backend.all_beliefs()

        assert len(all_after) <= len(all_before)
        backend.close()


# ======================================================================
# Inhibition with embeddings
# ======================================================================

@pytest.mark.asyncio
class TestInhibitionWithEmbeddings:
    async def test_semantic_inhibition(self, tmp_dir, mock_embedder):
        """Lateral inhibition uses embedding similarity."""
        backend = SqliteMemoryBackend(tmp_dir / "inh.db")
        loop = MemoryLoop(backend=backend, embedder=mock_embedder, inhibition_rate=0.05)

        await loop.backend.store(Belief(
            capability="train", description="batch small works best",
            valence=Valence.POSITIVE, confidence=1.0,
            embedding=await _mock_embed("batch small works best"),
        ))
        await loop.backend.store(Belief(
            capability="train", description="batch large works best",
            valence=Valence.POSITIVE, confidence=1.0,
            embedding=await _mock_embed("batch large works best"),
        ))

        await loop.assimilate(Observation(
            capability="train", description="batch small works best",
            valence=Valence.POSITIVE,
        ))

        beliefs = await loop.backend.all_beliefs()
        competitor = next(b for b in beliefs if "large" in b.description)
        assert competitor.confirmation_count == 1
        backend.close()


# ======================================================================
# New imports
# ======================================================================

class TestNewImports:
    def test_new_modules_importable(self):
        import ganglion.memory as gm
        for name in [
            "cosine_similarity", "Embedder", "CallableEmbedder",
            "SentenceTransformerEmbedder", "get_embedder", "set_embedder",
        ]:
            assert hasattr(gm, name), f"{name} not exported from ganglion.memory"

    def test_removed_modules_not_exported(self):
        """Deleted modules should not be in __all__."""
        import ganglion.memory as gm
        for name in [
            "synthesize", "jaccard_similarity", "tokenize",
            "spread_activation", "temporal_neighbors",
            "JsonMemoryBackend", "FederatedMemoryBackend", "PeerDiscovery",
        ]:
            assert name not in gm.__all__, f"{name} should be removed from __all__"


# ======================================================================
# Counterfactual evaluation
# ======================================================================

class TestCounterfactual:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        import ganglion.memory.wrap as mod
        mod._default_memory = None
        yield
        mod._default_memory = None

    def test_no_context_single_call(self, tmp_dir):
        """When no memory context exists, only one LLM call happens."""
        call_count = 0

        def agent(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"response to: {prompt}"

        from ganglion.memory.wrap import memory
        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        result = wrapped("hello")
        assert "response to: hello" in result
        assert call_count == 1

    def test_with_context_two_calls(self, tmp_dir):
        """When memory context exists, two LLM calls happen (with and without)."""
        call_count = 0

        def agent(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"response {call_count}"

        from ganglion.memory.wrap import memory
        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))

        wrapped("hello")

        call_count = 0
        wrapped("hello")
        assert call_count == 2

    def test_async_no_context_single_call(self, tmp_dir):
        """Async: when no memory context, only one call happens."""
        import asyncio
        call_count = 0

        async def agent(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"response to: {prompt}"

        from ganglion.memory.wrap import memory
        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        result = asyncio.run(wrapped("hello"))
        assert "response to: hello" in result
        assert call_count == 1

    def test_clean_first_ordering(self, tmp_dir):
        """Clean call happens before memory check."""
        calls = []

        def agent(prompt: str) -> str:
            calls.append(prompt)
            return f"response to: {prompt}"

        from ganglion.memory.wrap import memory
        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))

        wrapped("hello")
        assert len(calls) == 1
        assert "Relevant experience" not in calls[0]
        assert "What we know" not in calls[0]

    def test_rollback_returns_clean_when_no_llm(self, tmp_dir):
        """Without LLM client, comparison returns 'same' — returns clean."""
        call_count = 0

        def agent(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"response {call_count}: {prompt}"

        from ganglion.memory.wrap import memory
        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))

        result1 = wrapped("hello")
        assert call_count == 1

        call_count = 0
        result2 = wrapped("hello")
        assert call_count == 2
        # Should return clean response (response 1 from this round)
        assert "response 1:" in result2


# ======================================================================
# Dependency tracking
# ======================================================================

@pytest.mark.asyncio
class TestDependencyTracking:
    async def test_produced_with_field_roundtrip(self):
        """produced_with survives to_dict/from_dict roundtrip."""
        b = Belief(
            id=1, capability="test", description="child",
            valence=Valence.POSITIVE, produced_with=(10, 20, 30),
        )
        d = b.to_dict()
        assert d["produced_with"] == [10, 20, 30]
        b2 = Belief.from_dict(d)
        assert b2.produced_with == (10, 20, 30)

    async def test_produced_with_stored_in_sqlite(self, tmp_dir):
        """SQLite backend preserves produced_with."""
        backend = SqliteMemoryBackend(tmp_dir / "dep.db")
        b = Belief(
            capability="test", description="child belief",
            valence=Valence.POSITIVE, produced_with=(5, 10),
        )
        await backend.store(b)
        beliefs = await backend.all_beliefs()
        assert beliefs[0].produced_with == (5, 10)
        backend.close()

    async def test_weaken_dependents_on_contradiction(self, tmp_dir):
        """When a parent belief dies, its dependents get weakened."""
        backend = SqliteMemoryBackend(tmp_dir / "dep2.db")
        loop = MemoryLoop(backend=backend, weaken_rate=0.5)

        parent = Belief(
            capability="test", description="parent approach",
            valence=Valence.POSITIVE, confidence=1.0,
        )
        await backend.store(parent)
        parent_id = parent.id

        child = Belief(
            capability="test", description="child conclusion",
            valence=Valence.POSITIVE, confidence=1.0,
            produced_with=(parent_id,),
        )
        await backend.store(child)

        neg = Observation(
            capability="test", description="parent approach",
            valence=Valence.NEGATIVE,
        )
        await loop.assimilate(neg)
        await loop.assimilate(neg)

        beliefs = await backend.all_beliefs()
        child_after = next(b for b in beliefs if b.description == "child conclusion")
        assert child_after.confidence < 1.0
        backend.close()

    async def test_weaken_dependents_recursive(self, tmp_dir):
        """Dependency weakening propagates through chains."""
        backend = SqliteMemoryBackend(tmp_dir / "dep3.db")
        loop = MemoryLoop(backend=backend, weaken_rate=0.5)

        grandparent = Belief(
            capability="test", description="grandparent idea",
            valence=Valence.POSITIVE, confidence=1.0,
        )
        await backend.store(grandparent)

        parent = Belief(
            capability="test", description="parent idea",
            valence=Valence.POSITIVE, confidence=1.0,
            produced_with=(grandparent.id,),
        )
        await backend.store(parent)

        child = Belief(
            capability="test", description="child idea",
            valence=Valence.POSITIVE, confidence=1.0,
            produced_with=(parent.id,),
        )
        await backend.store(child)

        neg = Observation(
            capability="test", description="grandparent idea",
            valence=Valence.NEGATIVE,
        )
        await loop.assimilate(neg)
        await loop.assimilate(neg)

        beliefs = await backend.all_beliefs()
        parent_after = next(b for b in beliefs if b.description == "parent idea")
        child_after = next(b for b in beliefs if b.description == "child idea")
        assert parent_after.confidence < 1.0
        assert child_after.confidence < 1.0
        backend.close()

    async def test_agent_stamps_dependencies(self, tmp_dir):
        """MemoryAgent.learn() stamps produced_with from _retrieved_beliefs."""
        from ganglion.memory.agent import MemoryAgent
        backend = SqliteMemoryBackend(tmp_dir / "dep4.db")
        loop = MemoryLoop(backend=backend)

        seed = Belief(
            capability="test", description="seed knowledge",
            valence=Valence.POSITIVE,
        )
        await backend.store(seed)

        agent = MemoryAgent(memory=loop, capability="test")
        await agent.remember()

        await agent.learn({
            "success": True,
            "description": "new conclusion based on seed",
        })

        beliefs = await backend.all_beliefs()
        new_belief = next(
            (b for b in beliefs if "new conclusion" in b.description), None
        )
        assert new_belief is not None
        assert seed.id in new_belief.produced_with
        backend.close()
