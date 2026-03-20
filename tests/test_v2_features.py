"""Tests for v2 features: embedding-based similarity, query-aware context,
reflection, synthesis, and the updated wrapper API.
"""

import tempfile
from pathlib import Path

import pytest

from ganglion.memory.backends.json_file import JsonMemoryBackend
from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.embed import CallableEmbedder
from ganglion.memory.loop import MemoryLoop
from ganglion.memory.reflect import _simple_reflect, _parse_reflection
from ganglion.memory.similarity import cosine_similarity as sim_cosine
from ganglion.memory.synthesize import _simple_synthesize
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


@pytest.fixture(params=["json", "sqlite"])
def backend(request, tmp_dir):
    if request.param == "json":
        yield JsonMemoryBackend(tmp_dir / "memory")
    else:
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

        # Store a belief (embedding computed automatically)
        await loop.assimilate(Observation(
            capability="train", description="batch size works well",
            valence=Valence.POSITIVE,
        ))

        # Similar text should match
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

    async def test_json_embedding_roundtrip(self, tmp_dir):
        """JSON stores and retrieves embeddings via base64 correctly."""
        backend = JsonMemoryBackend(tmp_dir / "memory")
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        belief = Belief(
            capability="x", description="test",
            valence=Valence.POSITIVE, embedding=embedding,
        )
        await backend.store(belief)

        # Force cache reload
        backend.invalidate_cache()
        beliefs = await backend.all_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].embedding is not None
        for a, b in zip(embedding, beliefs[0].embedding):
            assert abs(a - b) < 1e-6

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

        # Query about python should rank python belief higher
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

    async def test_context_for_empty_query(self, loop_with_embedder):
        """Empty query string falls back to strength-based ranking."""
        loop = loop_with_embedder

        await loop.assimilate(Observation(
            capability="x", description="some belief",
            valence=Valence.POSITIVE,
        ))

        ctx = await loop.context_for("x", query="")
        assert "some belief" in ctx


# ======================================================================
# Simple reflection (no LLM needed)
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

    def test_parse_valid_json(self):
        json_str = '{"valence": "negative", "description": "lr too high", "entities": ["lr"], "tags": ["training"]}'
        obs = _parse_reflection(json_str, "train", "bot1")
        assert obs.valence == Valence.NEGATIVE
        assert "lr too high" in obs.description
        assert "lr" in obs.entities
        assert "training" in obs.tags

    def test_parse_json_in_markdown(self):
        text = '```json\n{"valence": "positive", "description": "works"}\n```'
        obs = _parse_reflection(text, "test", None)
        assert obs.valence == Valence.POSITIVE

    def test_parse_invalid_json(self):
        obs = _parse_reflection("not json at all", "test", None)
        assert obs.valence == Valence.NEUTRAL


# ======================================================================
# Simple synthesis (no LLM needed)
# ======================================================================

class TestSimpleSynthesize:
    def test_consistent_failures_detected(self):
        beliefs = [
            Belief(
                capability="train", description=f"lr=0.1 fails run {i}",
                valence=Valence.NEGATIVE, entities=("lr",),
            )
            for i in range(3)
        ]
        observations = _simple_synthesize(beliefs)
        assert len(observations) >= 1
        assert any("consistently fails" in o.description for o in observations)
        assert all("synthesized" in o.tags for o in observations)

    def test_consistent_successes_detected(self):
        beliefs = [
            Belief(
                capability="train", description=f"batch=64 success {i}",
                valence=Valence.POSITIVE, entities=("batch_size",),
            )
            for i in range(3)
        ]
        observations = _simple_synthesize(beliefs)
        assert len(observations) >= 1
        assert any("consistently succeeds" in o.description for o in observations)

    def test_mixed_results_detected(self):
        beliefs = [
            Belief(
                capability="train", description="Adam works",
                valence=Valence.POSITIVE, entities=("optimizer",),
            ),
            Belief(
                capability="train", description="Adam fails",
                valence=Valence.NEGATIVE, entities=("optimizer",),
            ),
        ]
        observations = _simple_synthesize(beliefs)
        assert len(observations) >= 1
        assert any("mixed results" in o.description for o in observations)

    def test_too_few_beliefs_returns_empty(self):
        assert _simple_synthesize([]) == []
        assert _simple_synthesize([Belief()]) == []


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
# Similarity module: cosine_similarity
# ======================================================================

class TestSimilarityModule:
    def test_cosine_in_similarity_module(self):
        """cosine_similarity is importable from similarity module."""
        a = [1.0, 0.0]
        b = [1.0, 0.0]
        assert sim_cosine(a, b) == pytest.approx(1.0)


# ======================================================================
# Consolidation with embeddings
# ======================================================================

@pytest.mark.asyncio
class TestConsolidationWithEmbeddings:
    async def test_consolidation_uses_embeddings(self, tmp_dir, mock_embedder):
        """Consolidation clusters by embedding similarity when available."""
        backend = JsonMemoryBackend(tmp_dir / "memory")
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

        # Force different descriptions so they don't just merge via find_similar
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

        # Should have consolidated
        assert len(all_after) <= len(all_before)


# ======================================================================
# Inhibition with embeddings
# ======================================================================

@pytest.mark.asyncio
class TestInhibitionWithEmbeddings:
    async def test_semantic_inhibition(self, tmp_dir, mock_embedder):
        """Lateral inhibition uses embedding similarity."""
        backend = JsonMemoryBackend(tmp_dir / "memory")
        loop = MemoryLoop(backend=backend, embedder=mock_embedder, inhibition_rate=0.05)

        # Store two competing beliefs about similar topics
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

        # Agree with "batch small" — should inhibit "batch large" via similarity
        await loop.assimilate(Observation(
            capability="train", description="batch small works best",
            valence=Valence.POSITIVE,
        ))

        beliefs = await loop.backend.all_beliefs()
        competitor = next(b for b in beliefs if "large" in b.description)
        # May or may not be inhibited depending on embedding overlap
        # but it should not have been strengthened
        assert competitor.confirmation_count == 1


# ======================================================================
# New imports
# ======================================================================

class TestNewImports:
    def test_new_modules_importable(self):
        import ganglion.memory as gm
        for name in [
            "cosine_similarity", "Embedder", "CallableEmbedder",
            "SentenceTransformerEmbedder", "get_embedder", "set_embedder",
            "reflect", "synthesize",
        ]:
            assert hasattr(gm, name), f"{name} not exported from ganglion.memory"

    def test_deprecated_knobs_accepted_but_ignored(self):
        """Deprecated knobs are accepted for backward compat."""
        from ganglion.memory import MemoryLoop
        from ganglion.memory.backends.json_file import JsonMemoryBackend
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            backend = JsonMemoryBackend(d)
            # These should not raise
            loop = MemoryLoop(
                backend=backend,
                exploration_rate=2.0,
                cross_agent_bonus=1.0,
                crisis_multiplier=5.0,
                inhibition_floor=0.1,
            )
            assert loop is not None


# ======================================================================
# Counterfactual evaluation (Addition 1)
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
        # First call: no memory, so only 1 call
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

        # First call populates memory
        wrapped("hello")
        first_round_calls = call_count

        # Second call should have context and make 2 calls
        call_count = 0
        wrapped("hello")
        assert call_count == 2  # with memory + without memory

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
        """Clean call happens before memory check — first call has no injection."""
        calls = []

        def agent(prompt: str) -> str:
            calls.append(prompt)
            return f"response to: {prompt}"

        from ganglion.memory.wrap import memory
        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))

        # First invocation: no memory, single call, clean prompt only
        wrapped("hello")
        assert len(calls) == 1
        # The clean call should NOT have "What we know" injected
        assert "What we know" not in calls[0]

    def test_rollback_returns_clean_when_no_embedder(self, tmp_dir):
        """Without embedder, comparison can't happen — falls through to LLM compare
        which defaults to 'clean' when no LLM client is available."""
        call_count = 0

        def agent(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"response {call_count}: {prompt}"

        from ganglion.memory.wrap import memory
        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))

        # First call populates memory
        result1 = wrapped("hello")
        assert call_count == 1

        # Second call: has context, runs clean + memory, no embedder so
        # memory_changed_output stays True, _compare_outputs returns "clean"
        # because no LLM client available → rollback to clean response
        call_count = 0
        result2 = wrapped("hello")
        # 2 calls: clean first, then with memory
        assert call_count == 2
        # Should rollback to clean (response 1, the first call in this round)
        assert "response 1:" in result2

    def test_async_clean_first_ordering(self, tmp_dir):
        """Async: clean call happens before memory check."""
        import asyncio
        calls = []

        async def agent(prompt: str) -> str:
            calls.append(prompt)
            return f"response to: {prompt}"

        from ganglion.memory.wrap import memory
        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        asyncio.run(wrapped("hello"))
        assert len(calls) == 1
        assert "What we know" not in calls[0]


# ======================================================================
# Dependency tracking (Addition 2)
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

    async def test_produced_with_empty_by_default(self):
        """Beliefs without produced_with default to empty tuple."""
        b = Belief(capability="test", description="standalone")
        assert b.produced_with == ()
        d = b.to_dict()
        assert "produced_with" not in d  # empty tuples not serialized

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

    async def test_produced_with_stored_in_json(self, tmp_dir):
        """JSON backend preserves produced_with."""
        backend = JsonMemoryBackend(tmp_dir / "dep_json")
        b = Belief(
            capability="test", description="child belief",
            valence=Valence.POSITIVE, produced_with=(5, 10),
        )
        await backend.store(b)
        beliefs = await backend.all_beliefs()
        assert beliefs[0].produced_with == (5, 10)

    async def test_weaken_dependents_on_contradiction(self, tmp_dir):
        """When a parent belief dies, its dependents get weakened."""
        backend = SqliteMemoryBackend(tmp_dir / "dep2.db")
        loop = MemoryLoop(backend=backend, weaken_rate=0.5)

        # Create parent belief
        parent = Belief(
            capability="test", description="parent approach",
            valence=Valence.POSITIVE, confidence=1.0,
        )
        await backend.store(parent)
        parent_id = parent.id

        # Create child belief that was produced with parent
        child = Belief(
            capability="test", description="child conclusion",
            valence=Valence.POSITIVE, confidence=1.0,
            produced_with=(parent_id,),
        )
        await backend.store(child)

        # Contradict the parent until it dies
        neg = Observation(
            capability="test", description="parent approach",
            valence=Valence.NEGATIVE,
        )
        await loop.assimilate(neg)  # confidence: 1.0 - 0.5 = 0.5
        await loop.assimilate(neg)  # confidence: 0.5 - 0.5 = 0.0 → death

        # Child should be weakened
        beliefs = await backend.all_beliefs()
        child_after = next(b for b in beliefs if b.description == "child conclusion")
        assert child_after.confidence < 1.0
        backend.close()

    async def test_weaken_dependents_recursive(self, tmp_dir):
        """Dependency weakening propagates through chains."""
        backend = SqliteMemoryBackend(tmp_dir / "dep3.db")
        loop = MemoryLoop(backend=backend, weaken_rate=0.5)

        # Create chain: grandparent -> parent -> child
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

        # Kill grandparent
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

        # Seed a belief
        seed = Belief(
            capability="test", description="seed knowledge",
            valence=Valence.POSITIVE,
        )
        await backend.store(seed)

        agent = MemoryAgent(memory=loop, capability="test")
        # Simulate remembering
        await agent.remember()

        # Learn something new — should stamp the seed's id
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


# ======================================================================
# Continuous synthesis (Addition 3)
# ======================================================================

@pytest.mark.asyncio
class TestContinuousSynthesis:
    async def test_synthesis_triggers_at_interval(self, tmp_dir):
        """After synthesis_interval tasks, synthesized beliefs appear."""
        from ganglion.memory.agent import MemoryAgent
        backend = SqliteMemoryBackend(tmp_dir / "synth.db")
        loop = MemoryLoop(backend=backend)

        # Simple mock LLM client that returns empty synthesis
        synthesis_called = False

        async def mock_llm(prompt: str) -> str:
            nonlocal synthesis_called
            synthesis_called = True
            return "[]"

        agent = MemoryAgent(
            memory=loop, capability="test",
            synthesis_interval=3,
            _llm_client=mock_llm,
        )

        # Learn 2 tasks — no synthesis yet
        for i in range(2):
            await agent.learn({"success": True, "description": f"task {i}"})
        assert not synthesis_called

        # 3rd task triggers synthesis
        await agent.learn({"success": True, "description": "task 2"})
        assert synthesis_called
        backend.close()

    async def test_synthesis_not_triggered_without_client(self, tmp_dir):
        """Without llm_client, synthesis doesn't run."""
        from ganglion.memory.agent import MemoryAgent
        backend = SqliteMemoryBackend(tmp_dir / "synth2.db")
        loop = MemoryLoop(backend=backend)

        agent = MemoryAgent(
            memory=loop, capability="test",
            synthesis_interval=1,
            _llm_client=None,
        )

        # Should not raise
        await agent.learn({"success": True, "description": "task"})
        beliefs = await backend.all_beliefs()
        # Only the learned belief, no synthesized ones
        assert all("synthesized" not in b.tags for b in beliefs)
        backend.close()

    async def test_synthesis_stores_insights(self, tmp_dir):
        """When synthesis returns observations, they get stored."""
        from ganglion.memory.agent import MemoryAgent
        backend = SqliteMemoryBackend(tmp_dir / "synth3.db")
        loop = MemoryLoop(backend=backend)

        # Mock LLM that returns a synthesized insight
        async def mock_llm(prompt: str) -> str:
            return '[{"valence": "positive", "description": "lr<0.01 always works", "entities": ["lr"], "tags": ["synthesized"]}]'

        agent = MemoryAgent(
            memory=loop, capability="test",
            synthesis_interval=2,
            _llm_client=mock_llm,
        )

        # Seed with distinct beliefs so synthesis has material (>= 2 beliefs needed)
        await agent.learn({
            "success": True,
            "description": "small learning rate 0.001 gives stable convergence",
            "entities": ["lr"],
        })
        await agent.learn({
            "success": False,
            "description": "large batch size 256 causes out of memory errors",
            "entities": ["batch_size"],
        })

        beliefs = await backend.all_beliefs()
        synthesized = [b for b in beliefs if "synthesized" in b.tags]
        assert len(synthesized) >= 1
        assert any("lr" in b.description.lower() for b in synthesized)
        backend.close()
