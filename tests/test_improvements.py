"""Tests for the Evo-Memory-inspired improvements.

Covers:
    - Experience type (structured task experiences)
    - Confidence-gated injection (should_inject)
    - Tiered memory retrieval (strategies, experiences, pitfalls)
    - Contradiction-driven learning (conditional beliefs)
    - Embed on task input (not belief description)
    - Active memory pruning (_refine_memory)
    - Few-shot context formatting
    - Structured experience reflection
"""

import tempfile
from pathlib import Path

import pytest

from ganglion.memory.agent import MemoryAgent, result_to_observation
from ganglion.memory.backends.json_file import JsonMemoryBackend
from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.embed import CallableEmbedder
from ganglion.memory.loop import MemoryLoop
from ganglion.memory.reflect import (
    _simple_experience,
    _parse_experience,
)
from ganglion.memory.types import Belief, Experience, Observation, Valence


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# Deterministic mock embedder (same as test_v2_features.py)
async def _mock_embed(text: str) -> list[float]:
    vec = [0.0] * 26
    for c in text.lower():
        if 'a' <= c <= 'z':
            vec[ord(c) - ord('a')] += 1.0
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
def loop(backend):
    return MemoryLoop(backend=backend)


@pytest.fixture
def loop_with_embedder(backend, mock_embedder):
    return MemoryLoop(backend=backend, embedder=mock_embedder)


# ======================================================================
# Experience type
# ======================================================================

class TestExperienceType:
    def test_experience_creation(self):
        exp = Experience(
            input_summary="Solve quadratic equation 2x^2-5x+1=0",
            output_summary="Applied quadratic formula, got x=2.28 and x=0.22",
            success=True,
            lesson="For quadratic equations, apply the formula directly",
            strategy_tags=("quadratic_formula", "direct_computation"),
            capability="math",
        )
        assert exp.success is True
        assert "quadratic" in exp.lesson
        assert len(exp.strategy_tags) == 2

    def test_experience_roundtrip(self):
        exp = Experience(
            input_summary="test task",
            output_summary="test output",
            success=False,
            lesson="don't do this",
            strategy_tags=("avoid",),
            capability="test",
            embedding=[0.1, 0.2, 0.3],
        )
        d = exp.to_dict()
        restored = Experience.from_dict(d)
        assert restored.input_summary == "test task"
        assert restored.success is False
        assert restored.lesson == "don't do this"
        assert restored.strategy_tags == ("avoid",)
        assert restored.embedding is not None
        assert len(restored.embedding) == 3

    def test_experience_without_embedding_roundtrip(self):
        exp = Experience(
            input_summary="task",
            output_summary="result",
            success=True,
            lesson="it works",
        )
        d = exp.to_dict()
        assert "embedding" not in d
        restored = Experience.from_dict(d)
        assert restored.embedding is None


# ======================================================================
# Confidence-gated injection
# ======================================================================

@pytest.mark.asyncio
class TestConfidenceGating:
    async def test_should_inject_returns_false_when_no_beliefs(
        self, loop_with_embedder, mock_embedder,
    ):
        """No stored beliefs → don't inject."""
        agent = MemoryAgent(
            memory=loop_with_embedder, capability="test",
        )
        result = await agent.should_inject("some query")
        assert result is False

    async def test_should_inject_returns_true_without_embedder(self, loop):
        """Without an embedder, default to inject (can't gate)."""
        agent = MemoryAgent(memory=loop, capability="test")
        result = await agent.should_inject("some query")
        assert result is True

    async def test_should_inject_true_for_relevant_query(
        self, loop_with_embedder,
    ):
        """High-similarity query passes the gate."""
        loop = loop_with_embedder
        await loop.assimilate(Observation(
            capability="test", description="batch size optimization",
            valence=Valence.POSITIVE,
        ))

        agent = MemoryAgent(
            memory=loop, capability="test",
            relevance_threshold=0.3,
        )
        # Very similar query
        result = await agent.should_inject("batch size works well")
        assert result is True

    async def test_should_inject_false_for_irrelevant_query(
        self, loop_with_embedder,
    ):
        """Low-similarity query is blocked."""
        loop = loop_with_embedder
        await loop.assimilate(Observation(
            capability="test", description="aaaa",
            valence=Valence.POSITIVE,
        ))

        agent = MemoryAgent(
            memory=loop, capability="test",
            relevance_threshold=0.9,
        )
        # Very different query
        result = await agent.should_inject("zzzz")
        assert result is False


# ======================================================================
# Contradiction-driven learning
# ======================================================================

@pytest.mark.asyncio
class TestContradictionDrivenLearning:
    async def test_contradiction_creates_conditional_belief(self, loop):
        """Contradictions generate conditional insight beliefs."""
        pos = Observation(
            capability="x", description="strategy A works",
            valence=Valence.POSITIVE,
        )
        neg = Observation(
            capability="x", description="strategy A works",
            valence=Valence.NEGATIVE,
        )

        await loop.assimilate(pos)
        delta = await loop.assimilate(neg)

        assert delta is not None
        assert delta.delta_type == "contradiction"

        beliefs = await loop.backend.all_beliefs()
        conditionals = [b for b in beliefs if "CONDITIONAL" in b.description]
        assert len(conditionals) == 1
        assert conditionals[0].valence == Valence.NEUTRAL
        assert conditionals[0].confidence == 1.5
        assert "contradiction_insight" in conditionals[0].tags

    async def test_contradiction_halves_weakening(self, loop):
        """Contradictions weaken at half rate to preserve information."""
        pos = Observation(
            capability="x", description="approach B",
            valence=Valence.POSITIVE,
        )
        await loop.assimilate(pos)

        beliefs = await loop.backend.all_beliefs()
        conf_before = beliefs[0].confidence

        neg = Observation(
            capability="x", description="approach B",
            valence=Valence.NEGATIVE,
        )
        await loop.assimilate(neg)

        beliefs = await loop.backend.all_beliefs()
        original = [b for b in beliefs if b.description == "approach B"][0]
        drop = conf_before - original.confidence
        # Should be weaken_rate * 0.5 = 0.3 * 0.5 = 0.15
        assert abs(drop - loop.weaken_rate * 0.5) < 0.01


# ======================================================================
# Embed on task input
# ======================================================================

@pytest.mark.asyncio
class TestEmbedOnInput:
    async def test_input_text_stored_in_config(self):
        obs = result_to_observation(
            capability="test",
            result={"success": True, "description": "it worked"},
            input_text="What is the capital of France?",
        )
        assert obs.config is not None
        assert obs.config["input_text"] == "What is the capital of France?"

    async def test_embed_uses_input_text_from_config(self, loop_with_embedder):
        """When input_text is in config, embed on that instead of description."""
        loop = loop_with_embedder

        obs = Observation(
            capability="test",
            description="Succeeded: Paris is the capital",
            valence=Valence.POSITIVE,
            config={"input_text": "capital of france"},
        )
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        assert len(beliefs) == 1
        # The embedding should be based on "capital of france"
        # not "Succeeded: Paris is the capital"
        expected_embed = await _mock_embed("capital of france")
        actual_embed = beliefs[0].embedding
        assert actual_embed is not None
        # Check vectors are close (they should be identical)
        for a, b in zip(expected_embed, actual_embed):
            assert abs(a - b) < 1e-6

    async def test_embed_falls_back_to_description(self, loop_with_embedder):
        """Without input_text in config, embed on description as before."""
        loop = loop_with_embedder

        obs = Observation(
            capability="test",
            description="some observation",
            valence=Valence.POSITIVE,
        )
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        expected_embed = await _mock_embed("some observation")
        actual_embed = beliefs[0].embedding
        for a, b in zip(expected_embed, actual_embed):
            assert abs(a - b) < 1e-6


# ======================================================================
# Active memory pruning
# ======================================================================

@pytest.mark.asyncio
class TestActiveMemoryPruning:
    async def test_refine_removes_unretrieved_weak_beliefs(self, loop):
        """Beliefs that were never retrieved and have low confidence get pruned."""
        agent = MemoryAgent(
            memory=loop, capability="test",
            refine_interval=5,
        )

        # Store a weak, never-retrieved belief directly
        weak = Belief(
            capability="test", description="never used",
            valence=Valence.POSITIVE, confidence=1.0,
            confirmation_count=1, last_retrieved=None,
        )
        await loop.backend.store(weak)

        removed = await agent._refine_memory()
        assert removed >= 1

        remaining = await loop.backend.all_beliefs()
        assert not any(b.description == "never used" for b in remaining)

    async def test_refine_keeps_strong_beliefs(self, loop):
        """Beliefs with high confidence or many confirmations survive pruning."""
        agent = MemoryAgent(memory=loop, capability="test")

        strong = Belief(
            capability="test", description="well confirmed",
            valence=Valence.POSITIVE, confidence=3.0,
            confirmation_count=5, last_retrieved=None,
        )
        await loop.backend.store(strong)

        await agent._refine_memory()

        remaining = await loop.backend.all_beliefs()
        assert any(b.description == "well confirmed" for b in remaining)

    async def test_learn_triggers_refine_at_interval(self, loop):
        """learn() triggers refinement every refine_interval tasks."""
        agent = MemoryAgent(
            memory=loop, capability="test",
            refine_interval=3,
        )

        # Store a weak belief that should be pruned
        await loop.backend.store(Belief(
            capability="test", description="prunable",
            valence=Valence.POSITIVE, confidence=1.0,
            confirmation_count=1, last_retrieved=None,
        ))

        # Learn 3 times to trigger refinement
        for i in range(3):
            await agent.learn({
                "success": True,
                "description": f"result {i}",
            })

        remaining = await loop.backend.all_beliefs()
        assert not any(b.description == "prunable" for b in remaining)


# ======================================================================
# Tiered memory retrieval
# ======================================================================

@pytest.mark.asyncio
class TestTieredRetrieval:
    async def test_remember_includes_pitfalls(self, loop):
        """Known failures are always included in context."""
        await loop.assimilate(Observation(
            capability="test", description="approach X causes OOM",
            valence=Valence.NEGATIVE,
        ))

        agent = MemoryAgent(memory=loop, capability="test")
        ctx = await agent.remember()
        assert "AVOID" in ctx
        assert "approach X causes OOM" in ctx

    async def test_remember_includes_foreign(self, loop):
        """Foreign knowledge still works in tiered retrieval."""
        await loop.assimilate(Observation(
            capability="test", description="beta found success",
            valence=Valence.POSITIVE, source="beta",
        ))

        agent = MemoryAgent(
            memory=loop, capability="test", bot_id="alpha",
            include_foreign=True,
        )
        ctx = await agent.remember()
        assert "other agents report" in ctx

    async def test_remember_backward_compatible(self, loop):
        """Without strategies, falls back to standard context."""
        await loop.assimilate(Observation(
            capability="test", description="batch=64 works",
            valence=Valence.POSITIVE,
        ))

        agent = MemoryAgent(memory=loop, capability="test")
        ctx = await agent.remember()
        assert "batch=64 works" in ctx


# ======================================================================
# Few-shot context formatting
# ======================================================================

@pytest.mark.asyncio
class TestFewShotFormatting:
    async def test_experience_format_with_input_text(self, loop):
        """Beliefs with input_text in config get few-shot formatting."""
        obs = Observation(
            capability="test",
            description="Using step-by-step reasoning works",
            valence=Valence.POSITIVE,
            config={"input_text": "Solve the quadratic equation"},
        )
        await loop.assimilate(obs)

        ctx = await loop.context_for("test")
        assert "Similar task" in ctx
        assert "Outcome" in ctx
        assert "Lesson" in ctx

    async def test_legacy_format_without_input_text(self, loop):
        """Beliefs without input_text keep the legacy format."""
        await loop.assimilate(Observation(
            capability="test", description="batch=64 works",
            valence=Valence.POSITIVE,
        ))

        ctx = await loop.context_for("test")
        assert "What we know" in ctx
        assert "What works" in ctx


# ======================================================================
# Structured experience reflection
# ======================================================================

class TestStructuredExperienceReflection:
    def test_simple_experience_success(self):
        exp = _simple_experience(
            "solve 2+2", "the answer is 4", True, "math",
        )
        assert exp.success is True
        assert "succeeded" in exp.lesson.lower()
        assert exp.capability == "math"

    def test_simple_experience_failure(self):
        exp = _simple_experience(
            "solve hard problem", "Error: timeout", False, "math",
        )
        assert exp.success is False
        assert "failed" in exp.lesson.lower()

    def test_parse_experience_valid_json(self):
        json_str = '''{
            "lesson": "Use elimination for multiple choice",
            "strategy_tags": ["elimination", "process_of_elimination"],
            "input_summary": "Multiple choice question about biology",
            "output_summary": "Selected answer C by eliminating others"
        }'''
        exp = _parse_experience(json_str, "test q", "answer C", True, "exam")
        assert "elimination" in exp.lesson
        assert "elimination" in exp.strategy_tags
        assert exp.success is True

    def test_parse_experience_invalid_json_fallback(self):
        exp = _parse_experience(
            "not json", "test input", "test output", False, "test",
        )
        assert exp.success is False
        assert exp.input_summary == "test input"

    def test_parse_experience_json_in_markdown(self):
        text = '```json\n{"lesson": "works", "strategy_tags": ["test"]}\n```'
        exp = _parse_experience(text, "in", "out", True, "test")
        assert exp.lesson == "works"


# ======================================================================
# Wrapper gating
# ======================================================================

@pytest.mark.asyncio
class TestWrapperGating:
    async def test_wrapper_gates_injection(self, tmp_dir, mock_embedder):
        """Wrapper skips injection when query is irrelevant."""
        from ganglion.memory.wrap import memory
        import ganglion.memory.wrap as mod
        mod._default_memory = None  # Reset singleton

        db_path = str(tmp_dir / "gate_test.db")

        async def agent(prompt: str) -> str:
            return f"response to: {prompt}"

        wrapped = memory(
            agent, capability="test", db_path=db_path,
            embedder=mock_embedder,
        )

        # First call — empty memory, gate returns False, no injection
        result = await wrapped("hello world")
        assert "response to: hello world" in result

        mod._default_memory = None  # Cleanup


# ======================================================================
# Import checks
# ======================================================================

class TestNewExports:
    def test_experience_importable(self):
        from ganglion.memory import Experience
        assert Experience is not None

    def test_reflect_experience_importable(self):
        from ganglion.memory import reflect_experience
        assert callable(reflect_experience)
