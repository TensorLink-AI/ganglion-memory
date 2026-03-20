"""Tests for the memory system.

Covers: types, SQLite backend, MemoryLoop (the core primitive),
agent integration, and the biological properties we care about:
    - Hebbian strengthening (confirmation)
    - Contradiction detection (delta emission)
    - Metric drift detection
    - Strength-based eviction (not age-based)
    - Entity profiling as a view
    - Cross-bot knowledge sharing
"""

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ganglion.memory.agent import MemoryAgent, between_runs, result_to_observation
from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.loop import MemoryLoop
from ganglion.memory.types import Belief, Delta, Observation, Valence


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sqlite_backend(tmp_dir):
    backend = SqliteMemoryBackend(tmp_dir / "memory.db")
    yield backend
    backend.close()


@pytest.fixture
def backend(tmp_dir):
    b = SqliteMemoryBackend(tmp_dir / "memory.db")
    yield b
    b.close()


@pytest.fixture
def loop(backend):
    return MemoryLoop(backend=backend)


# ======================================================================
# Types
# ======================================================================

class TestTypes:
    def test_observation_to_dict(self):
        obs = Observation(
            capability="mining",
            description="batch_size=64",
            valence=Valence.POSITIVE,
            entities=("subnet-18",),
            metric_value=0.82,
            metric_name="score",
        )
        d = obs.to_dict()
        assert d["capability"] == "mining"
        assert d["valence"] == "positive"
        assert d["entities"] == ["subnet-18"]

    def test_belief_roundtrip(self):
        b = Belief(
            id=1,
            capability="mining",
            description="batch=64",
            valence=Valence.POSITIVE,
            confidence=2.5,
            confirmation_count=3,
            entities=("sn18",),
            metric_value=0.9,
        )
        d = b.to_dict()
        b2 = Belief.from_dict(d)
        assert b2.capability == "mining"
        assert b2.valence == Valence.POSITIVE
        assert b2.confidence == 2.5
        assert b2.confirmation_count == 3
        assert "sn18" in b2.entities

    def test_belief_strength_ranking(self):
        """Recent + confirmed beats old + confirmed beats recent + weak."""
        recent_strong = Belief(
            capability="x", description="a", valence=Valence.POSITIVE,
            confidence=2.0, confirmation_count=5,
            last_confirmed=datetime.now(UTC),
        )
        old_strong = Belief(
            capability="x", description="b", valence=Valence.POSITIVE,
            confidence=2.0, confirmation_count=5,
            last_confirmed=datetime.now(UTC) - timedelta(days=30),
        )
        recent_weak = Belief(
            capability="x", description="c", valence=Valence.POSITIVE,
            confidence=0.5, confirmation_count=1,
            last_confirmed=datetime.now(UTC),
        )
        assert recent_strong.strength > old_strong.strength > recent_weak.strength

    def test_belief_is_pattern_and_antipattern(self):
        assert Belief(valence=Valence.POSITIVE).is_pattern
        assert not Belief(valence=Valence.POSITIVE).is_antipattern
        assert Belief(valence=Valence.NEGATIVE).is_antipattern

    def test_delta_summary_metric_shift(self):
        d = Delta(
            old_belief=Belief(capability="mining", metric_name="score", metric_value=0.8),
            new_observation=Observation(
                capability="mining", description="x",
                valence=Valence.POSITIVE, metric_value=0.5,
            ),
            delta_type="metric_shift",
            magnitude=-0.375,
        )
        assert "shifted" in d.summary
        assert "score" in d.summary

    def test_delta_summary_contradiction(self):
        d = Delta(
            old_belief=Belief(capability="mining", description="old approach"),
            new_observation=Observation(
                capability="mining", description="new approach",
                valence=Valence.NEGATIVE,
            ),
            delta_type="contradiction",
        )
        assert "contradicted" in d.summary

    def test_long_term_memory_consolidation(self):
        """Consolidated beliefs (high confirmation_count) resist recency decay."""
        old_consolidated = Belief(
            capability="x", description="consolidated",
            valence=Valence.POSITIVE,
            confidence=5.0, confirmation_count=10,
            last_confirmed=datetime.now(UTC) - timedelta(days=60),
        )
        new_weak = Belief(
            capability="x", description="new",
            valence=Valence.POSITIVE,
            confidence=1.0, confirmation_count=1,
            last_confirmed=datetime.now(UTC),
        )
        assert old_consolidated.strength > new_weak.strength
        assert old_consolidated.strength >= 5.0 * 10 * 0.5

    def test_belief_input_context_field(self):
        """input_context field roundtrips correctly."""
        b = Belief(
            capability="test", description="lesson",
            valence=Valence.POSITIVE,
            input_context="what was the task",
        )
        d = b.to_dict()
        assert d["input_context"] == "what was the task"
        b2 = Belief.from_dict(d)
        assert b2.input_context == "what was the task"

    def test_belief_input_context_default(self):
        b = Belief(capability="test", description="x")
        assert b.input_context == ""


# ======================================================================
# Backend tests (SQLite only — JSON backend removed)
# ======================================================================

@pytest.mark.asyncio
class TestBackend:
    async def test_store_and_find_similar(self, backend):
        obs = Observation(capability="train", description="Good approach", valence=Valence.POSITIVE)
        assert await backend.find_similar(obs) is None

        belief = Belief(capability="train", description="Good approach", valence=Valence.POSITIVE)
        await backend.store(belief)

        found = await backend.find_similar(obs)
        assert found is not None
        assert found.description == "Good approach"

    async def test_store_and_query_by_capability(self, backend):
        await backend.store(Belief(capability="train", description="A", valence=Valence.POSITIVE))
        await backend.store(Belief(capability="eval", description="B", valence=Valence.POSITIVE))

        results = await backend.query(capability="train")
        assert len(results) == 1
        assert results[0].description == "A"

    async def test_query_by_valence(self, backend):
        await backend.store(Belief(capability="x", description="good", valence=Valence.POSITIVE))
        await backend.store(Belief(capability="x", description="bad", valence=Valence.NEGATIVE))

        pos = await backend.query(capability="x", valence=Valence.POSITIVE)
        assert len(pos) == 1
        assert pos[0].description == "good"

    async def test_query_by_entities(self, backend):
        await backend.store(Belief(
            capability="x", description="A", valence=Valence.POSITIVE,
            entities=("subnet-18",),
        ))
        await backend.store(Belief(
            capability="x", description="B", valence=Valence.POSITIVE,
            entities=("subnet-1",),
        ))

        results = await backend.query(entities=("subnet-18",))
        assert len(results) == 1
        assert results[0].description == "A"

    async def test_query_exclude_source(self, backend):
        await backend.store(Belief(
            capability="x", description="from alpha", valence=Valence.POSITIVE,
            source="alpha",
        ))
        await backend.store(Belief(
            capability="x", description="from beta", valence=Valence.POSITIVE,
            source="beta",
        ))

        results = await backend.query(exclude_source="alpha")
        assert len(results) == 1
        assert results[0].source == "beta"

    async def test_update(self, backend):
        belief = Belief(capability="x", description="A", valence=Valence.POSITIVE)
        await backend.store(belief)

        belief.confidence = 5.0
        belief.confirmation_count = 10
        await backend.update(belief)

        results = await backend.query(capability="x")
        assert results[0].confidence == 5.0
        assert results[0].confirmation_count == 10

    async def test_remove(self, backend):
        belief = Belief(capability="x", description="A", valence=Valence.POSITIVE)
        await backend.store(belief)
        assert len(await backend.all_beliefs()) == 1

        await backend.remove(belief)
        assert len(await backend.all_beliefs()) == 0

    async def test_all_beliefs(self, backend):
        for i in range(5):
            await backend.store(Belief(capability="x", description=f"b{i}", valence=Valence.POSITIVE))
        assert len(await backend.all_beliefs()) == 5

    async def test_query_limit(self, backend):
        for i in range(10):
            await backend.store(Belief(capability="x", description=f"b{i}", valence=Valence.POSITIVE))
        results = await backend.query(capability="x", limit=3)
        assert len(results) == 3

    async def test_query_by_tags(self, backend):
        await backend.store(Belief(
            capability="x", description="A", valence=Valence.POSITIVE,
            tags=("agent_design",),
        ))
        await backend.store(Belief(
            capability="x", description="B", valence=Valence.POSITIVE,
            tags=("strategy",),
        ))

        results = await backend.query(tags=("agent_design",))
        assert len(results) == 1
        assert results[0].description == "A"

    async def test_sqlite_roundtrip_all_beliefs(self, sqlite_backend):
        """SQLite store + all_beliefs roundtrip."""
        belief = Belief(
            capability="mining", description="test roundtrip",
            valence=Valence.POSITIVE, confidence=2.0,
            confirmation_count=3, entities=("sn18",),
            metric_value=0.9, metric_name="score",
            source="alpha", tags=("tag1",),
        )
        await sqlite_backend.store(belief)
        beliefs = await sqlite_backend.all_beliefs()
        assert len(beliefs) == 1
        b = beliefs[0]
        assert b.description == "test roundtrip"
        assert b.confidence == 2.0
        assert b.confirmation_count == 3

    async def test_sqlite_input_context_roundtrip(self, sqlite_backend):
        """input_context survives SQLite store/retrieve."""
        belief = Belief(
            capability="test", description="lesson",
            valence=Valence.POSITIVE,
            input_context="the original task",
        )
        await sqlite_backend.store(belief)
        beliefs = await sqlite_backend.all_beliefs()
        assert beliefs[0].input_context == "the original task"

    async def test_find_similar_fuzzy_match(self, backend):
        """Token-based similarity matches rephrased descriptions."""
        await backend.store(Belief(
            capability="train",
            description="batch_size=64 works well for training",
            valence=Valence.POSITIVE,
        ))

        obs = Observation(
            capability="train",
            description="batch 64 works well for training runs",
            valence=Valence.POSITIVE,
        )
        found = await backend.find_similar(obs, threshold=0.7)
        assert found is not None

    async def test_find_similar_no_match_dissimilar(self, backend):
        """Dissimilar descriptions don't match."""
        await backend.store(Belief(
            capability="train",
            description="batch_size=64 works",
            valence=Valence.POSITIVE,
        ))

        obs = Observation(
            capability="train",
            description="learning_rate=0.001 is optimal",
            valence=Valence.POSITIVE,
        )
        found = await backend.find_similar(obs)
        assert found is None

    async def test_find_similar_lower_threshold(self, backend):
        """Lowering threshold allows fuzzier matches."""
        await backend.store(Belief(
            capability="train",
            description="batch_size=64 works great",
            valence=Valence.POSITIVE,
        ))

        obs = Observation(
            capability="train",
            description="batch 64 great results observed",
            valence=Valence.POSITIVE,
        )
        found_loose = await backend.find_similar(obs, threshold=0.3)
        assert found_loose is not None


# ======================================================================
# MemoryLoop — the core primitive
# ======================================================================

@pytest.mark.asyncio
class TestMemoryLoop:
    async def test_novel_observation_creates_belief(self, loop):
        obs = Observation(capability="mining", description="batch=64", valence=Valence.POSITIVE)
        delta = await loop.assimilate(obs)
        assert delta is None

        s = await loop.summary()
        assert s["total_beliefs"] == 1
        assert s["patterns"] == 1

    async def test_hebbian_strengthening(self, loop):
        """Repeated agreement increments confirmation_count."""
        obs = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, metric_value=0.8,
        )
        await loop.assimilate(obs)
        await loop.assimilate(obs)
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].confirmation_count == 3

    async def test_contradiction_weakens_belief(self, loop):
        """Opposite valence weakens existing belief."""
        pos = Observation(capability="mining", description="batch=64", valence=Valence.POSITIVE)
        neg = Observation(capability="mining", description="batch=64", valence=Valence.NEGATIVE)

        await loop.assimilate(pos)
        delta = await loop.assimilate(neg)

        assert delta is not None
        assert delta.delta_type == "contradiction"

    async def test_contradiction_kills_weak_belief(self, loop):
        """Repeated contradiction causes apoptosis."""
        loop.weaken_rate = 0.5

        pos = Observation(capability="mining", description="batch=64", valence=Valence.POSITIVE)
        neg = Observation(capability="mining", description="batch=64", valence=Valence.NEGATIVE)

        await loop.assimilate(pos)
        await loop.assimilate(neg)
        await loop.assimilate(neg)

        beliefs = await loop.backend.all_beliefs()
        assert any(b.valence == Valence.NEGATIVE and b.confidence == 1.0 for b in beliefs)

    async def test_metric_shift_detection(self, loop):
        """Detects drift even when valence agrees."""
        loop.metric_shift_threshold = 0.1

        obs1 = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, metric_name="score", metric_value=0.8,
        )
        obs2 = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, metric_name="score", metric_value=0.5,
        )

        await loop.assimilate(obs1)
        delta = await loop.assimilate(obs2)

        assert delta is not None
        assert delta.delta_type == "metric_shift"
        assert delta.magnitude is not None
        assert delta.magnitude > 0.1

    async def test_drain_deltas(self, loop):
        loop.metric_shift_threshold = 0.01
        obs1 = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, metric_value=0.8,
        )
        obs2 = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, metric_value=0.5,
        )
        await loop.assimilate(obs1)
        await loop.assimilate(obs2)

        deltas = await loop.drain_deltas()
        assert len(deltas) >= 1
        assert len(await loop.drain_deltas()) == 0

    async def test_entity_merging(self, loop):
        """Entities accumulate across observations."""
        obs1 = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        )
        obs2 = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-1",),
        )
        await loop.assimilate(obs1)
        await loop.assimilate(obs2)

        beliefs = await loop.backend.all_beliefs()
        assert "subnet-18" in beliefs[0].entities
        assert "subnet-1" in beliefs[0].entities

    async def test_strength_based_eviction(self, loop):
        """forget() removes weakest beliefs, not oldest."""
        loop.max_beliefs = 3

        for i in range(5):
            b = Belief(
                capability="x", description=f"b{i}", valence=Valence.POSITIVE,
                confidence=float(i + 1),
                confirmation_count=i + 1,
            )
            await loop.backend.store(b)

        removed = await loop.forget()
        assert removed == 2

        remaining = await loop.backend.all_beliefs()
        assert len(remaining) == 3
        descriptions = {b.description for b in remaining}
        assert "b0" not in descriptions
        assert "b1" not in descriptions

    async def test_stable_entity_merge_order(self, loop):
        """Entity merging preserves existing order, appends new items."""
        obs1 = Observation(
            capability="mining", description="test order",
            valence=Valence.POSITIVE, entities=("a", "b", "c"),
        )
        obs2 = Observation(
            capability="mining", description="test order",
            valence=Valence.POSITIVE, entities=("d", "b", "e"),
        )

        await loop.assimilate(obs1)
        await loop.assimilate(obs2)

        beliefs = await loop.backend.all_beliefs()
        assert beliefs[0].entities == ("a", "b", "c", "d", "e")

    async def test_concurrent_delta_safety(self, loop):
        """Concurrent assimilate calls produce correct delta counts."""
        loop.metric_shift_threshold = 0.01

        base_obs = Observation(
            capability="mining", description="concurrent test",
            valence=Valence.POSITIVE, metric_name="score", metric_value=1.0,
        )
        await loop.assimilate(base_obs)

        for i in range(10):
            obs = Observation(
                capability=f"cap_{i}",
                description=f"unique obs {i}",
                valence=Valence.POSITIVE,
                metric_name="score",
                metric_value=0.5,
            )
            await loop.assimilate(obs)

        async def make_contradiction(idx: int) -> Delta | None:
            obs = Observation(
                capability=f"cap_{idx}",
                description=f"unique obs {idx}",
                valence=Valence.NEGATIVE,
            )
            return await loop.assimilate(obs)

        results = await asyncio.gather(*[make_contradiction(i) for i in range(10)])
        contradiction_deltas = [r for r in results if r is not None]

        assert len(contradiction_deltas) == 10

        drained = await loop.drain_deltas()
        assert len(drained) == 10
        assert len(await loop.drain_deltas()) == 0

    async def test_input_context_stored_on_novel_belief(self, loop):
        """Novel observation with input_text in config stores input_context."""
        obs = Observation(
            capability="test", description="lesson learned",
            valence=Valence.POSITIVE,
            config={"input_text": "what was the task"},
        )
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        assert beliefs[0].input_context == "what was the task"


# ======================================================================
# Prompt context generation
# ======================================================================

@pytest.mark.asyncio
class TestPromptContext:
    async def test_context_for_includes_patterns_and_antipatterns(self, loop):
        await loop.assimilate(Observation(
            capability="mining", description="batch=64 works",
            valence=Valence.POSITIVE, metric_value=0.85, metric_name="score",
        ))
        await loop.assimilate(Observation(
            capability="mining", description="batch=256 OOM",
            valence=Valence.NEGATIVE,
        ))

        ctx = await loop.context_for("mining")
        assert "What works" in ctx
        assert "batch=64 works" in ctx
        assert "What fails" in ctx
        assert "batch=256 OOM" in ctx

    async def test_context_for_empty(self, loop):
        ctx = await loop.context_for("nonexistent")
        assert ctx == ""

    async def test_entity_profile(self, loop):
        await loop.assimilate(Observation(
            capability="mining", description="subnet 18 rewards consistency",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        ))
        await loop.assimilate(Observation(
            capability="mining", description="subnet 18 penalises rapid changes",
            valence=Valence.NEGATIVE, entities=("subnet-18",),
        ))

        profile = await loop.entity_profile("subnet-18")
        assert "Profile: subnet-18" in profile
        assert "+" in profile
        assert "-" in profile

    async def test_entity_profile_empty(self, loop):
        profile = await loop.entity_profile("unknown")
        assert "No knowledge" in profile

    async def test_context_excludes_source(self, loop):
        await loop.assimilate(Observation(
            capability="mining", description="from alpha",
            valence=Valence.POSITIVE, source="alpha",
        ))
        await loop.assimilate(Observation(
            capability="mining", description="from beta",
            valence=Valence.POSITIVE, source="beta",
        ))

        ctx = await loop.context_for("mining", exclude_source="alpha")
        assert "from beta" in ctx
        assert "from alpha" not in ctx

    async def test_retrieve_for_returns_beliefs(self, loop):
        """retrieve_for() returns both context string and raw beliefs."""
        await loop.assimilate(Observation(
            capability="mining", description="batch=64 works",
            valence=Valence.POSITIVE,
        ))

        ctx, beliefs = await loop.retrieve_for("mining")
        assert "Relevant experience" in ctx
        assert len(beliefs) >= 1
        assert beliefs[0].description == "batch=64 works"

    async def test_retrieve_for_empty(self, loop):
        ctx, beliefs = await loop.retrieve_for("nonexistent")
        assert ctx == ""
        assert beliefs == []

    async def test_format_context_v3_few_shot(self, loop):
        """_format_context_v3 produces few-shot format for beliefs with experience."""
        b = Belief(
            capability="test", description="Task succeeded: solve equation",
            valence=Valence.POSITIVE,
            config={"input_text": "solve x^2 = 4", "output_text": "x = 2"},
        )
        ctx = loop._format_context_v3([b])
        assert "Prior task" in ctx
        assert "solve x^2 = 4" in ctx
        assert "Lesson" in ctx

    async def test_format_context_v3_legacy(self, loop):
        """_format_context_v3 handles legacy beliefs without experience context."""
        b = Belief(
            capability="test", description="batch=64 works",
            valence=Valence.POSITIVE, metric_name="score", metric_value=0.9,
        )
        ctx = loop._format_context_v3([b])
        assert "batch=64 works" in ctx
        assert "score" in ctx


# ======================================================================
# Agent integration
# ======================================================================

@pytest.mark.asyncio
class TestAgentIntegration:
    async def test_result_to_observation(self):
        obs = result_to_observation(
            capability="mining",
            result={
                "success": True,
                "description": "batch=64 lr=0.001",
                "metric_name": "score",
                "metric_value": 0.82,
                "config": {"batch_size": 64},
                "entities": ["subnet-18"],
            },
            source="alpha",
            run_id="run-001",
        )
        assert obs.valence == Valence.POSITIVE
        assert obs.capability == "mining"
        assert "subnet-18" in obs.entities
        assert obs.source == "alpha"

    async def test_result_to_observation_failure(self):
        obs = result_to_observation(
            capability="mining",
            result={
                "success": False,
                "description": "batch=256",
                "error": "OOM on small GPU",
            },
        )
        assert obs.valence == Valence.NEGATIVE
        assert "OOM" in obs.description

    async def test_memory_agent_remember_and_learn(self, loop):
        agent = MemoryAgent(
            memory=loop,
            capability="mining",
            bot_id="alpha",
            entities=("subnet-18",),
        )

        ctx = await agent.remember()
        assert ctx == ""

        delta = await agent.learn({
            "success": True,
            "description": "batch=64 works",
            "metric_name": "score",
            "metric_value": 0.82,
        })
        assert delta is None

        ctx = await agent.remember()
        assert "batch=64 works" in ctx

    async def test_cross_bot_knowledge(self, loop):
        alpha = MemoryAgent(memory=loop, capability="mining", bot_id="alpha")
        beta = MemoryAgent(memory=loop, capability="mining", bot_id="beta")

        await alpha.learn({
            "success": True,
            "description": "batch=64 works",
        })

        ctx = await beta.remember()
        assert "batch=64 works" in ctx
        assert "other agents report" in ctx

    async def test_between_runs(self, loop):
        loop.metric_shift_threshold = 0.01
        await loop.assimilate(Observation(
            capability="x", description="A",
            valence=Valence.POSITIVE, metric_value=1.0,
        ))
        await loop.assimilate(Observation(
            capability="x", description="A",
            valence=Valence.POSITIVE, metric_value=0.5,
        ))

        deltas = await between_runs(loop)
        assert len(deltas) >= 1

    async def test_learn_stamps_input_output(self, loop):
        """learn() stamps input_text and output_text into config."""
        agent = MemoryAgent(memory=loop, capability="test")
        await agent.learn(
            {"success": True, "description": "solved it"},
            input_text="what is 2+2",
            output_text="4",
        )

        beliefs = await loop.backend.all_beliefs()
        b = beliefs[0]
        assert b.config is not None
        assert b.config.get("input_text") == "what is 2+2"
        assert b.config.get("output_text") == "4"
        assert b.input_context == "what is 2+2"


# ======================================================================
# Salience (built into MemoryLoop.assimilate)
# ======================================================================

@pytest.mark.asyncio
class TestSalience:
    async def test_novel_surprising_observation_gets_boosted(self, loop):
        """Novel observation with surprising metric gets confidence > 1.0."""
        for i in range(5):
            await loop.assimilate(Observation(
                capability="mining", description=f"peer_{i}",
                valence=Valence.POSITIVE, metric_value=0.5 + i * 0.01,
            ))

        obs = Observation(
            capability="mining", description="outlier strategy",
            valence=Valence.POSITIVE, metric_value=0.99,
        )
        delta = await loop.assimilate(obs)
        assert delta is None

        beliefs = await loop.backend.all_beliefs()
        outlier = next(b for b in beliefs if b.description == "outlier strategy")
        assert outlier.confidence > 1.0

    async def test_salience_disabled(self, loop):
        """With salience=False, novel observations get confidence=1.0."""
        loop.salience = False
        for i in range(5):
            await loop.assimilate(Observation(
                capability="mining", description=f"peer_{i}",
                valence=Valence.POSITIVE, metric_value=0.5 + i * 0.01,
            ))

        obs = Observation(
            capability="mining", description="outlier strategy",
            valence=Valence.POSITIVE, metric_value=0.99,
        )
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        outlier = next(b for b in beliefs if b.description == "outlier strategy")
        assert outlier.confidence == 1.0


# ======================================================================
# Inhibition (built into MemoryLoop.assimilate)
# ======================================================================

@pytest.mark.asyncio
class TestInhibition:
    async def test_agreement_weakens_competitors(self, loop):
        """Agreement on one belief weakens competitors with overlapping features."""
        await loop.backend.store(Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=1.0, confirmation_count=1,
        ))
        await loop.backend.store(Belief(
            capability="mining", description="batch=128",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=1.0, confirmation_count=1,
        ))

        obs = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        )
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        comp = next(b for b in beliefs if b.description == "batch=128")
        assert comp.confidence < 1.0

    async def test_inhibition_disabled(self, loop):
        """inhibition_rate=0.0 disables inhibition entirely."""
        loop.inhibition_rate = 0.0

        await loop.backend.store(Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=1.0, confirmation_count=1,
        ))
        await loop.backend.store(Belief(
            capability="mining", description="batch=128",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=1.0, confirmation_count=1,
        ))

        obs = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        )
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        comp = next(b for b in beliefs if b.description == "batch=128")
        assert comp.confidence == 1.0


# ======================================================================
# Consolidation (built into MemoryLoop.forget)
# ======================================================================

@pytest.mark.asyncio
class TestConsolidation:
    async def test_forget_merges_similar_cluster(self, loop):
        """forget() merges similar beliefs before evicting."""
        loop.max_beliefs = 1
        loop.consolidation_threshold = 0.5

        for i in range(3):
            await loop.backend.store(Belief(
                capability="mining", description=f"strategy_{i}",
                valence=Valence.POSITIVE,
                entities=("subnet-18",), tags=("gpu",),
                confirmation_count=1, confidence=1.0,
                metric_value=0.8 + i * 0.01,
                metric_name="score",
            ))

        await loop.forget()

        remaining = await loop.backend.all_beliefs()
        assert len(remaining) == 1
        merged = remaining[0]
        assert merged.confirmation_count == 3
        assert "consolidated" in merged.tags
        assert "subnet-18" in merged.entities

    async def test_no_merge_below_jaccard_threshold(self, loop):
        """Beliefs with no feature overlap don't merge."""
        loop.max_beliefs = 100
        loop.consolidation_threshold = 0.5

        await loop.backend.store(Belief(
            capability="mining", description="a",
            valence=Valence.POSITIVE, entities=("A",),
        ))
        await loop.backend.store(Belief(
            capability="mining", description="b",
            valence=Valence.POSITIVE, entities=("B",),
        ))
        await loop.backend.store(Belief(
            capability="mining", description="c",
            valence=Valence.POSITIVE, entities=("C",),
        ))

        await loop.forget()
        assert len(await loop.backend.all_beliefs()) == 3


# ======================================================================
# Strategy bundling via run_id
# ======================================================================

@pytest.mark.asyncio
class TestStrategyBundling:
    async def test_run_id_creates_strategy_bundle(self, loop):
        """Beliefs from the same run share a tag and can be retrieved together."""
        await loop.assimilate(Observation(
            capability="training", description="batch=64",
            valence=Valence.POSITIVE, run_id="run-042",
        ))
        await loop.assimilate(Observation(
            capability="training", description="cosine lr",
            valence=Valence.POSITIVE, run_id="run-042",
        ))
        await loop.assimilate(Observation(
            capability="training", description="unrelated approach",
            valence=Valence.POSITIVE, run_id="run-099",
        ))

        bundle = await loop.backend.query(tags=("run:run-042",))
        assert len(bundle) == 2
        descriptions = {b.description for b in bundle}
        assert "batch=64" in descriptions
        assert "cosine lr" in descriptions
        assert "unrelated approach" not in descriptions


# ======================================================================
# Crisis detection
# ======================================================================

@pytest.mark.asyncio
class TestCrisisDetection:
    async def test_crisis_mode_accelerates_weakening(self, loop):
        """Consecutive contradictions make the system more plastic."""
        for _ in range(5):
            await loop.assimilate(Observation(
                capability="x", description="old strategy",
                valence=Valence.POSITIVE,
            ))

        beliefs = await loop.backend.all_beliefs()
        conf_before = beliefs[0].confidence

        for i in range(3):
            await loop.assimilate(Observation(
                capability="x", description="old strategy",
                valence=Valence.NEGATIVE,
            ))

        beliefs = await loop.backend.all_beliefs()
        old = [b for b in beliefs if "old strategy" in b.description][0]
        total_drop = conf_before - old.confidence

        # With crisis (kicks in at contradiction 3): 0.3 + 0.3 + 0.9 = 1.5
        assert total_drop > 0.9 * 1.3

    async def test_agreement_resets_crisis(self, loop):
        """One agreement resets the contradiction streak."""
        await loop.assimilate(Observation(
            capability="x", description="strategy A",
            valence=Valence.POSITIVE,
        ))

        await loop.assimilate(Observation(
            capability="x", description="strategy A",
            valence=Valence.NEGATIVE,
        ))
        await loop.assimilate(Observation(
            capability="x", description="strategy A",
            valence=Valence.NEGATIVE,
        ))
        assert loop._contradiction_streak == 2

        await loop.assimilate(Observation(
            capability="x", description="strategy A",
            valence=Valence.POSITIVE,
        ))
        assert loop._contradiction_streak == 0

    async def test_streak_decays_on_forget(self, loop):
        """Between-runs decay prevents slow-burn false crises."""
        await loop.assimilate(Observation(
            capability="x", description="A",
            valence=Valence.POSITIVE,
        ))
        await loop.assimilate(Observation(
            capability="x", description="A",
            valence=Valence.NEGATIVE,
        ))
        await loop.assimilate(Observation(
            capability="x", description="A",
            valence=Valence.NEGATIVE,
        ))
        assert loop._contradiction_streak == 2

        await loop.forget()
        assert loop._contradiction_streak == 1

        await loop.forget()
        assert loop._contradiction_streak == 0
