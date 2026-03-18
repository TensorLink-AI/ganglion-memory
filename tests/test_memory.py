"""Tests for the memory system.

Covers: types, JSON backend, SQLite backend, MemoryLoop (the core primitive),
agent integration, federation, and the biological properties we care about:
    - Hebbian strengthening (confirmation)
    - Contradiction detection (delta emission)
    - Metric drift detection
    - Strength-based eviction (not age-based)
    - Entity profiling as a view
    - Cross-bot knowledge sharing
"""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ganglion.memory.agent import MemoryAgent, between_runs, result_to_observation
from ganglion.memory.backends.federated import FederatedMemoryBackend, PeerDiscovery
from ganglion.memory.backends.json_file import JsonMemoryBackend
from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.loop import MemoryLoop
from ganglion.memory.types import Belief, Delta, Observation, Valence


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def json_backend(tmp_dir):
    return JsonMemoryBackend(tmp_dir / "memory")


@pytest.fixture
def sqlite_backend(tmp_dir):
    return SqliteMemoryBackend(tmp_dir / "memory.db")


@pytest.fixture(params=["json", "sqlite"])
def backend(request, tmp_dir):
    """Run every backend test against both JSON and SQLite."""
    if request.param == "json":
        return JsonMemoryBackend(tmp_dir / "memory")
    return SqliteMemoryBackend(tmp_dir / "memory.db")


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


# ======================================================================
# Backend tests (parametrized: JSON + SQLite)
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


# ======================================================================
# MemoryLoop — the core primitive
# ======================================================================

@pytest.mark.asyncio
class TestMemoryLoop:
    async def test_novel_observation_creates_belief(self, loop):
        obs = Observation(capability="mining", description="batch=64", valence=Valence.POSITIVE)
        delta = await loop.assimilate(obs)
        assert delta is None  # novel, no contradiction

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
        loop.weaken_rate = 0.5  # aggressive weakening

        pos = Observation(capability="mining", description="batch=64", valence=Valence.POSITIVE)
        neg = Observation(capability="mining", description="batch=64", valence=Valence.NEGATIVE)

        await loop.assimilate(pos)  # confidence=1.0
        await loop.assimilate(neg)  # confidence=0.5
        await loop.assimilate(neg)  # confidence=0.0 → death → new negative belief

        beliefs = await loop.backend.all_beliefs()
        # Original weakened + new replacement
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
        # Second drain should be empty
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

        # Store 5 beliefs with different strengths
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
        # The weakest (b0, b1) should be gone
        descriptions = {b.description for b in remaining}
        assert "b0" not in descriptions
        assert "b1" not in descriptions


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
        assert "+" in profile  # positive
        assert "-" in profile  # negative

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

        # Empty memory
        ctx = await agent.remember()
        assert ctx == ""

        # Learn something
        delta = await agent.learn({
            "success": True,
            "description": "batch=64 works",
            "metric_name": "score",
            "metric_value": 0.82,
        })
        assert delta is None  # novel

        # Now memory has content
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


# ======================================================================
# Federated backend
# ======================================================================

@pytest.mark.asyncio
class TestFederation:
    async def test_peer_discovery(self, tmp_dir):
        base = tmp_dir / "shared"
        base.mkdir()

        alpha_backend = JsonMemoryBackend(base / "alpha")
        beta_backend = JsonMemoryBackend(base / "beta")

        await alpha_backend.store(Belief(
            capability="mining", description="Alpha discovery",
            valence=Valence.POSITIVE, source="alpha",
        ))
        await beta_backend.store(Belief(
            capability="mining", description="Beta discovery",
            valence=Valence.POSITIVE, source="beta",
        ))

        discovery = PeerDiscovery(base, "alpha")
        peer_beliefs = await discovery.query_all(capability="mining")
        descriptions = {b.description for b in peer_beliefs}
        assert "Beta discovery" in descriptions
        assert "Alpha discovery" not in descriptions

    async def test_federated_query_merges(self, tmp_dir):
        base = tmp_dir / "shared"
        base.mkdir()

        alpha_local = JsonMemoryBackend(base / "alpha")
        alpha_fed = FederatedMemoryBackend(
            alpha_local, PeerDiscovery(base, "alpha"),
        )

        beta_local = JsonMemoryBackend(base / "beta")

        await alpha_local.store(Belief(
            capability="mining", description="Alpha pattern",
            valence=Valence.POSITIVE, source="alpha",
        ))
        await beta_local.store(Belief(
            capability="mining", description="Beta pattern",
            valence=Valence.POSITIVE, source="beta",
        ))

        results = await alpha_fed.query(capability="mining")
        descriptions = {b.description for b in results}
        assert "Alpha pattern" in descriptions
        assert "Beta pattern" in descriptions

    async def test_federated_exclude_source(self, tmp_dir):
        base = tmp_dir / "shared"
        base.mkdir()

        alpha_local = JsonMemoryBackend(base / "alpha")
        alpha_fed = FederatedMemoryBackend(
            alpha_local, PeerDiscovery(base, "alpha"),
        )

        beta_local = JsonMemoryBackend(base / "beta")

        await alpha_local.store(Belief(
            capability="mining", description="Alpha",
            valence=Valence.POSITIVE, source="alpha",
        ))
        await beta_local.store(Belief(
            capability="mining", description="Beta",
            valence=Valence.POSITIVE, source="beta",
        ))

        results = await alpha_fed.query(capability="mining", exclude_source="alpha")
        descriptions = {b.description for b in results}
        assert "Beta" in descriptions
        assert "Alpha" not in descriptions
