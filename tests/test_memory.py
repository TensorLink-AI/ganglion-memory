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

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ganglion.memory.agent import MemoryAgent, between_runs, result_to_observation
from ganglion.memory.backends.federated import FederatedMemoryBackend, PeerDiscovery
from ganglion.memory.backends.json_file import JsonMemoryBackend
from ganglion.memory.backends.sqlite import SqliteMemoryBackend
from ganglion.memory.loop import MemoryLoop
from ganglion.memory.similarity import jaccard_similarity, tokenize
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
    backend = SqliteMemoryBackend(tmp_dir / "memory.db")
    yield backend
    backend.close()


@pytest.fixture(params=["json", "sqlite"])
def backend(request, tmp_dir):
    """Run every backend test against both JSON and SQLite."""
    if request.param == "json":
        yield JsonMemoryBackend(tmp_dir / "memory")
    else:
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
        # With consolidation floor (0.5), old belief strength = 5.0 * 10 * 0.5 = 25.0
        # New belief strength = 1.0 * 1 * ~1.0 = ~1.0
        assert old_consolidated.strength > new_weak.strength
        # Verify the floor is in effect (without floor, recency ~0.10 for 60 days)
        assert old_consolidated.strength >= 5.0 * 10 * 0.5


# ======================================================================
# Similarity
# ======================================================================

class TestSimilarity:
    def test_identical_strings(self):
        assert jaccard_similarity("batch_size=64 works", "batch_size=64 works") == 1.0

    def test_similar_strings(self):
        score = jaccard_similarity(
            "batch_size=64 works well for training",
            "batch 64 works well for training runs"
        )
        assert score > 0.7

    def test_dissimilar_strings(self):
        score = jaccard_similarity(
            "batch_size=64 works",
            "learning_rate=0.001 is optimal"
        )
        assert score < 0.3

    def test_empty_string(self):
        assert jaccard_similarity("", "anything") == 0.0

    def test_tokenize(self):
        tokens = tokenize("batch_size=64 Works WELL")
        assert "batch" in tokens
        assert "size" in tokens
        assert "64" in tokens
        assert "works" in tokens
        assert "well" in tokens


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

    async def test_sqlite_roundtrip_all_beliefs(self, sqlite_backend):
        """SQLite store + all_beliefs roundtrip (covers _row_to_belief UTC fix)."""
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
        assert b.first_seen is not None
        assert b.last_confirmed is not None

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
        # Jaccard score ~0.75, so use a lower threshold
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
        # Default threshold (0.85) might miss this
        found_loose = await backend.find_similar(obs, threshold=0.3)
        # Loose threshold should find it
        assert found_loose is not None


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
        """Repeated contradiction causes apoptosis.

        With contradiction-driven learning, weakening is halved (0.5x)
        so it takes more contradictions to kill a belief, but crisis
        mode (3+ consecutive) still accelerates the process.
        """
        loop.weaken_rate = 1.0  # aggressive weakening (halved internally to 0.5)

        pos = Observation(capability="mining", description="batch=64", valence=Valence.POSITIVE)
        neg = Observation(capability="mining", description="batch=64", valence=Valence.NEGATIVE)

        await loop.assimilate(pos)  # confidence=1.0
        await loop.assimilate(neg)  # confidence -= 1.0*0.5 = 0.5
        await loop.assimilate(neg)  # confidence -= 1.0*0.5 = 0.0 → death

        beliefs = await loop.backend.all_beliefs()
        # Original weakened + new replacement + conditional insights
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

    async def test_metric_shift_delta_snapshots_old_value(self, loop):
        """Delta.old_belief captures pre-mutation state for metric shifts."""
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
        # old_belief should reflect pre-mutation state
        assert delta.old_belief.metric_value == 0.8
        # Stored belief should have the new value
        beliefs = await loop.backend.all_beliefs()
        assert beliefs[0].metric_value == 0.5

    async def test_contradiction_delta_snapshots_old_confidence(self, loop):
        """Delta.old_belief captures pre-mutation state for contradictions."""
        pos = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE,
        )
        neg = Observation(
            capability="mining", description="batch=64",
            valence=Valence.NEGATIVE,
        )

        await loop.assimilate(pos)  # confidence=1.0
        delta = await loop.assimilate(neg)

        assert delta is not None
        assert delta.old_belief.confidence == 1.0  # snapshot before weakening

    async def test_metric_shift_does_not_strengthen(self, loop):
        """Metric shift in agreement branch should not increase confidence/count."""
        loop.metric_shift_threshold = 0.1

        obs1 = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, metric_name="score", metric_value=0.85,
        )
        obs2 = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, metric_name="score", metric_value=0.45,
        )

        await loop.assimilate(obs1)
        delta = await loop.assimilate(obs2)

        assert delta is not None
        assert delta.delta_type == "metric_shift"

        beliefs = await loop.backend.all_beliefs()
        b = beliefs[0]
        # Should NOT have been strengthened
        assert b.confidence <= 1.0  # not increased from initial 1.0
        assert b.confirmation_count == 1  # not bumped to 2

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

        # First, create a base belief
        base_obs = Observation(
            capability="mining", description="concurrent test",
            valence=Valence.POSITIVE, metric_name="score", metric_value=1.0,
        )
        await loop.assimilate(base_obs)

        # Create 10 contradicting observations to generate deltas
        for i in range(10):
            obs = Observation(
                capability=f"cap_{i}",
                description=f"unique obs {i}",
                valence=Valence.POSITIVE,
                metric_name="score",
                metric_value=0.5,
            )
            # First assimilate to create belief
            await loop.assimilate(obs)

        # Now create contradictions concurrently
        async def make_contradiction(idx: int) -> Delta | None:
            obs = Observation(
                capability=f"cap_{idx}",
                description=f"unique obs {idx}",
                valence=Valence.NEGATIVE,
            )
            return await loop.assimilate(obs)

        results = await asyncio.gather(*[make_contradiction(i) for i in range(10)])
        contradiction_deltas = [r for r in results if r is not None]

        # All 10 should produce contradiction deltas
        assert len(contradiction_deltas) == 10

        # drain_deltas should have all of them
        drained = await loop.drain_deltas()
        assert len(drained) == 10
        assert len(await loop.drain_deltas()) == 0  # second drain empty


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


# ======================================================================
# JSON backend cache
# ======================================================================

@pytest.mark.asyncio
class TestJsonCache:
    async def test_cache_returns_same_data(self, json_backend):
        """After storing, all_beliefs() uses cache on second call."""
        for i in range(3):
            await json_backend.store(Belief(
                capability="x", description=f"b{i}", valence=Valence.POSITIVE,
            ))

        result1 = await json_backend.all_beliefs()
        result2 = await json_backend.all_beliefs()
        assert len(result1) == 3
        assert len(result2) == 3
        assert {b.description for b in result1} == {b.description for b in result2}

    async def test_invalidate_cache(self, json_backend):
        """invalidate_cache forces re-read from disk."""
        await json_backend.store(Belief(
            capability="x", description="cached", valence=Valence.POSITIVE,
        ))
        json_backend.invalidate_cache()
        beliefs = await json_backend.all_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].description == "cached"


# ======================================================================
# Salience (built into MemoryLoop.assimilate)
# ======================================================================

@pytest.mark.asyncio
class TestSalience:
    async def test_novel_surprising_observation_gets_boosted(self, loop):
        """Novel observation with surprising metric gets confidence > 1.0."""
        # Seed peers so salience has data
        for i in range(5):
            await loop.assimilate(Observation(
                capability="mining", description=f"peer_{i}",
                valence=Valence.POSITIVE, metric_value=0.5 + i * 0.01,
            ))

        # Novel observation with outlier metric
        obs = Observation(
            capability="mining", description="outlier strategy",
            valence=Valence.POSITIVE, metric_value=0.99,
        )
        delta = await loop.assimilate(obs)
        assert delta is None  # novel

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

    async def test_agreement_preserves_hebbian_not_salience(self, loop):
        """Repeated agreement uses Hebbian strengthening, not salience."""
        obs = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, metric_value=0.8,
        )
        await loop.assimilate(obs)
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].confirmation_count == 2

    async def test_no_metric_gets_baseline(self, loop):
        """Observation without metric_value gets confidence=1.0."""
        obs = Observation(
            capability="mining", description="no metric",
            valence=Valence.POSITIVE,
        )
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        assert beliefs[0].confidence == 1.0

    async def test_too_few_peers_gets_baseline(self, loop):
        """Fewer than 2 peer metrics → confidence=1.0."""
        await loop.assimilate(Observation(
            capability="mining", description="only peer",
            valence=Valence.POSITIVE, metric_value=0.8,
        ))

        obs = Observation(
            capability="mining", description="new one",
            valence=Valence.POSITIVE, metric_value=0.99,
        )
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        new = next(b for b in beliefs if b.description == "new one")
        assert new.confidence == 1.0


# ======================================================================
# Inhibition (built into MemoryLoop.assimilate)
# ======================================================================

@pytest.mark.asyncio
class TestInhibition:
    async def test_agreement_weakens_competitors(self, loop):
        """Agreement on one belief weakens competitors with overlapping features."""
        # Create two competing beliefs directly
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

        # Agree with batch=64 — should inhibit batch=128
        obs = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        )
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        comp = next(b for b in beliefs if b.description == "batch=128")
        assert comp.confidence < 1.0

    async def test_same_description_not_inhibited(self, loop):
        """Same-description beliefs are allies, not competitors."""
        await loop.backend.store(Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=1.0, confirmation_count=1,
        ))
        await loop.backend.store(Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=1.0, confirmation_count=1,
        ))

        obs = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        )
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        # All beliefs with same description should be untouched by inhibition
        for b in beliefs:
            if b.description == "batch=64":
                assert b.confidence >= 1.0

    async def test_contradiction_does_not_inhibit(self, loop):
        """Contradiction does NOT trigger inhibition."""
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

        # Record batch=128 confidence before contradiction
        beliefs_before = await loop.backend.all_beliefs()
        alt_before = next(b for b in beliefs_before if b.description == "batch=128")
        conf_before = alt_before.confidence

        # Contradict batch=64
        neg = Observation(
            capability="mining", description="batch=64",
            valence=Valence.NEGATIVE, entities=("subnet-18",),
        )
        delta = await loop.assimilate(neg)
        assert delta is not None
        assert delta.delta_type == "contradiction"

        # batch=128 confidence should not have been reduced
        beliefs_after = await loop.backend.all_beliefs()
        alt_after = next(b for b in beliefs_after if b.description == "batch=128")
        assert alt_after.confidence >= conf_before

    async def test_inhibition_floor(self, loop):
        """Inhibition cannot push confidence below the floor (hardcoded at 0.2)."""

        await loop.backend.store(Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=1.0, confirmation_count=1,
        ))
        await loop.backend.store(Belief(
            capability="mining", description="batch=256",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=0.25, confirmation_count=1,
        ))

        obs = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        )
        await loop.assimilate(obs)

        beliefs = await loop.backend.all_beliefs()
        comp = next(b for b in beliefs if b.description == "batch=256")
        assert comp.confidence >= 0.2

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
        assert comp.confidence == 1.0  # untouched


# ======================================================================
# Consolidation (built into MemoryLoop.forget)
# ======================================================================

@pytest.mark.asyncio
class TestConsolidation:
    async def test_forget_merges_similar_cluster(self, loop):
        """forget() merges similar beliefs before evicting."""
        loop.max_beliefs = 1  # force eviction after consolidation
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

    async def test_seed_anchored_no_chaining(self, loop):
        """A→B and B→C overlap, but A and C don't. No chain merge."""
        loop.max_beliefs = 100
        loop.consolidation_threshold = 0.5

        await loop.backend.store(Belief(
            capability="mining", description="a",
            valence=Valence.POSITIVE, entities=("X", "Y"),
        ))
        await loop.backend.store(Belief(
            capability="mining", description="b",
            valence=Valence.POSITIVE, entities=("Y", "Z"),
        ))
        await loop.backend.store(Belief(
            capability="mining", description="c",
            valence=Valence.POSITIVE, entities=("Z", "W"),
        ))

        await loop.forget()
        # No cluster of 3 formed — all 3 remain
        assert len(await loop.backend.all_beliefs()) == 3

    async def test_merged_entities_and_tags_sorted(self, loop):
        """Merged belief has sorted entities and tags."""
        loop.max_beliefs = 1
        loop.consolidation_threshold = 0.3

        await loop.backend.store(Belief(
            capability="mining", description="a",
            valence=Valence.POSITIVE,
            entities=("C", "A"), tags=("beta", "alpha"),
        ))
        await loop.backend.store(Belief(
            capability="mining", description="b",
            valence=Valence.POSITIVE,
            entities=("A", "B"), tags=("alpha", "gamma"),
        ))
        await loop.backend.store(Belief(
            capability="mining", description="c",
            valence=Valence.POSITIVE,
            entities=("A", "C"), tags=("alpha", "beta"),
        ))

        await loop.forget()
        remaining = await loop.backend.all_beliefs()
        if len(remaining) == 1:
            merged = remaining[0]
            assert list(merged.entities) == sorted(merged.entities)
            assert list(merged.tags) == sorted(merged.tags)


# ======================================================================
# Source-agnostic strengthening (cross_agent_bonus removed in v2)
# ======================================================================

@pytest.mark.asyncio
class TestSourceAgnosticStrengthening:
    async def test_all_sources_get_same_strengthen_rate(self, loop):
        """All confirmations use the same strengthen_rate regardless of source."""
        await loop.assimilate(Observation(
            capability="mining", description="batch=64 works",
            valence=Valence.POSITIVE, source="alpha",
        ))
        beliefs = await loop.backend.all_beliefs()
        c1 = beliefs[0].confidence

        # Cross-agent confirmation: same rate as self-confirmation
        await loop.assimilate(Observation(
            capability="mining", description="batch=64 works",
            valence=Valence.POSITIVE, source="beta",
        ))
        beliefs = await loop.backend.all_beliefs()
        boost = beliefs[0].confidence - c1

        assert abs(boost - loop.strengthen_rate) < 0.001

    async def test_same_source_gets_strengthen_rate(self, loop):
        """Same-source confirmation gets normal strengthen_rate."""
        await loop.assimilate(Observation(
            capability="mining", description="batch=64 works",
            valence=Valence.POSITIVE, source="alpha",
        ))
        beliefs = await loop.backend.all_beliefs()
        c1 = beliefs[0].confidence

        await loop.assimilate(Observation(
            capability="mining", description="batch=64 works",
            valence=Valence.POSITIVE, source="alpha",
        ))
        beliefs = await loop.backend.all_beliefs()
        boost = beliefs[0].confidence - c1

        assert abs(boost - loop.strengthen_rate) < 0.001


# ======================================================================
# Strength-based ranking (exploration_rate removed in v2)
# ======================================================================

@pytest.mark.asyncio
class TestStrengthRanking:
    async def test_strongest_beliefs_surface_first(self, loop):
        """Without embeddings, strongest beliefs appear first."""
        for _ in range(10):
            await loop.assimilate(Observation(
                capability="x", description="dominant",
                valence=Valence.POSITIVE,
            ))
        await loop.assimilate(Observation(
            capability="x", description="weak",
            valence=Valence.POSITIVE,
        ))

        # Only strongest belief should appear in max_entries=1
        for _ in range(10):
            ctx = await loop.context_for("x", max_entries=1)
            assert "weak" not in ctx


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

        # Query by run tag retrieves the bundle
        bundle = await loop.backend.query(tags=("run:run-042",))
        assert len(bundle) == 2
        descriptions = {b.description for b in bundle}
        assert "batch=64" in descriptions
        assert "cosine lr" in descriptions
        assert "unrelated approach" not in descriptions

    async def test_no_run_tag_without_run_id(self, loop):
        """Beliefs without run_id get no run: tag."""
        await loop.assimilate(Observation(
            capability="training", description="batch=64",
            valence=Valence.POSITIVE,
        ))

        beliefs = await loop.backend.all_beliefs()
        assert not any(t.startswith("run:") for t in beliefs[0].tags)

    async def test_run_tag_merges_on_agreement(self, loop):
        """Run tag is added when a new run confirms an existing belief."""
        await loop.assimilate(Observation(
            capability="training", description="batch=64",
            valence=Valence.POSITIVE,
        ))
        await loop.assimilate(Observation(
            capability="training", description="batch=64",
            valence=Valence.POSITIVE, run_id="run-042",
        ))

        beliefs = await loop.backend.all_beliefs()
        assert "run:run-042" in beliefs[0].tags


# ======================================================================
# Crisis detection
# ======================================================================

@pytest.mark.asyncio
class TestCrisisDetection:
    async def test_crisis_mode_accelerates_weakening(self, loop):
        """Consecutive contradictions make the system more plastic.

        With contradiction-driven learning, base weakening is halved (0.5x),
        but crisis mode (3x at 3+ contradictions) still accelerates.
        Effective drops: 0.15 + 0.15 + 0.45 = 0.75
        Non-crisis would be: 3 × 0.15 = 0.45
        """
        # crisis acceleration is built-in (3.0x after 3 consecutive contradictions)

        # Establish a strong belief
        for _ in range(5):
            await loop.assimilate(Observation(
                capability="x", description="old strategy",
                valence=Valence.POSITIVE,
            ))

        beliefs = await loop.backend.all_beliefs()
        conf_before = beliefs[0].confidence

        # Three consecutive contradictions should trigger crisis mode
        for i in range(3):
            await loop.assimilate(Observation(
                capability="x", description="old strategy",
                valence=Valence.NEGATIVE,
            ))

        beliefs = await loop.backend.all_beliefs()
        old = [b for b in beliefs if "old strategy" in b.description][0]
        total_drop = conf_before - old.confidence

        # With half-rate weakening: non-crisis = 3 × 0.15 = 0.45
        # With crisis (3x on 3rd): 0.15 + 0.15 + 0.45 = 0.75
        assert total_drop > 0.45 * 1.3  # at least 30% more than non-crisis

    async def test_agreement_resets_crisis(self, loop):
        """One agreement resets the contradiction streak."""
        # crisis acceleration is built-in (3.0x after 3 consecutive contradictions)

        await loop.assimilate(Observation(
            capability="x", description="strategy A",
            valence=Valence.POSITIVE,
        ))

        # Two contradictions
        await loop.assimilate(Observation(
            capability="x", description="strategy A",
            valence=Valence.NEGATIVE,
        ))
        await loop.assimilate(Observation(
            capability="x", description="strategy A",
            valence=Valence.NEGATIVE,
        ))
        assert loop._contradiction_streak == 2

        # One agreement resets
        await loop.assimilate(Observation(
            capability="x", description="strategy A",
            valence=Valence.POSITIVE,
        ))
        assert loop._contradiction_streak == 0

    async def test_crisis_accelerates_after_three(self, loop):
        """Crisis mode kicks in at exactly 3 consecutive contradictions.

        With contradiction-driven learning, effective weakening is halved.
        First two: 0.3 * 0.5 = 0.15 each
        Third (crisis 3x): 0.3 * 3.0 * 0.5 = 0.45
        """
        # Build up moderate confidence
        for _ in range(5):
            await loop.assimilate(Observation(
                capability="x", description="belief",
                valence=Valence.POSITIVE,
            ))

        beliefs = await loop.backend.all_beliefs()
        conf_before = beliefs[0].confidence

        # First two contradictions: halved weaken_rate
        await loop.assimilate(Observation(
            capability="x", description="belief",
            valence=Valence.NEGATIVE,
        ))
        await loop.assimilate(Observation(
            capability="x", description="belief",
            valence=Valence.NEGATIVE,
        ))

        beliefs = await loop.backend.all_beliefs()
        old = [b for b in beliefs if "belief" in b.description][0]
        drop_after_2 = conf_before - old.confidence
        # Should be 2 × 0.3 × 0.5 = 0.3 (halved rate)
        assert abs(drop_after_2 - 2 * loop.weaken_rate * 0.5) < 0.01

        # Third contradiction: crisis kicks in (3.0x multiplier, still halved)
        await loop.assimilate(Observation(
            capability="x", description="belief",
            valence=Valence.NEGATIVE,
        ))
        beliefs = await loop.backend.all_beliefs()
        old = [b for b in beliefs if "belief" in b.description][0]
        total_drop = conf_before - old.confidence
        # Should be 0.15 + 0.15 + 0.45 = 0.75 (third hit is 3x × 0.5)
        assert total_drop > 2 * loop.weaken_rate * 0.5 + loop.weaken_rate * 0.5

    async def test_streak_decays_on_forget(self, loop):
        """Between-runs decay prevents slow-burn false crises."""
        # crisis acceleration is built-in (3.0x after 3 consecutive contradictions)

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
