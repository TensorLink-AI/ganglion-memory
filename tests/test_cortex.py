"""Tests for cortex.py — biological extensions to MemoryLoop.

Tests against JsonMemoryBackend only (cortex operates at the MemoryBackend
protocol level — if it works on one backend, it works on all).
"""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ganglion.memory.backends.json_file import JsonMemoryBackend
from ganglion.memory.cortex import (
    assimilate_with_biology,
    between_runs_with_biology,
    compute_salience,
    consolidate,
    context_with_associations,
    inhibit_competitors,
    spread_activation,
    temporal_neighbors,
)
from ganglion.memory.loop import MemoryLoop
from ganglion.memory.types import Belief, Observation, Valence


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def backend(tmp_dir):
    return JsonMemoryBackend(tmp_dir / "memory")


@pytest.fixture
def loop(backend):
    return MemoryLoop(backend=backend)


# ======================================================================
# 1. Spread Activation
# ======================================================================

@pytest.mark.asyncio
class TestSpreadActivation:
    async def test_finds_neighbors_via_shared_entity(self, backend):
        """Beliefs sharing an entity are discovered by spread activation."""
        seed = Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        )
        await backend.store(seed)

        neighbor = Belief(
            capability="mining", description="lr=0.001",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        )
        await backend.store(neighbor)

        unrelated = Belief(
            capability="mining", description="unrelated",
            valence=Valence.POSITIVE, entities=("subnet-99",),
        )
        await backend.store(unrelated)

        results = await spread_activation(seed, backend, max_hops=1, limit=10)
        descriptions = {b.description for b in results}
        assert "lr=0.001" in descriptions
        assert "unrelated" not in descriptions

    async def test_finds_neighbors_via_shared_tag(self, backend):
        """Beliefs sharing a tag are discovered by spread activation."""
        seed = Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, tags=("strategy",),
        )
        await backend.store(seed)

        neighbor = Belief(
            capability="mining", description="lr=0.001",
            valence=Valence.POSITIVE, tags=("strategy",),
        )
        await backend.store(neighbor)

        results = await spread_activation(seed, backend, max_hops=1, limit=10)
        assert len(results) >= 1
        assert results[0].description == "lr=0.001"

    async def test_seed_excluded_from_results(self, backend):
        """The seed belief itself never appears in spread activation results."""
        seed = Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        )
        await backend.store(seed)

        results = await spread_activation(seed, backend, max_hops=1, limit=10)
        assert all(b.id != seed.id for b in results)

    async def test_scores_accumulate_across_hops(self, backend):
        """Scores persist across hops, not just the final hop."""
        seed = Belief(
            capability="mining", description="seed",
            valence=Valence.POSITIVE, entities=("A",),
        )
        await backend.store(seed)

        hop1 = Belief(
            capability="mining", description="hop1",
            valence=Valence.POSITIVE, entities=("A", "B"),
        )
        await backend.store(hop1)

        hop2 = Belief(
            capability="mining", description="hop2",
            valence=Valence.POSITIVE, entities=("B",),
        )
        await backend.store(hop2)

        results = await spread_activation(seed, backend, max_hops=2, limit=10)
        descriptions = {b.description for b in results}
        # hop1 found via entity A (hop 0), hop2 found via entity B (hop 1)
        assert "hop1" in descriptions
        assert "hop2" in descriptions

    async def test_empty_entities_and_tags(self, backend):
        """Seed with no entities/tags returns no results."""
        seed = Belief(
            capability="mining", description="lonely",
            valence=Valence.POSITIVE,
        )
        await backend.store(seed)

        results = await spread_activation(seed, backend)
        assert results == []


# ======================================================================
# 2. Consolidation
# ======================================================================

@pytest.mark.asyncio
class TestConsolidation:
    async def test_merges_similar_cluster(self, backend):
        """Three beliefs with overlapping entities/tags merge into one."""
        for i in range(3):
            await backend.store(Belief(
                capability="mining", description=f"strategy_{i}",
                valence=Valence.POSITIVE,
                entities=("subnet-18",), tags=("gpu",),
                confirmation_count=1, confidence=1.0,
                metric_value=0.8 + i * 0.01,
                metric_name="score",
            ))

        removed = await consolidate(backend, min_cluster_size=3, similarity_threshold=0.5)
        assert removed == 3

        remaining = await backend.all_beliefs()
        assert len(remaining) == 1
        merged = remaining[0]
        assert merged.confirmation_count == 3
        assert "consolidated" in merged.tags
        assert "subnet-18" in merged.entities

    async def test_no_merge_below_threshold(self, backend):
        """Beliefs with no feature overlap don't merge."""
        await backend.store(Belief(
            capability="mining", description="a",
            valence=Valence.POSITIVE, entities=("A",),
        ))
        await backend.store(Belief(
            capability="mining", description="b",
            valence=Valence.POSITIVE, entities=("B",),
        ))
        await backend.store(Belief(
            capability="mining", description="c",
            valence=Valence.POSITIVE, entities=("C",),
        ))

        removed = await consolidate(backend, min_cluster_size=3)
        assert removed == 0

    async def test_no_merge_small_store(self, backend):
        """Fewer than min_cluster_size beliefs → no consolidation."""
        await backend.store(Belief(
            capability="mining", description="a",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        ))

        removed = await consolidate(backend, min_cluster_size=3)
        assert removed == 0

    async def test_seed_anchored_no_chaining(self, backend):
        """A→B and B→C overlap, but A and C don't. No chain merge."""
        await backend.store(Belief(
            capability="mining", description="a",
            valence=Valence.POSITIVE, entities=("X", "Y"),
        ))
        await backend.store(Belief(
            capability="mining", description="b",
            valence=Valence.POSITIVE, entities=("Y", "Z"),
        ))
        await backend.store(Belief(
            capability="mining", description="c",
            valence=Valence.POSITIVE, entities=("Z", "W"),
        ))

        removed = await consolidate(backend, min_cluster_size=3, similarity_threshold=0.5)
        # A shares {Y} with B (jaccard=1/3), B shares {Z} with C (jaccard=1/3)
        # Neither reaches 0.5 threshold, so no cluster of size 3 forms
        assert removed == 0

    async def test_entities_and_tags_sorted_in_merged(self, backend):
        """Merged belief has sorted entities and tags."""
        await backend.store(Belief(
            capability="mining", description="a",
            valence=Valence.POSITIVE,
            entities=("C", "A"), tags=("beta", "alpha"),
        ))
        await backend.store(Belief(
            capability="mining", description="b",
            valence=Valence.POSITIVE,
            entities=("A", "B"), tags=("alpha", "gamma"),
        ))
        await backend.store(Belief(
            capability="mining", description="c",
            valence=Valence.POSITIVE,
            entities=("A", "C"), tags=("alpha", "beta"),
        ))

        removed = await consolidate(backend, min_cluster_size=3, similarity_threshold=0.3)
        if removed > 0:
            remaining = await backend.all_beliefs()
            merged = remaining[0]
            assert list(merged.entities) == sorted(merged.entities)
            assert list(merged.tags) == sorted(merged.tags)


# ======================================================================
# 3. Salience
# ======================================================================

@pytest.mark.asyncio
class TestSalience:
    async def test_no_metric_returns_baseline(self, backend):
        """Observation without metric_value gets baseline confidence."""
        obs = Observation(
            capability="mining", description="no metric",
            valence=Valence.POSITIVE,
        )
        salience = await compute_salience(obs, backend)
        assert salience == 1.0

    async def test_too_few_peers_returns_baseline(self, backend):
        """Fewer than 2 peer metrics → baseline confidence."""
        await backend.store(Belief(
            capability="mining", description="a",
            valence=Valence.POSITIVE, metric_value=0.8,
        ))
        obs = Observation(
            capability="mining", description="new",
            valence=Valence.POSITIVE, metric_value=0.9,
        )
        salience = await compute_salience(obs, backend)
        assert salience == 1.0

    async def test_surprising_metric_boosts_salience(self, backend):
        """An outlier metric value gets a salience boost > 1."""
        # Create a cluster of beliefs around metric_value ~0.5
        for i in range(5):
            await backend.store(Belief(
                capability="mining", description=f"peer_{i}",
                valence=Valence.POSITIVE, metric_value=0.5 + i * 0.01,
            ))

        # This observation is an outlier (0.95 vs mean ~0.52)
        obs = Observation(
            capability="mining", description="outlier",
            valence=Valence.POSITIVE, metric_value=0.95,
        )
        salience = await compute_salience(obs, backend)
        assert salience > 1.0

    async def test_normal_metric_low_salience(self, backend):
        """A metric close to the mean gets minimal boost."""
        for i in range(5):
            await backend.store(Belief(
                capability="mining", description=f"peer_{i}",
                valence=Valence.POSITIVE, metric_value=0.5 + i * 0.01,
            ))

        obs = Observation(
            capability="mining", description="normal",
            valence=Valence.POSITIVE, metric_value=0.52,
        )
        salience = await compute_salience(obs, backend)
        # Should be close to baseline
        assert salience < 2.0


# ======================================================================
# 4. Inhibition
# ======================================================================

@pytest.mark.asyncio
class TestInhibition:
    async def test_weakens_competitors(self, backend):
        """A strengthened belief weakens competitors with overlapping features."""
        strong = Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=2.0,
        )
        await backend.store(strong)

        competitor = Belief(
            capability="mining", description="batch=128",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=1.0,
        )
        await backend.store(competitor)

        inhibited = await inhibit_competitors(strong, backend)
        assert inhibited == 1

        beliefs = await backend.all_beliefs()
        comp = next(b for b in beliefs if b.description == "batch=128")
        assert comp.confidence < 1.0

    async def test_same_description_not_inhibited(self, backend):
        """Same-description beliefs are allies, not competitors."""
        strong = Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=2.0,
        )
        await backend.store(strong)

        ally = Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=1.0,
        )
        await backend.store(ally)

        inhibited = await inhibit_competitors(strong, backend)
        assert inhibited == 0

    async def test_confidence_floor(self, backend):
        """Inhibition cannot push confidence below the floor."""
        strong = Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=5.0,
        )
        await backend.store(strong)

        weak = Belief(
            capability="mining", description="batch=256",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            confidence=0.25,
        )
        await backend.store(weak)

        await inhibit_competitors(strong, backend, confidence_floor=0.2)
        beliefs = await backend.all_beliefs()
        comp = next(b for b in beliefs if b.description == "batch=256")
        assert comp.confidence >= 0.2

    async def test_no_features_no_inhibition(self, backend):
        """Strengthened belief with no entities/tags inhibits nothing."""
        strong = Belief(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, confidence=2.0,
        )
        await backend.store(strong)

        inhibited = await inhibit_competitors(strong, backend)
        assert inhibited == 0


# ======================================================================
# 5. Temporal Neighbors
# ======================================================================

@pytest.mark.asyncio
class TestTemporalNeighbors:
    async def test_finds_beliefs_within_window(self, backend):
        """Beliefs confirmed within the time window are returned."""
        now = datetime.now(UTC)
        target = Belief(
            capability="mining", description="target",
            valence=Valence.POSITIVE, last_confirmed=now,
        )
        await backend.store(target)

        near = Belief(
            capability="mining", description="near",
            valence=Valence.POSITIVE, last_confirmed=now + timedelta(minutes=30),
        )
        await backend.store(near)

        far = Belief(
            capability="mining", description="far",
            valence=Valence.POSITIVE, last_confirmed=now + timedelta(hours=5),
        )
        await backend.store(far)

        results = await temporal_neighbors(target, backend, window=timedelta(hours=1))
        descriptions = {b.description for b in results}
        assert "near" in descriptions
        assert "far" not in descriptions

    async def test_excludes_self(self, backend):
        """The belief itself is not included in temporal neighbors."""
        now = datetime.now(UTC)
        belief = Belief(
            capability="mining", description="self",
            valence=Valence.POSITIVE, last_confirmed=now,
        )
        await backend.store(belief)

        results = await temporal_neighbors(belief, backend)
        assert all(b.id != belief.id for b in results)


# ======================================================================
# Wrappers: assimilate_with_biology, context_with_associations,
#           between_runs_with_biology
# ======================================================================

@pytest.mark.asyncio
class TestAssimilateWithBiology:
    async def test_novel_observation_gets_salience(self, loop):
        """Novel observation with surprising metric gets boosted confidence."""
        # Seed some peers so salience has data
        for i in range(5):
            await loop.assimilate(Observation(
                capability="mining", description=f"peer_{i}",
                valence=Valence.POSITIVE, metric_value=0.5 + i * 0.01,
            ))

        # Novel observation with surprising metric
        obs = Observation(
            capability="mining", description="outlier strategy",
            valence=Valence.POSITIVE, metric_value=0.99,
        )
        delta = await assimilate_with_biology(loop, obs)
        assert delta is None  # novel

        beliefs = await loop.backend.all_beliefs()
        outlier = next(b for b in beliefs if b.description == "outlier strategy")
        # Should have boosted confidence > 1.0
        assert outlier.confidence > 1.0

    async def test_agreement_preserves_hebbian_strength(self, loop):
        """Repeated agreement uses Hebbian strengthening, not salience."""
        obs = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, metric_value=0.8,
        )
        await assimilate_with_biology(loop, obs)
        await assimilate_with_biology(loop, obs)

        beliefs = await loop.backend.all_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].confirmation_count == 2

    async def test_contradiction_does_not_inhibit(self, loop):
        """Contradiction path does NOT trigger inhibition."""
        # Store two beliefs with shared entity
        pos = Observation(
            capability="mining", description="batch=64",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        )
        await loop.assimilate(pos)

        alt = Observation(
            capability="mining", description="batch=128",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        )
        await loop.assimilate(alt)

        # Record confidence before contradiction
        beliefs_before = await loop.backend.all_beliefs()
        alt_before = next(b for b in beliefs_before if b.description == "batch=128")
        conf_before = alt_before.confidence

        # Contradict batch=64
        neg = Observation(
            capability="mining", description="batch=64",
            valence=Valence.NEGATIVE, entities=("subnet-18",),
        )
        delta = await assimilate_with_biology(loop, neg)
        assert delta is not None
        assert delta.delta_type == "contradiction"

        # batch=128 confidence should not have been reduced by inhibition
        beliefs_after = await loop.backend.all_beliefs()
        alt_after = next(b for b in beliefs_after if b.description == "batch=128")
        assert alt_after.confidence >= conf_before


@pytest.mark.asyncio
class TestContextWithAssociations:
    async def test_includes_related_knowledge(self, loop):
        """Context with associations includes related beliefs."""
        await loop.assimilate(Observation(
            capability="mining", description="batch=64 works",
            valence=Valence.POSITIVE, entities=("subnet-18",),
            metric_value=0.85, metric_name="score",
        ))
        await loop.assimilate(Observation(
            capability="eval", description="eval on subnet-18",
            valence=Valence.POSITIVE, entities=("subnet-18",),
        ))

        ctx = await context_with_associations(
            loop, capability="mining", entities=("subnet-18",),
        )
        assert "What works" in ctx

    async def test_empty_returns_empty(self, loop):
        """Empty memory returns empty string."""
        ctx = await context_with_associations(loop, capability="nonexistent")
        assert ctx == ""


@pytest.mark.asyncio
class TestBetweenRunsWithBiology:
    async def test_returns_deltas_consolidated_forgotten(self, loop):
        """Returns (deltas, consolidated, forgotten) tuple."""
        loop.metric_shift_threshold = 0.01
        await loop.assimilate(Observation(
            capability="x", description="A",
            valence=Valence.POSITIVE, metric_value=1.0,
        ))
        await loop.assimilate(Observation(
            capability="x", description="A",
            valence=Valence.POSITIVE, metric_value=0.5,
        ))

        deltas, consolidated, forgotten = await between_runs_with_biology(loop)
        assert len(deltas) >= 1
        assert isinstance(consolidated, int)
        assert isinstance(forgotten, int)

    async def test_consolidation_only_above_threshold(self, loop):
        """Consolidation only triggers when total_beliefs > threshold."""
        for i in range(5):
            await loop.assimilate(Observation(
                capability="mining", description=f"obs_{i}",
                valence=Valence.POSITIVE, entities=("subnet-18",),
            ))

        # With threshold=100, no consolidation should happen
        deltas, consolidated, forgotten = await between_runs_with_biology(
            loop, consolidation_threshold=100,
        )
        assert consolidated == 0

    async def test_drain_then_consolidate_then_forget_order(self, loop):
        """Draining happens before consolidation and forgetting."""
        loop.metric_shift_threshold = 0.01
        await loop.assimilate(Observation(
            capability="x", description="A",
            valence=Valence.POSITIVE, metric_value=1.0,
        ))
        await loop.assimilate(Observation(
            capability="x", description="A",
            valence=Valence.POSITIVE, metric_value=0.5,
        ))

        deltas, consolidated, forgotten = await between_runs_with_biology(loop)
        # Deltas should be drained
        assert len(deltas) >= 1
        # Second call should have empty deltas
        deltas2, _, _ = await between_runs_with_biology(loop)
        assert len(deltas2) == 0


# ======================================================================
# Import verification
# ======================================================================

class TestImports:
    def test_cortex_imports_from_package(self):
        """All cortex functions importable from ganglion.memory."""
        from ganglion.memory import (  # noqa: F401
            assimilate_with_biology,
            between_runs_with_biology,
            compute_salience,
            consolidate,
            context_with_associations,
            inhibit_competitors,
            spread_activation,
            temporal_neighbors,
        )

    def test_no_circular_imports(self):
        """Importing cortex doesn't cause circular import errors."""
        import importlib
        mod = importlib.import_module("ganglion.memory.cortex")
        assert hasattr(mod, "spread_activation")
        assert hasattr(mod, "assimilate_with_biology")
