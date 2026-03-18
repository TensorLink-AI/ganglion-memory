"""Tests for cortex.py — advanced retrieval capabilities.

Tests against JsonMemoryBackend only (cortex operates at the MemoryBackend
protocol level — if it works on one backend, it works on all).
"""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ganglion.memory.backends.json_file import JsonMemoryBackend
from ganglion.memory.cortex import spread_activation, temporal_neighbors
from ganglion.memory.types import Belief, Valence


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def backend(tmp_dir):
    return JsonMemoryBackend(tmp_dir / "memory")


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
# 2. Temporal Neighbors
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
# Import verification
# ======================================================================

class TestImports:
    def test_cortex_imports_from_package(self):
        """Cortex functions importable from ganglion.memory."""
        from ganglion.memory import spread_activation, temporal_neighbors  # noqa: F401

    def test_no_circular_imports(self):
        """Importing cortex doesn't cause circular import errors."""
        import importlib
        mod = importlib.import_module("ganglion.memory.cortex")
        assert hasattr(mod, "spread_activation")
        assert hasattr(mod, "temporal_neighbors")

    def test_removed_wrappers_not_exported(self):
        """Deleted wrapper functions are not in the public API."""
        import ganglion.memory as gm
        assert not hasattr(gm, "assimilate_with_biology")
        assert not hasattr(gm, "context_with_associations")
        assert not hasattr(gm, "between_runs_with_biology")
        assert not hasattr(gm, "compute_salience")
        assert not hasattr(gm, "consolidate")
        assert not hasattr(gm, "inhibit_competitors")
