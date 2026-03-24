"""Tests for ReMem-style refine tools.

6 tests covering merge, split, rewrite, and forget operations.
"""

import tempfile
from pathlib import Path

import pytest

from ganglion.memory.backends.sqlite import SqliteBackend
from ganglion.memory.core import Memory
from ganglion.memory.embed import CallableEmbedder
from ganglion.memory.refine import (
    forget_experience,
    merge_experiences,
    rewrite_experience,
    split_experience,
)
from ganglion.memory.types import Experience


async def _mock_embed(text: str) -> list[float]:
    """Deterministic mock: character frequency vector (26 dims)."""
    vec = [0.0] * 26
    for c in text.lower():
        if "a" <= c <= "z":
            vec[ord(c) - ord("a")] += 1.0
    total = sum(v * v for v in vec) ** 0.5
    if total > 0:
        vec = [v / total for v in vec]
    return vec


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def mem(tmp_dir):
    backend = SqliteBackend(tmp_dir / "refine.db")
    m = Memory(backend=backend, embedder=CallableEmbedder(_mock_embed))
    yield m
    backend.close()


# ======================================================================
# Merge
# ======================================================================


@pytest.mark.asyncio
class TestMerge:
    async def test_merge_two(self, mem):
        """Merge two experiences into one."""
        e1 = await mem.add("approach A works", tags=("mining",))
        e2 = await mem.add("approach B works zzz", tags=("mining",))
        merged = await merge_experiences(mem, [e1.id, e2.id], "A and B both work")
        assert merged.content == "A and B both work"
        assert merged.id is not None
        assert await mem.count() == 1

    async def test_merge_sums_counts(self, mem):
        """Merged experience sums confirmation and contradiction counts."""
        e1 = await mem.add("xxxx works", tags=("t",))
        await mem.confirm(e1.id)
        e2 = await mem.add("yyyy works", tags=("t",))
        await mem.contradict(e2.id)
        merged = await merge_experiences(mem, [e1.id, e2.id], "merged result")
        assert merged.confirmation_count == 3  # e1 started at 1 + 1 confirm, e2 = 1
        assert merged.contradiction_count == 1

    async def test_merge_single_raises(self, mem):
        """Merging fewer than 2 experiences raises ValueError."""
        e1 = await mem.add("solo", tags=("t",))
        with pytest.raises(ValueError, match="at least 2"):
            await merge_experiences(mem, [e1.id], "nope")


# ======================================================================
# Split
# ======================================================================


@pytest.mark.asyncio
class TestSplit:
    async def test_split_into_two(self, mem):
        """Split one experience into two, preserving source and tags."""
        e = await mem.add("combined fact", tags=("mining",), source="bot")
        parts = await split_experience(mem, e.id, ["fact A", "fact B"])
        assert len(parts) == 2
        assert parts[0].source == "bot"
        assert parts[1].tags == ("mining",)
        # Original should be deleted
        assert await mem.get(e.id) is None
        assert await mem.count() == 2


# ======================================================================
# Rewrite
# ======================================================================


@pytest.mark.asyncio
class TestRewrite:
    async def test_rewrite_content(self, mem):
        """Rewrite preserves counts but updates content."""
        e = await mem.add("old content zzzz", tags=("t",))
        await mem.confirm(e.id)
        updated = await rewrite_experience(mem, e.id, "new content")
        assert updated.content == "new content"
        assert updated.confirmation_count == 2  # preserved from confirm


# ======================================================================
# Forget
# ======================================================================


@pytest.mark.asyncio
class TestForget:
    async def test_forget_deletes(self, mem):
        """Forget removes the experience."""
        e = await mem.add("to forget", tags=("t",))
        assert await mem.count() == 1
        await forget_experience(mem, e.id)
        assert await mem.count() == 0
