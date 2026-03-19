"""Tests for the embedding module and cosine similarity."""

import pytest

from ganglion.memory.embed import (
    CallableEmbedder,
    cosine_similarity,
    reset_embedder,
    set_embedder,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 1.0, 0.1]
        assert cosine_similarity(a, b) > 0.95

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


@pytest.mark.asyncio
class TestCallableEmbedder:
    async def test_wraps_async_callable(self):
        async def mock_embed(text: str) -> list[float]:
            return [float(len(text)), 0.5, 0.1]

        embedder = CallableEmbedder(mock_embed)
        result = await embedder.embed("hello")
        assert len(result) == 3
        assert result[0] == 5.0

    async def test_batch_embed(self):
        async def mock_embed(text: str) -> list[float]:
            return [float(len(text))]

        embedder = CallableEmbedder(mock_embed)
        results = await embedder.embed_batch(["a", "ab", "abc"])
        assert len(results) == 3
        assert results[0] == [1.0]
        assert results[2] == [3.0]


class TestGlobalEmbedder:
    def test_set_and_reset(self):
        set_embedder(None)
        reset_embedder()
        # Should not raise
