"""Tests for the one-line wrapper API."""

import tempfile
from pathlib import Path

import pytest

from ganglion.memory.wrap import memory, _default_judge


@pytest.fixture(autouse=True)
def reset_default_memory():
    """Reset the module-level singleton between tests."""
    import ganglion.memory.wrap as mod
    mod._default_memory = None
    yield
    mod._default_memory = None


class TestMemoryWrap:
    def test_wraps_sync_function(self, tmp_dir):
        def agent(prompt: str) -> str:
            return f"response to: {prompt}"

        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        result = wrapped("hello")
        assert "response to: hello" in result
        assert wrapped.__name__ == "agent"

    def test_wraps_async_function(self, tmp_dir):
        import asyncio

        async def agent(prompt: str) -> str:
            return f"async response to: {prompt}"

        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        result = asyncio.run(wrapped("hello"))
        assert "async response to: hello" in result

    def test_injects_into_openai_messages(self, tmp_dir):
        def agent(messages=None):
            return messages[0]["content"]

        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))

        # First call — empty memory, no injection
        result = wrapped(messages=[
            {"role": "system", "content": "base prompt"},
            {"role": "user", "content": "hi"},
        ])
        assert "base prompt" in result

    def test_injects_into_anthropic_system(self, tmp_dir):
        def agent(system="", messages=None):
            return system

        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        result = wrapped(system="base prompt", messages=[])
        assert "base prompt" in result

    def test_injects_into_string_arg(self, tmp_dir):
        def agent(prompt: str) -> str:
            return prompt

        wrapped = memory(agent, capability="test", db_path=str(tmp_dir / "m.db"))
        result = wrapped("base prompt")
        assert "base prompt" in result

    def test_works_as_decorator(self, tmp_dir):
        db = str(tmp_dir / "m.db")

        @memory
        def agent(prompt: str) -> str:
            return f"decorated: {prompt}"

        result = agent("test")
        assert "decorated: test" in result

    def test_custom_judge(self, tmp_dir):
        def agent(prompt: str) -> dict:
            return {"score": 0.95, "text": "good result"}

        def judge(response):
            return {
                "success": response["score"] > 0.5,
                "description": response["text"],
                "metric_name": "score",
                "metric_value": response["score"],
            }

        wrapped = memory(agent, capability="test", judge=judge, db_path=str(tmp_dir / "m.db"))
        result = wrapped("test")
        assert result["score"] == 0.95

    def test_preserves_function_name(self, tmp_dir):
        def my_special_agent(x):
            return x

        wrapped = memory(my_special_agent, capability="test", db_path=str(tmp_dir / "m.db"))
        assert wrapped.__name__ == "my_special_agent"


class TestDefaultJudge:
    def test_string_response(self):
        result = _default_judge("hello world")
        assert result["success"] is True
        assert result["description"] == "hello world"

    def test_dict_passthrough(self):
        result = _default_judge({"success": False, "description": "failed"})
        assert result["success"] is False

    def test_truncates_long_strings(self):
        result = _default_judge("x" * 1000)
        assert len(result["description"]) <= 500


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
