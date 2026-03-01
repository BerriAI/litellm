"""
Tests for OpenRouter cost propagation through streaming chunk builder.

OpenRouter returns a ``cost`` field on ``usage`` in streaming chunks.
The streaming chunk builder must extract and propagate this value to the
final ``Usage`` object so that callers receive accurate cost information.

Fixes: https://github.com/BerriAI/litellm/issues/16021
"""

from litellm.litellm_core_utils.streaming_chunk_builder_utils import (
    ChunkProcessor,
)
from litellm.types.utils import Usage


def _make_processor(chunks=None):
    """Create a ChunkProcessor with minimal valid chunks."""
    if chunks is None:
        chunks = [{"choices": [{"delta": {"content": ""}}]}]
    return ChunkProcessor(chunks=chunks)


class TestUsageChunkCostExtraction:
    """_usage_chunk_calculation_helper must extract the cost field."""

    def setup_method(self):
        self.processor = _make_processor()

    def test_extracts_cost_from_usage_chunk(self):
        usage = Usage(prompt_tokens=10, completion_tokens=20, cost=0.00123)
        result = self.processor._usage_chunk_calculation_helper(usage)
        assert result["cost"] == 0.00123

    def test_cost_none_when_absent(self):
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        result = self.processor._usage_chunk_calculation_helper(usage)
        assert result["cost"] is None

    def test_cost_zero_preserved(self):
        usage = Usage(prompt_tokens=0, completion_tokens=0, cost=0.0)
        result = self.processor._usage_chunk_calculation_helper(usage)
        assert result["cost"] == 0.0


class TestCalculateUsagePerChunkCost:
    """_calculate_usage_per_chunk must propagate cost."""

    def setup_method(self):
        self.processor = _make_processor()

    def test_cost_propagated_from_chunk(self):
        chunks = [
            {"usage": Usage(prompt_tokens=10, completion_tokens=5)},
            {"usage": Usage(prompt_tokens=10, completion_tokens=20, cost=0.05)},
        ]
        result = self.processor._calculate_usage_per_chunk(chunks)
        assert result["cost"] == 0.05

    def test_last_cost_wins(self):
        """When multiple chunks report cost, the last non-None value wins."""
        chunks = [
            {"usage": Usage(prompt_tokens=10, completion_tokens=5, cost=0.01)},
            {"usage": Usage(prompt_tokens=10, completion_tokens=20, cost=0.05)},
        ]
        result = self.processor._calculate_usage_per_chunk(chunks)
        assert result["cost"] == 0.05

    def test_cost_none_when_no_chunks_have_cost(self):
        chunks = [
            {"usage": Usage(prompt_tokens=10, completion_tokens=20)},
        ]
        result = self.processor._calculate_usage_per_chunk(chunks)
        assert result["cost"] is None


class TestCalculateUsageCost:
    """calculate_usage must set cost on the returned Usage object."""

    def setup_method(self):
        self.processor = _make_processor()

    def test_cost_set_on_final_usage(self):
        chunks = [
            {"usage": Usage(prompt_tokens=63, completion_tokens=255, cost=0.00123)},
        ]
        usage = self.processor.calculate_usage(
            chunks=chunks,
            model="openrouter/x-ai/grok-4-fast",
            completion_output="test output",
        )
        assert usage.cost == 0.00123
        assert usage.prompt_tokens == 63
        assert usage.completion_tokens == 255

    def test_cost_none_when_not_provided(self):
        chunks = [
            {"usage": Usage(prompt_tokens=10, completion_tokens=20)},
        ]
        usage = self.processor.calculate_usage(
            chunks=chunks,
            model="openrouter/x-ai/grok-4-fast",
            completion_output="test",
        )
        assert getattr(usage, "cost", None) is None

    def test_cost_zero_preserved_on_final_usage(self):
        chunks = [
            {"usage": Usage(prompt_tokens=10, completion_tokens=0, cost=0.0)},
        ]
        usage = self.processor.calculate_usage(
            chunks=chunks,
            model="openrouter/x-ai/grok-4-fast",
            completion_output="",
        )
        assert usage.cost == 0.0

    def test_cost_from_middle_chunk_propagated(self):
        """Cost may appear on any chunk (typically the last), not just first."""
        chunks = [
            {"usage": Usage(prompt_tokens=10, completion_tokens=5)},
            {"usage": Usage(prompt_tokens=10, completion_tokens=10, cost=0.042)},
            {"usage": Usage(prompt_tokens=10, completion_tokens=15)},
        ]
        usage = self.processor.calculate_usage(
            chunks=chunks,
            model="openrouter/anthropic/claude-sonnet-4-20250514",
            completion_output="multi chunk output",
        )
        assert usage.cost == 0.042

    def test_openrouter_realistic_streaming_scenario(self):
        """Simulate realistic OpenRouter streaming: only the final chunk has usage+cost."""
        intermediate_chunks = [
            {},  # content chunks without usage
            {},
            {},
        ]
        final_chunk = {
            "usage": Usage(
                prompt_tokens=63,
                completion_tokens=255,
                cost=0.00492,
            ),
        }
        all_chunks = intermediate_chunks + [final_chunk]
        usage = self.processor.calculate_usage(
            chunks=all_chunks,
            model="openrouter/x-ai/grok-4-fast",
            completion_output="Hello! How can I help you today?",
        )
        assert usage.cost == 0.00492
        assert usage.prompt_tokens == 63
        assert usage.completion_tokens == 255
        assert usage.total_tokens == 318
