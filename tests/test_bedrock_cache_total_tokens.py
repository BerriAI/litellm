"""
Unit tests for the Bedrock _transform_usage total_tokens fix.

Verifies that total_tokens is recalculated AFTER cache-token inflation so that
    total_tokens == prompt_tokens + completion_tokens
always holds, regardless of whether cacheReadInputTokens or cacheWriteInputTokens
are present in the Bedrock usage payload.
"""
import pytest
from unittest.mock import MagicMock
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig


def _make_transform():
    """Return a bound _transform_usage method with minimal mock dependencies."""
    config = AmazonConverseConfig()
    return config._transform_usage


class TestBedrockTransformUsageTotalTokens:
    """total_tokens must equal prompt_tokens + completion_tokens after cache-token inflation."""

    def test_no_cache_tokens_baseline(self):
        """Without cache tokens, total_tokens should equal inputTokens + outputTokens."""
        transform = _make_transform()
        usage = transform({
            "inputTokens": 100,
            "outputTokens": 50,
            "totalTokens": 150,
        })
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_cache_read_tokens_total_includes_cache(self):
        """Cache-read tokens added to input_tokens must be reflected in total_tokens.

        Before the fix, total_tokens was 150 (from Bedrock's totalTokens which excluded
        cache tokens). After the fix it must be 180 = 130 prompt + 50 completion.
        """
        transform = _make_transform()
        usage = transform({
            "inputTokens": 100,
            "cacheReadInputTokens": 30,
            "outputTokens": 50,
            "totalTokens": 150,  # Bedrock's native value — intentionally stale
        })
        assert usage.prompt_tokens == 130           # 100 + 30 cache-read
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 180            # was 150 before the fix
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_cache_write_tokens_total_includes_cache(self):
        """Cache-write tokens added to input_tokens must be reflected in total_tokens."""
        transform = _make_transform()
        usage = transform({
            "inputTokens": 100,
            "cacheWriteInputTokens": 20,
            "outputTokens": 50,
            "totalTokens": 150,
        })
        assert usage.prompt_tokens == 120           # 100 + 20 cache-write
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 170            # was 150 before the fix
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_both_cache_read_and_write_tokens(self):
        """Both cache-read and cache-write tokens must both roll up into total_tokens."""
        transform = _make_transform()
        usage = transform({
            "inputTokens": 100,
            "cacheReadInputTokens": 30,
            "cacheWriteInputTokens": 20,
            "outputTokens": 50,
            "totalTokens": 150,
        })
        assert usage.prompt_tokens == 150           # 100 + 30 + 20
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 200            # was 150 before the fix
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_prompt_tokens_details_preserved(self):
        """Cache fields must be accessible via prompt_tokens_details as well."""
        transform = _make_transform()
        usage = transform({
            "inputTokens": 100,
            "cacheReadInputTokens": 30,
            "cacheWriteInputTokens": 20,
            "outputTokens": 50,
            "totalTokens": 150,
        })
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 30

    def test_zero_tokens_edge_case(self):
        """Zero-token payload must not raise and must satisfy the invariant."""
        transform = _make_transform()
        usage = transform({
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0,
        })
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
