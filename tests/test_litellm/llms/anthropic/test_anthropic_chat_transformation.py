"""
Unit tests for Anthropic chat transformation, including top-level automatic
prompt caching via the `cache_control` parameter.

Reference: https://platform.claude.com/docs/en/build-with-claude/prompt-caching#automatic-caching
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import pytest

from litellm.llms.anthropic.chat.transformation import AnthropicConfig


class TestAutomaticCachingSupport:
    """Tests for top-level cache_control (automatic caching) in Anthropic transformation."""

    def test_cache_control_in_supported_params(self):
        """cache_control must appear in get_supported_openai_params so LiteLLM routes it."""
        config = AnthropicConfig()
        supported = config.get_supported_openai_params("claude-opus-4-6")
        assert "cache_control" in supported

    def test_cache_control_in_supported_params_other_models(self):
        """cache_control should be listed as supported for all Anthropic models."""
        config = AnthropicConfig()
        for model in [
            "claude-3-5-haiku-20241022",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ]:
            supported = config.get_supported_openai_params(model)
            assert "cache_control" in supported, (
                f"cache_control missing from supported params for model {model}"
            )

    def test_map_openai_params_cache_control_ephemeral(self):
        """map_openai_params should pass cache_control through to optional_params."""
        config = AnthropicConfig()
        result = config.map_openai_params(
            non_default_params={"cache_control": {"type": "ephemeral"}},
            optional_params={},
            model="claude-opus-4-6",
            drop_params=False,
        )
        assert result.get("cache_control") == {"type": "ephemeral"}

    def test_map_openai_params_cache_control_with_ttl(self):
        """map_openai_params should support cache_control with 1-hour TTL."""
        config = AnthropicConfig()
        result = config.map_openai_params(
            non_default_params={"cache_control": {"type": "ephemeral", "ttl": "1h"}},
            optional_params={},
            model="claude-opus-4-6",
            drop_params=False,
        )
        assert result.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}

    def test_map_openai_params_cache_control_not_passed_when_absent(self):
        """cache_control should not appear in optional_params when not provided."""
        config = AnthropicConfig()
        result = config.map_openai_params(
            non_default_params={"max_tokens": 100},
            optional_params={},
            model="claude-opus-4-6",
            drop_params=False,
        )
        assert "cache_control" not in result

    def test_transform_request_includes_cache_control(self):
        """transform_request should include top-level cache_control in the final request body."""
        config = AnthropicConfig()
        messages = [{"role": "user", "content": "Hello"}]
        result = config.transform_request(
            model="claude-opus-4-6",
            messages=messages,
            optional_params={
                "max_tokens": 100,
                "cache_control": {"type": "ephemeral"},
            },
            litellm_params={},
            headers={},
        )
        assert "cache_control" in result
        assert result["cache_control"] == {"type": "ephemeral"}

    def test_transform_request_cache_control_with_ttl(self):
        """transform_request should pass cache_control with TTL unchanged."""
        config = AnthropicConfig()
        messages = [{"role": "user", "content": "Hello"}]
        result = config.transform_request(
            model="claude-opus-4-6",
            messages=messages,
            optional_params={
                "max_tokens": 100,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
            litellm_params={},
            headers={},
        )
        assert result.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}
