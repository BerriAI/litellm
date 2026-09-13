"""
Tests for issue #40890: /v1/messages passthrough drops adaptive thinking.

When using the native passthrough config (OpenAILikeAnthropicMessagesConfig),
adaptive thinking fields (thinking: {"type": "adaptive"}, output_config.effort)
must be forwarded unchanged, not dropped by model-map translations.
"""
import pytest
from unittest.mock import MagicMock

from litellm.llms.openai_like.messages.transformation import (
    OpenAILikeAnthropicMessagesConfig,
)


class TestOpenAILikeAnthropicMessagesTransform:
    """Test that native passthrough preserves adaptive thinking fields."""

    def _make_config(self, cache_control_ttl=False):
        return OpenAILikeAnthropicMessagesConfig(cache_control_ttl=cache_control_ttl)

    def _default_params(self, **overrides):
        base = {
            "max_tokens": 1024,
        }
        base.update(overrides)
        return base

    def test_adaptive_thinking_preserved(self):
        """Adaptive thinking config must be forwarded unchanged."""
        config = self._make_config()
        params = self._default_params(
            thinking={"type": "adaptive"},
        )
        messages = [{"role": "user", "content": "Hello"}]
        litellm_params = {"model": "claude-opus-4-6"}

        request = config.transform_anthropic_messages_request(
            model="claude-opus-4-6",
            messages=messages,
            anthropic_messages_optional_request_params=params,
            litellm_params=litellm_params,
            headers={},
        )

        # Adaptive thinking must be preserved in the output
        assert request.get("thinking") == {"type": "adaptive"}, (
            f"thinking was dropped or modified: {request.get('thinking')}"
        )

    def test_output_config_effort_preserved(self):
        """output_config.effort must be forwarded unchanged."""
        config = self._make_config()
        params = self._default_params(
            output_config={"effort": "high"},
        )
        messages = [{"role": "user", "content": "Hello"}]
        litellm_params = {"model": "claude-opus-4-6"}

        request = config.transform_anthropic_messages_request(
            model="claude-opus-4-6",
            messages=messages,
            anthropic_messages_optional_request_params=params,
            litellm_params=litellm_params,
            headers={},
        )

        # output_config with effort must be preserved
        assert request.get("output_config") == {"effort": "high"}, (
            f"output_config was dropped or modified: {request.get('output_config')}"
        )

    def test_budget_tokens_preserved(self):
        """thinking.budget_tokens must be forwarded unchanged."""
        config = self._make_config()
        params = self._default_params(
            thinking={"type": "enabled", "budget_tokens": 5000},
        )
        messages = [{"role": "user", "content": "Hello"}]
        litellm_params = {"model": "claude-opus-4-6"}

        request = config.transform_anthropic_messages_request(
            model="claude-opus-4-6",
            messages=messages,
            anthropic_messages_optional_request_params=params,
            litellm_params=litellm_params,
            headers={},
        )

        # thinking config must be preserved
        assert request.get("thinking") == {"type": "enabled", "budget_tokens": 5000}, (
            f"thinking was modified: {request.get('thinking')}"
        )

    def test_standard_params_preserved(self):
        """Standard params like temperature, top_p must be forwarded."""
        config = self._make_config()
        params = self._default_params(
            temperature=0.7,
            top_p=0.9,
        )
        messages = [{"role": "user", "content": "Hello"}]
        litellm_params = {"model": "claude-opus-4-6"}

        request = config.transform_anthropic_messages_request(
            model="claude-opus-4-6",
            messages=messages,
            anthropic_messages_optional_request_params=params,
            litellm_params=litellm_params,
            headers={},
        )

        assert request.get("temperature") == 0.7
        assert request.get("top_p") == 0.9

    def test_max_tokens_required(self):
        """Request must fail if max_tokens is missing."""
        config = self._make_config()
        params = {}  # No max_tokens
        messages = [{"role": "user", "content": "Hello"}]
        litellm_params = {"model": "claude-opus-4-6"}

        with pytest.raises(Exception, match="max_tokens is required"):
            config.transform_anthropic_messages_request(
                model="claude-opus-4-6",
                messages=messages,
                anthropic_messages_optional_request_params=params,
                litellm_params=litellm_params,
                headers={},
            )

    def test_cache_control_normalization_still_applied(self):
        """Cache control normalization should still be applied when TTL not supported."""
        config = self._make_config(cache_control_ttl=False)
        params = self._default_params(
            system=[
                {
                    "type": "text",
                    "text": "You are a helpful assistant",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        )
        messages = [{"role": "user", "content": "Hello"}]
        litellm_params = {"model": "claude-opus-4-6"}

        request = config.transform_anthropic_messages_request(
            model="claude-opus-4-6",
            messages=messages,
            anthropic_messages_optional_request_params=params,
            litellm_params=litellm_params,
            headers={},
        )

        # TTL should be stripped (non-Anthropic backend)
        system = request.get("system", [])
        if system and isinstance(system, list):
            cache_control = system[0].get("cache_control", {})
            assert "ttl" not in cache_control, (
                f"cache_control.ttl should be stripped: {cache_control}"
            )

    def test_cache_control_ttl_preserved_when_supported(self):
        """Cache control TTL should be preserved when cache_control_ttl=True."""
        config = self._make_config(cache_control_ttl=True)
        params = self._default_params(
            system=[
                {
                    "type": "text",
                    "text": "You are a helpful assistant",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        )
        messages = [{"role": "user", "content": "Hello"}]
        litellm_params = {"model": "claude-opus-4-6"}

        request = config.transform_anthropic_messages_request(
            model="claude-opus-4-6",
            messages=messages,
            anthropic_messages_optional_request_params=params,
            litellm_params=litellm_params,
            headers={},
        )

        # TTL should be preserved
        system = request.get("system", [])
        if system and isinstance(system, list):
            cache_control = system[0].get("cache_control", {})
            assert cache_control.get("ttl") == "1h", (
                f"cache_control.ttl should be preserved: {cache_control}"
            )
