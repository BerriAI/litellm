"""
Tests for _map_reasoning_effort in AnthropicConfig.

Verifies that reasoning_effort=None returns None for all models,
including Claude Opus 4.6.
"""

from litellm.llms.anthropic.chat.transformation import AnthropicConfig


class TestMapReasoningEffort:
    def test_none_returns_none_for_opus_4_6(self):
        """reasoning_effort=None should return None for Opus 4.6, not adaptive."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort=None, model="claude-opus-4-6"
        )
        assert result is None

    def test_none_returns_none_for_other_models(self):
        """reasoning_effort=None should return None for non-Opus models."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort=None, model="claude-3-7-sonnet-20250219"
        )
        assert result is None

    def test_opus_4_6_returns_adaptive_for_low(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="low", model="claude-opus-4-6"
        )
        assert result["type"] == "adaptive"

    def test_opus_4_6_returns_adaptive_for_high(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="high", model="claude-opus-4-6"
        )
        assert result["type"] == "adaptive"

    def test_other_model_low_returns_enabled_with_budget(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="low", model="claude-3-7-sonnet-20250219"
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    def test_other_model_high_returns_enabled_with_budget(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="high", model="claude-3-7-sonnet-20250219"
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    def test_none_string_returns_none_for_opus_4_6(self):
        """reasoning_effort='none' should return None for Opus 4.6."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="none", model="claude-opus-4-6"
        )
        assert result is None

    def test_none_string_returns_none_for_other_models(self):
        """reasoning_effort='none' should return None for non-Opus models."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="none", model="claude-3-7-sonnet-20250219"
        )
        assert result is None
