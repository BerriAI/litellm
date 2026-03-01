"""
Tests for _map_reasoning_effort in AnthropicConfig.

Verifies that reasoning_effort=None returns None for all models,
including Claude Opus 4.6.
"""

from litellm.llms.anthropic.chat.transformation import AnthropicConfig


class TestMapReasoningEffortToOutputConfig:
    def test_xhigh_maps_to_high(self):
        """xhigh reasoning_effort should map to high in output_config."""
        result = AnthropicConfig._map_reasoning_effort_to_output_config("xhigh")
        assert result == "high"

    def test_max_maps_to_max(self):
        """max reasoning_effort should map to max in output_config."""
        result = AnthropicConfig._map_reasoning_effort_to_output_config("max")
        assert result == "max"

    def test_unknown_value_returns_none(self):
        """Unknown reasoning_effort values should return None."""
        result = AnthropicConfig._map_reasoning_effort_to_output_config("unknown")
        assert result is None

    def test_low_maps_to_low(self):
        """low reasoning_effort should map to low in output_config."""
        result = AnthropicConfig._map_reasoning_effort_to_output_config("low")
        assert result == "low"

    def test_minimal_maps_to_low(self):
        """minimal reasoning_effort should map to low in output_config."""
        result = AnthropicConfig._map_reasoning_effort_to_output_config("minimal")
        assert result == "low"

    def test_medium_maps_to_medium(self):
        """medium reasoning_effort should map to medium in output_config."""
        result = AnthropicConfig._map_reasoning_effort_to_output_config("medium")
        assert result == "medium"

    def test_high_maps_to_high(self):
        """high reasoning_effort should map to high in output_config."""
        result = AnthropicConfig._map_reasoning_effort_to_output_config("high")
        assert result == "high"


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
            reasoning_effort=None, model="claude-4-sonnet-20250514"
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
            reasoning_effort="low", model="claude-4-sonnet-20250514"
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    def test_other_model_high_returns_enabled_with_budget(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="high", model="claude-4-sonnet-20250514"
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
            reasoning_effort="none", model="claude-4-sonnet-20250514"
        )
        assert result is None
