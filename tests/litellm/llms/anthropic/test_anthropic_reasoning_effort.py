"""
Tests for _map_reasoning_effort and _map_reasoning_effort_to_anthropic_effort
in AnthropicConfig.

Verifies that reasoning_effort=None returns None for all models,
including Claude Opus 4.6, and that effort levels are correctly mapped
to Anthropic output_config effort values.
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


class TestMapReasoningEffortToAnthropicEffort:
    """Tests for _map_reasoning_effort_to_anthropic_effort (output_config mapping)."""

    def test_low_maps_to_low(self):
        result = AnthropicConfig._map_reasoning_effort_to_anthropic_effort(
            "low", "claude-sonnet-4-6"
        )
        assert result == "low"

    def test_minimal_maps_to_low(self):
        result = AnthropicConfig._map_reasoning_effort_to_anthropic_effort(
            "minimal", "claude-sonnet-4-6"
        )
        assert result == "low"

    def test_medium_maps_to_medium(self):
        result = AnthropicConfig._map_reasoning_effort_to_anthropic_effort(
            "medium", "claude-sonnet-4-6"
        )
        assert result == "medium"

    def test_high_maps_to_high(self):
        result = AnthropicConfig._map_reasoning_effort_to_anthropic_effort(
            "high", "claude-sonnet-4-6"
        )
        assert result == "high"

    def test_none_returns_none(self):
        result = AnthropicConfig._map_reasoning_effort_to_anthropic_effort(
            "none", "claude-sonnet-4-6"
        )
        assert result is None

    def test_xhigh_maps_to_max_for_opus_4_6(self):
        result = AnthropicConfig._map_reasoning_effort_to_anthropic_effort(
            "xhigh", "claude-opus-4-6"
        )
        assert result == "max"

    def test_xhigh_maps_to_high_for_sonnet_4_6(self):
        """Sonnet 4.6 does not support max effort, so xhigh maps to high."""
        result = AnthropicConfig._map_reasoning_effort_to_anthropic_effort(
            "xhigh", "claude-sonnet-4-6"
        )
        assert result == "high"


class TestIsEffortSupportedModel:
    """Tests for _is_effort_supported_model."""

    def test_opus_4_6(self):
        assert AnthropicConfig._is_effort_supported_model("claude-opus-4-6") is True

    def test_sonnet_4_6(self):
        assert AnthropicConfig._is_effort_supported_model("claude-sonnet-4-6") is True

    def test_opus_4_5(self):
        assert (
            AnthropicConfig._is_effort_supported_model("claude-opus-4-5-20251101")
            is True
        )

    def test_sonnet_4_5_not_supported(self):
        assert (
            AnthropicConfig._is_effort_supported_model("claude-sonnet-4-5-20250929")
            is False
        )

    def test_sonnet_3_7_not_supported(self):
        assert AnthropicConfig._is_effort_supported_model("claude-3-7-sonnet") is False
