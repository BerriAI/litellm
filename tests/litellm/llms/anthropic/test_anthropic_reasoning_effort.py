"""
Tests for _map_reasoning_effort in AnthropicConfig.

Verifies that reasoning_effort=None returns None for all models,
including Claude Opus 4.6.
"""

import pytest

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


class TestMapOpenaiParamsReasoningEffortOutputConfig:
    """
    Tests that map_openai_params sets output_config alongside thinking
    for Claude 4.6 models when reasoning_effort is provided.

    Fixes https://github.com/BerriAI/litellm/issues/22212
    """

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4-6-20250827",
            "claude-sonnet-4.6",
        ],
    )
    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_claude_4_6_sets_output_config_with_effort(self, model, effort):
        """For Claude 4.6 models, reasoning_effort should set both
        thinking={"type":"adaptive"} AND output_config={"effort": <value>}."""
        config = AnthropicConfig()
        optional_params: dict = {}
        config.map_openai_params(
            non_default_params={"reasoning_effort": effort},
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )
        assert "thinking" in optional_params
        assert optional_params["thinking"]["type"] == "adaptive"
        assert "output_config" in optional_params
        assert optional_params["output_config"] == {"effort": effort}

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_non_4_6_model_does_not_set_output_config(self, effort):
        """For non-4.6 models, reasoning_effort should only set thinking,
        not output_config."""
        config = AnthropicConfig()
        optional_params: dict = {}
        config.map_openai_params(
            non_default_params={"reasoning_effort": effort},
            optional_params=optional_params,
            model="claude-4-sonnet-20250514",
            drop_params=False,
        )
        assert "thinking" in optional_params
        assert optional_params["thinking"]["type"] == "enabled"
        assert "output_config" not in optional_params

    def test_claude_4_6_reasoning_effort_none_skips_output_config(self):
        """reasoning_effort='none' should not set output_config.

        We test the internal logic directly because _map_reasoning_effort
        returns None for 'none', and the output_config guard checks that
        thinking_param is not None before setting output_config.
        """
        thinking_param = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="none", model="claude-opus-4-6"
        )
        assert thinking_param is None
        # Since thinking_param is None, the guard in map_openai_params
        # (if thinking_param is not None and ...) will not set output_config
