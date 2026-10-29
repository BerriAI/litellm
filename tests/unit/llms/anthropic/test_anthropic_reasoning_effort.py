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
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort=None, model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result is None

    def test_none_returns_none_for_other_models(self):
        """reasoning_effort=None should return None for non-Opus models."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort=None, model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
        )
        assert result is None

    def test_opus_4_6_returns_adaptive_for_low(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="low", model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result["type"] == "adaptive"

    def test_opus_4_6_returns_adaptive_for_high(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="high", model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result["type"] == "adaptive"

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_adaptive_mapping_requests_summarized_display(self, effort):
        """Regression LIT-5714: adaptive thinking without ``display`` makes Anthropic
        return a blank thinking block, so reasoning_effort callers always got
        ``reasoning_content: ""``."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort=effort, model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result["display"] == "summarized"

    def test_other_model_low_returns_enabled_with_budget(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="low", model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    def test_other_model_high_returns_enabled_with_budget(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="high", model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    def test_none_string_returns_none_for_opus_4_6(self):
        """reasoning_effort='none' should return None for Opus 4.6."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="none", model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result is None

    def test_none_string_returns_none_for_other_models(self):
        """reasoning_effort='none' should return None for non-Opus models."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="none", model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
        )
        assert result is None

    def test_none_string_returns_disabled_for_haiku_5_5(self):
        """reasoning_effort='none' must explicitly disable thinking on Claude
        models that default thinking on (Haiku 5.5); omission leaves the
        model's default in effect."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="none", model="claude-haiku-5-5", custom_llm_provider="anthropic"
        )
        assert result == {"type": "disabled"}

    def test_none_string_returns_disabled_for_opus_5(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="none", model="claude-opus-5", custom_llm_provider="anthropic"
        )
        assert result == {"type": "disabled"}

    def test_none_string_returns_disabled_for_always_on_model(self):
        """Always-on models map to 'disabled' too; transform_request drops it
        via maybe_drop_disabled_thinking (the API-documented remedy)."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="none", model="claude-sonnet-5-5", custom_llm_provider="anthropic"
        )
        assert result == {"type": "disabled"}

    def test_none_string_map_openai_params_sets_thinking_without_output_config(self):
        params = AnthropicConfig().map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="claude-haiku-5-5",
            drop_params=False,
        )
        assert params.get("thinking") == {"type": "disabled"}
        assert "output_config" not in params

    def test_disabled_thinking_dropped_for_always_on_model(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        optional_params = {"thinking": {"type": "disabled"}}
        AnthropicModelInfo.maybe_drop_disabled_thinking(
            model="claude-sonnet-5-5",
            optional_params=optional_params,
            custom_llm_provider="anthropic",
        )
        assert "thinking" not in optional_params

    def test_disabled_thinking_kept_for_haiku_5_5(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        optional_params = {"thinking": {"type": "disabled"}}
        AnthropicModelInfo.maybe_drop_disabled_thinking(
            model="claude-haiku-5-5",
            optional_params=optional_params,
            custom_llm_provider="anthropic",
        )
        assert optional_params["thinking"] == {"type": "disabled"}
