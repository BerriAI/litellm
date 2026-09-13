"""
Tests for _map_reasoning_effort in AnthropicConfig.

Verifies that reasoning_effort=None returns None for all models,
including Claude Opus 4.6.
"""

from unittest.mock import patch

import pytest

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


class TestMapReasoningEffort:
    def test_none_returns_none_for_opus_4_6(self):
        """reasoning_effort=None should return None for Opus 4.6, not adaptive."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort=None, model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result is None

    def test_none_returns_none_for_other_models(self):
        """reasoning_effort=None should return None for non-Opus models."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort=None, model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
        )
        assert result is None

    def test_opus_4_6_returns_adaptive_for_low(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="low", model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result["type"] == "adaptive"

    def test_opus_4_6_returns_adaptive_for_high(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="high", model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result["type"] == "adaptive"

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_adaptive_mapping_requests_summarized_display(self, effort):
        """Regression LIT-5714: adaptive thinking without ``display`` makes Anthropic
        return a blank thinking block, so reasoning_effort callers always got
        ``reasoning_content: ""``."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort=effort, model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result["display"] == "summarized"

    def test_other_model_low_returns_enabled_with_budget(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="low", model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    def test_other_model_high_returns_enabled_with_budget(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="high", model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    def test_none_string_returns_none_for_opus_4_6(self):
        """reasoning_effort='none' should return None for Opus 4.6."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="none", model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result is None

    def test_none_string_returns_none_for_other_models(self):
        """reasoning_effort='none' should return None for non-Opus models."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="none", model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
        )
        assert result is None


def _mock_model_info(**flags):
    return flags


class TestMapReasoningEffortDegradation:
    def test_max_stays_max_when_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
                supports_reasoning=True,
                supports_max_reasoning_effort=True,
                supports_xhigh_reasoning_effort=True,
            ),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="max",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert result["type"] == "enabled"
            assert result["budget_tokens"] == DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET

    def test_max_degrades_to_xhigh_when_only_xhigh_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
                supports_reasoning=True,
                supports_max_reasoning_effort=False,
                supports_xhigh_reasoning_effort=True,
            ),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="max",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert result["budget_tokens"] == DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET

    def test_max_degrades_to_high_when_neither_max_nor_xhigh_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
                supports_reasoning=True,
                supports_max_reasoning_effort=False,
                supports_xhigh_reasoning_effort=False,
            ),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="max",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert result["budget_tokens"] == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET

    def test_max_passthrough_for_unknown_model(self):
        with patch(
            "litellm.utils.get_model_info",
            side_effect=Exception("model not found"),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="max",
                model="unknown-glm-4.6",
                custom_llm_provider="anthropic",
            )
            assert result["budget_tokens"] == DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET

    def test_xhigh_stays_xhigh_when_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
                supports_reasoning=True,
                supports_xhigh_reasoning_effort=True,
            ),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="xhigh",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert result["budget_tokens"] == DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET

    def test_xhigh_degrades_to_high_when_unsupported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
                supports_reasoning=True,
                supports_xhigh_reasoning_effort=False,
            ),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="xhigh",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert result["budget_tokens"] == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET

    def test_xhigh_passthrough_for_unknown_model(self):
        with patch(
            "litellm.utils.get_model_info",
            side_effect=Exception("model not found"),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="xhigh",
                model="unknown-deepseek",
                custom_llm_provider="anthropic",
            )
            assert result["budget_tokens"] == DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET

    def test_minimal_stays_minimal_when_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
                supports_reasoning=True,
                supports_minimal_reasoning_effort=True,
            ),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="minimal",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert result["budget_tokens"] == max(DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET, 1024)

    def test_minimal_degrades_to_low_when_unsupported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
                supports_reasoning=True,
                supports_minimal_reasoning_effort=False,
            ),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="minimal",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert result["budget_tokens"] == DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET

    def test_high_unchanged(self):
        with patch(
            "litellm.utils.get_model_info",
            side_effect=Exception("model not found"),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="high",
                model="unknown-model",
                custom_llm_provider="anthropic",
            )
            assert result["budget_tokens"] == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET

    def test_medium_unchanged(self):
        with patch(
            "litellm.utils.get_model_info",
            side_effect=Exception("model not found"),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="medium",
                model="unknown-model",
                custom_llm_provider="anthropic",
            )
            assert result["budget_tokens"] == DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET

    def test_low_unchanged(self):
        with patch(
            "litellm.utils.get_model_info",
            side_effect=Exception("model not found"),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="low",
                model="unknown-model",
                custom_llm_provider="anthropic",
            )
            assert result["budget_tokens"] == DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET

    def test_none_returns_none(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="none",
            model="any-model",
            custom_llm_provider="anthropic",
        )
        assert result is None

    def test_none_value_returns_none(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort=None,
            model="any-model",
            custom_llm_provider="anthropic",
        )
        assert result is None

    def test_adaptive_model_short_circuits_before_degradation(self):
        with patch(
            "litellm.llms.anthropic.chat.transformation.AnthropicConfig._is_adaptive_thinking_model",
            return_value=True,
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="max",
                model="claude-opus-4-6",
                custom_llm_provider="anthropic",
            )
            assert result["type"] == "adaptive"


class TestApplyOutputConfigDegradation:
    def test_max_degrades_to_high_when_unsupported(self):
        with (
            patch(
                "litellm.llms.anthropic.chat.transformation.AnthropicConfig._validate_effort_for_model",
                return_value="effort='max' is not supported by this model. Got model: test",
            ),
            patch(
                "litellm.llms.anthropic.chat.transformation.AnthropicConfig._is_adaptive_thinking_model",
                return_value=True,
            ),
            patch(
                "litellm.utils.get_model_info",
                return_value=_mock_model_info(
                    supports_reasoning=True,
                    supports_max_reasoning_effort=False,
                    supports_xhigh_reasoning_effort=False,
                ),
            ),
        ):
            cfg = AnthropicConfig()
            data: dict = {}
            optional_params = {"output_config": {"effort": "max"}}
            cfg._apply_output_config(data, "test-model", optional_params)
            assert data["output_config"]["effort"] == "high"

    def test_xhigh_degrades_to_high_when_unsupported(self):
        with (
            patch(
                "litellm.llms.anthropic.chat.transformation.AnthropicConfig._validate_effort_for_model",
                return_value="effort='xhigh' is not supported by this model. Got model: test",
            ),
            patch(
                "litellm.llms.anthropic.chat.transformation.AnthropicConfig._is_adaptive_thinking_model",
                return_value=True,
            ),
            patch(
                "litellm.utils.get_model_info",
                return_value=_mock_model_info(
                    supports_reasoning=True,
                    supports_xhigh_reasoning_effort=False,
                ),
            ),
        ):
            cfg = AnthropicConfig()
            data: dict = {}
            optional_params = {"output_config": {"effort": "xhigh"}}
            cfg._apply_output_config(data, "test-model", optional_params)
            assert data["output_config"]["effort"] == "high"

    def test_max_stays_max_when_supported(self):
        with patch(
            "litellm.llms.anthropic.chat.transformation.AnthropicConfig._validate_effort_for_model",
            return_value=None,
        ):
            cfg = AnthropicConfig()
            data: dict = {}
            optional_params = {"output_config": {"effort": "max"}}
            cfg._apply_output_config(data, "test-model", optional_params)
            assert data["output_config"]["effort"] == "max"

    def test_no_output_config_is_noop(self):
        cfg = AnthropicConfig()
        data: dict = {}
        cfg._apply_output_config(data, "test-model", {})
        assert "output_config" not in data

    def test_invalid_effort_value_still_raises(self):
        with patch(
            "litellm.llms.anthropic.chat.transformation.AnthropicConfig._is_adaptive_thinking_model",
            return_value=True,
        ):
            cfg = AnthropicConfig()
            with pytest.raises(Exception):
                cfg._apply_output_config({}, "test-model", {"output_config": {"effort": "bogus"}})
