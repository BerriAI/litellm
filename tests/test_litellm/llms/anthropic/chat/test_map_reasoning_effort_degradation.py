"""
Unit tests for reasoning effort degradation in AnthropicConfig.

Covers:
- Patch 1: _map_reasoning_effort normalizes unsupported effort levels via
  normalize_reasoning_effort_value (max->xhigh->high, xhigh->high, minimal->low).
- Patch 2a: _apply_output_config degrades unsupported effort instead of raising
  BadRequestError.
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


def _mock_model_info(**flags):
    """Return a mock model_info dict with given capability flags."""
    return flags


class TestMapReasoningEffortDegradation:
    """Patch 1: _map_reasoning_effort degrades unsupported effort levels."""

    # --- max degradation chain ---

    def test_max_stays_max_when_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
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
            assert (
                result["budget_tokens"] == DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET
            )

    def test_max_degrades_to_xhigh_when_only_xhigh_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
                supports_max_reasoning_effort=False,
                supports_xhigh_reasoning_effort=True,
            ),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="max",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert (
                result["budget_tokens"]
                == DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET
            )

    def test_max_degrades_to_high_when_neither_max_nor_xhigh_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
                supports_max_reasoning_effort=False,
                supports_xhigh_reasoning_effort=False,
            ),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="max",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert (
                result["budget_tokens"] == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
            )

    def test_max_passthrough_for_unknown_model(self):
        """Unknown model (get_model_info raises) -> max passes through unchanged
        (passthrough-on-unknown semantics)."""
        with patch(
            "litellm.utils.get_model_info",
            side_effect=Exception("model not found"),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="max",
                model="unknown-glm-4.6",
                custom_llm_provider="anthropic",
            )
            assert (
                result["budget_tokens"] == DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET
            )

    # --- xhigh degradation chain ---

    def test_xhigh_stays_xhigh_when_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(supports_xhigh_reasoning_effort=True),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="xhigh",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert (
                result["budget_tokens"]
                == DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET
            )

    def test_xhigh_degrades_to_high_when_unsupported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(supports_xhigh_reasoning_effort=False),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="xhigh",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert (
                result["budget_tokens"] == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
            )

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
            assert (
                result["budget_tokens"]
                == DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET
            )

    # --- minimal degradation chain ---

    def test_minimal_stays_minimal_when_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(supports_minimal_reasoning_effort=True),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="minimal",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert result["budget_tokens"] == max(
                DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET, 1024
            )

    def test_minimal_degrades_to_low_when_unsupported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(supports_minimal_reasoning_effort=False),
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="minimal",
                model="test-model",
                custom_llm_provider="anthropic",
            )
            assert (
                result["budget_tokens"] == DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET
            )

    # --- passthrough values (no degradation) ---

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
            assert (
                result["budget_tokens"] == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
            )

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
            assert (
                result["budget_tokens"]
                == DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET
            )

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
            assert (
                result["budget_tokens"] == DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET
            )

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

    # --- adaptive model short-circuit ---

    def test_adaptive_model_short_circuits_before_degradation(self):
        """Adaptive thinking models return {type: adaptive} and skip degradation."""
        with patch(
            "litellm.llms.anthropic.chat.transformation.AnthropicConfig._is_adaptive_thinking_model",
            return_value=True,
        ):
            result = AnthropicConfig._map_reasoning_effort(
                reasoning_effort="max",
                model="claude-opus-4-6",
                custom_llm_provider="anthropic",
            )
            assert result == {"type": "adaptive"}


class TestApplyOutputConfigDegradation:
    """Patch 2a: _apply_output_config degrades instead of raising for
    unsupported effort levels on adaptive-thinking models."""

    def _build_config(self):
        cfg = AnthropicConfig()
        return cfg

    def test_max_degrades_to_high_when_unsupported(self):
        """output_config.effort=max should degrade to high, not raise."""
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
                    supports_max_reasoning_effort=False,
                    supports_xhigh_reasoning_effort=False,
                ),
            ),
        ):
            cfg = self._build_config()
            data = {}
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
                return_value=_mock_model_info(supports_xhigh_reasoning_effort=False),
            ),
        ):
            cfg = self._build_config()
            data = {}
            optional_params = {"output_config": {"effort": "xhigh"}}
            cfg._apply_output_config(data, "test-model", optional_params)
            assert data["output_config"]["effort"] == "high"

    def test_max_stays_max_when_supported(self):
        """When _validate_effort_for_model passes (no gate_error), effort is unchanged."""
        with patch(
            "litellm.llms.anthropic.chat.transformation.AnthropicConfig._validate_effort_for_model",
            return_value=None,
        ):
            cfg = self._build_config()
            data = {}
            optional_params = {"output_config": {"effort": "max"}}
            cfg._apply_output_config(data, "test-model", optional_params)
            assert data["output_config"]["effort"] == "max"

    def test_no_output_config_is_noop(self):
        cfg = self._build_config()
        data = {}
        optional_params = {}
        cfg._apply_output_config(data, "test-model", optional_params)
        assert "output_config" not in data

    def test_invalid_effort_value_still_raises(self):
        """Genuinely invalid effort values (not in valid_efforts) still raise."""
        with patch(
            "litellm.llms.anthropic.chat.transformation.AnthropicConfig._is_adaptive_thinking_model",
            return_value=True,
        ):
            cfg = self._build_config()
            data = {}
            optional_params = {"output_config": {"effort": "bogus"}}
            with pytest.raises(Exception):
                cfg._apply_output_config(data, "test-model", optional_params)
