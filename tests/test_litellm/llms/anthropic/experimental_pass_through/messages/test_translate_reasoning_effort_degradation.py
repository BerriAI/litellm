"""
Unit tests for Patch 2b: _translate_reasoning_effort_to_anthropic degrades
unsupported effort levels instead of raising on the /v1/messages path.
"""

from unittest.mock import patch


from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


def _mock_model_info(**flags):
    return flags


class TestTranslateReasoningEffortDegradation:
    """Patch 2b: messages path degrades unsupported effort levels."""

    def test_xhigh_degrades_to_high_for_non_adaptive_model(self):
        """Non-adaptive model with supports_xhigh=False: xhigh -> high (via Patch 1
        in _map_reasoning_effort), producing a legacy thinking block with high budget.
        No 400 raised."""
        with (
            patch(
                "litellm.llms.anthropic.common_utils.AnthropicModelInfo._is_adaptive_thinking_model",
                return_value=False,
            ),
            patch(
                "litellm.utils.get_model_info",
                return_value=_mock_model_info(supports_xhigh_reasoning_effort=False),
            ),
        ):
            optional_params = {"reasoning_effort": "xhigh"}
            AnthropicMessagesConfig._translate_reasoning_effort_to_anthropic(
                model="unknown-glm-4.6",
                optional_params=optional_params,
                custom_llm_provider="anthropic",
            )
            assert "thinking" in optional_params
            thinking = optional_params["thinking"]
            assert thinking["type"] == "enabled"
            assert (
                thinking["budget_tokens"]
                == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
            )
            # adaptive path not taken
            assert "output_config" not in optional_params

    def test_max_degrades_to_high_for_non_adaptive_model(self):
        with (
            patch(
                "litellm.llms.anthropic.common_utils.AnthropicModelInfo._is_adaptive_thinking_model",
                return_value=False,
            ),
            patch(
                "litellm.utils.get_model_info",
                return_value=_mock_model_info(
                    supports_max_reasoning_effort=False,
                    supports_xhigh_reasoning_effort=False,
                ),
            ),
        ):
            optional_params = {"reasoning_effort": "max"}
            AnthropicMessagesConfig._translate_reasoning_effort_to_anthropic(
                model="unknown-deepseek",
                optional_params=optional_params,
                custom_llm_provider="anthropic",
            )
            assert (
                optional_params["thinking"]["budget_tokens"]
                == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
            )

    def test_minimal_degrades_to_low_for_non_adaptive_model(self):
        with (
            patch(
                "litellm.llms.anthropic.common_utils.AnthropicModelInfo._is_adaptive_thinking_model",
                return_value=False,
            ),
            patch(
                "litellm.utils.get_model_info",
                return_value=_mock_model_info(supports_minimal_reasoning_effort=False),
            ),
        ):
            optional_params = {"reasoning_effort": "minimal"}
            AnthropicMessagesConfig._translate_reasoning_effort_to_anthropic(
                model="unknown-kimi",
                optional_params=optional_params,
                custom_llm_provider="anthropic",
            )
            assert (
                optional_params["thinking"]["budget_tokens"]
                == DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET
            )

    def test_high_unchanged_for_non_adaptive_model(self):
        with (
            patch(
                "litellm.llms.anthropic.common_utils.AnthropicModelInfo._is_adaptive_thinking_model",
                return_value=False,
            ),
            patch(
                "litellm.utils.get_model_info",
                side_effect=Exception("not found"),
            ),
        ):
            optional_params = {"reasoning_effort": "high"}
            AnthropicMessagesConfig._translate_reasoning_effort_to_anthropic(
                model="unknown-model",
                optional_params=optional_params,
                custom_llm_provider="anthropic",
            )
            assert (
                optional_params["thinking"]["budget_tokens"]
                == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
            )

    def test_none_effort_clears_params(self):
        with patch(
            "litellm.llms.anthropic.common_utils.AnthropicModelInfo._is_adaptive_thinking_model",
            return_value=False,
        ):
            optional_params = {
                "reasoning_effort": "none",
                "thinking": {"type": "enabled"},
            }
            AnthropicMessagesConfig._translate_reasoning_effort_to_anthropic(
                model="any-model",
                optional_params=optional_params,
                custom_llm_provider="anthropic",
            )
            assert "thinking" not in optional_params
            assert "output_config" not in optional_params

    def test_adaptive_model_xhigh_degrades_instead_of_raising(self):
        """Adaptive model with supports_xhigh=False: xhigh -> high in
        output_config.effort, no AnthropicError raised."""
        with (
            patch(
                "litellm.llms.anthropic.common_utils.AnthropicModelInfo._is_adaptive_thinking_model",
                return_value=True,
            ),
            patch(
                "litellm.llms.anthropic.chat.transformation.AnthropicConfig._validate_effort_for_model",
                return_value="effort='xhigh' is not supported by this model. Got model: test",
            ),
            patch(
                "litellm.utils.get_model_info",
                return_value=_mock_model_info(supports_xhigh_reasoning_effort=False),
            ),
        ):
            optional_params = {"reasoning_effort": "xhigh"}
            # Should not raise
            AnthropicMessagesConfig._translate_reasoning_effort_to_anthropic(
                model="claude-fake-4-6",
                optional_params=optional_params,
                custom_llm_provider="anthropic",
            )
            assert optional_params["output_config"]["effort"] == "high"
            assert optional_params["thinking"] == {"type": "adaptive"}

    def test_adaptive_model_max_degrades_to_high(self):
        with (
            patch(
                "litellm.llms.anthropic.common_utils.AnthropicModelInfo._is_adaptive_thinking_model",
                return_value=True,
            ),
            patch(
                "litellm.llms.anthropic.chat.transformation.AnthropicConfig._validate_effort_for_model",
                return_value="effort='max' is not supported by this model. Got model: test",
            ),
            patch(
                "litellm.utils.get_model_info",
                return_value=_mock_model_info(
                    supports_max_reasoning_effort=False,
                    supports_xhigh_reasoning_effort=False,
                ),
            ),
        ):
            optional_params = {"reasoning_effort": "max"}
            AnthropicMessagesConfig._translate_reasoning_effort_to_anthropic(
                model="claude-fake-4-6",
                optional_params=optional_params,
                custom_llm_provider="anthropic",
            )
            assert optional_params["output_config"]["effort"] == "high"

    def test_adaptive_model_max_stays_when_supported(self):
        """Regression: when _validate_effort_for_model returns None (max supported),
        effort stays at max."""
        with (
            patch(
                "litellm.llms.anthropic.common_utils.AnthropicModelInfo._is_adaptive_thinking_model",
                return_value=True,
            ),
            patch(
                "litellm.llms.anthropic.chat.transformation.AnthropicConfig._validate_effort_for_model",
                return_value=None,
            ),
        ):
            optional_params = {"reasoning_effort": "max"}
            AnthropicMessagesConfig._translate_reasoning_effort_to_anthropic(
                model="claude-opus-4-7",
                optional_params=optional_params,
                custom_llm_provider="anthropic",
            )
            assert optional_params["output_config"]["effort"] == "max"
            assert optional_params["thinking"] == {"type": "adaptive"}

    def test_xhigh_stays_xhigh_when_supported(self):
        """Regression: when supports_xhigh=True, xhigh is preserved."""
        with (
            patch(
                "litellm.llms.anthropic.common_utils.AnthropicModelInfo._is_adaptive_thinking_model",
                return_value=False,
            ),
            patch(
                "litellm.utils.get_model_info",
                return_value=_mock_model_info(supports_xhigh_reasoning_effort=True),
            ),
        ):
            optional_params = {"reasoning_effort": "xhigh"}
            AnthropicMessagesConfig._translate_reasoning_effort_to_anthropic(
                model="claude-opus-4-7",
                optional_params=optional_params,
                custom_llm_provider="anthropic",
            )
            assert (
                optional_params["thinking"]["budget_tokens"]
                == DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET
            )
