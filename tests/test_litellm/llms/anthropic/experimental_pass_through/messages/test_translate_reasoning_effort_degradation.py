from unittest.mock import patch

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
)
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


def _mock_model_info(**flags):
    return flags


class TestTranslateReasoningEffortDegradation:
    def test_xhigh_degrades_to_high_for_non_adaptive_model(self):
        with (
            patch(
                "litellm.llms.anthropic.common_utils.AnthropicModelInfo._is_adaptive_thinking_model",
                return_value=False,
            ),
            patch(
                "litellm.utils.get_model_info",
                return_value=_mock_model_info(
                    supports_reasoning=True,
                    supports_xhigh_reasoning_effort=False,
                ),
            ),
        ):
            optional_params = {"reasoning_effort": "xhigh"}
            AnthropicMessagesConfig._translate_reasoning_effort_to_anthropic(
                model="unknown-glm-4.6",
                optional_params=optional_params,
                max_tokens=None,
                custom_llm_provider="anthropic",
            )
            assert optional_params["thinking"]["type"] == "enabled"
            assert optional_params["thinking"]["budget_tokens"] == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
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
                    supports_reasoning=True,
                    supports_max_reasoning_effort=False,
                    supports_xhigh_reasoning_effort=False,
                ),
            ),
        ):
            optional_params = {"reasoning_effort": "max"}
            AnthropicMessagesConfig._translate_reasoning_effort_to_anthropic(
                model="unknown-deepseek",
                optional_params=optional_params,
                max_tokens=None,
                custom_llm_provider="anthropic",
            )
            assert optional_params["thinking"]["budget_tokens"] == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
