"""Regression tests for BerriAI/litellm#31449.

1. Non-function tools must be skipped, not fail the whole request.
2. ``reasoning_effort`` must not be forwarded to OCI models that reject it
   (xAI Grok, OpenAI non-reasoning models), so ``drop_params`` works.
"""

import pytest

import litellm
from litellm.llms.oci.chat.generic import adapt_tool_definition_to_oci_standard
from litellm.llms.oci.chat.transformation import (
    OCIChatConfig,
    _model_supports_reasoning_effort,
)
from litellm.llms.oci.common_utils import OCIError
from litellm.types.llms.oci import OCIVendors

FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
BUILTIN_TOOL = {"type": "web_search_preview"}
SERVER_TOOL = {"type": "computer_use_preview", "display_width": 1024}


class TestNonFunctionToolsAreSkipped:
    def test_mixed_tools_keep_function_tools_only(self):
        result = adapt_tool_definition_to_oci_standard([BUILTIN_TOOL, FUNCTION_TOOL, SERVER_TOOL], OCIVendors.GENERIC)
        assert [tool.name for tool in result] == ["get_weather"]
        assert result[0].type == "FUNCTION"

    def test_only_non_function_tools_yield_empty_list(self):
        assert adapt_tool_definition_to_oci_standard([BUILTIN_TOOL], OCIVendors.GENERIC) == []

    def test_tool_without_type_is_skipped_not_keyerror(self):
        assert adapt_tool_definition_to_oci_standard([{"name": "x"}], OCIVendors.GENERIC) == []

    def test_request_drops_tools_and_tool_choice_when_none_remain(self):
        config = OCIChatConfig()
        params = config._get_optional_params(
            OCIVendors.GENERIC,
            {"tools": [BUILTIN_TOOL], "tool_choice": "auto"},
            model="xai.grok-4.20",
        )
        assert "tools" not in params
        assert "toolChoice" not in params
        assert "tool_choice" not in params

    def test_request_keeps_function_tools_and_tool_choice(self):
        config = OCIChatConfig()
        params = config._get_optional_params(
            OCIVendors.GENERIC,
            {"tools": [BUILTIN_TOOL, FUNCTION_TOOL], "tool_choice": "auto"},
            model="xai.grok-4.20",
        )
        assert [tool.name for tool in params["tools"]] == ["get_weather"]
        assert params["toolChoice"] == {"type": "AUTO"}


class TestReasoningEffortGating:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("xai.grok-4.20", False),
            ("oci/xai.grok-4.20-0309-reasoning", False),
            ("xai.grok-code-fast-1", False),
            ("openai.gpt-4o", False),
            ("openai.gpt-4.1", False),
            ("openai.gpt-5", True),
            ("oci/openai.gpt-5-mini", True),
            ("openai.gpt-oss-120b", True),
            ("meta.llama-3.3-70b-instruct", True),
            ("google.gemini-2.5-flash", True),
        ],
    )
    def test_model_supports_reasoning_effort(self, model, expected):
        assert _model_supports_reasoning_effort(model) is expected

    def test_supported_params_exclude_reasoning_effort_for_grok(self):
        supported = OCIChatConfig().get_supported_openai_params("xai.grok-4.20")
        assert "reasoning_effort" not in supported
        assert "tools" in supported

    def test_supported_params_include_reasoning_effort_for_gpt5(self):
        assert "reasoning_effort" in OCIChatConfig().get_supported_openai_params("openai.gpt-5")

    @pytest.mark.parametrize("model", ["xai.grok-4.20", "openai.gpt-4o"])
    def test_reasoning_effort_dropped_under_drop_params(self, model):
        result = OCIChatConfig().map_openai_params(
            non_default_params={"reasoning_effort": "low", "temperature": 0.1},
            optional_params={},
            model=model,
            drop_params=True,
        )
        assert "reasoningEffort" not in result
        assert "reasoning_effort" not in result
        assert result["temperature"] == 0.1

    def test_reasoning_effort_raises_without_drop_params_for_grok(self, monkeypatch):
        monkeypatch.setattr(litellm, "drop_params", False)
        with pytest.raises(OCIError, match=r"reasoning_effort.*xai\.grok-4\.20"):
            OCIChatConfig().map_openai_params(
                non_default_params={"reasoning_effort": "low"},
                optional_params={},
                model="xai.grok-4.20",
                drop_params=False,
            )

    def test_reasoning_effort_still_forwarded_for_reasoning_models(self):
        result = OCIChatConfig().map_openai_params(
            non_default_params={"reasoning_effort": "low"},
            optional_params={},
            model="openai.gpt-5",
            drop_params=False,
        )
        assert result["reasoningEffort"] == "low"
