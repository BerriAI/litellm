"""Regression tests for BerriAI/litellm#31449.

1. Non-function tools must be skipped, not fail the whole request, and a
   ``tool_choice`` that forces one of the skipped tools must not survive.
2. ``reasoning_effort`` must only be forwarded to OCI models whose catalog
   entry carries ``supports_reasoning`` (gpt-5 family, Gemini 2.5, gpt-oss);
   xAI Grok and Meta Llama entries lack it, so ``drop_params`` works and
   callers get a clear error otherwise.
3. Flagging Gemini 2.5 as a reasoning model must not move it to
   ``maxCompletionTokens`` (OCI accepts but ignores that cap for Gemini).
"""

import pytest

import litellm
from litellm.llms.oci.chat.generic import adapt_tool_definition_to_oci_standard
from litellm.llms.oci.chat.transformation import (
    OCIChatConfig,
    _model_supports_reasoning_effort,
    _model_uses_max_completion_tokens,
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
FORCE_BUILTIN = {"type": "function", "function": {"name": "web_search_preview"}}
FORCE_FUNCTION = {"type": "function", "function": {"name": "get_weather"}}

GEMINI_MODELS = ["google.gemini-2.5-flash", "google.gemini-2.5-pro", "google.gemini-2.5-flash-lite"]
GPT_OSS_MODELS = ["openai.gpt-oss-120b", "openai.gpt-oss-20b"]


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
        params = OCIChatConfig()._get_optional_params(
            OCIVendors.GENERIC,
            {"tools": [BUILTIN_TOOL], "tool_choice": "auto"},
            model="xai.grok-4.20",
        )
        assert "tools" not in params
        assert "toolChoice" not in params
        assert "tool_choice" not in params

    def test_request_keeps_function_tools_and_tool_choice(self):
        params = OCIChatConfig()._get_optional_params(
            OCIVendors.GENERIC,
            {"tools": [BUILTIN_TOOL, FUNCTION_TOOL], "tool_choice": "auto"},
            model="xai.grok-4.20",
        )
        assert [tool.name for tool in params["tools"]] == ["get_weather"]
        assert params["toolChoice"] == {"type": "AUTO"}

    @pytest.mark.parametrize("tool_choice", [FORCE_BUILTIN, "web_search_preview"])
    def test_tool_choice_forcing_a_skipped_tool_is_dropped(self, tool_choice):
        params = OCIChatConfig()._get_optional_params(
            OCIVendors.GENERIC,
            {"tools": [BUILTIN_TOOL, FUNCTION_TOOL], "tool_choice": tool_choice},
            model="xai.grok-4.20",
        )
        assert [tool.name for tool in params["tools"]] == ["get_weather"]
        assert "toolChoice" not in params
        assert "tool_choice" not in params

    @pytest.mark.parametrize("tool_choice", [FORCE_FUNCTION, "get_weather"])
    def test_tool_choice_forcing_a_kept_tool_is_preserved(self, tool_choice):
        params = OCIChatConfig()._get_optional_params(
            OCIVendors.GENERIC,
            {"tools": [BUILTIN_TOOL, FUNCTION_TOOL], "tool_choice": tool_choice},
            model="xai.grok-4.20",
        )
        assert params["toolChoice"] == {"type": "FUNCTION", "name": "get_weather"}

    def test_tool_choice_without_tools_is_left_for_the_service(self):
        # Nothing was filtered, so nothing is masked: OCI reports the caller's mistake.
        params = OCIChatConfig()._get_optional_params(
            OCIVendors.GENERIC,
            {"tool_choice": FORCE_FUNCTION},
            model="xai.grok-4.20",
        )
        assert params["toolChoice"] == {"type": "FUNCTION", "name": "get_weather"}


class TestReasoningEffortGating:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("xai.grok-4.20", False),
            ("oci/xai.grok-4.20-multi-agent", False),
            ("xai.grok-code-fast-1", False),
            ("xai.grok-4", False),
            ("meta.llama-3.3-70b-instruct", False),
            ("meta.llama-4-scout-17b-16e-instruct", False),
            ("openai.gpt-5", True),
            ("oci/openai.gpt-5-mini", True),
            ("openai.gpt-oss-120b", True),
            ("openai.gpt-oss-20b", True),
            ("google.gemini-2.5-flash", True),
            ("google.gemini-2.5-pro", True),
            ("google.gemini-2.5-flash-lite", True),
        ],
    )
    def test_gate_follows_catalog_supports_reasoning(self, local_model_cost_map, model, expected):
        assert _model_supports_reasoning_effort(model) is expected

    def test_models_absent_from_catalog_are_forwarded(self, local_model_cost_map):
        assert "oci/meta.llama-9-future-instruct" not in litellm.model_cost
        assert _model_supports_reasoning_effort("meta.llama-9-future-instruct") is True
        assert _model_supports_reasoning_effort("") is True

    def test_gate_is_catalog_driven(self, local_model_cost_map, monkeypatch):
        # Flipping the flag on the catalog entry flips the gate; no code change needed.
        entry = {**litellm.model_cost["oci/xai.grok-4.20"], "supports_reasoning": True}
        monkeypatch.setitem(litellm.model_cost, "oci/xai.grok-4.20", entry)
        litellm.get_model_info.cache_clear()
        assert _model_supports_reasoning_effort("xai.grok-4.20") is True

    def test_supported_params_exclude_reasoning_effort_for_grok(self, local_model_cost_map):
        supported = OCIChatConfig().get_supported_openai_params("xai.grok-4.20")
        assert "reasoning_effort" not in supported
        assert "tools" in supported

    @pytest.mark.parametrize("model", ["openai.gpt-5", "google.gemini-2.5-flash", "openai.gpt-oss-120b"])
    def test_supported_params_include_reasoning_effort_for_reasoning_models(self, local_model_cost_map, model):
        assert "reasoning_effort" in OCIChatConfig().get_supported_openai_params(model)

    @pytest.mark.parametrize("model", ["xai.grok-4.20", "meta.llama-3.3-70b-instruct"])
    def test_reasoning_effort_dropped_under_drop_params(self, local_model_cost_map, model):
        result = OCIChatConfig().map_openai_params(
            non_default_params={"reasoning_effort": "low", "temperature": 0.1},
            optional_params={},
            model=model,
            drop_params=True,
        )
        assert "reasoningEffort" not in result
        assert "reasoning_effort" not in result
        assert result["temperature"] == 0.1

    def test_reasoning_effort_raises_without_drop_params_for_grok(self, local_model_cost_map, monkeypatch):
        monkeypatch.setattr(litellm, "drop_params", False)
        with pytest.raises(OCIError, match=r"reasoning_effort.*xai\.grok-4\.20"):
            OCIChatConfig().map_openai_params(
                non_default_params={"reasoning_effort": "low"},
                optional_params={},
                model="xai.grok-4.20",
                drop_params=False,
            )

    @pytest.mark.parametrize("model", ["openai.gpt-5", "google.gemini-2.5-flash", "openai.gpt-oss-120b"])
    def test_reasoning_effort_still_forwarded_for_reasoning_models(self, local_model_cost_map, model):
        result = OCIChatConfig().map_openai_params(
            non_default_params={"reasoning_effort": "low"},
            optional_params={},
            model=model,
            drop_params=False,
        )
        assert result["reasoningEffort"] == "low"


class TestCatalogFlagsDoNotMoveMaxTokens:
    @pytest.mark.parametrize("model", GEMINI_MODELS + GPT_OSS_MODELS)
    def test_reasoning_models_outside_openai_commercial_keep_max_tokens(self, local_model_cost_map, model):
        assert litellm.get_model_info(model, custom_llm_provider="oci")["supports_reasoning"] is True
        assert _model_uses_max_completion_tokens(model) is False
        params = OCIChatConfig()._get_optional_params(OCIVendors.GENERIC, {"max_tokens": 64}, model=model)
        assert params["maxTokens"] == 64
        assert "maxCompletionTokens" not in params

    def test_gpt5_still_routes_to_max_completion_tokens(self, local_model_cost_map):
        assert _model_uses_max_completion_tokens("openai.gpt-5") is True
