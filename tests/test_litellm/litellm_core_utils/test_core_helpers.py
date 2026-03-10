"""Tests for litellm_core_utils.core_helpers module."""

import pytest

from litellm.litellm_core_utils.core_helpers import map_finish_reason, reconstruct_model_name


class TestMapFinishReason:
    @pytest.mark.parametrize(
        "value",
        [
            "stop",
            "length",
            "function_call",
            "tool_calls",
            "content_filter",
            "finish_reason_unspecified",
            "eos",
            "guardrail_intervened",
            "malformed_function_call",
        ],
    )
    def test_known_openai_values_pass_through(self, value: str) -> None:
        assert map_finish_reason(value) == value

    def test_anthropic_tool_use_maps_to_tool_calls(self) -> None:
        assert map_finish_reason("tool_use") == "tool_calls"

    def test_anthropic_max_tokens_maps_to_length(self) -> None:
        assert map_finish_reason("max_tokens") == "length"

    def test_anthropic_end_turn_maps_to_stop(self) -> None:
        assert map_finish_reason("end_turn") == "stop"

    def test_cohere_complete_maps_to_stop(self) -> None:
        assert map_finish_reason("COMPLETE") == "stop"

    def test_cohere_max_tokens_maps_to_length(self) -> None:
        assert map_finish_reason("MAX_TOKENS") == "length"

    def test_cohere_error_toxic_maps_to_content_filter(self) -> None:
        assert map_finish_reason("ERROR_TOXIC") == "content_filter"

    def test_vertex_ai_stop_maps_to_stop(self) -> None:
        assert map_finish_reason("STOP") == "stop"

    def test_vertex_ai_safety_maps_to_content_filter(self) -> None:
        assert map_finish_reason("SAFETY") == "content_filter"

    def test_vertex_ai_finish_reason_unspecified_maps_correctly(self) -> None:
        assert map_finish_reason("FINISH_REASON_UNSPECIFIED") == "finish_reason_unspecified"

    def test_vertex_ai_malformed_function_call_maps_correctly(self) -> None:
        assert map_finish_reason("MALFORMED_FUNCTION_CALL") == "malformed_function_call"

    def test_unknown_value_maps_to_finish_reason_unspecified(self) -> None:
        assert map_finish_reason("some_unknown_reason") == "finish_reason_unspecified"

    def test_empty_string_maps_to_finish_reason_unspecified(self) -> None:
        assert map_finish_reason("") == "finish_reason_unspecified"

    def test_zhipuai_glm_network_error_regression(self) -> None:
        assert map_finish_reason("network_error") == "finish_reason_unspecified"


def test_reconstruct_model_name_prefers_deployment_value():
    """Ensure deployment metadata wins when reconstructing the model name."""

    metadata = {"deployment": "vertex_ai/gemini-1.5-flash"}

    result = reconstruct_model_name(
        model_name="gemini-1.5-flash",
        custom_llm_provider="vertex_ai",
        metadata=metadata,
    )

    assert result == "vertex_ai/gemini-1.5-flash"


def test_reconstruct_model_name_adds_bedrock_prefix_when_missing():
    """Bedrock model names without prefixes should gain the provider prefix."""

    metadata = {}

    result = reconstruct_model_name(
        model_name="us.anthropic.claude-3-sonnet",
        custom_llm_provider="bedrock",
        metadata=metadata,
    )

    assert result == "bedrock/us.anthropic.claude-3-sonnet"


def test_reconstruct_model_name_returns_original_for_other_providers():
    """Non-Bedrock providers should not prepend anything."""

    metadata = {}

    result = reconstruct_model_name(
        model_name="claude-3-sonnet",
        custom_llm_provider="anthropic",
        metadata=metadata,
    )

    assert result == "claude-3-sonnet"
