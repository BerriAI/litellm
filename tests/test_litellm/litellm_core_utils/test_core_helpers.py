"""Tests for litellm_core_utils.core_helpers module."""

import pytest

from litellm.litellm_core_utils.core_helpers import (
    _FINISH_REASON_MAP,
    MCP_INTERNAL_REQUEST_KEYS,
    filter_internal_params,
    map_finish_reason,
    reconstruct_model_name,
    redact_nested_match_and_regex_keys,
)
from litellm.types.utils import all_litellm_params


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


# ---------------------------------------------------------------------------
# map_finish_reason tests
# ---------------------------------------------------------------------------

VALID_OPENAI_FINISH_REASONS = {
    "stop",
    "length",
    "tool_calls",
    "function_call",
    "content_filter",
}


class TestMapFinishReasonAnthropic:
    @pytest.mark.parametrize(
        "provider_reason,expected",
        [
            ("stop_sequence", "stop"),
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
            ("compaction", "length"),
            ("content_filtered", "content_filter"),
        ],
    )
    def test_anthropic_finish_reasons(self, provider_reason: str, expected: str) -> None:
        assert map_finish_reason(provider_reason) == expected

    def test_refusal(self):
        assert map_finish_reason("refusal") == "content_filter"


class TestMapFinishReasonGemini:
    @pytest.mark.parametrize(
        "gemini_reason,expected",
        [
            ("STOP", "stop"),
            ("MAX_TOKENS", "length"),
            ("SAFETY", "content_filter"),
            ("RECITATION", "content_filter"),
            ("FINISH_REASON_UNSPECIFIED", "stop"),
            ("MALFORMED_FUNCTION_CALL", "stop"),
            ("LANGUAGE", "content_filter"),
            ("OTHER", "content_filter"),
            ("BLOCKLIST", "content_filter"),
            ("PROHIBITED_CONTENT", "content_filter"),
            ("SPII", "content_filter"),
            ("IMAGE_SAFETY", "content_filter"),
            ("IMAGE_PROHIBITED_CONTENT", "content_filter"),
            ("TOO_MANY_TOOL_CALLS", "stop"),
            ("MALFORMED_RESPONSE", "stop"),
        ],
    )
    def test_gemini_finish_reasons(self, gemini_reason, expected):
        assert map_finish_reason(gemini_reason) == expected


class TestMapFinishReasonCohere:
    def test_complete(self):
        assert map_finish_reason("COMPLETE") == "stop"

    def test_error_toxic(self):
        assert map_finish_reason("ERROR_TOXIC") == "content_filter"

    def test_error(self):
        assert map_finish_reason("ERROR") == "stop"


class TestMapFinishReasonHuggingFace:
    def test_eos_token(self):
        assert map_finish_reason("eos_token") == "stop"

    def test_eos(self):
        assert map_finish_reason("eos") == "stop"


class TestMapFinishReasonBedrock:
    def test_guardrail_intervened(self):
        assert map_finish_reason("guardrail_intervened") == "content_filter"


class TestMapFinishReasonZhipu:
    def test_network_error(self):
        assert map_finish_reason("network_error") == "stop"

    def test_sensitive(self):
        assert map_finish_reason("sensitive") == "content_filter"


class TestMapFinishReasonOpenAIPassthrough:
    @pytest.mark.parametrize("reason", ["stop", "length", "tool_calls", "function_call", "content_filter"])
    def test_openai_values_pass_through(self, reason):
        assert map_finish_reason(reason) == reason


class TestMapFinishReasonUnknown:
    def test_unknown_value_defaults_to_stop(self):
        assert map_finish_reason("some_unknown_value") == "stop"

    def test_empty_string_defaults_to_stop(self):
        assert map_finish_reason("") == "stop"


class TestFinishReasonMapOutputsAreValid:
    def test_all_mapped_values_are_valid_openai_reasons(self):
        """Every value in _FINISH_REASON_MAP must be a valid OpenAI finish reason."""
        for provider_reason, openai_reason in _FINISH_REASON_MAP.items():
            assert openai_reason in VALID_OPENAI_FINISH_REASONS, (
                f"Mapped value '{openai_reason}' (from '{provider_reason}') is not a valid OpenAI finish reason"
            )


class TestRedactNestedMatchAndRegexKeys:
    def test_redacts_match_and_regex_recursively(self):
        payload = {
            "assessments": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": [{"type": "NAME", "match": "secret-name", "action": "BLOCKED"}]
                    },
                    "wordPolicy": {"customWords": [{"match": "badword", "action": "BLOCKED"}]},
                }
            ],
            "regex": "should-redact-key-named-regex",
        }
        out = redact_nested_match_and_regex_keys(payload)
        assert out["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0]["match"] == "[REDACTED]"
        assert out["assessments"][0]["wordPolicy"]["customWords"][0]["match"] == ("[REDACTED]")
        assert out["regex"] == "[REDACTED]"
        assert payload["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0]["match"] == "secret-name"

    def test_passes_through_none_and_str(self):
        assert redact_nested_match_and_regex_keys(None) is None
        assert redact_nested_match_and_regex_keys("plain") == "plain"


class TestFilterInternalParams:
    # MCP handler plumbing keys, also registered in all_litellm_params (#30301)
    INTERNAL_KEYS = {
        "skip_mcp_handler",
        "mcp_handler_context",
        "_skip_mcp_handler",
    }

    def test_registry_strips_every_known_internal_key(self):
        seeded = {k: "leak" for k in self.INTERNAL_KEYS}
        seeded["model"] = "gpt-4"
        seeded["temperature"] = 0.2
        out = filter_internal_params(seeded)
        for k in self.INTERNAL_KEYS:
            assert k not in out, f"{k} leaked through filter_internal_params"
        assert out == {"model": "gpt-4", "temperature": 0.2}

    def test_internal_keys_are_registered_in_all_litellm_params(self):
        for k in self.INTERNAL_KEYS:
            assert k in all_litellm_params, f"{k} missing from all_litellm_params"

    def test_stream_chunk_size_is_not_filtered(self):
        # stream_chunk_size is read downstream (converse streaming), not internal
        assert filter_internal_params({"stream_chunk_size": 2048}) == {"stream_chunk_size": 2048}

    def test_additional_internal_params_layer_on_top(self):
        out = filter_internal_params(
            {"keep": 1, "skip_mcp_handler": 2, "provider_only": 3},
            additional_internal_params={"provider_only"},
        )
        assert out == {"keep": 1}

    def test_registry_not_mutated_by_additional_params(self):
        baseline = set(MCP_INTERNAL_REQUEST_KEYS)
        filter_internal_params({"x": 1}, additional_internal_params={"adhoc_key"})
        assert MCP_INTERNAL_REQUEST_KEYS == baseline

    def test_non_dict_passes_through(self):
        assert filter_internal_params("not-a-dict") == "not-a-dict"
        assert filter_internal_params([1, 2, 3]) == [1, 2, 3]

    def test_existing_mcp_keys_still_filtered(self):
        out = filter_internal_params({"skip_mcp_handler": True, "mcp_handler_context": {}, "model": "gpt-4"})
        assert out == {"model": "gpt-4"}
