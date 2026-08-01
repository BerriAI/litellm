"""Tests for litellm_core_utils.core_helpers module."""

import pytest

from litellm.litellm_core_utils.core_helpers import (
    _FINISH_REASON_MAP,
    get_litellm_metadata_from_kwargs,
    get_or_create_metadata_bucket,
    map_finish_reason,
    reconstruct_model_name,
    redact_nested_match_and_regex_keys,
)


def _spend_log_identity(kwargs: dict) -> dict:
    """Mirror how spend logging derives model / model_id from request kwargs."""
    metadata = get_litellm_metadata_from_kwargs(kwargs)
    return {
        "model": reconstruct_model_name(
            kwargs.get("model") or "", kwargs.get("custom_llm_provider"), metadata or {}
        ),
        "model_id": metadata.get("model_info", {}).get("id", ""),
        "model_group": metadata.get("model_group", ""),
    }


def test_spend_log_deployment_identity_consistent_with_litellm_metadata():
    """Regression for #35472: an openai/ passthrough deployment must log the same
    model / model_id whether or not the request also carries litellm_metadata.

    The router writes the deployment identity into `metadata`, but spend logging
    reads `litellm_metadata` when present, so those keys must be carried over."""
    router_metadata = {
        "deployment": "openai/anthropic/claude-sonnet-5",
        "model_info": {"id": "9da5dfc9-2223-4f77-b3c9-f9100d9cb2a0"},
        "model_group": "my-model",
        "user_api_key_hash": "abc",
    }

    without_litellm_metadata = _spend_log_identity(
        {
            "model": "anthropic/claude-sonnet-5",
            "custom_llm_provider": "openai",
            "litellm_params": {"metadata": dict(router_metadata)},
        }
    )
    with_litellm_metadata = _spend_log_identity(
        {
            "model": "anthropic/claude-sonnet-5",
            "custom_llm_provider": "openai",
            "litellm_params": {
                "metadata": dict(router_metadata),
                "litellm_metadata": {"tags": ["t1"]},
            },
        }
    )

    expected = {
        "model": "openai/anthropic/claude-sonnet-5",
        "model_id": "9da5dfc9-2223-4f77-b3c9-f9100d9cb2a0",
        "model_group": "my-model",
    }
    assert without_litellm_metadata == expected
    assert with_litellm_metadata == expected


def test_get_litellm_metadata_from_kwargs_does_not_overwrite_existing_identity():
    """Caller-supplied identity keys in litellm_metadata must win over metadata."""
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "router-group", "deployment": "router-dep"},
            "litellm_metadata": {"model_group": "caller-group"},
        }
    }

    metadata = get_litellm_metadata_from_kwargs(kwargs)

    assert metadata["model_group"] == "caller-group"
    assert metadata["deployment"] == "router-dep"


class TestGetOrCreateMetadataBucket:
    """The single owner every guardrail writer and reader shares, so the response
    header and the spend log can never disagree about which dict a record lives in."""

    def test_prefers_litellm_metadata_when_both_present(self):
        request_data = {"metadata": {"user_id": "caller"}, "litellm_metadata": {}}

        key, bucket = get_or_create_metadata_bucket(request_data)

        assert key == "litellm_metadata"
        assert bucket is request_data["litellm_metadata"]

    def test_uses_metadata_when_litellm_metadata_absent(self):
        request_data = {"metadata": {"user_id": "caller"}}

        key, bucket = get_or_create_metadata_bucket(request_data)

        assert key == "metadata"
        assert bucket is request_data["metadata"]

    def test_creates_the_bucket_in_place_when_missing(self):
        request_data: dict = {}

        key, bucket = get_or_create_metadata_bucket(request_data)

        assert key == "metadata"
        assert request_data["metadata"] is bucket
        bucket["k"] = "v"
        assert request_data["metadata"]["k"] == "v"

    def test_replaces_a_non_dict_bucket(self):
        request_data = {"litellm_metadata": None}

        key, bucket = get_or_create_metadata_bucket(request_data)

        assert key == "litellm_metadata"
        assert isinstance(request_data["litellm_metadata"], dict)
        assert bucket is request_data["litellm_metadata"]


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
    def test_anthropic_finish_reasons(
        self, provider_reason: str, expected: str
    ) -> None:
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
    @pytest.mark.parametrize(
        "reason", ["stop", "length", "tool_calls", "function_call", "content_filter"]
    )
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
                f"Mapped value '{openai_reason}' (from '{provider_reason}') "
                f"is not a valid OpenAI finish reason"
            )


class TestRedactNestedMatchAndRegexKeys:
    def test_redacts_match_and_regex_recursively(self):
        payload = {
            "assessments": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": [
                            {"type": "NAME", "match": "secret-name", "action": "BLOCKED"}
                        ]
                    },
                    "wordPolicy": {
                        "customWords": [{"match": "badword", "action": "BLOCKED"}]
                    },
                }
            ],
            "regex": "should-redact-key-named-regex",
        }
        out = redact_nested_match_and_regex_keys(payload)
        assert out["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0][
            "match"
        ] == "[REDACTED]"
        assert out["assessments"][0]["wordPolicy"]["customWords"][0]["match"] == (
            "[REDACTED]"
        )
        assert out["regex"] == "[REDACTED]"
        assert payload["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][
            0
        ]["match"] == "secret-name"

    def test_passes_through_none_and_str(self):
        assert redact_nested_match_and_regex_keys(None) is None
        assert redact_nested_match_and_regex_keys("plain") == "plain"
