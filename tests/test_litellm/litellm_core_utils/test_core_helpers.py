"""Tests for litellm_core_utils.core_helpers module."""

import pytest

from litellm.litellm_core_utils.core_helpers import (
    _FINISH_REASON_MAP,
    map_finish_reason,
    reconstruct_model_name,
    redact_nested_match_and_regex_keys,
)


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
                            {
                                "type": "NAME",
                                "match": "secret-name",
                                "action": "BLOCKED",
                            }
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
        assert (
            out["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0][
                "match"
            ]
            == "[REDACTED]"
        )
        assert out["assessments"][0]["wordPolicy"]["customWords"][0]["match"] == (
            "[REDACTED]"
        )
        assert out["regex"] == "[REDACTED]"
        assert (
            payload["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0][
                "match"
            ]
            == "secret-name"
        )

    def test_passes_through_none_and_str(self):
        assert redact_nested_match_and_regex_keys(None) is None
        assert redact_nested_match_and_regex_keys("plain") == "plain"


# ---------------------------------------------------------------------------
# Regression tests for LIT-3302
# (header-derived spend-tracking attributes dropped when both
# `metadata` and `litellm_metadata` are present on a request, e.g. /v1/messages)
# ---------------------------------------------------------------------------


class TestAddMissingSpendMetadataToLitellmMetadata:
    """Regression coverage for `add_missing_spend_metadata_to_litellm_metadata`.

    The proxy puts spend-tracking + header-derived fields on `metadata`, while
    some endpoints (e.g. Anthropic `/v1/messages`) also populate
    `litellm_metadata`. The helper must preserve the proxy-set fields when
    merging so they reach the spend logs.
    """

    def test_preserves_spend_logs_metadata_from_request_header(self):
        """`spend_logs_metadata` (set from `x-litellm-spend-logs-metadata`) must
        survive a merge when `litellm_metadata` is also present."""
        from litellm.litellm_core_utils.core_helpers import (
            add_missing_spend_metadata_to_litellm_metadata,
        )

        header_attrs = {
            "billing_uuid": "9af2b4e0-1c11-4d8b-bcb3-1234567890ab",
            "agent_id": "agent-cogni-7",
            "agent_owner_id": "owner-acme-42",
            "request_type": "chat_completion",
        }
        metadata = {
            "user_api_key": "sk-hashed-abc",
            "user_api_key_alias": "test-key",
            "spend_logs_metadata": header_attrs,
        }
        litellm_metadata = {"user_id": "anthropic-end-user"}

        merged = add_missing_spend_metadata_to_litellm_metadata(
            litellm_metadata, metadata
        )

        assert merged["spend_logs_metadata"] == header_attrs
        assert merged["user_api_key"] == "sk-hashed-abc"
        assert merged["user_api_key_alias"] == "test-key"
        # Original litellm_metadata content stays untouched.
        assert merged["user_id"] == "anthropic-end-user"

    def test_preserves_other_proxy_header_metadata(self):
        """`agent_id`, `trace_id`, `session_id`, `requester_ip_address` are all
        proxy-populated and must also survive the merge."""
        from litellm.litellm_core_utils.core_helpers import (
            add_missing_spend_metadata_to_litellm_metadata,
        )

        metadata = {
            "user_api_key_team_id": "team-1",
            "agent_id": "agent-7",
            "trace_id": "trace-abc",
            "session_id": "sess-xyz",
            "requester_ip_address": "10.0.0.1",
            "spend_logs_metadata": {"foo": "bar"},
        }
        litellm_metadata = {"user_id": "u"}

        merged = add_missing_spend_metadata_to_litellm_metadata(
            litellm_metadata, metadata
        )

        for key in (
            "agent_id",
            "trace_id",
            "session_id",
            "requester_ip_address",
            "spend_logs_metadata",
            "user_api_key_team_id",
        ):
            assert merged[key] == metadata[key]

    def test_does_not_leak_unrelated_metadata_keys(self):
        """Client-supplied (request body) keys on `metadata` that are not
        spend-tracking or proxy-set must NOT clobber `litellm_metadata`."""
        from litellm.litellm_core_utils.core_helpers import (
            add_missing_spend_metadata_to_litellm_metadata,
        )

        metadata = {
            "user_api_key": "sk-hashed",
            "user_id": "client-1",  # client-supplied body field
            "custom_unrelated": "noise",
        }
        litellm_metadata = {"user_id": "proxy-set-user"}

        merged = add_missing_spend_metadata_to_litellm_metadata(
            litellm_metadata, metadata
        )

        # user_api_key still copied
        assert merged["user_api_key"] == "sk-hashed"
        # client-supplied body fields NOT propagated
        assert merged["user_id"] == "proxy-set-user"
        assert "custom_unrelated" not in merged

    def test_get_litellm_metadata_from_kwargs_round_trip(self):
        """End-to-end through `get_litellm_metadata_from_kwargs`: the merged
        dict must include `spend_logs_metadata` for the LIT-3302 scenario."""
        from litellm.litellm_core_utils.core_helpers import (
            get_litellm_metadata_from_kwargs,
        )

        kwargs = {
            "litellm_params": {
                "metadata": {
                    "user_api_key": "sk-hashed",
                    "spend_logs_metadata": {
                        "billing_uuid": "uuid-1",
                        "agent_id": "agent-1",
                        "agent_owner_id": "owner-1",
                        "request_type": "chat_completion",
                    },
                },
                "litellm_metadata": {
                    "user_id": "client",  # forces the merge path
                },
            }
        }

        result = get_litellm_metadata_from_kwargs(kwargs)
        assert "spend_logs_metadata" in result
        assert result["spend_logs_metadata"]["billing_uuid"] == "uuid-1"
        assert result["user_api_key"] == "sk-hashed"

    def test_does_not_overwrite_proxy_set_value_when_litellm_metadata_already_has_it(
        self,
    ):
        """Inverse case: on `/v1/messages` (and other routes where
        `_metadata_variable_name == "litellm_metadata"`) the proxy-trusted
        values live on `litellm_metadata`. A client can forge body
        `metadata.<key>` for one of the header-derived keys; the merge must
        NOT overwrite the real proxy-set value with the forged one."""
        from litellm.litellm_core_utils.core_helpers import (
            add_missing_spend_metadata_to_litellm_metadata,
        )

        # Proxy-set values on litellm_metadata (e.g. /v1/messages)
        litellm_metadata = {
            "user_api_key": "sk-hashed",
            "agent_id": "agent-real-from-proxy",
            "trace_id": "trace-real",
            "session_id": "session-real",
            "requester_ip_address": "10.0.0.1",
            "spend_logs_metadata": {"billing_uuid": "real-bu"},
        }
        # Client-forged values on the body metadata
        metadata = {
            "agent_id": "agent-forged",
            "trace_id": "trace-forged",
            "session_id": "session-forged",
            "requester_ip_address": "1.2.3.4",
            "spend_logs_metadata": {"billing_uuid": "forged-bu"},
        }

        merged = add_missing_spend_metadata_to_litellm_metadata(
            litellm_metadata, metadata
        )

        # Every proxy-set value is preserved unchanged.
        assert merged["agent_id"] == "agent-real-from-proxy"
        assert merged["trace_id"] == "trace-real"
        assert merged["session_id"] == "session-real"
        assert merged["requester_ip_address"] == "10.0.0.1"
        assert merged["spend_logs_metadata"] == {"billing_uuid": "real-bu"}

    def test_preserve_keys_use_setdefault_semantics(self):
        """Even with only one proxy key already present on litellm_metadata,
        only that one is left untouched; the others (absent on litellm_metadata)
        are propagated from metadata."""
        from litellm.litellm_core_utils.core_helpers import (
            add_missing_spend_metadata_to_litellm_metadata,
        )

        litellm_metadata = {"agent_id": "agent-real"}  # only agent_id pre-set
        metadata = {
            "agent_id": "forged-agent",
            "trace_id": "trace-from-metadata",  # not on litellm_metadata
            "spend_logs_metadata": {"k": "v"},  # not on litellm_metadata
        }

        merged = add_missing_spend_metadata_to_litellm_metadata(
            litellm_metadata, metadata
        )

        assert merged["agent_id"] == "agent-real"  # not overwritten
        assert merged["trace_id"] == "trace-from-metadata"  # newly copied
        assert merged["spend_logs_metadata"] == {"k": "v"}  # newly copied
