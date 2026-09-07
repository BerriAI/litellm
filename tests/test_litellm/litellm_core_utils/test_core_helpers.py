"""Tests for litellm_core_utils.core_helpers module."""

import pytest

from litellm.litellm_core_utils.core_helpers import (
    _FINISH_REASON_MAP,
    get_or_create_metadata_bucket,
    map_finish_reason,
    reconstruct_model_name,
    redact_nested_match_and_regex_keys,
)


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


class TestMapFinishReasonGenericError:
    def test_lowercase_error_is_explicitly_mapped(self):
        assert "error" in _FINISH_REASON_MAP
        assert map_finish_reason("error") == "stop"

    def test_lowercase_error_does_not_warn(self, mocker):
        warn = mocker.patch(
            "litellm.litellm_core_utils.core_helpers.verbose_logger.warning"
        )
        assert map_finish_reason("error") == "stop"
        warn.assert_not_called()


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


class TestIsExpectedClientError:
    def test_status_ranges(self):
        from litellm.litellm_core_utils.core_helpers import is_expected_client_error

        class WithStatusCode(Exception):
            def __init__(self, status_code):
                self.status_code = status_code

        class WithCode(Exception):
            def __init__(self, code):
                self.code = code

        assert is_expected_client_error(WithStatusCode(400)) is True
        assert is_expected_client_error(WithStatusCode(429)) is True
        assert is_expected_client_error(WithStatusCode(499)) is True
        assert is_expected_client_error(WithStatusCode(500)) is False
        assert is_expected_client_error(WithStatusCode(399)) is False
        assert is_expected_client_error(WithCode("403")) is True
        assert is_expected_client_error(WithCode("invalid_request_error")) is False
        assert is_expected_client_error(Exception("no status")) is False
        assert is_expected_client_error(None) is False

    def test_provider_originated_4xx_is_not_expected(self):
        """Regression for LIT-6163: a 4xx the provider returned is an upstream or
        deployment problem, so it keeps its traceback; only the proxy's own
        pre-call rejections (no llm_provider) are expected client errors."""
        from litellm.exceptions import AuthenticationError, RateLimitError
        from litellm.litellm_core_utils.core_helpers import is_expected_client_error
        from litellm.llms.anthropic.common_utils import AnthropicError
        from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError

        provider_auth_failure = AuthenticationError(
            message="AnthropicException - API key is invalid.", llm_provider="anthropic", model="claude-haiku-4-5"
        )
        assert is_expected_client_error(provider_auth_failure) is False

        provider_rate_limit = RateLimitError(message="rate limited upstream", llm_provider="openai", model="gpt-4o")
        assert is_expected_client_error(provider_rate_limit) is False

        unmapped_provider_failure = AnthropicError(status_code=401, message='{"type":"authentication_error"}')
        assert is_expected_client_error(unmapped_provider_failure) is False

        proxy_rate_limit = ProxyRateLimitError(
            detail={"error": "Max parallel requests reached"}, model="claude-haiku-4-5", llm_provider="anthropic"
        )
        assert proxy_rate_limit.llm_provider == "anthropic"
        assert is_expected_client_error(proxy_rate_limit) is True

        class RouterRejection(Exception):
            def __init__(self):
                self.status_code = 429
                self.llm_provider = ""

        assert is_expected_client_error(RouterRejection()) is True

    def test_budget_rejection_decorated_with_provider_is_expected(self):
        """The auth handler stamps the requested model's provider onto the proxy's
        own BudgetExceededError before logging it, which must not turn a key-over-budget
        429 into a provider error that keeps its traceback."""
        from litellm.exceptions import BudgetExceededError, RateLimitError, RateLimitErrorCategory
        from litellm.litellm_core_utils.core_helpers import is_expected_client_error

        over_budget = BudgetExceededError(current_cost=0.01, max_budget=0.0, llm_provider="anthropic")
        assert over_budget.llm_provider == "anthropic"
        assert is_expected_client_error(over_budget) is True

        litellm_limit = RateLimitError(
            message="key over rpm", llm_provider="anthropic", model="claude-haiku-4-5",
            category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
        )
        assert is_expected_client_error(litellm_limit) is True

        vendor_limit = RateLimitError(
            message="rate limited upstream", llm_provider="anthropic", model="claude-haiku-4-5",
            category=RateLimitErrorCategory.VENDOR_RATE_LIMIT,
        )
        assert is_expected_client_error(vendor_limit) is False
