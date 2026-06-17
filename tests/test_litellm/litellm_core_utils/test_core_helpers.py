"""Tests for litellm_core_utils.core_helpers module."""

import logging

import httpx
import pytest

from litellm.litellm_core_utils.core_helpers import (
    _FINISH_REASON_MAP,
    LITELLM_INTERNAL_PARAMS,
    RESPONSE_COST_HEADER,
    bind_budget_reservation_to_callbacks,
    budget_reservation_from_metadata,
    drop_params_env_flag,
    drop_params_flag,
    filter_internal_params,
    get_or_create_metadata_bucket,
    get_provider_response_headers_from_hidden_params,
    map_finish_reason,
    normalize_drop_params,
    reconstruct_model_name,
    redact_nested_match_and_regex_keys,
    set_provider_response_headers_in_hidden_params,
    unbind_budget_reservation_from_callbacks,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import (
    ImageResponse,
    TranscriptionResponse,
    all_litellm_params,
)


class TestBudgetReservationBinding:
    """The request-end release skips a reservation a cost callback has claimed, so the claim
    must land on the one dict auth stamped, through whichever metadata field or auth object
    carries it, and a failed call must be able to hand it back."""

    @staticmethod
    def _reservation() -> dict:
        return {"reserved_cost": 0.5, "entries": [], "finalized": False, "callback_bound": False}

    @pytest.mark.parametrize("metadata_variable_name", ["metadata", "litellm_metadata"])
    def test_reservation_stamped_on_the_metadata_is_bound(self, metadata_variable_name: str):
        reservation = self._reservation()

        bind_budget_reservation_to_callbacks({metadata_variable_name: {"user_api_key_budget_reservation": reservation}})

        assert reservation["callback_bound"] is True

    def test_reservation_reachable_only_through_the_auth_object_is_bound(self):
        reservation = self._reservation()
        user_api_key_auth = UserAPIKeyAuth(token="hashed")
        user_api_key_auth.budget_reservation = reservation

        bind_budget_reservation_to_callbacks({"metadata": {"user_api_key_auth": user_api_key_auth}})

        assert reservation["callback_bound"] is True

    def test_reservation_reachable_only_through_a_dumped_auth_object_is_bound(self):
        reservation = self._reservation()

        bind_budget_reservation_to_callbacks({"metadata": {"user_api_key_auth": {"budget_reservation": reservation}}})

        assert reservation["callback_bound"] is True

    def test_unbind_hands_a_claimed_reservation_back(self):
        reservation = self._reservation()
        litellm_params = {"litellm_metadata": {"user_api_key_budget_reservation": reservation}}
        bind_budget_reservation_to_callbacks(litellm_params)

        unbind_budget_reservation_from_callbacks(litellm_params)

        assert reservation["callback_bound"] is False

    def test_request_without_a_reservation_binds_nothing(self):
        metadata = {"user_api_key_auth": UserAPIKeyAuth(token="hashed")}

        bind_budget_reservation_to_callbacks({"metadata": metadata, "litellm_metadata": None})

        assert budget_reservation_from_metadata(metadata) is None
        assert "user_api_key_budget_reservation" not in metadata


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
            ("NO_IMAGE", "content_filter"),
            ("IMAGE_RECITATION", "content_filter"),
            ("IMAGE_OTHER", "content_filter"),
            ("ESCALATION", "content_filter"),
            ("UNEXPECTED_TOOL_CALL", "stop"),
            ("MISSING_THOUGHT_SIGNATURE", "stop"),
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


class TestMapFinishReasonGenericError:
    def test_lowercase_error_is_explicitly_mapped(self):
        assert "error" in _FINISH_REASON_MAP
        assert map_finish_reason("error") == "stop"

    def test_lowercase_error_does_not_warn(self, mocker):
        warn = mocker.patch("litellm.litellm_core_utils.core_helpers.verbose_logger.warning")
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

    def test_redacts_custom_keys_without_changing_default_keys(self):
        payload = {
            "keyword": "secret-keyword",
            "snippet": "secret-snippet",
            "match": "secret-match",
            "regex": "secret-regex",
            "nested": [{"keyword": "nested-keyword", "match": "nested-match"}],
        }

        custom_keys = redact_nested_match_and_regex_keys(payload, keys=("keyword", "snippet"))
        default_keys = redact_nested_match_and_regex_keys(payload)

        assert custom_keys["keyword"] == "[REDACTED]"
        assert custom_keys["snippet"] == "[REDACTED]"
        assert custom_keys["nested"][0]["keyword"] == "[REDACTED]"
        assert custom_keys["match"] == "secret-match"
        assert custom_keys["regex"] == "secret-regex"
        assert default_keys["match"] == "[REDACTED]"
        assert default_keys["regex"] == "[REDACTED]"
        assert default_keys["keyword"] == "secret-keyword"
        assert default_keys["snippet"] == "secret-snippet"


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("True", True),
        (" TRUE ", True),
        ("false", False),
        ("False", False),
        ("yes", True),
        ("off", False),
        ("1", True),
        (1, True),
        (0, False),
        (None, None),
        ("", None),
        ("os.environ/DROP_PARAMS", None),
        ("v2:gcm:not-a-flag", None),
        (2, None),
    ],
)
def test_normalize_drop_params(value, expected):
    assert normalize_drop_params(value) is expected


@pytest.mark.parametrize("value, expected", [("true", True), ("off", False), (None, False)])
def test_drop_params_flag_returns_a_bool_without_a_warning(value, expected, caplog):
    with caplog.at_level(logging.WARNING, logger="drop-params-test"):
        assert drop_params_flag(value, "LITELLM_DROP_PARAMS", logging.getLogger("drop-params-test")) is expected
    assert caplog.text == ""


@pytest.mark.parametrize("value", ["temperature", "ture", 2])
def test_drop_params_flag_treats_non_flag_values_as_off_with_a_warning(value, caplog):
    with caplog.at_level(logging.WARNING, logger="drop-params-test"):
        assert drop_params_flag(value, "LITELLM_DROP_PARAMS", logging.getLogger("drop-params-test")) is False
    assert f"LITELLM_DROP_PARAMS={value!r} is not a flag value, treating it as off" in caplog.text


@pytest.mark.parametrize(
    "environ, expected",
    [
        ({}, False),
        ({"LITELLM_DROP_PARAMS": ""}, False),
        ({"LITELLM_DROP_PARAMS": "   "}, False),
        ({"LITELLM_DROP_PARAMS": "true"}, True),
        ({"LITELLM_DROP_PARAMS": " False "}, False),
        ({"LITELLM_DROP_PARAMS": "0"}, False),
    ],
)
def test_drop_params_env_flag_reads_a_flag_without_a_warning(environ, expected, caplog):
    with caplog.at_level(logging.WARNING, logger="drop-params-test"):
        assert drop_params_env_flag(environ, logging.getLogger("drop-params-test")) is expected
    assert caplog.text == ""


@pytest.mark.parametrize("configured", ["temperature", "temperature,top_p", "enabled"])
def test_drop_params_env_flag_keeps_a_non_flag_value_on_with_a_warning(configured, caplog):
    with caplog.at_level(logging.WARNING, logger="drop-params-test"):
        assert drop_params_env_flag({"LITELLM_DROP_PARAMS": configured}, logging.getLogger("drop-params-test")) is True
    assert (
        f"LITELLM_DROP_PARAMS={configured!r} is not a flag value, treating it as on. Set it to true or false"
        in caplog.text
    )


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
            message="key over rpm",
            llm_provider="anthropic",
            model="claude-haiku-4-5",
            category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
        )
        assert is_expected_client_error(litellm_limit) is True

        vendor_limit = RateLimitError(
            message="rate limited upstream",
            llm_provider="anthropic",
            model="claude-haiku-4-5",
            category=RateLimitErrorCategory.VENDOR_RATE_LIMIT,
        )
        assert is_expected_client_error(vendor_limit) is False


class TestProviderResponseHeadersInHiddenParams:
    def test_records_raw_headers_and_the_processed_additional_headers(self):
        response = ImageResponse()
        response._hidden_params = {"additional_headers": {RESPONSE_COST_HEADER: 0.04}}

        set_provider_response_headers_in_hidden_params(
            response, httpx.Headers({"X-Request-Id": "req_img", "x-ratelimit-remaining-requests": "41"})
        )

        assert response._hidden_params["headers"] == {
            "x-request-id": "req_img",
            "x-ratelimit-remaining-requests": "41",
        }
        additional_headers = response._hidden_params["additional_headers"]
        assert additional_headers["llm_provider-x-request-id"] == "req_img"
        assert additional_headers["x-ratelimit-remaining-requests"] == "41"
        assert additional_headers[RESPONSE_COST_HEADER] == 0.04

    def test_litellm_owned_additional_headers_win_over_provider_headers(self):
        response = TranscriptionResponse(text="hi")
        response._hidden_params = {"additional_headers": {"llm_provider-x-request-id": "kept"}}

        set_provider_response_headers_in_hidden_params(response, {"x-request-id": "provider"})

        assert response._hidden_params["additional_headers"]["llm_provider-x-request-id"] == "kept"
        assert response._hidden_params["headers"] == {"x-request-id": "provider"}

    def test_getter_returns_the_recorded_headers(self):
        response = ImageResponse()

        set_provider_response_headers_in_hidden_params(response, {"x-request-id": "req_img"})

        assert get_provider_response_headers_from_hidden_params(response) == {"x-request-id": "req_img"}

    @pytest.mark.parametrize(
        "hidden_params",
        [
            None,
            "headers",
            {"additional_headers": {}},
            {"headers": "x-request-id: req_img"},
            {"headers": {"x-request-id": 7}},
        ],
    )
    def test_getter_returns_none_without_a_string_header_mapping(self, hidden_params):
        response = ImageResponse()
        response._hidden_params = hidden_params

        assert get_provider_response_headers_from_hidden_params(response) is None

    def test_getter_returns_none_for_an_object_without_hidden_params(self):
        assert get_provider_response_headers_from_hidden_params(object()) is None

    def test_headers_never_leak_into_a_sibling_response(self):
        recorded = TranscriptionResponse()
        sibling = TranscriptionResponse()

        set_provider_response_headers_in_hidden_params(recorded, {"x-request-id": "req_stt"})

        assert get_provider_response_headers_from_hidden_params(sibling) is None
        assert "additional_headers" not in sibling._hidden_params


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
        baseline = set(LITELLM_INTERNAL_PARAMS)
        filter_internal_params({"x": 1}, additional_internal_params={"adhoc_key"})
        assert LITELLM_INTERNAL_PARAMS == baseline

    def test_non_dict_passes_through(self):
        assert filter_internal_params("not-a-dict") == "not-a-dict"
        assert filter_internal_params([1, 2, 3]) == [1, 2, 3]

    def test_existing_mcp_keys_still_filtered(self):
        out = filter_internal_params({"skip_mcp_handler": True, "mcp_handler_context": {}, "model": "gpt-4"})
        assert out == {"model": "gpt-4"}
