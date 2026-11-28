import functools
import json
import traceback
from collections.abc import Awaitable, Callable
from types import MappingProxyType
from typing import Any, Final, Literal, Optional, Union
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from fastapi import HTTPException
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm import completion
from litellm.anthropic_interface.exceptions.exception_mapping_utils import AnthropicExceptionMapping
from litellm.exceptions import GuardrailRaisedException
from litellm.litellm_core_utils.exception_mapping_utils import (
    ExceptionCheckers,
    _get_body_error_code,
    _get_response_headers,
    exception_type,
    extract_and_raise_litellm_exception,
    get_error_message,
    provider_origin_status,
)
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.github_copilot.authenticator import DEFAULT_GITHUB_CLIENT_ID
from litellm.llms.github_copilot.github_host import github_host_endpoints
from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession
from litellm.llms.groq.chat.transformation import GroqChatCompletionStreamingHandler
from litellm.llms.litellm_proxy.responses.transformation import LiteLLMProxyResponsesAPIConfig
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.proxy.common_utils.openai_error_payload import (
    openai_error_param,
    openai_error_type,
    relayed_provider_specific_fields,
)
from litellm.proxy.common_utils.responses_stream_errors import ResponsesStreamErrorState
from litellm.types.utils import LlmProviders


@pytest.mark.parametrize("shape", ["sdk", "http"])
@pytest.mark.parametrize("code", ["cyber_policy", "invalid_prompt"])
def test_openai_cyber_policy_uses_the_structured_error_code(shape: str, code: str) -> None:
    body: Final = {"code": code, "message": "Request mentions cyber_policy", "type": "invalid_request_error"}
    request: Final = httpx.Request("POST", "https://provider.example/v1/responses")
    response: Final = httpx.Response(400, request=request, headers={"x-request-id": "cyber-policy-request"})
    original: Final = OpenAIError(
        status_code=400,
        message=json.dumps({"error": body}),
        request=request,
        response=response,
        body=body if shape == "sdk" else None,
    )
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(
            model="gpt-6-astra",
            original_exception=original,
            custom_llm_provider="openai",
            completion_kwargs={},
            extra_kwargs={},
        )
    assert isinstance(caught.value, litellm.ContentPolicyViolationError) == (code == "cyber_policy")
    if code == "cyber_policy":
        assert caught.value.code == code
        assert caught.value.body == body
        assert caught.value.status_code == 400
        assert caught.value.response.status_code == 400
        assert caught.value.request.url == request.url
        assert caught.value.response.headers["x-request-id"] == "cyber-policy-request"


def _debug_info_for_api_base(configured_api_base: str) -> str:
    request: Final = httpx.Request("POST", "https://gateway.example/v1/chat/completions")
    original: Final = OpenAIError(
        status_code=500, message="upstream failed", request=request, response=httpx.Response(500, request=request)
    )
    with pytest.raises(litellm.InternalServerError) as caught:
        exception_type(
            model="gpt-5.4-mini",
            original_exception=original,
            custom_llm_provider="openai",
            completion_kwargs={},
            extra_kwargs={"api_base": configured_api_base},
        )
    debug_info: Final = caught.value.litellm_debug_info
    assert isinstance(debug_info, str)
    return debug_info


@pytest.mark.parametrize(
    ("configured_api_base", "shown_api_base"),
    [
        ("https://qa-user:function-secret@gateway.example/v1?code=function-secret", "https://gateway.example/v1"),
        ("https://gateway.example/v1#function-secret", "https://gateway.example/v1"),
        ("qa-user:function-secret@gateway.example:8080/v1", "gateway.example:8080/v1"),
        ("https://gateway.example?code=function/secret", "https://gateway.example"),
        ("https://gateway.example?user=alice@example.org&code=function-secret", "https://gateway.example"),
        ("https://gateway.example?email=ops@other-host.example", "https://gateway.example"),
        ("qa-user:function-secret@gateway.example/v1?next=https://other.example/y", "gateway.example/v1"),
        (
            "https://qa-user:function-secret@gateway.example/models/claude-opus-5-5@default",
            "https://gateway.example/models/claude-opus-5-5@default",
        ),
    ],
    ids=[
        "userinfo_and_query",
        "fragment",
        "no_scheme",
        "slash_in_query_without_path",
        "at_sign_in_query_without_path",
        "at_sign_names_other_host",
        "no_scheme_with_url_in_query",
        "userinfo_and_at_sign_in_path",
    ],
)
def test_api_base_line_keeps_credentials_out(configured_api_base: str, shown_api_base: str) -> None:
    debug_info: Final = _debug_info_for_api_base(configured_api_base)
    assert f"API Base: `{shown_api_base}`" in debug_info
    assert "function-secret" not in debug_info
    assert "secret" not in debug_info.split("API Base:", 1)[1].split("\n", 1)[0]


@pytest.mark.parametrize(
    "configured_api_base",
    [
        "https://qa-user:qa?function-secret@gateway.example/v1",
        "https://qa-user:qa#function-secret@gateway.example/v1",
        "https://qa-user:qa/function-secret@gateway.example/v1",
        "https://qa-user:qa/function?secret@gateway.example/v1",
        "https://qa.user:qa/function-secret@gateway.example/v1",
        "https://qa-user:1234/function-secret@gateway.example/v1",
        "https://function-secret/more@gateway.example/v1",
        "qa-user:qa://function-secret@gateway.example",
    ],
    ids=[
        "question_mark_in_password",
        "hash_in_password",
        "slash_in_password",
        "slash_and_question_mark_in_password",
        "dotted_username_and_slash_in_password",
        "numeric_password_with_slash",
        "slash_in_username",
        "no_scheme_with_scheme_separator_in_password",
    ],
)
def test_api_base_line_is_left_out_when_a_password_breaks_the_url(configured_api_base: str) -> None:
    debug_info: Final = _debug_info_for_api_base(configured_api_base)
    assert "API Base:" not in debug_info
    assert not any(part in debug_info for part in ("qa-user", "qa.user", "function", "secret"))


@pytest.mark.parametrize(
    "api_base",
    [
        "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5"
        "/publishers/anthropic/models/claude-opus-5-5@default:rawPredict",
        "us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1"
        "/publishers/mistralai/models/mistral-large@2407:rawPredict",
        "https://gateway.ai.cloudflare.com/v1/account/gateway/workers-ai/@cf/meta/llama-3.1-8b-instruct",
        "http://localhost:4000/v1/models/claude@default",
        "https://[::1]:8080/v1/models/claude@default",
    ],
    ids=["vertex_partner_model", "vertex_without_scheme", "workers_ai_path", "localhost", "ipv6"],
)
def test_api_base_line_keeps_an_at_sign_in_the_path(api_base: str) -> None:
    assert f"API Base: `{api_base}`" in _debug_info_for_api_base(api_base)


@pytest.mark.parametrize(
    ("provider_error", "mapped_class"),
    [
        (
            httpx.ReadError("peer closed connection", request=httpx.Request("POST", "https://relay.example/v1/messages")),
            litellm.APIConnectionError,
        ),
        (RuntimeError("stream decoder failed"), litellm.APIConnectionError),
        (AnthropicError(status_code=500, message="[Errno 104] Connection reset by peer"), litellm.InternalServerError),
        (AnthropicError(status_code=429, message="rate_limit_error"), litellm.RateLimitError),
    ],
    ids=["unmapped_with_request", "unmapped_without_request", "anthropic_500", "anthropic_429"],
)
def test_mapped_provider_error_carries_the_deployment_debug_info(
    provider_error: Exception, mapped_class: type[Exception]
) -> None:
    with pytest.raises(mapped_class) as caught:
        exception_type(
            model="claude-opus-5-5",
            original_exception=provider_error,
            custom_llm_provider="anthropic",
            completion_kwargs={"messages": [{"role": "user", "content": "CONFIDENTIAL patient record"}]},
            extra_kwargs={
                "api_base": "https://relay.example",
                "metadata": {"model_group": "claude-alias", "deployment": "anthropic/claude-opus-5-5"},
            },
        )
    debug_info: Final = caught.value.litellm_debug_info
    assert isinstance(debug_info, str)
    assert "API Base: `https://relay.example`" in debug_info
    assert "model_group: `claude-alias`" in debug_info
    assert "deployment: `anthropic/claude-opus-5-5`" in debug_info
    assert "CONFIDENTIAL" not in debug_info


_ROUTER_METADATA: Final = {
    "api_base": "https://relay.example/v1",
    "model_group": "claude-alias",
    "deployment": "anthropic/claude-haiku-5-5",
}


@pytest.mark.parametrize(
    ("provider_error", "custom_llm_provider", "extra_kwargs"),
    [
        (
            AnthropicError(status_code=529, message="Overloaded"),
            "anthropic",
            {"litellm_metadata": _ROUTER_METADATA},
        ),
        (
            OpenAIError(status_code=500, message="upstream failed"),
            "hosted_vllm",
            {
                "api_base": "https://relay.example/v1",
                "metadata": {"user_id": "set by the caller"},
                "litellm_metadata": _ROUTER_METADATA,
            },
        ),
    ],
    ids=["native_messages_without_api_base", "bridged_responses_with_request_metadata"],
)
def test_router_metadata_under_litellm_metadata_names_the_deployment(
    provider_error: Exception, custom_llm_provider: str, extra_kwargs: dict[str, object]
) -> None:
    with pytest.raises(litellm.InternalServerError) as caught:
        exception_type(
            model="claude-haiku-5-5",
            original_exception=provider_error,
            custom_llm_provider=custom_llm_provider,
            extra_kwargs=extra_kwargs,
        )
    debug_info: Final = caught.value.litellm_debug_info
    assert isinstance(debug_info, str)
    assert "API Base: `https://relay.example/v1`" in debug_info
    assert "model_group: `claude-alias`" in debug_info
    assert "deployment: `anthropic/claude-haiku-5-5`" in debug_info


_KEY_METADATA: Final = {"user_api_key_alias": "alias-key", "user_api_key_team_alias": "team-a"}


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {"metadata": _ROUTER_METADATA, "litellm_metadata": _KEY_METADATA},
        {"metadata": _KEY_METADATA, "litellm_metadata": _ROUTER_METADATA},
    ],
    ids=["chat_with_guardrail_litellm_metadata", "generic_route_with_key_fields_in_metadata"],
)
def test_debug_info_reads_lines_from_both_metadata_buckets(extra_kwargs: dict[str, object]) -> None:
    with pytest.raises(litellm.InternalServerError) as caught:
        exception_type(
            model="claude-haiku-5-5",
            original_exception=AnthropicError(status_code=529, message="Overloaded"),
            custom_llm_provider="anthropic",
            extra_kwargs=extra_kwargs,
        )
    debug_info: Final = caught.value.litellm_debug_info
    assert isinstance(debug_info, str)
    assert "Key Name: `alias-key`" in debug_info
    assert "Team: `team-a`" in debug_info
    assert "API Base: `https://relay.example/v1`" in debug_info
    assert "model_group: `claude-alias`" in debug_info
    assert "deployment: `anthropic/claude-haiku-5-5`" in debug_info


def test_litellm_metadata_wins_over_caller_metadata_for_the_same_key() -> None:
    with pytest.raises(litellm.InternalServerError) as caught:
        exception_type(
            model="claude-haiku-5-5",
            original_exception=AnthropicError(status_code=529, message="Overloaded"),
            custom_llm_provider="anthropic",
            extra_kwargs={
                "metadata": {"deployment": "set by the caller", "api_base": "https://caller.example"},
                "litellm_metadata": _ROUTER_METADATA,
            },
        )
    debug_info: Final = caught.value.litellm_debug_info
    assert isinstance(debug_info, str)
    assert "deployment: `anthropic/claude-haiku-5-5`" in debug_info
    assert "API Base: `https://relay.example/v1`" in debug_info
    assert "caller" not in debug_info


def test_mapper_that_passes_its_debug_info_keeps_the_prompt_line() -> None:
    request: Final = httpx.Request("POST", "https://gateway.example/v1/chat/completions")
    original: Final = OpenAIError(
        status_code=500, message="upstream failed", request=request, response=httpx.Response(500, request=request)
    )
    with pytest.raises(litellm.InternalServerError) as caught:
        exception_type(
            model="gpt-5.4-mini",
            original_exception=original,
            custom_llm_provider="openai",
            completion_kwargs={"messages": [{"role": "user", "content": "hello there"}]},
            extra_kwargs={"metadata": {"deployment": "openai/gpt-5.4-mini"}},
        )
    debug_info: Final = caught.value.litellm_debug_info
    assert isinstance(debug_info, str)
    assert "Messages: `[{'role': 'user', 'content': 'hello there'}]`" in debug_info
    assert "deployment: `openai/gpt-5.4-mini`" in debug_info


def test_debug_info_has_no_messages_line_when_the_request_has_no_messages() -> None:
    assert "Messages" not in _debug_info_for_api_base("https://gateway.example/v1")


@pytest.mark.parametrize(
    "already_mapped",
    [
        litellm.InternalServerError(
            message="Upstream generation aborted (finish_reason=abort)", model="gpt-5.4-mini", llm_provider="openai"
        ),
        litellm.Timeout(
            message="Stream exceeded max streaming duration of 10s", model="gpt-5.4-mini", llm_provider="openai"
        ),
    ],
    ids=["finish_reason_abort", "max_streaming_duration"],
)
def test_already_mapped_error_gains_the_deployment_debug_info(already_mapped: Exception) -> None:
    returned: Final = exception_type(
        model="gpt-5.4-mini",
        original_exception=already_mapped,
        custom_llm_provider="openai",
        completion_kwargs={"messages": [{"role": "user", "content": "CONFIDENTIAL patient record"}]},
        extra_kwargs={
            "api_base": "https://relay.example/v1",
            "metadata": {"model_group": "gpt-alias", "deployment": "openai/gpt-5.4-mini"},
        },
    )
    assert returned is already_mapped
    debug_info: Final = returned.litellm_debug_info
    assert isinstance(debug_info, str)
    assert "API Base: `https://relay.example/v1`" in debug_info
    assert "deployment: `openai/gpt-5.4-mini`" in debug_info
    assert "CONFIDENTIAL" not in debug_info


def test_error_instance_raised_by_several_requests_names_none_of_them() -> None:
    shared: Final = litellm.RateLimitError(message="mock rate limit", llm_provider="openai", model="gpt-5.4-mini")

    def debug_info_for(alias: str) -> object:
        with pytest.raises(litellm.RateLimitError) as caught:
            completion(
                model="openai/gpt-5.4-mini",
                messages=[{"role": "user", "content": "hi"}],
                mock_response=shared,
                metadata={"user_api_key_alias": alias, "model_group": f"group-{alias}"},
            )
        assert caught.value is shared
        return caught.value.litellm_debug_info

    first: Final = debug_info_for("alias-a")
    assert isinstance(first, str) and "Key Name: `alias-a`" in first
    assert debug_info_for("alias-b") is None
    assert debug_info_for("alias-c") is None


_RELAY_DEPLOYMENT_KWARGS: Final = {
    "api_base": "https://relay.example/v1",
    "metadata": {"deployment": "openai/gpt-5.4-mini"},
}


@pytest.mark.parametrize(
    ("first_kwargs", "second_kwargs"),
    [
        ({"litellm_call_id": "call-1", **_RELAY_DEPLOYMENT_KWARGS}, {}),
        (_RELAY_DEPLOYMENT_KWARGS, {"litellm_call_id": "call-1"}),
    ],
    ids=["second-mapping-has-no-request", "first-mapping-had-no-call-id"],
)
def test_mapping_again_for_the_same_request_keeps_the_deployment_already_named(
    first_kwargs: dict[str, object], second_kwargs: dict[str, object]
) -> None:
    already_mapped: Final = litellm.InternalServerError(
        message="Upstream generation aborted (finish_reason=abort)", model="gpt-5.4-mini", llm_provider="openai"
    )
    exception_type(
        model="gpt-5.4-mini",
        original_exception=already_mapped,
        custom_llm_provider="openai",
        extra_kwargs=first_kwargs,
    )
    exception_type(
        model="gpt-5.4-mini",
        original_exception=already_mapped,
        custom_llm_provider="openai",
        completion_kwargs={},
        extra_kwargs=second_kwargs,
    )
    debug_info: Final = already_mapped.litellm_debug_info
    assert isinstance(debug_info, str)
    assert "API Base: `https://relay.example/v1`" in debug_info
    assert "deployment: `openai/gpt-5.4-mini`" in debug_info


class _ErrorWhoseMessageCannotBeRead(Exception):
    @property
    def message(self) -> str:
        raise RuntimeError("message unavailable")


def test_error_that_breaks_the_mapping_still_names_the_deployment() -> None:
    with pytest.raises(litellm.APIConnectionError) as caught:
        exception_type(
            model="claude-opus-5-5",
            original_exception=_ErrorWhoseMessageCannotBeRead("upstream failed"),
            custom_llm_provider="anthropic",
            completion_kwargs={},
            extra_kwargs={"api_base": "https://relay.example", "metadata": {"deployment": "anthropic/claude-opus-5-5"}},
        )
    debug_info: Final = caught.value.litellm_debug_info
    assert isinstance(debug_info, str)
    assert "API Base: `https://relay.example`" in debug_info
    assert "deployment: `anthropic/claude-opus-5-5`" in debug_info


# Test cases for is_error_str_context_window_exceeded
# Tuple format: (error_message, expected_result)
context_window_test_cases = [
    # Positive cases (should return True)
    (
        '"{\n  "error": {\n    "message": "Your input exceeds the context window of this model. '
        'Please adjust your input and try again.",\n    "type": "invalid_request_error",\n'
        '    "param": "input",\n    "code": "context_length_exceeded"\n  }\n}"',
        True,
    ),
    (
        "An error occurred: The input exceeds the model's maximum context limit of 8192 tokens.",
        True,
    ),
    (
        "Some text before, this model's maximum context length is 4096 tokens. Some text after.",
        True,
    ),
    (
        "Validation Error: string too long. expected a string with maximum length 1000.",
        True,
    ),
    ("Your prompt is longer than the model's context length of 2048.", True),
    ("AWS Bedrock Error: The request payload size has exceed context limit.", True),
    (
        "Input tokens exceed the configured limit of 272000 tokens. Your messages resulted in 509178 tokens. Please reduce the length of the messages.",
        True,
    ),
    (
        "`inputs` tokens + `max_new_tokens` must be <= 4096",
        True,
    ),
    (
        "request (67311 tokens) exceeds the available context size (65536 tokens), try increasing it",
        True,
    ),
    # Gemini 2.5/3 format
    (
        "The input token count exceeds the maximum number of tokens allowed 1048576.",
        True,
    ),
    (
        'GeminiException BadRequestError - {\n  "error": {\n    "code": 400,\n    "message": "The input token count exceeds the maximum number of tokens allowed 1048576.",\n    "status": "INVALID_ARGUMENT"\n  }\n}\n',
        True,
    ),
    # Gemini 2.0 Flash format (includes input token count in message)
    (
        "The input token count (2800010) exceeds the maximum number of tokens allowed (1048575).",
        True,
    ),
    (
        'GeminiException BadRequestError - {\n  "error": {\n    "code": 400,\n    "message": "The input token count (2800010) exceeds the maximum number of tokens allowed (1048575).",\n    "status": "INVALID_ARGUMENT"\n  }\n}\n',
        True,
    ),
    # Test case insensitivity
    ("ERROR: THIS MODEL'S MAXIMUM CONTEXT LENGTH IS 1024.", True),
    # Cerebras context window error format
    # See: https://github.com/BerriAI/litellm/issues/XXXX
    (
        "Current length is 132784 while limit is 131000",
        True,
    ),
    (
        "CerebrasException - Please reduce the length of the messages or completion. Current length is 50000 while limit is 40000",
        True,
    ),
    (
        "Invalid 'input[0]': maximum input length is 8192 tokens.",
        True,
    ),
    (
        "OpenAIException - Error code: 400 - {'error': {'message': \"Invalid 'input[0]': maximum input length is 8192 tokens.\", 'type': 'invalid_request_error'}}",
        True,
    ),
    (
        "Invalid 'metadata': maximum input length is 512 characters.",
        False,
    ),
    (
        '{"error":{"message":"Input too long: 150055 input tokens, limit is 131072 for this model","code":"upstream_rejected"}}',
        True,
    ),
    (
        "Input should be 'error' or 'truncate', field: 'context_length_exceeded_behavior', value: 'foo'",
        False,
    ),
    # Negative cases (should return False)
    ("A generic API error occurred.", False),
    ("Invalid API Key provided.", False),
    ("Rate limit reached for requests.", False),
    ("The context is large, but acceptable.", False),
    ("", False),  # Empty string
    # OpenAI user param length validation - not a context window error
    (
        "Invalid 'user': string too long. Expected a string with maximum length 64, but got a string with length 123 instead.",
        False,
    ),
    (
        '{"error": {"message": "Invalid \'user\': string too long.", "code": "string_above_max_length"}}',
        False,
    ),
]


@pytest.mark.parametrize("error_str, expected", context_window_test_cases)
def test_is_error_str_context_window_exceeded(error_str, expected):
    """
    Tests the is_error_str_context_window_exceeded function with various error strings.
    """
    assert ExceptionCheckers.is_error_str_context_window_exceeded(error_str) == expected


class TestExceptionCheckers:
    """Test the ExceptionCheckers utility methods"""

    def test_is_error_str_rate_limit_ignores_embedded_numbers(self):
        """An arbitrary 429 inside user-provided payload must not trigger rate-limit detection"""

        error_str = "Invalid user message={'role': 'user', 'content': [{'text': 'payload429snippet'}]}"
        result = ExceptionCheckers.is_error_str_rate_limit(error_str)
        assert result is False

    def test_is_error_str_rate_limit_detects_true_rate_limit(self):
        """A real rate-limit error string should still be detected"""

        error_str = "RateLimitError: OpenAIException - You exceeded your current quota. (status code 429)"
        result = ExceptionCheckers.is_error_str_rate_limit(error_str)
        assert result is True

    def test_bare_429_in_body_is_ignored_when_status_code_says_otherwise(self):
        """A 429 echoed back inside a 400's body is not a rate limit.

        Word boundaries don't help: 429 is an ordinary token id (" that" in several
        tokenisers), so an echoed prompt_token_ids array reads as a standalone 429.
        """
        error_str = (
            '{"error":{"message":"`tools` must not be an empty array",'
            '"type":"invalid_request_error"},'
            '"prompt_token_ids":[9906,429,1234]}'
        )
        assert ExceptionCheckers.is_error_str_rate_limit(error_str, status_code=400) is False

    def test_bare_429_still_detected_without_a_status_code(self):
        """With no status available, a standalone 429 still counts (unchanged behaviour)."""

        assert ExceptionCheckers.is_error_str_rate_limit("HTTP 429 Too Many Requests") is True
        assert ExceptionCheckers.is_error_str_rate_limit("HTTP 429 Too Many Requests", status_code=None) is True
        assert ExceptionCheckers.is_error_str_rate_limit("HTTP 429 Too Many Requests", status_code=429) is True

    def test_non_integer_status_code_does_not_suppress_bare_429(self):
        """A non-integer status counts as unknown, not as a contradiction."""

        assert ExceptionCheckers.is_error_str_rate_limit("HTTP 429 Too Many Requests", status_code="not-an-int") is True

    def test_rate_limit_phrase_is_honoured_under_a_non_429_status(self):
        """Phrase matching stays ungated: some providers report a real rate limit in
        the text under a non-429 status (#11455)."""

        assert (
            ExceptionCheckers.is_error_str_rate_limit("FireworksException - rate limit exceeded", status_code=400)
            is True
        )

    def test_is_azure_content_policy_violation_error_with_policy_violation_text(self):
        """Test detection of Azure content policy violation with explicit policy violation text"""

        error_strings = [
            "invalid_request_error content_policy_violation occurred",
            "The response was filtered due to the prompt triggering Azure OpenAI's content management policy",
            "Your task failed as a result of our safety system detecting harmful content",
            "The model produced invalid content that violates our policy",
            "Request blocked due to content_filter_policy restrictions",
        ]

        for error_str in error_strings:
            result = ExceptionCheckers.is_azure_content_policy_violation_error(error_str)
            assert result is True, f"Should detect policy violation in: {error_str}"

    def test_is_azure_content_policy_violation_error_case_insensitive(self):
        """Test that content policy violation detection is case insensitive"""

        error_strings = [
            "INVALID_REQUEST_ERROR CONTENT_POLICY_VIOLATION",
            "The Response Was Filtered Due To The Prompt Triggering Azure OpenAI's Content Management",
            "YOUR TASK FAILED AS A RESULT OF OUR SAFETY SYSTEM",
            "Content_Filter_Policy restriction detected",
        ]

        for error_str in error_strings:
            result = ExceptionCheckers.is_azure_content_policy_violation_error(error_str)
            assert result is True, f"Should detect policy violation in uppercase: {error_str}"

    def test_is_azure_content_policy_violation_error_with_non_policy_errors(self):
        """Test that non-policy violation errors are not detected as policy violations"""

        error_strings = [
            "Invalid API key provided",
            "Rate limit exceeded for current model",
            "Model not found: gpt-nonexistent",
            "Request timeout occurred",
            "Authentication failed",
            "Insufficient quota remaining",
            "Bad request format",
            "Internal server error occurred",
        ]

        for error_str in error_strings:
            result = ExceptionCheckers.is_azure_content_policy_violation_error(error_str)
            assert result is False, f"Should NOT detect policy violation in: {error_str}"

    def test_is_azure_content_policy_violation_error_with_partial_matches(self):
        """Test that partial keyword matches work correctly"""

        # These should match because they contain the required substrings
        positive_cases = [
            "Error: content_policy_violation detected in request",
            "Safety content management, your task failed as a result of our safety system",
            "the model produced invalid content",
        ]

        for error_str in positive_cases:
            result = ExceptionCheckers.is_azure_content_policy_violation_error(error_str)
            assert result is True, f"Should detect policy violation in: {error_str}"

        # These should not match even though they contain similar words
        negative_cases = [
            "Invalid content format in request",  # "invalid" but not "invalid content"
            "Policy configuration error",  # "policy" but not policy violation context
            "Content type not supported",  # "content" but not content filter context
            "Management API unavailable",  # "management" but not content management context
        ]

        for error_str in negative_cases:
            result = ExceptionCheckers.is_azure_content_policy_violation_error(error_str)
            assert result is False, f"Should NOT detect policy violation in: {error_str}"


gemini_context_window_test_cases = [
    # Gemini 2.0 Flash format (includes input token count in message)
    (
        "The input token count (2800010) exceeds the maximum number of tokens allowed (1048575).",
        True,
    ),
    # Gemini 2.5/3 format
    (
        "The input token count exceeds the maximum number of tokens allowed (1048576).",
        True,
    ),
    ("A generic error occurred.", False),
]


@pytest.mark.parametrize("error_message, should_raise_context_window", gemini_context_window_test_cases)
def test_gemini_context_window_error_mapping(error_message, should_raise_context_window):
    """
    Tests that the exception_type function correctly maps Gemini's
    context window exceeded errors to litellm.ContextWindowExceededError.
    """
    model = "gemini/gemini-2.0-flash"
    custom_llm_provider = "gemini"

    # Create a generic exception with the specific error message
    original_exception = Exception(error_message)

    if should_raise_context_window:
        with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
            exception_type(
                model=model,
                original_exception=original_exception,
                custom_llm_provider=custom_llm_provider,
            )
        # Check if the raised exception is indeed a ContextWindowExceededError
        assert isinstance(excinfo.value, litellm.ContextWindowExceededError)
    else:
        # For the negative case, we expect it to raise a generic APIConnectionError
        with pytest.raises(litellm.APIConnectionError):
            exception_type(
                model=model,
                original_exception=original_exception,
                custom_llm_provider=custom_llm_provider,
            )


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    ("error_message", "expected_type"),
    [
        (
            (
                "Requested token count exceeds the model's maximum context length of 131072 tokens. "
                "You requested a total of 140000 tokens: 130000 tokens from the input messages and "
                "10000 tokens for the completion. Please reduce the number of tokens in the input "
                "messages or the completion to fit within the limit."
            ),
            litellm.ContextWindowExceededError,
        ),
        (
            (
                "requested token count exceeds the model's maximum context length of 8192 tokens. "
                "you requested a total of 9000 tokens"
            ),
            litellm.ContextWindowExceededError,
        ),
        ("max_tokens must be a positive integer", litellm.BadRequestError),
    ],
    ids=["sglang-context-overflow", "lowercase-context-overflow", "unrelated-bad-request"],
)
@pytest.mark.asyncio
async def test_sglang_context_window_error_mapping(
    sync_mode: bool,
    streaming: bool,
    error_message: str,
    expected_type: type[litellm.BadRequestError],
) -> None:
    def reject_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            request=request,
            json={"object": "error", "message": error_message, "type": "BadRequestError", "param": None, "code": 400},
        )

    transport: Final = httpx.MockTransport(reject_request)
    if sync_mode:
        with (
            httpx.Client(transport=transport) as http_client,
            OpenAI(api_key="sk-test", max_retries=0, http_client=http_client) as client,
            pytest.raises(litellm.BadRequestError) as caught,
        ):
            _call_openai_api_sync("openai/sglang-model", "chat_completion", streaming, client)
    else:
        async with (
            httpx.AsyncClient(transport=transport) as async_http_client,
            AsyncOpenAI(api_key="sk-test", max_retries=0, http_client=async_http_client) as async_client,
        ):
            with pytest.raises(litellm.BadRequestError) as caught:
                await _call_openai_api_async("openai/sglang-model", "chat_completion", streaming, async_client)

    assert type(caught.value) is expected_type
    assert caught.value.status_code == 400
    assert error_message in str(caught.value)
    assert caught.value.llm_provider == "openai"


def test_lemonade_context_window_error_mapping():
    """Lemonade's llama.cpp backend should map context overflows to LiteLLM's standard error."""

    model = "lemonade/Qwen3.6-35B-A3B-GGUF"
    error_message = (
        '{"error":{"code":"context_length_exceeded","message":"request '
        "(80010 tokens) exceeds the available context size (65536 tokens), "
        'try increasing it","status_code":400,"type":"invalid_request_error"}}'
    )
    original_exception = OpenAIError(
        status_code=400,
        message=error_message,
        headers={},
    )

    with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
        exception_type(
            model=model,
            original_exception=original_exception,
            custom_llm_provider="lemonade",
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.llm_provider == "lemonade"
    assert excinfo.value.model == model


def test_openai_compatible_400_with_bare_429_in_body_maps_to_bad_request():
    """A provider 400 whose echoed body contains a 429 must stay a 400.

    ``is_error_str_rate_limit`` runs before the status-code branch for
    openai-compatible providers, so a validation error echoing the request back came
    out as RateLimitError, which tells the caller to retry a request that cannot
    succeed and books the failure against provider throttling.
    """
    error_message = (
        '{"error":{"message":"`tools` must not be an empty array",'
        '"type":"invalid_request_error","code":400},'
        '"prompt_token_ids":[9906,429,1234]}'
    )
    original_exception = OpenAIError(
        status_code=400,
        message=error_message,
        headers={},
    )

    with pytest.raises(litellm.BadRequestError) as excinfo:
        exception_type(
            model="deepseek-ai/DeepSeek-V3",
            original_exception=original_exception,
            custom_llm_provider="deepinfra",
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.llm_provider == "deepinfra"


def test_openai_compatible_429_still_maps_to_rate_limit():
    """A real 429 still maps to RateLimitError."""
    original_exception = OpenAIError(
        status_code=429,
        message='{"error":{"message":"Too Many Requests","type":"rate_limit_error"}}',
        headers={},
    )

    with pytest.raises(litellm.RateLimitError) as excinfo:
        exception_type(
            model="deepseek-ai/DeepSeek-V3",
            original_exception=original_exception,
            custom_llm_provider="deepinfra",
        )

    assert excinfo.value.status_code == 429


@pytest.mark.parametrize(
    "error_message",
    [
        "AnthropicException - prompt is too long: 250000 tokens > 200000 maximum",
        "AnthropicException - input length and max_tokens exceed context limit: "
        "200000 + 8000 > 200000, decrease input length or max_tokens and try again",
    ],
)
def test_anthropic_context_window_error_mapping(error_message):
    """Anthropic context-window overflows (input too long, or input + max_tokens
    over the context limit) must map to ContextWindowExceededError (400) even when
    the upstream exception carries no ``status_code`` attribute. Previously only
    "prompt is too long" was special-cased, so the "exceed context limit" phrasing
    fell through to a generic APIConnectionError (500)."""
    original_exception = Exception(error_message)

    with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
        exception_type(
            model="claude-sonnet-4-5",
            original_exception=original_exception,
            custom_llm_provider="anthropic",
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.llm_provider == "anthropic"


@pytest.mark.parametrize("upstream_status_code", [400, 500])
def test_vertex_ai_anthropic_prompt_too_long_maps_to_context_window_exceeded(upstream_status_code: int) -> None:
    original_exception: Final = VertexAIError(
        status_code=upstream_status_code,
        message='{"type":"error","error":{"type":"invalid_request_error",'
        '"message":"prompt is too long: 210123 tokens > 200000 maximum"}}',
    )

    with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
        exception_type(
            model="claude-opus-5-5",
            original_exception=original_exception,
            custom_llm_provider="vertex_ai",
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.llm_provider == "vertex_ai"
    assert excinfo.value.model == "claude-opus-5-5"


# Test cases for Vertex AI RateLimitError mapping
# As per https://github.com/BerriAI/litellm/issues/16189
vertex_rate_limit_test_cases = [
    ("429 Quota exceeded for model", True),
    ("Resource exhausted. Please try again later.", True),
    (
        "429 Unable to submit request because the service is temporarily out of capacity.",
        True,
    ),
    ("A generic error occurred.", False),  # Negative case
]


@pytest.mark.parametrize("error_message, should_raise_rate_limit", vertex_rate_limit_test_cases)
def test_vertex_ai_rate_limit_error_mapping(error_message, should_raise_rate_limit):
    """
    Tests that the exception_type function correctly maps Vertex AI's
    "Resource exhausted" error to a litellm.RateLimitError.
    """
    model = "gemini/gemini-2.5-flash"
    custom_llm_provider = "vertex_ai"

    # Create a generic exception with the specific error message
    original_exception = Exception(error_message)

    if should_raise_rate_limit:
        with pytest.raises(litellm.RateLimitError) as excinfo:
            exception_type(
                model=model,
                original_exception=original_exception,
                custom_llm_provider=custom_llm_provider,
            )
        # Check if the raised exception is indeed a RateLimitError
        assert isinstance(excinfo.value, litellm.RateLimitError)
    else:
        # For the negative case, we expect it to raise a generic APIConnectionError
        with pytest.raises(litellm.APIConnectionError):
            exception_type(
                model=model,
                original_exception=original_exception,
                custom_llm_provider=custom_llm_provider,
            )


def test_vertex_status_code_429_ignores_403_token_in_message() -> None:
    original_exception: Final = BaseLLMException(status_code=429, message="scripted vertex rejection f4039")

    with pytest.raises(litellm.RateLimitError) as excinfo:
        exception_type(
            model="gemini/gemini-2.5-flash",
            original_exception=original_exception,
            custom_llm_provider="vertex_ai",
        )

    assert excinfo.value.status_code == 429


def test_vertex_status_code_403_maps_to_403() -> None:
    original_exception: Final = BaseLLMException(status_code=403, message="permission denied")

    with pytest.raises(litellm.PermissionDeniedError) as excinfo:
        exception_type(
            model="gemini/gemini-2.5-flash",
            original_exception=original_exception,
            custom_llm_provider="vertex_ai",
        )

    assert excinfo.value.status_code == 403


def test_vertex_status_code_403_with_text_token_maps_to_bad_request() -> None:
    original_exception: Final = BaseLLMException(
        status_code=403,
        message='{"error":{"code":403,"message":"Permission denied"}}',
    )

    with pytest.raises(litellm.BadRequestError) as excinfo:
        exception_type(
            model="gemini/gemini-2.5-flash",
            original_exception=original_exception,
            custom_llm_provider="vertex_ai",
        )

    assert excinfo.value.status_code == 403


def test_vertex_text_403_without_status_code_still_maps_to_bad_request() -> None:
    with pytest.raises(litellm.BadRequestError) as excinfo:
        exception_type(
            model="gemini/gemini-2.5-flash",
            original_exception=Exception("403 Forbidden: permission denied"),
            custom_llm_provider="vertex_ai",
        )

    assert excinfo.value.status_code == 403


def test_vertex_status_substring_without_status_code_does_not_map_to_403() -> None:
    with pytest.raises(litellm.APIConnectionError) as excinfo:
        exception_type(
            model="gemini/gemini-2.5-flash",
            original_exception=Exception("scripted vertex rejection f4039a"),
            custom_llm_provider="vertex_ai",
        )

    assert excinfo.value.status_code != 403


def test_vertex_wrapped_500_with_403_body_maps_to_bad_request() -> None:
    original_exception: Final = BaseLLMException(
        status_code=500,
        message='Error code: 403 - {"error": {"code": 403, "message": "Permission denied"}}',
    )

    with pytest.raises(litellm.BadRequestError) as excinfo:
        exception_type(
            model="gemini/gemini-2.5-flash",
            original_exception=original_exception,
            custom_llm_provider="vertex_ai",
        )

    assert excinfo.value.status_code == 403


def test_vertex_wrapped_500_with_embedded_403_text_does_not_map_to_403() -> None:
    original_exception: Final = BaseLLMException(status_code=500, message="scripted vertex rejection f4039")

    with pytest.raises(litellm.InternalServerError) as excinfo:
        exception_type(
            model="gemini/gemini-2.5-flash",
            original_exception=original_exception,
            custom_llm_provider="vertex_ai",
        )

    assert excinfo.value.status_code == 500


def test_vertex_wrapped_500_with_body_code_429_and_403_token_maps_to_rate_limit() -> None:
    original_exception: Final = BaseLLMException(
        status_code=500,
        message='{"error":{"code":429,"message":"upstream unavailable (request id: 403-xyz)"}}',
    )

    with pytest.raises(litellm.RateLimitError) as excinfo:
        exception_type(
            model="gemini/gemini-2.5-flash",
            original_exception=original_exception,
            custom_llm_provider="vertex_ai",
        )

    assert excinfo.value.status_code == 429


def test_vertex_wrapped_500_with_body_code_403_maps_to_bad_request() -> None:
    original_exception: Final = BaseLLMException(
        status_code=500,
        message='{"error":{"code":403,"message":"Permission denied"}}',
    )

    with pytest.raises(litellm.BadRequestError) as excinfo:
        exception_type(
            model="gemini/gemini-2.5-flash",
            original_exception=original_exception,
            custom_llm_provider="vertex_ai",
        )

    assert excinfo.value.status_code == 403


class TestGetBodyErrorCode:
    """Unit tests for _get_body_error_code helper."""

    def test_parses_int_code(self):
        body = '{"error":{"message":"high demand","type":"upstream_error","param":"","code":429}}'
        assert _get_body_error_code(body) == 429

    def test_parses_string_code(self):
        # some gateways serialize code as a string
        body = '{"error":{"message":"x","code":"503"}}'
        assert _get_body_error_code(body) == 503

    def test_returns_none_on_non_json(self):
        assert _get_body_error_code("not json") is None

    def test_returns_none_when_no_error_key(self):
        assert _get_body_error_code('{"ok":true}') is None

    def test_returns_none_when_no_code_key(self):
        assert _get_body_error_code('{"error":{"message":"x"}}') is None


# Test cases for Gemini upstream-error body-code mapping.
#
# Body code 429 wrapped in a 5xx HTTP envelope (e.g. new-api gateways)
# must map to RateLimitError so Router retries kick in. A 4xx HTTP
# envelope with body code:429 must NOT — it falls through to whatever
# the HTTP status code maps to (BadRequestError, AuthenticationError,
# etc.), matching upstream's existing semantics.
gemini_body_code_429_test_cases = [
    # (status_code, error_body, expected_exception_type, description)
    (
        500,
        '{"error":{"message":" This model is currently experiencing high demand.'
        " Spikes in demand are usually temporary. Please try again later."
        ' (request id: x)","type":"upstream_error","param":"","code":429}}',
        litellm.RateLimitError,
        "HTTP 500 envelope with body code:429 -> RateLimitError",
    ),
    (
        503,
        '{"error":{"message":"upstream unavailable","type":"upstream_error","param":"","code":429}}',
        litellm.RateLimitError,
        "HTTP 503 envelope with body code:429 -> RateLimitError",
    ),
    (
        502,
        '{"error":{"message":"bad gateway","code":429}}',
        litellm.RateLimitError,
        "HTTP 502 envelope with body code:429 -> RateLimitError",
    ),
    (
        500,
        '{"error":{"message":"server boom","code":500}}',
        litellm.InternalServerError,
        "HTTP 500 with body code:500 stays InternalServerError",
    ),
    (
        500,
        "plain text 500 error",
        litellm.InternalServerError,
        "HTTP 500 with non-JSON body falls through to status_code mapping",
    ),
    (
        400,
        '{"error":{"message":"malformed","code":429}}',
        litellm.BadRequestError,
        "HTTP 400 with body code:429 must NOT be promoted to RateLimitError",
    ),
    (
        401,
        '{"error":{"message":"bad key","code":429}}',
        litellm.AuthenticationError,
        "HTTP 401 with body code:429 must NOT be promoted to RateLimitError",
    ),
]


@pytest.mark.parametrize(
    "status_code, error_body, expected_exception, description",
    gemini_body_code_429_test_cases,
)
def test_gemini_upstream_error_body_code_429_maps_to_rate_limit(
    status_code, error_body, expected_exception, description
):
    """
    Body code 429 inside a 5xx envelope -> RateLimitError so Router
    retries kick in. Body code 429 inside a 4xx envelope must fall
    through to the HTTP-status-code branch (P1 from greptile review).
    """
    model = "gemini/gemini-2.5-flash"
    custom_llm_provider = "gemini"

    # Build an exception that looks like what _handle_error produces:
    # a BaseLLMException-style object with .status_code and .message
    class _FakeGeminiError(Exception):
        def __init__(self, status_code, message):
            self.status_code = status_code
            self.message = message
            super().__init__(message)

    original_exception = _FakeGeminiError(status_code=status_code, message=error_body)

    with pytest.raises(expected_exception) as excinfo:
        exception_type(
            model=model,
            original_exception=original_exception,
            custom_llm_provider=custom_llm_provider,
        )
    assert isinstance(excinfo.value, expected_exception), description


class TestExtractAndRaiseLitellmException:
    """Tests for extract_and_raise_litellm_exception function"""

    def test_extract_and_raise_api_connection_error_without_response(self):
        """
        Test that APIConnectionError can be raised without response parameter.

        This is a regression test for the bug where extract_and_raise_litellm_exception
        would fail with TypeError when trying to raise APIConnectionError with a
        response parameter, since APIConnectionError doesn't accept that parameter.

        Relevant Issue: https://github.com/BerriAI/litellm/issues/XXXXX
        """
        error_str = "litellm.APIConnectionError: GeminiException - some error message"

        with pytest.raises(litellm.APIConnectionError) as excinfo:
            extract_and_raise_litellm_exception(
                response=None,
                error_str=error_str,
                model="gemini/gemini-3-pro-preview",
                custom_llm_provider="gemini",
            )

        assert "APIConnectionError" in str(excinfo.value)

    def test_extract_and_raise_bad_request_error_with_response(self):
        """
        Test that BadRequestError can be raised with response parameter.

        BadRequestError does accept the response parameter, so this should work.
        """
        error_str = "litellm.BadRequestError: Invalid request format"

        with pytest.raises(litellm.BadRequestError) as excinfo:
            extract_and_raise_litellm_exception(
                response=None,
                error_str=error_str,
                model="gpt-4",
                custom_llm_provider="openai",
            )

        assert "BadRequestError" in str(excinfo.value)

    def test_extract_and_raise_context_window_exceeded_error(self):
        """
        Test that ContextWindowExceededError can be raised.
        """
        error_str = "litellm.ContextWindowExceededError: Token limit exceeded"

        with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
            extract_and_raise_litellm_exception(
                response=None,
                error_str=error_str,
                model="gpt-4",
                custom_llm_provider="openai",
            )

        assert "ContextWindowExceededError" in str(excinfo.value)

    def test_no_exception_raised_for_non_litellm_error(self):
        """
        Test that no exception is raised for non-litellm error strings.
        """
        error_str = "Some generic error that is not a litellm exception"

        # Should not raise any exception
        result = extract_and_raise_litellm_exception(
            response=None,
            error_str=error_str,
            model="gpt-4",
            custom_llm_provider="openai",
        )

        assert result is None


class ModelError(Exception):
    """Mimics replicate's SDK exception, whose mapping keys on the class name."""


class CohereConnectionError(Exception):
    """Mimics cohere's SDK exception, whose mapping keys on the class name."""


def test_replicate_model_error_maps_to_bad_request():
    """The replicate branch keys on ``type(original_exception).__name__ ==
    "ModelError"`` rather than on the error string. The dispatch now lives in a
    per-provider helper, so this class-name value has to be threaded into the
    helper; if it is not, the bare name ``exception_type`` resolves to the
    module-level function and the comparison is always False, silently mismapping
    to APIConnectionError."""
    original_exception = ModelError("the deployed model failed to return a prediction")

    with pytest.raises(litellm.BadRequestError) as excinfo:
        exception_type(
            model="replicate/meta/llama-2-70b-chat",
            original_exception=original_exception,
            custom_llm_provider="replicate",
        )

    assert excinfo.value.llm_provider == "replicate"


def test_cohere_connection_error_maps_to_rate_limit():
    """The cohere branch keys on ``"CohereConnectionError" in
    type(original_exception).__name__``. With the dispatch extracted into a helper
    the class-name value must be passed in; otherwise ``in`` runs against the
    module-level ``exception_type`` function object and raises TypeError, which the
    outer handler swallows into a generic APIConnectionError."""
    original_exception = CohereConnectionError("connection reset by peer")
    original_exception.message = "connection reset by peer"

    with pytest.raises(litellm.RateLimitError) as excinfo:
        exception_type(
            model="command-r",
            original_exception=original_exception,
            custom_llm_provider="cohere",
        )

    assert excinfo.value.llm_provider == "cohere"


class ReplicateError(Exception):
    """Mimics a replicate HTTP error carrying a status_code and response."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://api.replicate.com/v1/predictions"),
        )


def test_replicate_422_maps_to_unprocessable_entity():
    """The replicate status-code ladder carried two identical ``status_code == 422``
    branches; the second was unreachable dead code. After dropping the duplicate the
    surviving branch must still map 422 to UnprocessableEntityError."""
    original_exception = ReplicateError("validation failed for the input", 422)

    with pytest.raises(litellm.UnprocessableEntityError) as excinfo:
        exception_type(
            model="replicate/meta/llama-2-70b-chat",
            original_exception=original_exception,
            custom_llm_provider="replicate",
        )

    assert excinfo.value.llm_provider == "replicate"


def test_upstream_4xx_without_model_maps_to_bad_request():
    """Responses API follow-ups (cancel/get/delete) call ``exception_type`` with
    ``model=None``; the provider mapping used to be gated on ``if model:``, so an
    upstream 400 like Azure's "Cannot cancel a synchronous response." fell through to
    the generic 500 APIConnectionError instead of surfacing as a 400."""
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    original_exception = BaseLLMException(
        status_code=400,
        message='{"error": {"message": "Cannot cancel a synchronous response.", "type": "invalid_request_error"}}',
    )

    with pytest.raises(litellm.BadRequestError) as excinfo:
        exception_type(
            model=None,
            original_exception=original_exception,
            custom_llm_provider="azure",
        )

    assert excinfo.value.status_code == 400
    assert "Cannot cancel a synchronous response." in excinfo.value.message


def test_azure_404_with_invalid_request_error_type_maps_to_not_found():
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    original_exception = BaseLLMException(
        status_code=404,
        message='{"error": {"message": "Response with id \'resp_abc\' not found.", "type": "invalid_request_error"}}',
    )

    with pytest.raises(litellm.NotFoundError) as excinfo:
        exception_type(
            model=None,
            original_exception=original_exception,
            custom_llm_provider="azure",
        )

    assert excinfo.value.status_code == 404
    assert "Response with id 'resp_abc' not found." in excinfo.value.message


@pytest.mark.parametrize("provider", ["openai", "groq", "deepseek"])
@pytest.mark.parametrize(
    "status_code, message, expected_exception",
    [
        (400, "Invalid 'temperature'", litellm.BadRequestError),
        (401, "Invalid API key", litellm.AuthenticationError),
        (403, "Project does not have access to this endpoint", litellm.PermissionDeniedError),
        (404, "The model does not exist", litellm.NotFoundError),
    ],
)
def test_openai_invalid_request_error_respects_status_code(
    provider: str, status_code: int, message: str, expected_exception: type[openai.APIStatusError]
) -> None:
    error_message: Final = json.dumps({"error": {"message": message, "type": "invalid_request_error"}})
    original_exception: Final = OpenAIError(status_code=status_code, message=error_message, headers={})

    with pytest.raises(openai.APIError) as excinfo:
        exception_type(
            model="some-model",
            original_exception=original_exception,
            custom_llm_provider=provider,
        )

    assert type(excinfo.value) is expected_exception
    assert excinfo.value.status_code == status_code
    assert message in excinfo.value.message


@pytest.mark.parametrize("provider", ["openai", "groq", "deepseek"])
def test_openai_invalid_request_error_without_a_status_code_maps_to_bad_request(provider: str) -> None:
    message: Final = "Missing required parameter: 'input'."
    original_exception: Final = Exception(json.dumps({"error": {"message": message, "type": "invalid_request_error"}}))

    with pytest.raises(openai.APIError) as excinfo:
        exception_type(
            model="some-model",
            original_exception=original_exception,
            custom_llm_provider=provider,
        )

    assert type(excinfo.value) is litellm.BadRequestError
    assert message in excinfo.value.message


@pytest.mark.parametrize("provider", ["openai", "groq", "deepseek"])
@pytest.mark.parametrize("status_code", [409, 422, 429, 500, 503])
def test_openai_invalid_request_error_keeps_bad_request_for_statuses_other_than_401_403_and_404(
    provider: str, status_code: int
) -> None:
    message: Final = "Cannot cancel a batch that is already completed."
    error_message: Final = json.dumps({"error": {"message": message, "type": "invalid_request_error"}})
    original_exception: Final = OpenAIError(status_code=status_code, message=error_message, headers={})

    with pytest.raises(openai.APIError) as excinfo:
        exception_type(
            model="some-model",
            original_exception=original_exception,
            custom_llm_provider=provider,
        )

    assert type(excinfo.value) is litellm.BadRequestError
    assert message in excinfo.value.message


def _openai_sdk_error(
    sdk_error: type[openai.APIStatusError], status_code: int, provider_error: dict[str, str | None]
) -> openai.APIStatusError:
    request: Final = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return sdk_error(
        f"Error code: {status_code} - {{'error': {provider_error}}}",
        response=httpx.Response(status_code, request=request),
        body=provider_error,
    )


@pytest.mark.parametrize(
    "sdk_error, status_code, expected_exception, param, code",
    [
        (openai.AuthenticationError, 401, litellm.AuthenticationError, "organization", "mismatched_organization"),
        (openai.PermissionDeniedError, 403, litellm.PermissionDeniedError, "model", "model_access_denied"),
        (openai.NotFoundError, 404, litellm.NotFoundError, "id", None),
    ],
)
def test_openai_invalid_request_error_401_403_and_404_keep_the_provider_error_fields(
    sdk_error: type[openai.APIStatusError],
    status_code: int,
    expected_exception: type[openai.APIStatusError],
    param: str,
    code: str | None,
) -> None:
    provider_error: Final = {
        "message": "The request was rejected.",
        "type": "invalid_request_error",
        "param": param,
        "code": code,
    }

    with pytest.raises(openai.APIError) as excinfo:
        exception_type(
            model="gpt-5-mini",
            original_exception=_openai_sdk_error(sdk_error, status_code, provider_error),
            custom_llm_provider="openai",
        )

    assert type(excinfo.value) is expected_exception
    assert (excinfo.value.status_code, excinfo.value.type, excinfo.value.param, excinfo.value.code) == (
        status_code,
        "invalid_request_error",
        param,
        code,
    )
    assert excinfo.value.body == provider_error
    assert openai_error_type(excinfo.value, status_code) == "invalid_request_error"
    assert excinfo.value.message.count(expected_exception.__name__) == 1


@pytest.mark.parametrize(
    "error_detail, expected_exception",
    [
        ("model_not_found", litellm.NotFoundError),
        ("content_policy_violation", litellm.ContentPolicyViolationError),
    ],
)
def test_openai_invalid_request_error_403_keeps_specialized_mappings(
    error_detail: str, expected_exception: type[openai.APIStatusError]
) -> None:
    error_message: Final = json.dumps(
        {"error": {"message": "Rejected.", "type": "invalid_request_error", "code": error_detail}}
    )
    original_exception: Final = OpenAIError(status_code=403, message=error_message, headers={})

    with pytest.raises(openai.APIError) as excinfo:
        exception_type(
            model="some-model",
            original_exception=original_exception,
            custom_llm_provider="openai",
        )

    assert type(excinfo.value) is expected_exception


def test_openai_incorrect_api_key_keeps_the_proxy_authentication_error_type() -> None:
    provider_error: Final = {
        "message": "Incorrect API key provided: sk-litellmqa.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key",
    }

    with pytest.raises(litellm.AuthenticationError) as excinfo:
        exception_type(
            model="gpt-5-mini",
            original_exception=_openai_sdk_error(openai.AuthenticationError, 401, provider_error),
            custom_llm_provider="openai",
        )

    assert openai_error_type(excinfo.value, 401) == "authentication_error"


def _openai_sdk_403(message: str) -> Exception:
    return _openai_sdk_error(openai.PermissionDeniedError, 403, {"message": message, "type": "permission_denied"})


def _openai_like_403(message: str) -> Exception:
    return OpenAIError(status_code=403, message=message, headers={})


@pytest.mark.parametrize("make_error", [_openai_sdk_403, _openai_like_403], ids=["openai_sdk", "openai_like_httpx"])
@pytest.mark.parametrize(
    ("provider", "exception_provider"),
    [("openai", "OpenAIException"), ("deepseek", "DeepseekException"), ("hostinger_router", "Hostinger_routerException")],
)
def test_an_openai_style_403_maps_to_permission_denied(
    provider: str, exception_provider: str, make_error: Callable[[str], Exception], quiet_exception_mapping
) -> None:
    with pytest.raises(litellm.PermissionDeniedError) as raised:
        exception_type(
            model="test-model",
            original_exception=make_error("Invalid API key"),
            custom_llm_provider=provider,
        )

    assert (
        type(raised.value),
        raised.value.status_code,
        raised.value.message,
        raised.value.llm_provider,
        raised.value.model,
        raised.value.response.status_code,
    ) == (
        litellm.PermissionDeniedError,
        403,
        f"litellm.PermissionDeniedError: {exception_provider} - Invalid API key",
        provider,
        "test-model",
        403,
    )


class _UpstreamHTTPError(Exception):
    def __init__(self, status_code: int, message: str = "upstream failure") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
        self.response = httpx.Response(status_code=status_code, request=self.request, text=message)


UPSTREAM_STATUS_CODES = (400, 401, 402, 403, 404, 408, 422, 429, 500, 503)

PERMISSION_DENIED = (litellm.PermissionDeniedError, 403)

OPENAI_SHAPED = {
    400: (litellm.BadRequestError, 400),
    401: (litellm.AuthenticationError, 401),
    402: (litellm.PaymentRequiredError, 402),
    403: PERMISSION_DENIED,
    404: (litellm.NotFoundError, 404),
    408: (litellm.Timeout, 408),
    422: (litellm.BadRequestError, 422),
    429: (litellm.RateLimitError, 429),
    500: (litellm.InternalServerError, 500),
    503: (litellm.ServiceUnavailableError, 503),
}

DEVIATIONS_FROM_THE_OPENAI_SHAPE = {
    "ai21": {402: (litellm.APIError, 402)},
    "azure": {402: (litellm.APIError, 402), 403: (litellm.APIError, 403), 500: (litellm.APIError, 500)},
    "azure_ai": {402: (litellm.APIError, 402)},
    "bedrock": {
        500: (litellm.ServiceUnavailableError, 503),
    },
    "databricks": {
        422: (litellm.BadRequestError, 400),
    },
    "deepseek": {402: (litellm.APIError, 402)},
    "fireworks_ai": {402: (litellm.APIError, 402)},
    "groq": {402: (litellm.APIError, 402)},
    "huggingface": {
        402: (litellm.APIError, 402),
        403: (litellm.APIError, 403),
        404: (litellm.APIError, 404),
        422: (litellm.APIError, 422),
        500: (litellm.APIError, 500),
    },
    "inferenceapis": {
        422: (litellm.BadRequestError, 400),
    },
    "mistral": {402: (litellm.APIError, 402)},
    "nlp_cloud": {
        402: (litellm.RateLimitError, 429),
        403: (litellm.AuthenticationError, 403),
        404: (litellm.APIError, 404),
        408: (litellm.APIError, 408),
        500: (litellm.APIError, 500),
        503: (litellm.APIError, 503),
    },
    "openai": {402: (litellm.APIError, 402)},
    "openrouter": {402: (litellm.APIError, 402), 403: (litellm.APIError, 403), 500: (litellm.APIError, 500)},
    "perplexity": {402: (litellm.APIError, 402)},
    "replicate": {
        403: (litellm.APIError, 500),
        404: (litellm.APIError, 500),
        402: (litellm.APIError, 500),
        422: (litellm.UnprocessableEntityError, 422),
        500: (litellm.ServiceUnavailableError, 503),
        503: (litellm.APIError, 500),
    },
    "runwayml": {402: (litellm.APIError, 402)},
    "sagemaker": {
        500: (litellm.ServiceUnavailableError, 503),
    },
    "together_ai": {402: (litellm.APIError, 402)},
    "xai": {402: (litellm.APIError, 402)},
}

PROVIDERS_WITH_A_HANDLER = (
    "ai21",
    "anthropic",
    "azure",
    "azure_ai",
    "bedrock",
    "cloudflare",
    "cohere",
    "databricks",
    "deepseek",
    "fireworks_ai",
    "gemini",
    "groq",
    "huggingface",
    "inferenceapis",
    "mistral",
    "nlp_cloud",
    "ollama",
    "openai",
    "openrouter",
    "perplexity",
    "replicate",
    "runwayml",
    "sagemaker",
    "together_ai",
    "vertex_ai",
    "vllm",
    "xai",
)

PROVIDER_ALIASES_WITH_A_HANDLER = (
    "aleph_alpha",
    "anthropic_text",
    "azure_text",
    "bedrock_mantle",
    "cohere_chat",
    "custom_openai",
    "lemonade",
    "litellm_proxy",
    "ollama_chat",
    "predibase",
    "sagemaker_chat",
    "text-completion-openai",
    "vertex_ai_beta",
    "watsonx",
)

PROVIDERS_WITHOUT_A_HANDLER = tuple(
    sorted(
        frozenset(provider.value for provider in LlmProviders)
        - frozenset(PROVIDERS_WITH_A_HANDLER)
        - frozenset(PROVIDER_ALIASES_WITH_A_HANDLER)
        - frozenset(litellm.openai_compatible_providers)
    )
)

MINIMAX_401_BODY = (
    '{"type":"error","error":{"type":"authorized_error","message":"login fail: Please carry the API secret key '
    'in the \'Authorization\' field of the request header (1004)","http_code":"401"},'
    '"request_id":"06ddc9ba97ee6340e38f10e09787f547"}'
)


def _expected_for(provider: str, status_code: int) -> tuple[type[Exception], int]:
    return DEVIATIONS_FROM_THE_OPENAI_SHAPE.get(provider, {}).get(status_code, OPENAI_SHAPED[status_code])


@pytest.fixture
def quiet_exception_mapping(monkeypatch):
    monkeypatch.setattr(litellm, "suppress_debug_info", True)


@pytest.mark.parametrize("status_code", UPSTREAM_STATUS_CODES)
@pytest.mark.parametrize("provider", PROVIDERS_WITH_A_HANDLER)
def test_an_upstream_status_maps_to_one_exception_per_provider(provider, status_code, quiet_exception_mapping):
    expected_class, expected_status = _expected_for(provider, status_code)

    with pytest.raises(openai.APIError) as raised:
        exception_type(
            model="test-model",
            original_exception=_UpstreamHTTPError(status_code=status_code),
            custom_llm_provider=provider,
        )

    assert type(raised.value) is expected_class
    assert raised.value.status_code == expected_status


@pytest.mark.parametrize("status_code", UPSTREAM_STATUS_CODES)
@pytest.mark.parametrize("provider", PROVIDERS_WITH_A_HANDLER)
def test_a_mapped_exception_keeps_the_provider_and_model_it_came_from(provider, status_code, quiet_exception_mapping):
    with pytest.raises(openai.APIError) as raised:
        exception_type(
            model="test-model",
            original_exception=_UpstreamHTTPError(status_code=status_code),
            custom_llm_provider=provider,
        )

    assert raised.value.llm_provider == provider
    assert raised.value.model == "test-model"


@pytest.mark.parametrize("provider", PROVIDERS_WITH_A_HANDLER)
def test_an_already_mapped_litellm_exception_passes_through_untouched(provider, quiet_exception_mapping):
    already_mapped = litellm.RateLimitError(message="already mapped", llm_provider=provider, model="test-model")

    returned = exception_type(
        model="test-model",
        original_exception=already_mapped,
        custom_llm_provider=provider,
    )

    assert returned is already_mapped


@pytest.mark.parametrize(
    "message",
    ['{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}', "upstream failure"],
)
def test_an_anthropic_529_stays_an_internal_server_error_that_carries_the_529(message, quiet_exception_mapping):
    with pytest.raises(litellm.InternalServerError) as raised:
        exception_type(
            model="claude-opus-5-5",
            original_exception=_UpstreamHTTPError(status_code=529, message=message),
            custom_llm_provider="anthropic",
        )

    assert (type(raised.value), raised.value.status_code, raised.value.provider_status_code) == (
        litellm.InternalServerError,
        500,
        529,
    )


@pytest.mark.parametrize("status_code", [400, 404])
def test_an_anthropic_client_error_naming_overloaded_carries_no_overload_status(status_code, quiet_exception_mapping):
    with pytest.raises(litellm.InternalServerError) as raised:
        exception_type(
            model="claude-opus-5-5",
            original_exception=_UpstreamHTTPError(status_code=status_code, message="model: claude-Overloaded"),
            custom_llm_provider="anthropic",
        )

    assert raised.value.provider_status_code is None


def test_an_anthropic_500_carries_its_own_status_as_the_provider_status(quiet_exception_mapping):
    with pytest.raises(litellm.InternalServerError) as raised:
        exception_type(
            model="claude-opus-5-5",
            original_exception=_UpstreamHTTPError(status_code=500),
            custom_llm_provider="anthropic",
        )

    assert (raised.value.status_code, raised.value.provider_status_code) == (500, 500)


@pytest.mark.parametrize("status_code", UPSTREAM_STATUS_CODES)
@pytest.mark.parametrize("provider", PROVIDERS_WITHOUT_A_HANDLER)
def test_a_provider_without_a_handler_maps_by_the_upstream_status(provider, status_code, quiet_exception_mapping):
    expected_class, expected_status = OPENAI_SHAPED[status_code]

    with pytest.raises(openai.APIError) as raised:
        exception_type(
            model="test-model",
            original_exception=_UpstreamHTTPError(status_code=status_code),
            custom_llm_provider=provider,
        )

    assert type(raised.value) is expected_class
    assert raised.value.status_code == expected_status
    assert raised.value.llm_provider == provider
    assert raised.value.model == "test-model"
    if status_code == 402:
        assert isinstance(raised.value, litellm.BadRequestError)
        assert raised.value.response.status_code == 402
        assert str(raised.value).startswith("litellm.PaymentRequiredError: ")
        assert "litellm.BadRequestError:" not in str(raised.value)
        assert repr(raised.value) == str(raised.value)


def test_payment_required_error_without_response_uses_402_response() -> None:
    assert litellm.PaymentRequiredError("x", "m", "p").response.status_code == 402


def test_a_minimax_bad_key_is_an_authentication_error(quiet_exception_mapping):
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    with pytest.raises(litellm.AuthenticationError) as raised:
        exception_type(
            model="MiniMax-M2.5",
            original_exception=BaseLLMException(status_code=401, message=MINIMAX_401_BODY),
            custom_llm_provider="minimax",
        )

    assert raised.value.status_code == 401
    assert raised.value.llm_provider == "minimax"
    assert raised.value.message.startswith("litellm.AuthenticationError: MinimaxException - ")
    assert "login fail" in raised.value.message


def test_an_exception_without_a_status_is_still_a_connection_error(quiet_exception_mapping):
    with pytest.raises(litellm.APIConnectionError):
        exception_type(
            model="MiniMax-M2.5",
            original_exception=RuntimeError("socket hung up"),
            custom_llm_provider="minimax",
        )


def test_an_unmapped_exception_with_no_model_or_provider_is_a_connection_error(quiet_exception_mapping):
    with pytest.raises(litellm.APIConnectionError) as raised:
        exception_type(
            model=None,
            original_exception=ValueError("boom"),
            custom_llm_provider=None,
        )

    assert "boom" in raised.value.message


def test_unmapped_sdk_exception_includes_bug_report_link(quiet_exception_mapping):
    with pytest.raises(litellm.APIConnectionError) as raised:
        exception_type(
            model="my-model",
            custom_llm_provider="minimax",
            original_exception=ValueError("boom"),
        )

    assert "https://github.com/BerriAI/litellm/issues/new?" in str(raised.value)
    assert "ValueError" in str(raised.value)


def test_unmapped_sdk_exception_bug_report_link_can_be_disabled(quiet_exception_mapping, monkeypatch):
    monkeypatch.setenv("LITELLM_DISABLE_BUG_REPORT_LINK", "true")

    with pytest.raises(litellm.APIConnectionError) as raised:
        exception_type(
            model="my-model",
            custom_llm_provider="minimax",
            original_exception=ValueError("boom"),
        )

    assert "https://github.com/BerriAI/litellm/issues/new?" not in str(raised.value)


def _raise_and_map(model: str | None, original_exception: Exception, custom_llm_provider: str | None) -> None:
    """Calls exception_type() from inside the except block, as litellm/main.py does,
    so traceback.format_exc() has a real stack."""
    try:
        raise original_exception
    except type(original_exception) as caught:
        exception_type(
            model=model,
            original_exception=caught,
            custom_llm_provider=custom_llm_provider,
        )


def test_an_unmapped_exception_message_keeps_traceback_for_sdk_callers(quiet_exception_mapping):
    """Direct SDK callers debug unmapped provider exceptions with this traceback;
    only the proxy's response boundary strips it."""
    with pytest.raises(litellm.APIConnectionError) as raised:
        _raise_and_map(
            model="MiniMax-M2.5",
            original_exception=RuntimeError("socket hung up"),
            custom_llm_provider="minimax",
        )

    assert "Traceback (most recent call last)" in raised.value.message
    assert "test_exception_mapping_utils.py" in raised.value.message


def test_an_unmapped_exception_with_no_model_or_provider_message_keeps_traceback(
    quiet_exception_mapping,
):
    with pytest.raises(litellm.APIConnectionError) as raised:
        _raise_and_map(
            model=None,
            original_exception=ValueError("boom"),
            custom_llm_provider=None,
        )

    assert "Traceback (most recent call last)" in raised.value.message


@pytest.mark.parametrize("custom_llm_provider", ["azure", "azure_text"])
def test_an_azure_stream_cut_maps_to_the_error_line_without_a_traceback(
    custom_llm_provider: str, quiet_exception_mapping
) -> None:
    stream_cut: Final = httpx.ReadError(
        "Response payload is not completed: <TransferEncodingError: 400, "
        "message='Not enough data to satisfy transfer length header.'>"
    )

    with pytest.raises(litellm.APIConnectionError) as raised:
        _raise_and_map(model="gpt-5.4-mini", original_exception=stream_cut, custom_llm_provider=custom_llm_provider)

    assert raised.value.message.endswith(f"APIConnectionError - {stream_cut}")
    assert "test_exception_mapping_utils.py" not in str(raised.value)


@pytest.mark.parametrize("custom_llm_provider", ["azure", "azure_text"])
def test_an_azure_connection_reset_with_no_message_names_the_error_type(
    custom_llm_provider: str, quiet_exception_mapping
) -> None:
    with pytest.raises(litellm.APIConnectionError) as raised:
        _raise_and_map(
            model="gpt-5.4-mini", original_exception=httpx.ReadError(""), custom_llm_provider=custom_llm_provider
        )

    assert raised.value.message.endswith("APIConnectionError - ReadError")


CONTEXT_WINDOW_MESSAGE = "This model's maximum context length is 4096 tokens."
CONTENT_POLICY_MESSAGE = '{"error": {"type": "invalid_request_error", "code": "content_policy_violation"}}'
TIMEOUT_MESSAGE = "Request timed out."

PROVIDERS_THAT_RECOGNISE_A_FULL_CONTEXT_WINDOW = (
    "ai21",
    "anthropic",
    "azure",
    "azure_ai",
    "databricks",
    "deepseek",
    "fireworks_ai",
    "gemini",
    "groq",
    "inferenceapis",
    "mistral",
    "openai",
    "perplexity",
    "runwayml",
    "together_ai",
    "vertex_ai",
    "xai",
)

PROVIDERS_THAT_RECOGNISE_A_CONTENT_POLICY_BLOCK = (
    "ai21",
    "azure",
    "azure_ai",
    "deepseek",
    "fireworks_ai",
    "groq",
    "mistral",
    "openai",
    "perplexity",
    "runwayml",
    "together_ai",
    "xai",
)


class _UpstreamErrorWithMessage(_UpstreamHTTPError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(status_code=status_code)
        self.args = (message,)
        self.message = message
        self.response = httpx.Response(status_code=status_code, request=self.request, text=message)


@pytest.mark.parametrize("provider", PROVIDERS_WITH_A_HANDLER)
def test_a_full_context_window_reaches_the_caller_as_the_router_needs_it(provider, quiet_exception_mapping):
    if provider in PROVIDERS_THAT_RECOGNISE_A_FULL_CONTEXT_WINDOW:
        expected_class, expected_status = litellm.ContextWindowExceededError, 400
    else:
        expected_class, expected_status = litellm.BadRequestError, 400

    with pytest.raises(openai.APIError) as raised:
        exception_type(
            model="test-model",
            original_exception=_UpstreamErrorWithMessage(CONTEXT_WINDOW_MESSAGE, 400),
            custom_llm_provider=provider,
        )

    assert type(raised.value) is expected_class
    assert raised.value.status_code == expected_status


@pytest.mark.parametrize("provider", PROVIDERS_WITH_A_HANDLER)
def test_a_content_policy_block_reaches_the_caller_as_the_router_needs_it(provider, quiet_exception_mapping):
    if provider in PROVIDERS_THAT_RECOGNISE_A_CONTENT_POLICY_BLOCK:
        expected_class, expected_status = litellm.ContentPolicyViolationError, 400
    else:
        expected_class, expected_status = litellm.BadRequestError, 400

    with pytest.raises(openai.APIError) as raised:
        exception_type(
            model="test-model",
            original_exception=_UpstreamErrorWithMessage(CONTENT_POLICY_MESSAGE, 400),
            custom_llm_provider=provider,
        )

    assert type(raised.value) is expected_class
    assert raised.value.status_code == expected_status


@pytest.mark.parametrize("provider", PROVIDERS_WITH_A_HANDLER)
def test_a_timed_out_request_is_a_timeout_for_every_provider(provider, quiet_exception_mapping):
    with pytest.raises(litellm.Timeout) as raised:
        exception_type(
            model="test-model",
            original_exception=_UpstreamErrorWithMessage(TIMEOUT_MESSAGE, 408),
            custom_llm_provider=provider,
        )

    assert raised.value.status_code == 408


def test_bedrock_mantle_400_maps_to_bad_request():
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    original_exception = BaseLLMException(
        status_code=400,
        message=(
            '{"error": {"code": "validation_error", "message": '
            "\"invalid request body: Invalid 'input': value did not match any expected variant\", "
            '"type": "invalid_request_error"}}'
        ),
    )

    with pytest.raises(litellm.BadRequestError) as excinfo:
        exception_type(
            model="gpt-5.6-terra",
            original_exception=original_exception,
            custom_llm_provider="bedrock_mantle",
        )

    assert excinfo.value.status_code == 400
    assert "Invalid 'input'" in excinfo.value.message
    assert type(excinfo.value) is litellm.BadRequestError


class _ErrorWithoutResponse(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def test_bedrock_mantle_context_overflow_without_a_provider_response_answers_400():
    with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
        exception_type(
            model="openai.gpt-5.6-sol",
            original_exception=_ErrorWithoutResponse(
                '{"error":{"code":"context_length_exceeded","message":"too long"}}', 400
            ),
            custom_llm_provider="bedrock_mantle",
        )

    assert excinfo.value.response.status_code == 400


def _invalid_model_identifier_from(url: str) -> BedrockError:
    return BedrockError(
        status_code=400,
        message='{"message":"The provided model identifier is invalid."}',
        response=httpx.Response(status_code=400, request=httpx.Request(method="POST", url=url)),
    )


@pytest.mark.parametrize(
    "path",
    ["/model/us.newvendor.typo-1/converse", "/model/us.newvendor.typo-1/converse-stream"],
)
def test_bedrock_converse_rejecting_an_id_with_no_invoke_provider_answers_404(path):
    with pytest.raises(litellm.NotFoundError) as excinfo:
        exception_type(
            model="us.newvendor.typo-1",
            original_exception=_invalid_model_identifier_from(f"https://bedrock-runtime.us-east-1.amazonaws.com{path}"),
            custom_llm_provider="bedrock",
        )

    assert excinfo.value.response.status_code == 404


@pytest.mark.parametrize(
    "original_exception",
    [
        _invalid_model_identifier_from("https://bedrock-runtime.us-east-1.amazonaws.com/model/claude-typo-9/invoke"),
        _ErrorWithoutResponse("The provided model identifier is invalid.", 400),
    ],
    ids=["invoke", "no-response"],
)
def test_bedrock_invalid_model_identifier_not_from_converse_stays_a_bad_request(original_exception):
    with pytest.raises(litellm.BadRequestError) as excinfo:
        exception_type(model="claude-typo-9", original_exception=original_exception, custom_llm_provider="bedrock")

    assert excinfo.value.status_code == 400


def test_bedrock_invalid_model_identifier_without_a_model_stays_a_bad_request():
    with pytest.raises(litellm.BadRequestError) as excinfo:
        exception_type(
            model=None,
            original_exception=_ErrorWithoutResponse("The provided model identifier is invalid.", 400),
            custom_llm_provider="bedrock",
        )

    assert excinfo.value.status_code == 400


def test_bedrock_mantle_context_overflow_maps_to_context_window_exceeded():
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    original_exception = BaseLLMException(
        status_code=400,
        message=(
            '{"error":{"code":"validation_error",'
            '"message":"prompt tokens (1055489) exceed model maximum (1050000) for openai.gpt-5.6-sol",'
            '"param":null,"type":"invalid_request_error"}}'
        ),
    )

    with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
        exception_type(
            model="openai.gpt-5.6-sol",
            original_exception=original_exception,
            custom_llm_provider="bedrock_mantle",
        )

    assert excinfo.value.status_code == 400
    assert "prompt is too long: 1055489 tokens > 1050000 maximum" in excinfo.value.message


def test_bedrock_mantle_openai_envelope_context_overflow_maps_to_context_window_exceeded():
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    original_exception = BaseLLMException(
        status_code=400,
        message=(
            '{"error":{"code":"context_length_exceeded",'
            '"message":"Your input exceeds the context window of this model. '
            'Please adjust your input and try again.",'
            '"param":"input","type":"invalid_request_error"}}'
        ),
    )

    with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
        exception_type(
            model="openai.gpt-5.6-sol",
            original_exception=original_exception,
            custom_llm_provider="bedrock_mantle",
        )

    assert excinfo.value.status_code == 400
    assert "prompt is too long" in excinfo.value.message


def test_bedrock_mantle_streamed_context_overflow_event_maps_to_context_window_exceeded():
    from litellm.responses.streaming_iterator import map_stream_error_to_exception

    mapped_exception = map_stream_error_to_exception(
        {
            "code": "context_length_exceeded",
            "message": "Your input exceeds the context window of this model. Please adjust your input and try again.",
        },
        model="openai.gpt-5.6-luna",
        custom_llm_provider="bedrock_mantle",
    )

    assert isinstance(mapped_exception, litellm.ContextWindowExceededError)
    assert mapped_exception.status_code == 400
    assert "prompt is too long" in mapped_exception.message


@pytest.mark.parametrize("original_status_code", [500, 503])
def test_bedrock_mantle_in_stream_context_overflow_answers_400_whatever_the_event_status_was(original_status_code):
    original_exception = OpenAIError(
        status_code=original_status_code,
        message="prompt tokens (229378) exceed model maximum (131072) for google.gemma-4-e2b",
    )

    with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
        exception_type(
            model="google.gemma-4-e2b",
            original_exception=original_exception,
            custom_llm_provider="bedrock_mantle",
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.response.status_code == 400


def test_bedrock_mantle_context_overflow_keeps_the_provider_400_response_body():
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    body = {
        "error": {
            "code": "context_length_exceeded",
            "message": "Your input exceeds the context window of this model.",
            "param": "input",
            "type": "invalid_request_error",
        }
    }
    provider_response = httpx.Response(
        status_code=400,
        json=body,
        request=httpx.Request("POST", "https://bedrock-mantle.us-east-1.amazonaws.com/v1/chat/completions"),
    )
    original_exception = BaseLLMException(status_code=400, message=json.dumps(body), response=provider_response)

    with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
        exception_type(
            model="openai.gpt-5.6-sol",
            original_exception=original_exception,
            custom_llm_provider="bedrock_mantle",
        )

    assert excinfo.value.response is provider_response
    assert excinfo.value.response.json() == body
    assert excinfo.value.status_code == 400


@pytest.mark.parametrize(
    "error_str",
    [
        '{"error":{"code":"context_length_exceeded","message":"Your input exceeds the context window."}}',
        "Error code: 400 - {'error': {'message': 'Your input exceeds the context window.', 'type': None, "
        "'code': 'context_length_exceeded'}}",
        '{"message":"BedrockException - {\\"error\\":{\\"code\\":\\"context_length_exceeded\\"}}"}',
    ],
)
def test_bedrock_mantle_overflow_code_maps_in_every_serialization_of_the_envelope(error_str):
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    with pytest.raises(litellm.ContextWindowExceededError):
        exception_type(
            model="openai.gpt-5.6-sol",
            original_exception=BaseLLMException(status_code=400, message=error_str),
            custom_llm_provider="bedrock_mantle",
        )


def test_bedrock_mantle_error_merely_quoting_the_overflow_code_stays_a_plain_400():
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    original_exception = BaseLLMException(
        status_code=400,
        message=(
            '{"error":{"code":"invalid_request_error",'
            '"message":"Invalid input value context_length_exceeded",'
            '"param":"input","type":"invalid_request_error"}}'
        ),
    )

    with pytest.raises(litellm.BadRequestError) as excinfo:
        exception_type(
            model="openai.gpt-5.6-sol",
            original_exception=original_exception,
            custom_llm_provider="bedrock_mantle",
        )

    assert not isinstance(excinfo.value, litellm.ContextWindowExceededError)
    assert excinfo.value.status_code == 400
    assert "Invalid input value context_length_exceeded" in excinfo.value.message


def test_branchless_provider_transport_error_maps_to_api_connection_error():
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    original_exception = BaseLLMException(status_code=500, message="[Errno 111] Connection refused")
    original_exception.status_code_is_synthesized = True

    with pytest.raises(litellm.APIConnectionError):
        exception_type(
            model="test-agent",
            original_exception=original_exception,
            custom_llm_provider="a2a",
        )


def test_openrouter_transport_error_maps_to_api_connection_error():
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    original_exception = BaseLLMException(status_code=500, message="[Errno 111] Connection refused")
    original_exception.status_code_is_synthesized = True

    with pytest.raises(litellm.APIConnectionError):
        exception_type(
            model="typesafe/jev-1.13",
            original_exception=original_exception,
            custom_llm_provider="openrouter",
        )


def test_openrouter_upstream_500_still_maps_to_api_error():
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    original_exception = BaseLLMException(status_code=500, message="upstream exploded")

    with pytest.raises(litellm.APIError) as excinfo:
        exception_type(
            model="typesafe/jev-1.13",
            original_exception=original_exception,
            custom_llm_provider="openrouter",
        )
    assert excinfo.value.status_code == 500


_OPAQUE_CREDENTIAL: Final = "opaque-upstream-credential"
_SHAPED_KEY: Final = "sk-9876543210abcdef9876543210"


def _openai_sdk_bad_request(body: dict[str, object]) -> openai.BadRequestError:  # mutable-ok: the SDK takes a dict
    request: Final = httpx.Request("POST", "https://backend.example.test/v1/chat/completions")
    response: Final = httpx.Response(400, json={"error": body}, request=request)
    return openai.BadRequestError(message=json.dumps({"error": body}), response=response, body=body)


def test_get_error_message_redacts_a_credential_beside_a_structured_message():
    structured: Final = {"detail": [{"loc": ["body", "max_tokens"], "msg": "must be >= 0"}]}

    message: Final = get_error_message(
        _openai_sdk_bad_request({"message": structured, "api_key": [_OPAQUE_CREDENTIAL]})
    )

    assert message is not None
    assert _OPAQUE_CREDENTIAL not in message
    assert json.loads(message)["message"] == structured


@pytest.mark.parametrize(
    ("body", "kept"),
    [
        ({"message": f"upstream rejected key {_SHAPED_KEY}", "code": 400}, "upstream rejected key"),
        ({"message": "Provider returned error", "metadata": {"raw": f"bad key {_SHAPED_KEY}"}}, "bad key"),
        (
            {"message": "Provider returned error", "metadata": {"raw": {"api_key": _OPAQUE_CREDENTIAL, "why": "denied"}}},
            "denied",
        ),
    ],
    ids=["message", "raw_text", "raw_object"],
)
def test_openrouter_error_message_redacts_a_credential_from_the_provider_body(
    body: dict[str, object],  # mutable-ok: the SDK takes a dict
    kept: str,
) -> None:
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(
            model="openrouter/qwen",
            original_exception=_openai_sdk_bad_request(body),
            custom_llm_provider="openrouter",
        )

    assert kept in caught.value.message
    assert [leak for leak in (_SHAPED_KEY, _OPAQUE_CREDENTIAL) if leak in caught.value.message] == []


@pytest.mark.parametrize("redaction_enabled", [True, False], ids=["redaction_on", "redaction_off"])
def test_openrouter_error_message_survives_a_raw_body_nested_past_the_recursion_limit(
    redaction_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm._logging._ENABLE_SECRET_REDACTION", redaction_enabled)
    raw: Final = functools.reduce(
        lambda inner, _: {"inner": inner},
        range(50_000),
        "deep-leaf",
    )
    request: Final = httpx.Request("POST", "https://backend.example.test/v1/chat/completions")
    response: Final = httpx.Response(400, json={"error": {"message": "rejected"}}, request=request)
    original: Final = openai.BadRequestError(
        message="rejected",
        response=response,
        body={"message": "rejected", "metadata": {"raw": raw}},
    )

    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(model="openrouter/qwen", original_exception=original, custom_llm_provider="openrouter")

    assert "rejected" in caught.value.message
    assert "deep-leaf" not in caught.value.message


@pytest.mark.parametrize("status_code", [401, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504])
@pytest.mark.parametrize("provider", ["openai", "azure", "openrouter", "hosted_vllm"])
def test_mapped_error_relays_the_provider_code_and_param_on_every_status(status_code: int, provider: str) -> None:
    request: Final = httpx.Request("POST", "https://backend.example.test/v1/chat/completions")
    response: Final = httpx.Response(status_code, json={"error": {"message": "rejected"}}, request=request)
    original: Final = openai.APIStatusError(
        message="rejected",
        response=response,
        body={"message": "rejected", "code": "provider_code", "param": "temperature"},
    )

    with pytest.raises(openai.APIError) as caught:
        exception_type(model="qwen", original_exception=original, custom_llm_provider=provider)

    assert (relayed_provider_specific_fields(caught.value) or {}).get("code") == "provider_code"
    assert openai_error_param(caught.value) == "temperature"


@pytest.mark.parametrize("provider", ["openai", "azure", "openrouter", "hosted_vllm"])
def test_mid_stream_error_relays_the_provider_code_and_param(provider: str) -> None:
    request: Final = httpx.Request("POST", "https://backend.example.test/v1/chat/completions")
    original: Final = openai.APIError(
        message="upstream broke",
        request=request,
        body={"message": "upstream broke", "code": "upstream_boom", "param": "temperature"},
    )

    with pytest.raises(openai.APIError) as caught:
        exception_type(model="qwen", original_exception=original, custom_llm_provider=provider)

    assert (relayed_provider_specific_fields(caught.value) or {}).get("code") == "upstream_boom"
    assert openai_error_param(caught.value) == "temperature"


def _logged_error_code(mapped: Exception) -> str:
    return StandardLoggingPayloadSetup.get_error_information(original_exception=mapped)["error_code"]


_PROVIDER_ERROR: Final = MappingProxyType(
    {"message": "rejected", "type": "invalid_request_error", "param": "temperature", "code": "provider_code"}
)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 409, 413, 422, 500, 502])
def test_openrouter_failure_logs_its_status_and_relays_the_provider_code(status_code: int) -> None:
    request: Final = httpx.Request("POST", "https://backend.example.test/v1/chat/completions")
    response: Final = httpx.Response(status_code, json={"error": dict(_PROVIDER_ERROR)}, request=request)
    original: Final = openai.APIStatusError(message="rejected", response=response, body=dict(_PROVIDER_ERROR))

    with pytest.raises(openai.APIError) as caught:
        exception_type(model="qwen", original_exception=original, custom_llm_provider="openrouter")

    assert _logged_error_code(caught.value) == str(status_code)
    assert (relayed_provider_specific_fields(caught.value) or {}).get("code") == "provider_code"
    assert openai_error_param(caught.value) == "temperature"


def test_azure_responses_failure_logs_its_status_and_relays_the_provider_code() -> None:
    with pytest.raises(BaseLLMException) as upstream:
        AzureOpenAIResponsesAPIConfig().get_error_class(json.dumps({"error": dict(_PROVIDER_ERROR)}), 400, {})

    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(model="gpt-5.6", original_exception=upstream.value, custom_llm_provider="azure")

    assert _logged_error_code(caught.value) == "400"
    assert (relayed_provider_specific_fields(caught.value) or {}).get("code") == "provider_code"
    assert openai_error_param(caught.value) == "temperature"


def _azure_responses_failure(code: str) -> BaseLLMException:
    with pytest.raises(BaseLLMException) as upstream:
        AzureOpenAIResponsesAPIConfig().get_error_class(
            json.dumps({"error": {**_PROVIDER_ERROR, "code": code}}), 400, {}
        )
    return upstream.value


def _openrouter_failure(code: str) -> openai.APIStatusError:
    request: Final = httpx.Request("POST", "https://backend.example.test/v1/chat/completions")
    error: Final = {**_PROVIDER_ERROR, "code": code}
    response: Final = httpx.Response(400, json={"error": error}, request=request)
    return openai.APIStatusError(message="rejected", response=response, body=error)


def test_azure_content_policy_failure_keeps_logging_its_code() -> None:
    with pytest.raises(litellm.ContentPolicyViolationError) as caught:
        exception_type(
            model="gpt-5.6",
            original_exception=_azure_responses_failure("content_policy_violation"),
            custom_llm_provider="azure",
        )

    assert _logged_error_code(caught.value) == "content_policy_violation"


@pytest.mark.parametrize(
    ("provider", "failure"), [("azure", _azure_responses_failure), ("openrouter", _openrouter_failure)]
)
def test_cyber_policy_code_outside_the_openai_mapping_logs_its_status(
    provider: str, failure: Callable[[str], Exception]
) -> None:
    with pytest.raises(openai.APIError) as caught:
        exception_type(model="gpt-5.6", original_exception=failure("cyber_policy"), custom_llm_provider=provider)

    assert _logged_error_code(caught.value) == "400"


def test_openai_sdk_failure_keeps_logging_the_provider_code() -> None:
    request: Final = httpx.Request("POST", "https://backend.example.test/v1/chat/completions")
    response: Final = httpx.Response(400, json={"error": dict(_PROVIDER_ERROR)}, request=request)
    original: Final = openai.BadRequestError(message="rejected", response=response, body=dict(_PROVIDER_ERROR))

    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(model="gpt-5.6", original_exception=original, custom_llm_provider="openai")

    assert _logged_error_code(caught.value) == "provider_code"


def test_branchless_provider_upstream_500_still_maps_to_internal_server_error():
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    original_exception = BaseLLMException(status_code=500, message="upstream exploded")

    with pytest.raises(litellm.InternalServerError):
        exception_type(
            model="test-agent",
            original_exception=original_exception,
            custom_llm_provider="a2a",
        )


def test_handle_error_marks_only_a_status_code_it_never_received():
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    handler = BaseLLMHTTPHandler()

    with pytest.raises(litellm.llms.base_llm.chat.transformation.BaseLLMException) as transport:
        raise handler.handle_error(e=httpx.ConnectError("Connection refused"), provider_config=None)
    assert transport.value.status_code == 500
    assert transport.value.status_code_is_synthesized is True

    request = httpx.Request(method="POST", url="https://example.invalid")
    upstream = httpx.HTTPStatusError(
        "server error",
        request=request,
        response=httpx.Response(status_code=500, request=request, text="upstream exploded"),
    )
    with pytest.raises(litellm.llms.base_llm.chat.transformation.BaseLLMException) as received:
        raise handler.handle_error(e=upstream, provider_config=None)
    assert received.value.status_code == 500
    assert received.value.status_code_is_synthesized is False


def test_bedrock_500_preserves_provider_response_headers():
    """A Bedrock 5xx must keep x-amzn-RequestId so AWS support can trace it (LIT-5428)."""
    provider_response = httpx.Response(
        status_code=500,
        headers={"x-amzn-RequestId": "req-map-500"},
        text='{"message":"Amazon Bedrock is unable to process your request."}',
        request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/"),
    )
    original_exception = BedrockError(
        status_code=500,
        message=provider_response.text,
        headers=provider_response.headers,
        response=provider_response,
    )

    with pytest.raises(litellm.ServiceUnavailableError) as exc_info:
        exception_type(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            original_exception=original_exception,
            custom_llm_provider="bedrock",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert exc_info.value.response.headers["x-amzn-requestid"] == "req-map-500"


@pytest.mark.parametrize(
    "custom_llm_provider, status_code, provider_message, expected_exception",
    [
        (
            "bedrock_mantle",
            400,
            (
                '{"error":{"code":"validation_error",'
                '"message":"prompt tokens (1055489) exceed model maximum (1050000) for openai.gpt-5.6-sol",'
                '"param":null,"type":"invalid_request_error"}}'
            ),
            litellm.ContextWindowExceededError,
        ),
        (
            "bedrock",
            400,
            '{"message":"Input is too long for requested model."}',
            litellm.ContextWindowExceededError,
        ),
        (
            "bedrock",
            400,
            '{"message":"Could not process image"}',
            litellm.BadRequestError,
        ),
    ],
)
def test_bedrock_classified_errors_preserve_provider_response_headers(
    custom_llm_provider, status_code, provider_message, expected_exception
):
    """Branches that classify a Bedrock error by its text must keep x-amzn-RequestId (LIT-5428)."""
    provider_response = httpx.Response(
        status_code=status_code,
        headers={"x-amzn-RequestId": "req-classified"},
        text=provider_message,
        request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/"),
    )
    original_exception = BedrockError(
        status_code=status_code,
        message=provider_message,
        headers=provider_response.headers,
        response=provider_response,
    )

    with pytest.raises(expected_exception) as exc_info:
        exception_type(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            original_exception=original_exception,
            custom_llm_provider=custom_llm_provider,
            completion_kwargs={},
            extra_kwargs={},
        )

    assert exc_info.value.response.headers["x-amzn-requestid"] == "req-classified"


@pytest.mark.parametrize(
    "status_code, expected_exception",
    [
        (400, litellm.BadRequestError),
        (503, litellm.ServiceUnavailableError),
        (500, litellm.InternalServerError),
    ],
)
def test_bedrock_unprocessable_image_keeps_provider_status_code(status_code, expected_exception):
    """An unprocessable image maps to the status Bedrock sent, so the 400 it returns stays a client error."""
    provider_message = '{"message":"The model returned the following errors: Could not process image"}'
    provider_response = httpx.Response(
        status_code=status_code,
        text=provider_message,
        request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/"),
    )
    original_exception = BedrockError(
        status_code=status_code,
        message=provider_message,
        headers=provider_response.headers,
        response=provider_response,
    )

    with pytest.raises(expected_exception) as exc_info:
        exception_type(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            original_exception=original_exception,
            custom_llm_provider="bedrock",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert exc_info.value.status_code == status_code


@pytest.mark.parametrize(
    "status_code, provider_message",
    [
        (504, '{"message":"Gateway timeout"}'),
        (408, '{"message":"Bedrock did not answer in time"}'),
        (408, '{"message":"Connect timeout on endpoint URL"}'),
    ],
)
def test_bedrock_timeout_mapping_preserves_provider_headers(status_code, provider_message):
    """A mapped bedrock timeout keeps the upstream response, like every other mapped bedrock error.

    The proxy prefixes those headers on the way out, while retry and cooldown
    logic still reads the raw retry-after off the response.
    """
    provider_response = httpx.Response(
        status_code=status_code,
        headers={"x-amzn-RequestId": "req-timeout", "set-cookie": "session=attacker"},
        text=provider_message,
        request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/"),
    )
    original_exception = BedrockError(
        status_code=status_code,
        message=provider_message,
        headers=provider_response.headers,
        response=provider_response,
    )

    with pytest.raises(litellm.Timeout) as exc_info:
        exception_type(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            original_exception=original_exception,
            custom_llm_provider="bedrock",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert exc_info.value.response.headers["x-amzn-requestid"] == "req-timeout"
    assert exc_info.value.headers is None


@pytest.mark.parametrize("status_code", [504, 408])
def test_bedrock_timeout_mapping_keeps_retry_after_readable(status_code):
    """Cooldown and retry timing read retry-after through _get_response_headers."""
    provider_response = httpx.Response(
        status_code=status_code,
        headers={"x-amzn-RequestId": "req-retry-after", "retry-after": "7"},
        text='{"message":"Bedrock did not answer in time"}',
        request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/"),
    )
    original_exception = BedrockError(
        status_code=status_code,
        message='{"message":"Bedrock did not answer in time"}',
        headers=provider_response.headers,
        response=provider_response,
    )

    with pytest.raises(litellm.Timeout) as exc_info:
        exception_type(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            original_exception=original_exception,
            custom_llm_provider="bedrock",
            completion_kwargs={},
            extra_kwargs={},
        )

    exception_headers = _get_response_headers(original_exception=exc_info.value)
    assert exception_headers is not None
    assert litellm.utils._get_retry_after_from_exception_header(response_headers=exception_headers) == 7


_GUARDRAIL_BLOCK_ERROR = {
    "message": "Content blocked: secret_project_codename pattern detected",
    "param": "None",
    "code": "400",
    "provider_specific_fields": {
        "error": "Content blocked: secret_project_codename pattern detected",
        "pattern": "secret_project_codename",
        "guardrail_name": "block-secret-project",
        "guardrail_mode": "pre_call",
    },
}


def _openai_handler_error(
    error_type: str,
    headers: dict[str, str] | list[tuple[str, str]],
    status_code: int = 400,
    message: str = _GUARDRAIL_BLOCK_ERROR["message"],
) -> OpenAIError:
    wire_error = {**_GUARDRAIL_BLOCK_ERROR, "type": error_type, "code": str(status_code), "message": message}
    return OpenAIError(
        status_code=status_code,
        message=f"Error code: {status_code} - {{'error': {wire_error}}}",
        headers=httpx.Headers(headers),
        body=wire_error,
    )


_PROXY_HEADERS = {"x-litellm-call-id": "call-guardrail", "x-litellm-applied-guardrails": "block-secret-project"}


@pytest.mark.parametrize(("error_type", "status_code"), [("None", 400), ("invalid_request_error", 400), ("None", 422)])
def test_litellm_proxy_guardrail_block_keeps_body_and_headers(error_type: str, status_code: int):
    with pytest.raises(litellm.BadRequestError) as exc_info:
        exception_type(
            model="claude-haiku-4-5",
            original_exception=_openai_handler_error(error_type, _PROXY_HEADERS, status_code=status_code),
            custom_llm_provider="litellm_proxy",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert exc_info.value.body["provider_specific_fields"]["guardrail_name"] == "block-secret-project"
    assert exc_info.value.body["type"] == error_type
    assert dict(exc_info.value.response.headers) == _PROXY_HEADERS


@pytest.mark.parametrize("relayed_class", [litellm.BadRequestError, litellm.ContentPolicyViolationError])
def test_litellm_proxy_relayed_litellm_error_keeps_body_and_headers(relayed_class: type[litellm.BadRequestError]):
    message = f"litellm.{relayed_class.__name__}: {_GUARDRAIL_BLOCK_ERROR['message']}"

    with pytest.raises(relayed_class) as exc_info:
        exception_type(
            model="claude-haiku-4-5",
            original_exception=_openai_handler_error("None", _PROXY_HEADERS, message=message),
            custom_llm_provider="litellm_proxy",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert type(exc_info.value) is relayed_class
    assert exc_info.value.body["provider_specific_fields"]["guardrail_name"] == "block-secret-project"
    assert dict(exc_info.value.response.headers) == _PROXY_HEADERS


def test_openai_compatible_vendor_400_keeps_body_but_not_headers():
    with pytest.raises(litellm.BadRequestError) as exc_info:
        exception_type(
            model="gpt-5.4-mini",
            original_exception=_openai_handler_error("vendor_specific_error", {"openai-organization": "org-1"}),
            custom_llm_provider="openai",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert exc_info.value.body["type"] == "vendor_specific_error"
    assert not exc_info.value.response.headers


@pytest.mark.parametrize(
    ("status_code", "mapped_class", "reported_type"),
    [(429, litellm.RateLimitError, "throttling_error"), (500, litellm.InternalServerError, "internal_server_error")],
)
def test_openai_429_and_500_keep_body_but_report_litellm_type(
    status_code: int, mapped_class: type[openai.APIError], reported_type: str
):
    with pytest.raises(mapped_class) as exc_info:
        exception_type(
            model="gpt-5.4-mini",
            original_exception=_openai_handler_error(
                "server_error", {}, status_code=status_code, message="upstream cannot complete this response"
            ),
            custom_llm_provider="openai",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert exc_info.value.body == {
        **_GUARDRAIL_BLOCK_ERROR,
        "type": "server_error",
        "code": str(status_code),
        "message": "upstream cannot complete this response",
    }
    assert exc_info.value.type == reported_type


def test_litellm_proxy_repeated_response_header_keeps_each_value():
    repeated = [("x-litellm-call-id", "call-guardrail"), ("set-cookie", "a=1"), ("set-cookie", "b=2")]

    with pytest.raises(litellm.BadRequestError) as exc_info:
        exception_type(
            model="claude-haiku-4-5",
            original_exception=_openai_handler_error("None", repeated),
            custom_llm_provider="litellm_proxy",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert exc_info.value.response.headers.multi_items() == repeated


@pytest.mark.parametrize(
    "block",
    [
        HTTPException(status_code=400, detail={"error": "Violated guardrail policy"}),
        HTTPException(status_code=422, detail={"error": "Violated guardrail policy"}),
        GuardrailRaisedException(guardrail_name="prompt-shield", message="Violated guardrail policy"),
    ],
    ids=["http_400", "http_422", "guardrail_raised"],
)
def test_guardrail_block_raised_inside_an_llm_call_is_returned_unmapped(block: Exception):
    returned = exception_type(
        model="gpt-5.6",
        original_exception=block,
        custom_llm_provider="openai",
        completion_kwargs={},
        extra_kwargs={},
    )

    assert returned is block


@pytest.mark.parametrize("provider", ["bedrock", "bedrock_mantle"])
@pytest.mark.parametrize(
    "failure", [ImportError("Run 'pip install boto3'."), ModuleNotFoundError(name="unrelated_dependency")]
)
def test_bedrock_import_errors_preserve_the_original_exception(provider, failure):
    assert exception_type(model="test-model", original_exception=failure, custom_llm_provider=provider) is failure


def test_guardrail_provider_failure_status_is_still_mapped():
    upstream_failure = HTTPException(status_code=401, detail={"error": "guardrail provider rejected the key"})

    with pytest.raises(litellm.AuthenticationError) as exc_info:
        exception_type(
            model="gpt-5.6",
            original_exception=upstream_failure,
            custom_llm_provider="openai",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert exc_info.value is not upstream_failure


class _MockProviderException(Exception):
    def __init__(self, status_code: int, llm_provider: str) -> None:
        super().__init__("This is an error message")
        self.text = "This is an error message"
        self.llm_provider = llm_provider
        self.status_code = status_code


def _provider_error_response(request: httpx.Request, status_code: int, body: bytes) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json", "retry-after": "30"},
        content=body,
        request=request,
    )


def _assert_retry_after_header(error: litellm.RateLimitError) -> None:
    response_headers: Final = error.litellm_response_headers
    assert response_headers is not None
    assert response_headers["retry-after"] == "30"


def _assert_sync_rate_limit(call: Callable[[], object]) -> None:
    with pytest.raises(litellm.RateLimitError) as exc_info:
        call()
    _assert_retry_after_header(exc_info.value)


async def _assert_async_rate_limit(call: Callable[[], Awaitable[object]]) -> None:
    with pytest.raises(litellm.RateLimitError) as exc_info:
        await call()
    _assert_retry_after_header(exc_info.value)


def _call_openai_api_sync(
    model: str,
    call_type: Literal["embedding", "chat_completion", "completion"],
    streaming: bool | None,
    client: openai.OpenAI | openai.AzureOpenAI,
) -> object:
    if call_type == "embedding":
        return litellm.embedding(
            model=model,
            input="Hello world!",
            client=client,
            num_retries=0,
        )
    if call_type == "chat_completion":
        return litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hello world"}],
            stream=streaming,
            client=client,
            num_retries=0,
        )
    if streaming is True:
        response: Final = litellm.text_completion(
            model=model,
            prompt="Hello world",
            stream=True,
            client=client,
            num_retries=0,
        )
        for _chunk in response:
            pass
        return response
    return litellm.text_completion(
        model=model,
        prompt="Hello world",
        stream=streaming,
        client=client,
        num_retries=0,
    )


async def _call_openai_api_async(
    model: str,
    call_type: Literal["embedding", "chat_completion", "completion"],
    streaming: bool | None,
    client: openai.AsyncOpenAI | openai.AsyncAzureOpenAI,
) -> object:
    if call_type == "embedding":
        return await litellm.aembedding(
            model=model,
            input="Hello world!",
            client=client,
            num_retries=0,
        )
    if call_type == "chat_completion":
        return await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Hello world"}],
            stream=streaming,
            client=client,
            num_retries=0,
        )
    if streaming is True:
        response: Final = await litellm.atext_completion(
            model=model,
            prompt="Hello world",
            stream=True,
            client=client,
            num_retries=0,
        )
        async for _chunk in response:
            pass
        return response
    return await litellm.atext_completion(
        model=model,
        prompt="Hello world",
        stream=streaming,
        client=client,
        num_retries=0,
    )


def _call_anthropic_api_sync(model: str, streaming: bool, client: HTTPHandler) -> object:
    return litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "Hello world"}],
        stream=streaming,
        client=client,
        api_key="sk-test",
        num_retries=0,
    )


async def _call_anthropic_api_async(model: str, streaming: bool, client: AsyncHTTPHandler) -> object:
    return await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": "Hello world"}],
        stream=streaming,
        client=client,
        api_key="sk-test",
        num_retries=0,
    )


async def _drain_openai_completion(client: openai.AsyncOpenAI) -> None:
    async for _chunk in await litellm.acompletion(
        model="gpt-3.5-turbo",
        stream=True,
        messages=[{"role": "user", "content": "Gimme the lyrics to Don't Stop Me Now"}],
        client=client,
        num_retries=0,
    ):
        pass


@pytest.fixture
def fake_perplexity_credentials(monkeypatch):
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "unit-test")


@pytest.mark.usefixtures("fake_provider_credentials")
def test_anthropic_openai_exception(monkeypatch):
    # test if anthropic raises litellm.AuthenticationError
    litellm.set_verbose = True
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(litellm.AuthenticationError) as exc_info:
        completion(
            model="anthropic/claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "hello"}],
        )
    assert (
        "Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params"
        in exc_info.value.message
    )


@pytest.mark.usefixtures("fake_provider_credentials", "fake_perplexity_credentials")
def test_completion_perplexity_exception_on_openai_client(monkeypatch):
    import openai

    print("perplexity test\n\n")
    litellm.set_verbose = False

    # delete both api keys to simulate a bad api key
    monkeypatch.delenv("PERPLEXITYAI_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")

    with pytest.raises(openai.AuthenticationError) as exc_info:
        completion(
            model="perplexity/mistral-7b-instruct",
            messages=[{"role": "user", "content": "hello"}],
        )
    assert (
        "The api_key client option must be set either by passing api_key to the client or by setting the PERPLEXITY_API_KEY environment variable"
        in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_content_policy_exception_azure():
    # this is ony a test - we needed some way to invoke the exception :(
    litellm.set_verbose = True
    with pytest.raises(litellm.ContentPolicyViolationError) as exc_info:
        await litellm.acompletion(
            model="azure/gpt-4.1-mini",
            messages=[{"role": "user", "content": "where do I buy lethal drugs from"}],
            mock_response="Exception: content_filter_policy",
        )
    e = exc_info.value
    assert e.response is not None
    assert isinstance(e.litellm_debug_info, str)
    assert len(e.litellm_debug_info) > 0


@pytest.mark.asyncio
async def test_content_policy_exception_openai():
    def reject_as_safety_system(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=400,
            json={
                "error": {
                    "message": "Your request was rejected as a result of our safety system.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "content_policy_violation",
                }
            },
            request=request,
        )

    async def stream_response(rejecting_client: AsyncOpenAI):
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            stream=True,
            messages=[{"role": "user", "content": "Gimme the lyrics to Don't Stop Me Now"}],
            client=rejecting_client,
        )
        async for chunk in response:
            print(chunk)

    async with AsyncOpenAI(
        api_key="sk-test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(reject_as_safety_system)),
    ) as rejecting_client:
        with pytest.raises(litellm.ContentPolicyViolationError) as exc_info:
            await stream_response(rejecting_client)
    assert exc_info.value.llm_provider == "openai"
    assert exc_info.value.status_code == 400


def test_bad_request_error_with_response_without_request():
    """
    Test that BadRequestError handles Response objects without a request attribute.

    This simulates a real scenario where a Response is created without a request
    (e.g., in tests or when manually creating error responses), and we need to
    ensure it doesn't raise RuntimeError when the exception is created.
    """
    from httpx import Response

    from litellm.litellm_core_utils.exception_mapping_utils import (
        extract_and_raise_litellm_exception,
    )

    # Create a Response without a request (simulates the scenario that was failing)
    response_without_request = Response(status_code=400, text="Bad Request")

    # Test that extract_and_raise_litellm_exception can handle this
    args = {
        "response": response_without_request,
        "error_str": "Error code: 400 - {'error': {'message': 'litellm.BadRequestError: Invalid request parameters', 'type': None, 'param': None, 'code': '400'}}",
        "model": "gpt-3.5-turbo",
        "custom_llm_provider": "openai",
    }

    # This should raise BadRequestError without RuntimeError
    with pytest.raises(litellm.BadRequestError) as exc_info:
        extract_and_raise_litellm_exception(**args)

    # Verify the exception was created successfully
    error = exc_info.value
    assert error is not None
    assert error.model == "gpt-3.5-turbo"
    assert error.llm_provider == "openai"

    # Verify the exception has a response (should be minimal error response)
    assert error.response is not None
    # The response should have a request (minimal error response has one)
    assert getattr(error.response, "_request", None) is not None
    # Should be able to access request property without RuntimeError
    assert error.response.request is not None


def test_context_window_exceeded_error_from_litellm_proxy():
    from httpx import Response

    from litellm.litellm_core_utils.exception_mapping_utils import (
        extract_and_raise_litellm_exception,
    )

    args = {
        "response": Response(status_code=400, text="Bad Request"),
        "error_str": "Error code: 400 - {'error': {'message': \"litellm.ContextWindowExceededError: litellm.BadRequestError: this is a mock context window exceeded error\\nmodel=gpt-3.5-turbo. context_window_fallbacks=None. fallbacks=None.\\n\\nSet 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks\\nReceived Model Group=gpt-3.5-turbo\\nAvailable Model Group Fallbacks=None\", 'type': None, 'param': None, 'code': '400'}}",
        "model": "gpt-3.5-turbo",
        "custom_llm_provider": "litellm_proxy",
    }
    with pytest.raises(litellm.ContextWindowExceededError):
        extract_and_raise_litellm_exception(**args)


@pytest.mark.parametrize(
    "provider",
    [
        "predibase",
        "vertex_ai_beta",
        "anthropic",
        "databricks",
        "watsonx",
        "fireworks_ai",
    ],
)
def test_exception_mapping(provider):
    """
    For predibase, run through a set of mock exceptions

    assert that they are being mapped correctly
    """
    litellm.set_verbose = True
    error_map = {
        400: litellm.BadRequestError,
        401: litellm.AuthenticationError,
        404: litellm.NotFoundError,
        408: litellm.Timeout,
        429: litellm.RateLimitError,
        500: litellm.InternalServerError,
        503: litellm.ServiceUnavailableError,
    }

    for code, expected_exception in error_map.items():
        mock_response = Exception()
        setattr(mock_response, "text", "This is an error message")
        setattr(mock_response, "llm_provider", provider)
        setattr(mock_response, "status_code", code)

        response: Any = None
        try:
            response = completion(
                model="{}/test-model".format(provider),
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response=mock_response,
            )
        except expected_exception:
            continue
        except Exception as e:
            traceback.print_exc()
            response = "{}".format(str(e))
        pytest.fail("Did not raise expected exception. Expected={}, Return={},".format(expected_exception, response))

    pass


def test_exceptions_base_class():
    with pytest.raises(litellm.RateLimitError) as exc_info:
        raise litellm.RateLimitError(
            message="BedrockException: Rate Limit Error",
            model="model",
            llm_provider="bedrock",
        )
    e = exc_info.value
    assert isinstance(e, litellm.RateLimitError)
    assert e.code == "429"
    assert e.type == "throttling_error"


def test_fireworks_ai_exception_mapping():
    """
    Comprehensive test for Fireworks AI exception mapping, including:
    1. Standard 429 rate limit errors
    2. Text-based rate limit detection (the main issue fixed)
    3. Generic 400 errors that should NOT be rate limits
    4. ExceptionCheckers utility function

    Related to: https://github.com/BerriAI/litellm/pull/11455
    Based on Fireworks AI documentation: https://docs.fireworks.ai/tools-sdks/python-client/api-reference
    """
    import litellm
    from litellm.litellm_core_utils.exception_mapping_utils import ExceptionCheckers
    from litellm.llms.fireworks_ai.common_utils import FireworksAIException

    # Test scenarios covering all important cases
    test_scenarios = [
        {
            "name": "Standard 429 rate limit with proper status code",
            "status_code": 429,
            "message": "Rate limit exceeded. Please try again in 60 seconds.",
            "expected_exception": litellm.RateLimitError,
        },
        {
            "name": "Status 400 with rate limit text (the main issue fixed)",
            "status_code": 400,
            "message": '{"error":{"object":"error","type":"invalid_request_error","message":"rate limit exceeded, please try again later"}}',
            "expected_exception": litellm.RateLimitError,
        },
        {
            "name": "Status 400 with generic invalid request (should NOT be rate limit)",
            "status_code": 400,
            "message": '{"error":{"type":"invalid_request_error","message":"Invalid parameter value"}}',
            "expected_exception": litellm.BadRequestError,
        },
    ]

    # Test each scenario
    for scenario in test_scenarios:
        mock_exception = FireworksAIException(
            status_code=scenario["status_code"], message=scenario["message"], headers={}
        )

        with pytest.raises(scenario["expected_exception"]) as exc_info:
            litellm.completion(
                model="fireworks_ai/llama-v3p1-70b-instruct",
                messages=[{"role": "user", "content": "Hello"}],
                mock_response=mock_exception,
            )
        if scenario["expected_exception"] == litellm.RateLimitError:
            error_str = str(exc_info.value)
            assert "rate limit" in error_str.lower() or "429" in error_str

    # Test ExceptionCheckers.is_error_str_rate_limit() method directly

    # Test cases that should return True (rate limit detected)
    rate_limit_strings = [
        "429 rate limit exceeded",
        "Rate limit exceeded, please try again later",
        "RATE LIMIT ERROR",
        "Error 429: rate limit",
        '{"error":{"type":"invalid_request_error","message":"rate limit exceeded, please try again later"}}',
        "HTTP 429 Too Many Requests",
    ]

    for error_str in rate_limit_strings:
        assert ExceptionCheckers.is_error_str_rate_limit(error_str), f"Should detect rate limit in: {error_str}"

    # Test cases that should return False (not rate limit)
    non_rate_limit_strings = [
        "400 Bad Request",
        "Authentication failed",
        "Invalid model specified",
        "Context window exceeded",
        "Internal server error",
        "",
        "Some other error message",
    ]

    for error_str in non_rate_limit_strings:
        assert not ExceptionCheckers.is_error_str_rate_limit(error_str), f"Should NOT detect rate limit in: {error_str}"

    # Test edge cases
    assert not ExceptionCheckers.is_error_str_rate_limit(None)  # type: ignore
    assert not ExceptionCheckers.is_error_str_rate_limit(42)  # type: ignore


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize(
    "provider, model, call_type, streaming",
    [
        ("openai", "text-embedding-ada-002", "embedding", None),
        ("openai", "gpt-3.5-turbo", "chat_completion", False),
        ("openai", "gpt-3.5-turbo", "chat_completion", True),
        ("openai", "gpt-3.5-turbo-instruct", "completion", True),
        ("azure", "azure/gpt-4.1-mini", "chat_completion", True),
    ],
)
@pytest.mark.asyncio
async def test_exception_with_headers(
    sync_mode: bool,
    provider: Literal["openai", "azure"],
    model: str,
    call_type: Literal["embedding", "chat_completion", "completion"],
    streaming: bool | None,
) -> None:
    error_body: Final = b'{"error":{"message":"Too many requests","type":"rate_limit_error"}}'
    transport: Final = httpx.MockTransport(
        lambda request: _provider_error_response(request, status_code=429, body=error_body)
    )

    if sync_mode:
        sync_http_client: Final = httpx.Client(transport=transport)
        with sync_http_client:
            if provider == "openai":
                sync_openai_client: Final = openai.OpenAI(
                    api_key="sk-test",
                    max_retries=0,
                    http_client=sync_http_client,
                )
                with sync_openai_client:
                    _assert_sync_rate_limit(
                        lambda: _call_openai_api_sync(model, call_type, streaming, sync_openai_client)
                    )
            else:
                sync_azure_client: Final = openai.AzureOpenAI(
                    api_key="sk-test",
                    azure_endpoint="https://example.invalid",
                    api_version=litellm.AZURE_DEFAULT_API_VERSION,
                    max_retries=0,
                    http_client=sync_http_client,
                )
                with sync_azure_client:
                    _assert_sync_rate_limit(
                        lambda: _call_openai_api_sync(model, call_type, streaming, sync_azure_client)
                    )
        return

    async_http_client: Final = httpx.AsyncClient(transport=transport)
    async with async_http_client:
        if provider == "openai":
            async_openai_client: Final = openai.AsyncOpenAI(
                api_key="sk-test",
                max_retries=0,
                http_client=async_http_client,
            )
            async with async_openai_client:
                await _assert_async_rate_limit(
                    lambda: _call_openai_api_async(model, call_type, streaming, async_openai_client)
                )
        else:
            async_azure_client: Final = openai.AsyncAzureOpenAI(
                api_key="sk-test",
                azure_endpoint="https://example.invalid",
                api_version=litellm.AZURE_DEFAULT_API_VERSION,
                max_retries=0,
                http_client=async_http_client,
            )
            async with async_azure_client:
                await _assert_async_rate_limit(
                    lambda: _call_openai_api_async(model, call_type, streaming, async_azure_client)
                )


@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    "provider, model, call_type",
    [
        ("anthropic", "claude-haiku-4-5-20251001", "chat_completion"),
    ],
)
@pytest.mark.usefixtures("fake_provider_credentials")
@pytest.mark.asyncio
async def test_exception_with_headers_httpx(sync_mode, provider, model, call_type, streaming):
    """
    User feedback: litellm says "No deployments available for selected model, Try again in 60 seconds"
    but Azure says to retry in at most 9s

    ```
    {"message": "litellm.proxy.proxy_server.embeddings(): Exception occured - No deployments available for selected model, Try again in 60 seconds. Passed model=text-embedding-ada-002. pre-call-checks=False, allowed_model_region=n/a, cooldown_list=[('b49cbc9314273db7181fe69b1b19993f04efb88f2c1819947c538bac08097e4c', {'Exception Received': 'litellm.RateLimitError: AzureException RateLimitError - Requests to the Embeddings_Create Operation under Azure OpenAI API version 2023-09-01-preview have exceeded call rate limit of your current OpenAI S0 pricing tier. Please retry after 9 seconds. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit.', 'Status Code': '429'})]", "level": "ERROR", "timestamp": "2024-08-22T03:25:36.900476"}
    ```
    """
    print(f"Received args: {locals()}")

    if sync_mode:
        client = HTTPHandler()
    else:
        client = AsyncHTTPHandler()

    data = {"model": model}
    data, original_function, mapped_target = _pre_call_utils_httpx(
        call_type=call_type,
        data=data,
        client=client,
        sync_mode=sync_mode,
        streaming=streaming,
    )

    cooldown_time = 30.0

    def _return_exception(*args, **kwargs):

        from httpx import Headers, HTTPStatusError, Request, Response

        # Create the Request object
        request = Request("POST", "http://0.0.0.0:9000/chat/completions")

        # Create the Response object with the necessary headers and status code
        response = Response(
            status_code=429,
            headers=Headers(
                {
                    "date": "Sat, 21 Sep 2024 22:56:53 GMT",
                    "server": "uvicorn",
                    "retry-after": "30",
                    "content-length": "30",
                    "content-type": "application/json",
                }
            ),
            request=request,
        )

        # Create and raise the HTTPStatusError exception
        raise HTTPStatusError(
            message="Error code: 429 - Rate Limit Error!",
            request=request,
            response=response,
        )

    with patch.object(
        mapped_target,
        "send",
        side_effect=_return_exception,
    ):
        new_retry_after_mock_client = MagicMock(return_value=-1)

        litellm.utils._get_retry_after_from_exception_header = new_retry_after_mock_client

        async def call_and_drain():
            if sync_mode:
                resp = original_function(**data, client=client)
                if streaming:
                    for chunk in resp:
                        continue
            else:
                resp = await original_function(**data, client=client)

                if streaming:
                    async for chunk in resp:
                        continue

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await call_and_drain()

        assert exc_info.value.litellm_response_headers is not None, "litellm_response_headers is None"
        print("e.litellm_response_headers", exc_info.value.litellm_response_headers)
        assert int(exc_info.value.litellm_response_headers["retry-after"]) == cooldown_time


@pytest.mark.usefixtures("fake_provider_credentials")
def test_openai_gateway_timeout_error():
    """
    Test that the OpenAI gateway timeout error is raised
    """
    openai_client = OpenAI()

    def _return_exception(*args, **kwargs):

        from httpx import Headers, Request, Response

        kwargs = {
            "request": Request("POST", "https://www.google.com"),
            "message": "Error code: 504 - Gateway Timeout Error!",
            "body": {"detail": "Gateway Timeout Error!"},
            "code": None,
            "param": None,
            "type": None,
            "response": Response(
                status_code=504,
                headers=Headers(
                    {
                        "date": "Sat, 21 Sep 2024 22:56:53 GMT",
                        "server": "uvicorn",
                        "content-length": "30",
                        "content-type": "application/json",
                    }
                ),
                request=Request("POST", "http://0.0.0.0:9000/chat/completions"),
            ),
            "status_code": 504,
            "request_id": None,
        }

        exception = Exception()
        for k, v in kwargs.items():
            setattr(exception, k, v)
        raise exception

    with pytest.raises(litellm.Timeout) as exc_info:
        with patch.object(
            openai_client,
            "post",
            side_effect=_return_exception,
        ):
            litellm.completion(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello world"}],
                client=openai_client,
            )
    e = exc_info.value
    assert e.status_code == 504


def _pre_call_utils_httpx(
    call_type: str,
    data: dict,
    client: Union[HTTPHandler, AsyncHTTPHandler],
    sync_mode: bool,
    streaming: Optional[bool],
):
    mapped_target: Any = client.client
    if call_type == "embedding":
        data["input"] = "Hello world!"

        if sync_mode:
            original_function = litellm.embedding
        else:
            original_function = litellm.aembedding
    elif call_type == "chat_completion":
        data["messages"] = [{"role": "user", "content": "Hello world"}]
        if streaming is True:
            data["stream"] = True

        if sync_mode:
            original_function = litellm.completion
        else:
            original_function = litellm.acompletion
    elif call_type == "completion":
        data["prompt"] = "Hello world"
        if streaming is True:
            data["stream"] = True
        if sync_mode:
            original_function = litellm.text_completion
        else:
            original_function = litellm.atext_completion

    return data, original_function, mapped_target


@pytest.mark.parametrize("provider", ["sagemaker", "sagemaker_chat", "aws_polly", "openai"])
@pytest.mark.parametrize("dependency", ["boto3", "botocore"])
def test_missing_aws_dependency_is_not_mapped_to_provider_failure(provider, dependency):
    failure = ModuleNotFoundError(f"No module named '{dependency}'", name=dependency)
    assert exception_type(model="test-model", original_exception=failure, custom_llm_provider=provider) is failure


@pytest.mark.parametrize("failure", [ImportError("broken import"), ModuleNotFoundError(name="unrelated_dependency")])
def test_non_aws_import_failure_keeps_provider_mapping(failure):
    with pytest.raises(litellm.APIConnectionError):
        exception_type(model="test-model", original_exception=failure, custom_llm_provider="openai")


@pytest.mark.parametrize(
    "transport_timeout,reason",
    [
        (httpx.ReadTimeout(""), "ReadTimeout"),
        (httpx.ReadTimeout("Timeout on reading data from socket"), "Timeout on reading data from socket"),
        (httpx.TimeoutException(""), "TimeoutException"),
    ],
    ids=["httpx_read_timeout", "aiohttp_read_timeout", "timeout_base_class"],
)
@pytest.mark.parametrize("provider", ["hosted_vllm", "openai", "anthropic"])
def test_transport_timeout_maps_to_timeout(
    transport_timeout: httpx.TimeoutException, reason: str, provider: str
) -> None:
    with pytest.raises(litellm.Timeout) as exc_info:
        exception_type(model="glm", original_exception=transport_timeout, custom_llm_provider=provider)

    assert exc_info.value.status_code == 408
    assert f"Request timed out. Error_str: {reason}" in str(exc_info.value)


def test_bedrock_server_error_keeps_the_aws_body_without_the_signed_request() -> None:
    body: Final = '{"message":"Bedrock is unable to process your request."}'
    signed_request: Final = httpx.Request(
        "POST",
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-opus-5-5/converse",
        headers={"authorization": "AWS4-HMAC-SHA256 Credential=AKIDSIGNED/20261029/us-east-1/bedrock/aws4_request"},
        content=b'{"messages":[{"role":"user","content":[{"text":"hi"}]}]}',
    )
    upstream: Final = httpx.Response(
        500, content=body, headers={"x-amzn-requestid": "req-aws"}, request=signed_request
    )

    with pytest.raises(litellm.ServiceUnavailableError) as raised:
        exception_type(
            model="bedrock/anthropic.claude-opus-5-5",
            original_exception=BedrockError(status_code=500, message=body, response=upstream),
            custom_llm_provider="bedrock",
        )

    assert raised.value.response.text == body
    assert raised.value.response.headers["x-amzn-requestid"] == "req-aws"
    assert "AKIDSIGNED" not in str(raised.value.request.headers)
    assert raised.value.request.content == b""


@pytest.mark.parametrize("provider", ("gemini", "vertex_ai"))
def test_gemini_server_error_response_carries_the_upstream_error_body(provider: str) -> None:
    body: Final = '{"error":{"code":500,"message":"Internal error encountered.","status":"INTERNAL"}}'

    with pytest.raises(litellm.InternalServerError) as raised:
        exception_type(
            model="gemini-3.1-pro",
            original_exception=VertexAIError(status_code=500, message=body, headers=None),
            custom_llm_provider=provider,
        )

    assert raised.value.response.status_code == 500
    assert raised.value.response.json() == json.loads(body)


@pytest.mark.parametrize(
    ("status_code", "message", "mapped_error"),
    (
        (500, "The server had an error processing your request. Sorry about that!", litellm.InternalServerError),
        (520, "Web server is returning an unknown error", litellm.InternalServerError),
        (504, "A timeout occurred", litellm.Timeout),
        (500, "Mistral API raised a streaming error", litellm.APIError),
    ),
)
def test_openai_error_matched_by_its_message_keeps_the_upstream_body(
    status_code: int, message: str, mapped_error: type[Exception]
) -> None:
    body: Final = json.dumps({"error": {"message": message, "type": "server_error", "param": None, "code": None}})
    sent_request: Final = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        headers={"authorization": "Bearer sk-openai-secret"},
        content=b'{"messages":[{"role":"user","content":"hi"}]}',
    )
    upstream: Final = httpx.Response(
        status_code, content=body, headers={"x-request-id": "req-openai"}, request=sent_request
    )

    with pytest.raises(mapped_error) as raised:
        exception_type(
            model="gpt-5.6",
            original_exception=openai.APIStatusError(message, response=upstream, body=json.loads(body)),
            custom_llm_provider="openai",
        )

    error_response: Final = getattr(raised.value, "response")
    assert error_response.text == body
    assert error_response.headers["x-request-id"] == "req-openai"
    assert "sk-openai-secret" not in str(error_response.request.headers)
    assert error_response.request.content == b""


def test_openai_compatible_error_relayed_as_json_text_keeps_its_param_and_plain_message(
    quiet_exception_mapping: None,
) -> None:
    upstream_message: Final = "max_tokens must be at least 1, got -5. (parameter=max_tokens, value=-5)"
    relayed_text: Final = json.dumps(
        {"error": {"message": upstream_message, "type": "BadRequestError", "param": "max_tokens", "code": 400}}
    )

    with pytest.raises(litellm.BadRequestError) as raised:
        exception_type(
            model="zai-org/GLM-5.3",
            original_exception=OpenAIError(status_code=400, message=relayed_text),
            custom_llm_provider="hosted_vllm",
        )

    assert raised.value.param == "max_tokens"
    assert raised.value.message == f"litellm.BadRequestError: Hosted_vllmException - {upstream_message}"


@pytest.mark.parametrize("message", ("plain text failure", '{"error": "plain text failure"}', "[1, 2]"))
def test_openai_compatible_error_without_a_relayed_error_object_keeps_its_message(
    message: str, quiet_exception_mapping: None
) -> None:
    with pytest.raises(litellm.BadRequestError) as raised:
        exception_type(
            model="zai-org/GLM-5.3",
            original_exception=OpenAIError(status_code=400, message=message),
            custom_llm_provider="hosted_vllm",
        )

    assert raised.value.param is None
    assert raised.value.message == f"litellm.BadRequestError: Hosted_vllmException - {message}"


@pytest.mark.parametrize("provider", ("openai", "hosted_vllm"))
@pytest.mark.parametrize(
    ("status", "error_type", "error_code", "mapped_error"),
    [
        (404, "not_found_error", "thread_not_found", litellm.NotFoundError),
        (400, "invalid_request_error", "thread_unsupported_request", litellm.BadRequestError),
    ],
)
def test_anthropic_thread_refusals_keep_their_details_on_openai_compatible_legs(
    provider: str,
    status: int,
    error_type: str,
    error_code: str,
    mapped_error: type[litellm.NotFoundError] | type[litellm.BadRequestError],
    quiet_exception_mapping: None,
) -> None:
    message: Final = json.dumps(
        {
            "type": "error",
            "error": {
                "type": error_type,
                "message": "Threads are not supported.",
                "details": {"error_code": error_code},
            },
        }
    )
    with pytest.raises(mapped_error) as raised:
        exception_type(
            model="test-model",
            original_exception=AnthropicError(status_code=status, message=message),
            custom_llm_provider=provider,
        )

    assert AnthropicExceptionMapping.embedded_error_details(raised.value.message) == {"error_code": error_code}


_PER_USER_SESSION: Final = GithubCopilotUserSession(token="ghu-session", endpoints=github_host_endpoints("github.com"))
_APP_SESSION: Final = GithubCopilotUserSession(
    token="ghs-app", endpoints=github_host_endpoints("github.com"), is_app=True
)


def _copilot_model_not_supported_message(request_kwargs: dict[str, object]) -> str:
    copilot_body: Final = {
        "error": {
            "message": "The requested model is not supported.",
            "code": "model_not_supported",
            "param": "model",
            "type": "invalid_request_error",
        }
    }
    request: Final = httpx.Request("POST", "https://api.githubcopilot.com/chat/completions")
    original: Final = OpenAIError(
        status_code=400,
        message=json.dumps(copilot_body),
        request=request,
        response=httpx.Response(400, request=request),
        body=copilot_body["error"],
    )
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(
            model="claude-opus-4.6",
            original_exception=original,
            custom_llm_provider="github_copilot",
            completion_kwargs={},
            extra_kwargs=request_kwargs,
        )
    assert caught.value.status_code == 400
    assert "The requested model is not supported." in caught.value.message
    assert "gho-" not in caught.value.message
    assert "ghu-" not in caught.value.message
    assert "ghs-" not in caught.value.message
    return caught.value.message


@pytest.mark.parametrize(
    ("request_kwargs", "account"),
    [
        pytest.param(
            {"litellm_credential_name": "shared-copilot"},
            "the GitHub account this LiteLLM proxy signed in to GitHub Copilot with",
            id="shared-device-login",
        ),
        pytest.param(
            {
                "litellm_credential_name": "team-copilot",
                "github_copilot_user_session": _PER_USER_SESSION,
                "secret_fields": {
                    "user_provider_credentials_user_id": "dev-alice",
                    "user_provider_credentials": {
                        "team-copilot": {"access_token": "gho-alice", "github_login": "alice-gh"},
                        "other-copilot": {"access_token": "gho-other", "github_login": "someone-else"},
                    },
                },
            },
            "GitHub account 'alice-gh', which LiteLLM user 'dev-alice' connected for LiteLLM credential "
            "'team-copilot'",
            id="per-user-connected-login",
        ),
        pytest.param(
            {
                "litellm_credential_name": "team-copilot",
                "github_copilot_user_session": _PER_USER_SESSION,
                "secret_fields": {
                    "user_provider_credentials_user_id": "dev-alice",
                    "user_provider_credentials": {"team-copilot": {"access_token": "gho-alice"}},
                },
            },
            "the GitHub account LiteLLM user 'dev-alice' connected for LiteLLM credential 'team-copilot'",
            id="per-user-without-saved-login",
        ),
        pytest.param(
            {"litellm_credential_name": "team-copilot", "github_copilot_user_session": _PER_USER_SESSION},
            "the GitHub account you connected for LiteLLM credential 'team-copilot'",
            id="per-user-outside-the-proxy",
        ),
        pytest.param(
            {"litellm_credential_name": "org-app", "github_copilot_user_session": _APP_SESSION},
            "the GitHub App installation on LiteLLM credential 'org-app'",
            id="github-app",
        ),
    ],
)
def test_github_copilot_model_not_supported_names_the_github_account_used(
    request_kwargs: dict[str, object], account: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("GITHUB_COPILOT_CLIENT_ID", raising=False)
    assert (
        f"GitHub Copilot rejected model 'claude-opus-4.6' because it is not available for {account}. That can be "
        "a different account from the one signed in to your editor. Ask your GitHub admin to enable the model in "
        "that account's Copilot model policy, or use a model its Copilot plan includes. This is not a LiteLLM error."
    ) in _copilot_model_not_supported_message(request_kwargs)


_CUSTOM_APP_HINT: Final = (
    "That can be a different account from the one signed in to your editor. LiteLLM signs in to GitHub Copilot "
    "with a custom GitHub App (GITHUB_COPILOT_CLIENT_ID). Check that the app's 'Copilot Requests' permission is set "
    "to 'Read-only'"
)


@pytest.mark.parametrize(
    "request_kwargs",
    [
        pytest.param({"litellm_credential_name": "shared-copilot"}, id="shared-device-login"),
        pytest.param(
            {"litellm_credential_name": "team-copilot", "github_copilot_user_session": _PER_USER_SESSION},
            id="per-user-device-login",
        ),
    ],
)
def test_github_copilot_model_not_supported_names_a_custom_sign_in_app(
    request_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GITHUB_COPILOT_CLIENT_ID", "Iv23custom")
    assert _CUSTOM_APP_HINT in _copilot_model_not_supported_message(request_kwargs)


@pytest.mark.parametrize(
    ("client_id", "request_kwargs"),
    [
        pytest.param(DEFAULT_GITHUB_CLIENT_ID, {"litellm_credential_name": "shared-copilot"}, id="default-app"),
        pytest.param(
            "Iv23custom",
            {"litellm_credential_name": "org-app", "github_copilot_user_session": _APP_SESSION},
            id="github-app-installation",
        ),
    ],
)
def test_github_copilot_model_not_supported_skips_the_app_hint_without_a_custom_sign_in_app(
    client_id: str, request_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GITHUB_COPILOT_CLIENT_ID", client_id)
    assert "GITHUB_COPILOT_CLIENT_ID" not in _copilot_model_not_supported_message(request_kwargs)


def test_openai_model_not_supported_keeps_the_upstream_message():
    request: Final = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    original: Final = OpenAIError(
        status_code=400,
        message='{"error": {"message": "The requested model is not supported.", "code": "model_not_supported"}}',
        request=request,
        response=httpx.Response(400, request=request),
    )
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(
            model="gpt-5.4",
            original_exception=original,
            custom_llm_provider="openai",
            completion_kwargs={},
            extra_kwargs={},
        )
    assert "GitHub Copilot" not in caught.value.message, caught.value.message


_COPILOT_APP_HINT: Final = "Check that the app's 'Copilot Requests' permission is set to 'Read-only'"


def _copilot_error(status_code: int, body: dict[str, str]) -> OpenAIError:
    request: Final = httpx.Request("POST", "https://api.githubcopilot.com/chat/completions")
    return OpenAIError(
        status_code=status_code,
        message=json.dumps({"error": body}),
        request=request,
        response=httpx.Response(status_code, request=request),
        body=body,
    )


@pytest.mark.parametrize(
    ("client_id", "hint_expected"),
    [(None, False), ("Iv1.b507a08c87ecfe98", False), ("Iv23liCustomApp", True)],
    ids=["default_app", "default_app_set_explicitly", "custom_github_app"],
)
def test_github_copilot_model_not_supported_tells_custom_app_users_to_check_copilot_requests(
    monkeypatch, client_id: str | None, hint_expected: bool
):
    if client_id is None:
        monkeypatch.delenv("GITHUB_COPILOT_CLIENT_ID", raising=False)
    else:
        monkeypatch.setenv("GITHUB_COPILOT_CLIENT_ID", client_id)
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(
            model="claude-sonnet-4.5",
            original_exception=_copilot_error(
                400, {"message": "The requested model is not supported.", "code": "model_not_supported"}
            ),
            custom_llm_provider="github_copilot",
            completion_kwargs={},
            extra_kwargs={},
        )
    assert caught.value.status_code == 400
    assert "GitHub Copilot rejected model 'claude-sonnet-4.5'" in caught.value.message, caught.value.message
    assert (_COPILOT_APP_HINT in caught.value.message) is hint_expected, caught.value.message
    assert caught.value.message.count(_COPILOT_APP_HINT) == (1 if hint_expected else 0), caught.value.message


def test_github_copilot_integration_mismatch_tells_custom_app_users_where_the_permission_lives(monkeypatch):
    monkeypatch.setenv("GITHUB_COPILOT_CLIENT_ID", "Iv23liCustomApp")
    upstream_text: Final = (
        "bad request: the Copilot-Integration-Id header does not match the GitHub App that issued this "
        "installation access token"
    )
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(
            model="gpt-4o",
            original_exception=_copilot_error(400, {"message": upstream_text}),
            custom_llm_provider="github_copilot",
            completion_kwargs={},
            extra_kwargs={},
        )
    assert caught.value.status_code == 400
    assert upstream_text in caught.value.message, caught.value.message
    assert _COPILOT_APP_HINT in caught.value.message, caught.value.message
    assert "https://github.com/settings/apps" in caught.value.message, caught.value.message


def test_github_copilot_app_installation_error_skips_the_device_login_app_hint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GITHUB_COPILOT_CLIENT_ID", "Iv23liCustomApp")
    upstream_text: Final = "bad request: Personal Access Tokens are not supported for this endpoint"
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(
            model="gpt-4o",
            original_exception=_copilot_error(400, {"message": upstream_text}),
            custom_llm_provider="github_copilot",
            completion_kwargs={},
            extra_kwargs={"litellm_credential_name": "org-app", "github_copilot_user_session": _APP_SESSION},
        )
    assert upstream_text in caught.value.message, caught.value.message
    assert "Copilot Requests" not in caught.value.message, caught.value.message


def test_openai_error_never_mentions_the_copilot_app_permission(monkeypatch):
    monkeypatch.setenv("GITHUB_COPILOT_CLIENT_ID", "Iv23liCustomApp")
    request: Final = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    original: Final = OpenAIError(
        status_code=400,
        message='{"error": {"message": "The requested model is not supported.", "code": "model_not_supported"}}',
        request=request,
        response=httpx.Response(400, request=request),
    )
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(
            model="gpt-5.4",
            original_exception=original,
            custom_llm_provider="openai",
            completion_kwargs={},
            extra_kwargs={},
        )
    assert "Copilot Requests" not in caught.value.message, caught.value.message


_PROVIDER_WIRE_BODY: Final = {"error": {"message": "wire says no", "type": "invalid_request_error"}}


@pytest.mark.parametrize(
    ("status_code", "mapped_error"),
    [
        (400, litellm.BadRequestError),
        (401, litellm.AuthenticationError),
        (404, litellm.NotFoundError),
        (429, litellm.RateLimitError),
    ],
)
def test_mapped_anthropic_error_exposes_the_provider_reply(status_code: int, mapped_error: type[Exception]):
    original: Final = AnthropicError(status_code=status_code, message=json.dumps(_PROVIDER_WIRE_BODY))
    original.response = httpx.Response(status_code, json=_PROVIDER_WIRE_BODY, request=original.request)

    with pytest.raises(mapped_error) as caught:
        exception_type(
            model="claude-sonnet-4-5",
            original_exception=original,
            custom_llm_provider="anthropic",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert caught.value.response.status_code == status_code
    assert caught.value.response.json() == _PROVIDER_WIRE_BODY


def test_mapped_anthropic_529_reports_one_status_beside_the_provider_reply(quiet_exception_mapping: None) -> None:
    wire_body: Final = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    original: Final = AnthropicError(status_code=529, message=json.dumps(wire_body))
    original.response = httpx.Response(529, json=wire_body, request=original.request)

    with pytest.raises(litellm.InternalServerError) as caught:
        exception_type(
            model="claude-sonnet-4-5",
            original_exception=original,
            custom_llm_provider="anthropic",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert (caught.value.status_code, caught.value.response.status_code) == (500, 500)
    assert caught.value.provider_status_code == 529
    assert caught.value.response.json() == wire_body


@pytest.mark.parametrize(
    ("provider", "wire_status", "message", "mapped_error"),
    [
        ("replicate", 402, json.dumps(_PROVIDER_WIRE_BODY), litellm.APIError),
        ("nlp_cloud", 522, json.dumps(_PROVIDER_WIRE_BODY), litellm.Timeout),
        ("openai", 524, "A timeout occurred", litellm.Timeout),
        ("openai", 503, "The server had an error processing your request.", litellm.InternalServerError),
        ("mistral", 400, "Mistral API raised a streaming error", litellm.APIError),
        ("openai", 504, "Request timed out", litellm.Timeout),
    ],
)
def test_an_error_mapped_to_another_status_reports_that_status_on_the_provider_reply(
    provider: str,
    wire_status: int,
    message: str,
    mapped_error: type[litellm.APIError | litellm.Timeout | litellm.InternalServerError],
    quiet_exception_mapping: None,
) -> None:
    request: Final = httpx.Request("POST", "https://provider.example/v1/predictions")
    original: Final = BaseLLMException(
        status_code=wire_status,
        message=message,
        request=request,
        response=httpx.Response(wire_status, json=_PROVIDER_WIRE_BODY, request=request),
    )

    with pytest.raises(mapped_error) as caught:
        exception_type(
            model="provider-model",
            original_exception=original,
            custom_llm_provider=provider,
            completion_kwargs={},
            extra_kwargs={},
        )

    assert caught.value.status_code != wire_status
    assert caught.value.response.status_code == caught.value.status_code
    assert caught.value.response.json() == _PROVIDER_WIRE_BODY


def test_a_streamed_groq_error_with_a_text_code_still_maps_to_an_api_error(quiet_exception_mapping: None) -> None:
    with pytest.raises(OpenAIError) as streamed_failure:
        GroqChatCompletionStreamingHandler(streaming_response=iter(()), sync_stream=True).chunk_parser(
            {"error": {"code": "tool_use_failed", "message": "Failed to call a function"}}
        )

    with pytest.raises(litellm.APIError) as caught:
        exception_type(
            model="llama-3.3-70b-versatile",
            original_exception=streamed_failure.value,
            custom_llm_provider="groq",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert type(caught.value) is litellm.APIError
    assert "Failed to call a function" in str(caught.value)


def test_litellm_proxy_error_keeps_the_reply_body_beside_the_forwarded_headers():
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    request: Final = httpx.Request("POST", "https://proxy.example/v1/responses")
    upstream: Final = httpx.Response(400, json=_PROVIDER_WIRE_BODY, headers=_PROXY_HEADERS, request=request)
    with pytest.raises(BaseLLMException) as provider_failure:
        BaseLLMHTTPHandler()._handle_error(
            e=httpx.HTTPStatusError("rejected", request=request, response=upstream), provider_config=None
        )

    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(
            model="gpt-5.6",
            original_exception=provider_failure.value,
            custom_llm_provider="litellm_proxy",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert caught.value.response.json() == _PROVIDER_WIRE_BODY
    assert dict(caught.value.response.headers) == dict(upstream.headers)


def test_provider_reply_does_not_reclassify_an_invalid_bedrock_model_identifier():
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    wire_body: Final = {"message": "The provided model identifier is invalid."}
    converse_like_url: Final = "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude/converse"
    request: Final = httpx.Request("POST", converse_like_url)
    rejection: Final = httpx.HTTPStatusError(
        "rejected", request=request, response=httpx.Response(400, json=wire_body, request=request)
    )

    with pytest.raises(BaseLLMException) as provider_failure:
        BaseLLMHTTPHandler()._handle_error(e=rejection, provider_config=None)
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            original_exception=provider_failure.value,
            custom_llm_provider="bedrock",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert type(caught.value) is litellm.BadRequestError
    assert caught.value.response.json() == wire_body


def _anthropic_failure_answered_with(reply_status: int) -> BaseLLMException:
    request: Final = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return BaseLLMException(
        status_code=400,
        message="invalid_request_error: rejected",
        request=request,
        response=httpx.Response(reply_status, content=b"provider reply", request=request),
    )


def test_only_an_http_error_reply_is_exposed_as_the_provider_error():
    with pytest.raises(litellm.BadRequestError) as answered_with_an_error:
        exception_type(
            model="claude-sonnet-4-5",
            original_exception=_anthropic_failure_answered_with(400),
            custom_llm_provider="anthropic",
            completion_kwargs={},
            extra_kwargs={},
        )
    with pytest.raises(litellm.BadRequestError) as answered_with_success:
        exception_type(
            model="claude-sonnet-4-5",
            original_exception=_anthropic_failure_answered_with(200),
            custom_llm_provider="anthropic",
            completion_kwargs={},
            extra_kwargs={},
        )

    assert answered_with_an_error.value.response.text == "provider reply"
    assert answered_with_success.value.response.status_code != 200
    assert answered_with_success.value.response.text == ""


@pytest.mark.parametrize(
    ("provider", "provider_status", "mapped_error", "public_status"),
    [
        ("anthropic", 529, litellm.InternalServerError, 500),
        ("anthropic", 429, litellm.RateLimitError, 429),
        ("openai", 529, litellm.APIError, 529),
        ("openai", 429, litellm.RateLimitError, 429),
    ],
)
def test_a_mapped_provider_error_keeps_the_providers_own_status_beside_its_public_one(
    provider: str,
    provider_status: int,
    mapped_error: type[litellm.InternalServerError | litellm.RateLimitError | litellm.APIError],
    public_status: int,
    quiet_exception_mapping: None,
) -> None:
    with pytest.raises(mapped_error, match="upstream failure") as raised:
        exception_type(
            model="model-under-test",
            original_exception=_UpstreamHTTPError(status_code=provider_status, message="upstream failure"),
            custom_llm_provider=provider,
        )

    assert type(raised.value) is mapped_error
    assert (raised.value.status_code, provider_origin_status(raised.value)) == (public_status, provider_status)


def test_an_error_litellm_raised_itself_carries_no_provider_status(quiet_exception_mapping: None) -> None:
    synthetic: Final = litellm.RateLimitError(message="deployment at capacity", llm_provider="", model="m")

    returned: Final = exception_type(model="m", original_exception=synthetic, custom_llm_provider="openai")

    assert returned is synthetic
    assert provider_origin_status(returned) is None


def test_a_status_litellm_made_up_for_a_provider_error_is_not_kept_as_the_providers(
    quiet_exception_mapping: None,
) -> None:
    made_up: Final = BaseLLMException(status_code=429, message="no status upstream", status_code_is_synthesized=True)

    with pytest.raises(litellm.RateLimitError, match="no status upstream") as raised:
        exception_type(model="m", original_exception=made_up, custom_llm_provider="openai")

    assert provider_origin_status(raised.value) is None


def _status_failure(status_code: int, error: dict[str, object]) -> openai.APIStatusError:
    request: Final = httpx.Request("POST", "https://backend.example.test/v1/chat/completions")
    response: Final = httpx.Response(status_code, json={"error": error}, request=request)
    return openai.APIStatusError(message="rejected", response=response, body=error)


@pytest.mark.parametrize("provider", ("openrouter", "openai", "hosted_vllm", "gemini"))
def test_mapped_error_never_carries_a_credential_in_its_param(provider: str) -> None:
    fake_key: Final = "sk-test-fake-0123456789abcdefghijklmn"
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(
            model="m",
            original_exception=_status_failure(400, {"message": "rejected", "param": fake_key}),
            custom_llm_provider=provider,
        )

    assert fake_key not in str(caught.value.param)


def test_mapped_error_keeps_a_plain_field_name_in_its_param() -> None:
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(
            model="m",
            original_exception=_status_failure(400, {"message": "rejected", "param": "messages[0].content"}),
            custom_llm_provider="openrouter",
        )

    assert caught.value.param == "messages[0].content"


def test_openrouter_rate_limit_keeps_its_own_code() -> None:
    with pytest.raises(litellm.RateLimitError) as caught:
        exception_type(
            model="m",
            original_exception=_status_failure(429, {"message": "busy", "code": "rate_limit_exceeded"}),
            custom_llm_provider="openrouter",
        )

    assert caught.value.code == "429"
    assert relayed_provider_specific_fields(caught.value) == {"code": "rate_limit_exceeded"}


def test_azure_responses_rate_limit_keeps_its_own_code() -> None:
    with pytest.raises(BaseLLMException) as upstream:
        AzureOpenAIResponsesAPIConfig().get_error_class(
            json.dumps({"error": {"message": "slow down", "code": "rate_limit_exceeded"}}), 429, {}
        )
    with pytest.raises(litellm.RateLimitError) as caught:
        exception_type(model="m", original_exception=upstream.value, custom_llm_provider="azure")

    assert caught.value.code == "429"
    assert relayed_provider_specific_fields(caught.value) == {"code": "rate_limit_exceeded"}


def test_openrouter_stream_failure_keeps_the_upstream_reason() -> None:
    error: Final = {"message": "Provider returned error", "code": 400, "metadata": {"raw": "bad schema"}}
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(model="m", original_exception=_status_failure(400, error), custom_llm_provider="openrouter")

    frame: Final = ResponsesStreamErrorState().format_failure(caught.value)

    assert frame is not None
    assert "bad schema" in json.loads(frame.split("data: ", 1)[1])["response"]["error"]["message"]


def test_flat_responses_failure_logs_its_status_and_relays_the_provider_code() -> None:
    with pytest.raises(BaseLLMException) as upstream:
        OpenAIResponsesAPIConfig().get_error_class(json.dumps({"code": "provider_code", "message": "rejected"}), 400, {})
    with pytest.raises(litellm.BadRequestError) as caught:
        exception_type(model="m", original_exception=upstream.value, custom_llm_provider="openai")

    assert _logged_error_code(caught.value) == "400"
    assert relayed_provider_specific_fields(caught.value) == {"code": "provider_code"}


@pytest.mark.parametrize(
    ("status_code", "logged_code"),
    [(400, "provider_code"), (422, "provider_code"), (500, "provider_code"), (401, "401"), (404, "404"), (503, "503")],
)
def test_litellm_proxy_flat_responses_failure_keeps_its_logged_code(status_code: int, logged_code: str) -> None:
    with pytest.raises(BaseLLMException) as upstream:
        LiteLLMProxyResponsesAPIConfig().get_error_class(
            json.dumps({"code": "provider_code", "message": "rejected"}), status_code, {}
        )
    with pytest.raises(openai.APIError) as caught:
        exception_type(model="m", original_exception=upstream.value, custom_llm_provider="litellm_proxy")

    assert _logged_error_code(caught.value) == logged_code


def test_a_deeply_nested_provider_error_keeps_its_status() -> None:
    from litellm.litellm_core_utils.exception_mapping_utils import exception_type
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    depth: Final = 200_000
    nested: Final = '{"error":' * depth + "1" + "}" * depth

    with pytest.raises(litellm.BadRequestError):
        exception_type(
            model="qwen",
            original_exception=BaseLLMException(status_code=400, message=nested),
            custom_llm_provider="hosted_vllm",
        )


def _botocore_credential_errors() -> tuple[Exception, ...]:
    from botocore.exceptions import (
        LoginRefreshRequired,
        NoCredentialsError,
        PartialCredentialsError,
        ProfileNotFound,
        SSOTokenLoadError,
        TokenRetrievalError,
        UnauthorizedSSOTokenError,
    )

    return (
        TokenRetrievalError(provider="sso", error_msg="Token has expired and refresh failed"),
        UnauthorizedSSOTokenError(),
        SSOTokenLoadError(error_msg="Token for example.awsapps.com does not exist"),
        LoginRefreshRequired(),
        NoCredentialsError(),
        PartialCredentialsError(provider="env", cred_var="AWS_SECRET_ACCESS_KEY"),
        ProfileNotFound(profile="missing-profile"),
    )


@pytest.mark.parametrize("custom_llm_provider", ["bedrock", "bedrock_mantle"])
@pytest.mark.parametrize("credential_error", _botocore_credential_errors(), ids=lambda error: type(error).__name__)
def test_bedrock_unusable_aws_credentials_answer_401_without_a_bug_report(
    credential_error: Exception, custom_llm_provider: str
) -> None:
    with pytest.raises(litellm.AuthenticationError) as excinfo:
        exception_type(
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            original_exception=credential_error,
            custom_llm_provider=custom_llm_provider,
        )

    assert excinfo.value.status_code == 401
    assert excinfo.value.response.status_code == 401
    assert str(credential_error) in excinfo.value.message
    assert "github.com/BerriAI/litellm/issues/new" not in excinfo.value.message


@pytest.mark.parametrize("custom_llm_provider", ["bedrock", "bedrock_mantle"])
def test_bedrock_aws_login_without_permission_answers_403(custom_llm_provider: str) -> None:
    from botocore.exceptions import LoginInsufficientPermissions

    with pytest.raises(litellm.PermissionDeniedError) as excinfo:
        exception_type(
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            original_exception=LoginInsufficientPermissions(),
            custom_llm_provider=custom_llm_provider,
        )

    assert excinfo.value.status_code == 403
    assert excinfo.value.response.status_code == 403


def test_bedrock_credential_source_outage_stays_a_retryable_connection_error() -> None:
    from botocore.exceptions import CredentialRetrievalError

    with pytest.raises(litellm.APIConnectionError) as excinfo:
        exception_type(
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            original_exception=CredentialRetrievalError(provider="container-role", error_msg="connection refused"),
            custom_llm_provider="bedrock",
        )

    assert excinfo.value.status_code == 500


def test_bedrock_mantle_without_usable_aws_credentials_answers_401() -> None:
    from botocore.exceptions import ProfileNotFound

    from litellm.llms.bedrock_mantle.common_utils import BedrockMantleAuthError

    signing_failure: Final = BedrockMantleAuthError()
    signing_failure.__cause__ = ProfileNotFound(profile="missing-profile")

    with pytest.raises(litellm.AuthenticationError) as excinfo:
        exception_type(
            model="openai.gpt-6-sol", original_exception=signing_failure, custom_llm_provider="bedrock_mantle"
        )

    assert excinfo.value.status_code == 401
    assert "github.com/BerriAI/litellm/issues/new" not in excinfo.value.message
