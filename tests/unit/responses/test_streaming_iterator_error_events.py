"""
Regression: in-stream error events (type="error", type="response.failed") must
raise instead of being returned as benign chunks, mirroring chat streaming
semantics (_handle_stream_fallback_error). The event's code, type and status go
through litellm.exception_type, so each event raises the same typed exception
the non-streaming path raises for that provider error: non-retriable 4xx
(except 429) raise that typed exception directly, so a context-length event
surfaces as ContextWindowExceededError(400) with no MidStreamFallbackError
wrapping, while 429, 5xx and ContentPolicyViolationError are wrapped in
MidStreamFallbackError so the Router's mid-stream fallback machinery fires and
its content_policy_fallbacks dispatch sees the trigger it matches on.

Status mapping must consider both the OpenAI error `type` (e.g.
"invalid_request_error") and `code` (e.g. "invalid_prompt",
"rate_limit_exceeded") fields — previously only `code` was read, so
type-classified client errors fell through to 500.

Also covers: ErrorEventError.param must accept dict payloads without raising a
Pydantic ValidationError (previously typed as Optional[str]).
"""

import asyncio
import json
from datetime import datetime
from importlib import import_module
from collections.abc import Mapping
from typing import Final, cast
from unittest.mock import Mock, patch

import httpx
import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.litellm_core_utils.litellm_logging import _get_provider_request_id
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import (
    _ERROR_CODE_HTTP_STATUS,
    BaseResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    ResponsesWebSocketStreaming,
    SyncResponsesAPIStreamingIterator,
    _status_code_for_error_fields,
)
from litellm.utils import ProviderConfigManager
from litellm.types.utils import StandardLoggingPayload
from litellm.types.llms.openai import (
    ErrorEvent,
    ErrorEventError,
    ResponseAPIUsage,
    ResponsesAPIStreamEvents,
)


def _make_iterator() -> BaseResponsesAPIStreamingIterator:
    mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_response = Mock()
    mock_response.headers = {}
    return BaseResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )


def _make_error_chunk(error_type: str, code: str, message: str = "err") -> ErrorEvent:
    error_obj = ErrorEventError(type=error_type, code=code, message=message)
    return ErrorEvent(type=ResponsesAPIStreamEvents.ERROR, sequence_number=0, error=error_obj)


@pytest.mark.parametrize("provider", ("openai", "azure", "bedrock"))
@pytest.mark.parametrize("header", ("x-amzn-RequestId", "x-request-id", None))
@pytest.mark.parametrize(
    "code,status", (("server_error", 500), ("rate_limit_exceeded", 429), ("invalid_request_error", 400))
)
def test_stream_error_preserves_provider_request_id(provider: str, header: str | None, code: str, status: int) -> None:
    stream_headers: Final = {"retry-after": "7", "content-type": "text/event-stream"}
    response: Final = httpx.Response(
        200, headers={**stream_headers, header: "provider-request-id"} if header else stream_headers
    )
    logging: Final = LiteLLMLoggingObj(
        model="test-model",
        messages=[],
        stream=True,
        call_type="responses",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="test-call",
        function_id="test-function",
    )
    iterator: Final = BaseResponsesAPIStreamingIterator(
        response=response,
        model="test-model",
        responses_api_provider_config=None,
        logging_obj=logging,
        custom_llm_provider=provider,
    )
    expected_id: Final = "provider-request-id" if header else None
    expected_headers: Final = {header.lower(): "provider-request-id"} if header else {}
    mapped: Final = iterator._map_error_event_exception({"type": code, "code": code, "message": "failure"})
    assert _get_provider_request_id(mapped) == expected_id
    assert dict(getattr(mapped, "litellm_response_headers", None) or {}) == expected_headers
    expected_status: Final = 503 if provider == "bedrock" and status == 500 else status
    with pytest.raises((litellm.BadRequestError, MidStreamFallbackError)) as raised:
        iterator._maybe_raise_for_error_event(_make_error_chunk(code, code, "failure"))
    assert raised.value.status_code == expected_status
    assert _get_provider_request_id(raised.value) == expected_id
    assert response.status_code == 200
    if status in (429, 500):
        assert isinstance(raised.value, MidStreamFallbackError)
        assert dict(raised.value.response.headers) == expected_headers
        assert raised.value.response.status_code == expected_status
        assert str(raised.value.response.request.url) == f"https://{provider}.com/v1/"
        assert raised.value.generated_content == ""
        assert raised.value.is_pre_first_chunk is True


def test_maybe_raise_for_error_event_wraps_unknown_error_in_mid_stream_fallback():
    iterator = _make_iterator()
    chunk = _make_error_chunk("server_error", "internal_error", "something went wrong")
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 500
    assert isinstance(exc_info.value.original_exception, litellm.InternalServerError)
    assert exc_info.value.original_exception.status_code == 500


def test_maybe_raise_for_error_event_maps_rate_limit_code_to_429_mid_stream_fallback():
    """429 is retriable: it must be wrapped so the Router can fall back, carrying the mapped RateLimitError."""
    iterator = _make_iterator()
    chunk = _make_error_chunk("tokens", "rate_limit_exceeded", "Too many requests")
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 429
    assert exc_info.value.generated_content == ""
    assert exc_info.value.is_pre_first_chunk is True
    assert isinstance(exc_info.value.original_exception, litellm.RateLimitError)
    assert exc_info.value.original_exception.status_code == 429


def test_maybe_raise_for_error_event_maps_invalid_request_type_to_400():
    """Client errors classified via the `type` field must raise BadRequestError directly (no fallback)."""
    iterator = _make_iterator()
    chunk = _make_error_chunk("invalid_request_error", "invalid_prompt", "bad request")
    with pytest.raises(litellm.BadRequestError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 400
    assert not isinstance(exc_info.value, MidStreamFallbackError)


def test_maybe_raise_for_error_event_maps_context_length_code_to_400():
    """Client errors classified via the `code` field alone must still map to 400."""
    iterator = _make_iterator()
    chunk = Mock()
    chunk.type = "error"
    chunk.error = {"code": "context_length_exceeded", "message": "too long"}
    with pytest.raises(litellm.BadRequestError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 400
    assert not isinstance(exc_info.value, MidStreamFallbackError)


def test_maybe_raise_for_error_event_raises_context_window_exceeded_directly():
    """A context-length error event maps to ContextWindowExceededError exactly like the non-streaming
    path and, being a non-retriable client error, is raised directly rather than wrapped for mid-stream
    fallback, preserving the direct-SDK 400 contract from issue #15785."""
    iterator = _make_iterator()
    chunk = _make_error_chunk(
        "invalid_request_error",
        "context_length_exceeded",
        "This model's maximum context length is 128000 tokens. However, your messages resulted in 130000 tokens.",
    )
    with pytest.raises(litellm.ContextWindowExceededError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 400
    assert not isinstance(exc_info.value, MidStreamFallbackError)
    assert "maximum context length" in str(exc_info.value)


CONTENT_POLICY_MESSAGE = "This content was flagged for possible cybersecurity risk. The response was halted mid-stream."


@pytest.mark.parametrize("custom_llm_provider", ["openai", "azure"])
def test_maybe_raise_for_error_event_wraps_content_policy_violation_for_content_policy_fallbacks(
    custom_llm_provider: str,
):
    """Regression: a content_policy_violation error event used to raise a bare APIError, so the Router's
    content_policy_fallbacks never fired. It must map to ContentPolicyViolationError (the same exception the
    non-streaming path raises) and be wrapped so the Router's mid-stream fallback catches it."""
    iterator = _make_iterator()
    iterator.custom_llm_provider = custom_llm_provider
    chunk = _make_error_chunk("invalid_request_error", "content_policy_violation", CONTENT_POLICY_MESSAGE)
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert isinstance(exc_info.value.original_exception, litellm.ContentPolicyViolationError)
    assert exc_info.value.original_exception.status_code == 400
    assert exc_info.value.status_code == 400
    assert exc_info.value.is_pre_first_chunk is True
    assert CONTENT_POLICY_MESSAGE in str(exc_info.value.original_exception)


def test_maybe_raise_for_response_failed_event_wraps_content_policy_violation():
    iterator = _make_iterator()
    chunk = _make_failed_chunk(
        {"type": "invalid_request_error", "code": "content_policy_violation", "message": CONTENT_POLICY_MESSAGE}
    )
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert isinstance(exc_info.value.original_exception, litellm.ContentPolicyViolationError)


@pytest.mark.parametrize(
    "error_type,error_code,expected_exception",
    [
        ("invalid_request_error", "content_policy_violation", litellm.ContentPolicyViolationError),
        ("invalid_request_error", "cyber_policy", litellm.ContentPolicyViolationError),
        ("tokens", "rate_limit_exceeded", litellm.RateLimitError),
        ("invalid_request_error", "insufficient_quota", litellm.RateLimitError),
        ("server_error", "internal_error", litellm.InternalServerError),
        ("invalid_request_error", "invalid_prompt", litellm.BadRequestError),
        ("invalid_request_error", "model_not_found", litellm.NotFoundError),
        ("server_error", "vector_store_timeout", litellm.Timeout),
    ],
)
def test_error_event_raises_the_same_typed_exception_as_the_non_streaming_path(
    error_type: str, error_code: str, expected_exception: type[Exception]
):
    iterator = _make_iterator()
    chunk = _make_error_chunk(error_type, error_code, "provider message")
    with pytest.raises((MidStreamFallbackError, expected_exception)) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    raised = exc_info.value
    typed_exception = raised.original_exception if isinstance(raised, MidStreamFallbackError) else raised
    assert type(typed_exception) is expected_exception
    assert "provider message" in str(typed_exception)


def test_maybe_raise_for_error_event_maps_insufficient_quota_to_429():
    """OpenAI returns HTTP 429 for insufficient_quota; it must not map to 400 even though its type
    is invalid_request_error-adjacent, and it must be wrapped for fallback."""
    iterator = _make_iterator()
    chunk = _make_error_chunk("invalid_request_error", "insufficient_quota", "You exceeded your current quota")
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 429
    assert isinstance(exc_info.value.original_exception, litellm.RateLimitError)


def test_maybe_raise_for_error_event_passes_through_normal_chunk():
    iterator = _make_iterator()
    chunk = Mock()
    chunk.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
    iterator._maybe_raise_for_error_event(chunk)  # must not raise


def test_error_event_error_param_accepts_dict():
    error_obj = ErrorEventError(
        type="invalid_request_error",
        code="context_length_exceeded",
        message="too long",
        param={"field": "messages", "index": 0},
    )
    assert isinstance(error_obj.param, dict)


def _make_async_iterator_with_events(events: list) -> ResponsesAPIStreamingIterator:
    sse_payload = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)

    async def mock_aiter_bytes():
        yield sse_payload

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    def transform(model, parsed_chunk, logging_obj):
        if parsed_chunk.get("type") == "error":
            return ErrorEvent(
                type=ResponsesAPIStreamEvents.ERROR,
                sequence_number=0,
                error=ErrorEventError(**parsed_chunk["error"]),
            )
        delta_event = Mock()
        delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        delta_event.delta = parsed_chunk.get("delta", "")
        return delta_event

    mock_config.transform_streaming_response.side_effect = transform

    return ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )


@pytest.mark.asyncio
async def test_async_iterator_raises_mid_stream_fallback_on_rate_limit_error_event():
    iterator = _make_async_iterator_with_events(
        [
            {
                "type": "error",
                "error": {"type": "tokens", "code": "rate_limit_exceeded", "message": "rate limited"},
            }
        ]
    )

    with pytest.raises(MidStreamFallbackError) as exc_info:
        async for _ in iterator:
            pass
    assert exc_info.value.status_code == 429
    assert exc_info.value.is_pre_first_chunk is True
    assert exc_info.value.generated_content == ""
    assert isinstance(exc_info.value.original_exception, litellm.RateLimitError)
    assert exc_info.value.original_exception.status_code == 429


@pytest.mark.asyncio
async def test_async_iterator_content_policy_violation_after_first_chunk_carries_generated_content():
    """The customer's case: text streams, then the provider halts the stream with a
    content_policy_violation error event. The iterator must surface ContentPolicyViolationError
    inside MidStreamFallbackError, together with the text already streamed."""
    iterator = _make_async_iterator_with_events(
        [
            {"type": "response.output_text.delta", "delta": "partial "},
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "content_policy_violation",
                    "message": CONTENT_POLICY_MESSAGE,
                },
            },
        ]
    )

    stream = aiter(iterator)
    first_chunk = await anext(stream)
    assert first_chunk is not None

    with pytest.raises(MidStreamFallbackError) as exc_info:
        await anext(stream)
    assert isinstance(exc_info.value.original_exception, litellm.ContentPolicyViolationError)
    assert exc_info.value.is_pre_first_chunk is False
    assert exc_info.value.generated_content == "partial "


@pytest.mark.asyncio
async def test_async_iterator_error_after_first_chunk_carries_generated_content():
    """An error after streamed output must expose the accumulated text so the router's
    fallback can build a continuation input instead of restarting from scratch."""
    iterator = _make_async_iterator_with_events(
        [
            {"type": "response.output_text.delta", "delta": "hello "},
            {"type": "response.output_text.delta", "delta": "world"},
            {
                "type": "error",
                "error": {"type": "server_error", "code": "internal_error", "message": "boom"},
            },
        ]
    )

    stream = aiter(iterator)
    first_chunk = await anext(stream)
    second_chunk = await anext(stream)
    assert first_chunk is not None and second_chunk is not None

    with pytest.raises(MidStreamFallbackError) as exc_info:
        await anext(stream)
    assert exc_info.value.status_code == 500
    assert exc_info.value.is_pre_first_chunk is False
    assert exc_info.value.generated_content == "hello world"


def test_maybe_raise_for_response_failed_event_with_dict_error():
    """response.failed chunks carry a dict error on .response.error; covers dict branch."""
    iterator = _make_iterator()
    mock_response_obj = Mock()
    mock_response_obj.error = {"type": "tokens", "code": "rate_limit_exceeded", "message": "throttled"}
    chunk = Mock()
    chunk.type = "response.failed"
    chunk.response = mock_response_obj
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 429


@pytest.mark.parametrize("code", [429, "429"])
def test_response_failed_numeric_code_maps_to_its_http_status(code: int | str):
    iterator = _make_iterator()
    mock_response_obj = Mock()
    mock_response_obj.error = {"code": code, "message": "throttled"}
    chunk = Mock()
    chunk.type = "response.failed"
    chunk.response = mock_response_obj
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 429
    assert isinstance(exc_info.value.original_exception, litellm.RateLimitError)


@pytest.mark.parametrize("code", ["cyber_policy", "unrecognized_failure"])
def test_response_failed_keeps_upstream_code_and_uses_content_policy_fallback_for_cyber_policy(code: str):
    iterator = _make_iterator()
    upstream_message = "This content was flagged for possible cybersecurity risk."
    mock_response_obj = Mock()
    mock_response_obj.error = {"code": code, "message": upstream_message}
    chunk = Mock()
    chunk.type = "response.failed"
    chunk.response = mock_response_obj
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    mapped = exc_info.value.original_exception
    expected = litellm.ContentPolicyViolationError if code == "cyber_policy" else litellm.InternalServerError
    assert isinstance(mapped, expected)
    assert mapped.code == code
    assert mapped.body == {"message": upstream_message, "type": None, "code": code}


def test_maybe_raise_for_error_event_null_error_obj():
    """error chunk with no error field: message and code default; wrapped as 500."""
    iterator = _make_iterator()
    chunk = Mock()
    chunk.type = "error"
    chunk.error = None
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 500
    assert "Response API in-stream error" in str(exc_info.value)


def _make_failed_chunk(error: dict, usage: ResponseAPIUsage | None = None) -> Mock:
    mock_response_obj = Mock()
    mock_response_obj.error = error
    mock_response_obj.usage = usage
    chunk = Mock()
    chunk.type = "response.failed"
    chunk.response = mock_response_obj
    return chunk


def test_handle_logging_failed_response_maps_rate_limit_to_429():
    """The exception logged to failure handlers must carry the mapped status, not a hardcoded 500."""
    iterator = _make_iterator()
    iterator.completed_response = _make_failed_chunk(
        {"type": "tokens", "code": "rate_limit_exceeded", "message": "throttled"}
    )
    with (
        patch.object(import_module("litellm.responses.streaming_iterator"), "run_async_function") as mock_run_async,
        patch.object(import_module("litellm.responses.streaming_iterator"), "executor"),
    ):
        iterator._handle_logging_failed_response()
    logged_exception = mock_run_async.call_args.kwargs["exception"]
    assert isinstance(logged_exception, litellm.RateLimitError)
    assert logged_exception.status_code == 429
    assert "throttled" in str(logged_exception)


def test_handle_logging_failed_response_maps_type_field_to_400():
    """Status derivation for failed-response logging must also read the error `type` field."""
    iterator = _make_iterator()
    iterator.completed_response = _make_failed_chunk(
        {"type": "invalid_request_error", "code": "invalid_prompt", "message": "bad prompt"}
    )
    with (
        patch.object(import_module("litellm.responses.streaming_iterator"), "run_async_function") as mock_run_async,
        patch.object(import_module("litellm.responses.streaming_iterator"), "executor"),
    ):
        iterator._handle_logging_failed_response()
    logged_exception = mock_run_async.call_args.kwargs["exception"]
    assert isinstance(logged_exception, litellm.BadRequestError)
    assert logged_exception.status_code == 400


def test_handle_logging_failed_response_logs_content_policy_violation():
    """Failure logging must record the same typed exception the stream raises, so logging
    integrations see a content policy violation instead of a generic APIError."""
    iterator = _make_iterator()
    iterator.completed_response = _make_failed_chunk(
        {"type": "invalid_request_error", "code": "content_policy_violation", "message": CONTENT_POLICY_MESSAGE}
    )
    with (
        patch.object(import_module("litellm.responses.streaming_iterator"), "run_async_function") as mock_run_async,
        patch.object(import_module("litellm.responses.streaming_iterator"), "executor"),
    ):
        iterator._handle_logging_failed_response()
    logged_exception = mock_run_async.call_args.kwargs["exception"]
    assert isinstance(logged_exception, litellm.ContentPolicyViolationError)
    assert logged_exception.status_code == 400
    assert CONTENT_POLICY_MESSAGE in str(logged_exception)


def test_handle_logging_failed_response_records_usage_and_cost():
    """Usage on a response.failed event must reach failure spend accounting via combined_usage_object."""
    iterator = _make_iterator()
    usage = ResponseAPIUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    chunk = _make_failed_chunk(
        {"type": "server_error", "code": "server_error", "message": "boom"},
        usage=usage,
    )
    iterator.completed_response = chunk
    iterator.logging_obj.response_cost_calculator.return_value = 0.0042
    with (
        patch.object(import_module("litellm.responses.streaming_iterator"), "run_async_function"),
        patch.object(import_module("litellm.responses.streaming_iterator"), "executor"),
    ):
        iterator._handle_logging_failed_response()
    combined_usage = iterator.logging_obj.model_call_details["combined_usage_object"]
    assert isinstance(combined_usage, litellm.Usage)
    assert combined_usage.prompt_tokens == 10
    assert combined_usage.completion_tokens == 5
    assert combined_usage.total_tokens == 15
    assert iterator.logging_obj.model_call_details["response_cost"] == 0.0042
    iterator.logging_obj.response_cost_calculator.assert_called_once_with(result=chunk.response)


class _FailureRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.failed: tuple[StandardLoggingPayload, ...] = ()

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.failed = (*self.failed, cast(StandardLoggingPayload, kwargs["standard_logging_object"]))


async def _first_failure_row(recorder: _FailureRecorder) -> StandardLoggingPayload:
    for _ in range(200):
        if recorder.failed:
            return recorder.failed[0]
        await asyncio.sleep(0.01)
    pytest.fail("the failed stream logged no failure row")


@pytest.mark.asyncio
async def test_failed_stream_bills_only_its_own_usage_over_an_earlier_attempts_stash():
    recorder: Final = _FailureRecorder()
    logging_obj: Final = LiteLLMLoggingObj(
        model="gpt-5.6",
        messages=[],
        stream=True,
        call_type="aresponses",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="responses-failed-over-stash",
        function_id="responses-failed-over-stash",
        dynamic_async_failure_callbacks=[recorder],
    )
    logging_obj.update_environment_variables(
        model="gpt-5.6", optional_params={}, litellm_params=get_litellm_params(custom_llm_provider="openai")
    )
    logging_obj.model_call_details["combined_usage_object"] = litellm.Usage(
        prompt_tokens=100, completion_tokens=50, total_tokens=150
    )
    logging_obj.model_call_details["response_cost"] = 0.5
    failed: Final = {
        "type": "response.failed",
        "sequence_number": 1,
        "response": {
            "id": "resp_failed_over_stash",
            "object": "response",
            "created_at": 1790000000,
            "status": "failed",
            "model": "gpt-5.6",
            "output": [],
            "error": {"code": "server_error", "message": "boom"},
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        },
    }
    iterator: Final = ResponsesAPIStreamingIterator(
        response=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=f"data: {json.dumps(failed)}\n\n".encode()
        ),
        model="gpt-5.6",
        responses_api_provider_config=ProviderConfigManager.get_provider_responses_api_config(
            provider="openai", model="gpt-5.6"
        ),
        logging_obj=logging_obj,
        custom_llm_provider="openai",
    )

    with pytest.raises(MidStreamFallbackError):
        await anext(aiter(iterator))

    row: Final = await _first_failure_row(recorder)
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="gpt-5.6", prompt_tokens=10, completion_tokens=5, custom_llm_provider="openai"
    )
    assert (row["prompt_tokens"], row["completion_tokens"]) == (10, 5)
    assert row["response_cost"] == pytest.approx(prompt_cost + completion_cost)


def test_handle_logging_failed_response_without_usage_skips_recording():
    iterator = _make_iterator()
    iterator.completed_response = _make_failed_chunk(
        {"type": "server_error", "code": "server_error", "message": "boom"}
    )
    with (
        patch.object(import_module("litellm.responses.streaming_iterator"), "run_async_function"),
        patch.object(import_module("litellm.responses.streaming_iterator"), "executor"),
    ):
        iterator._handle_logging_failed_response()
    assert "combined_usage_object" not in iterator.logging_obj.model_call_details
    iterator.logging_obj.response_cost_calculator.assert_not_called()


def test_sync_iterator_raises_mid_stream_fallback_on_rate_limit_error_event():
    """SyncResponsesAPIStreamingIterator must wrap retriable error events for fallback."""
    error_payload = {
        "type": "error",
        "error": {"type": "tokens", "code": "rate_limit_exceeded", "message": "throttled"},
    }
    sse_bytes = f"data: {json.dumps(error_payload)}\n\n".encode()

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.iter_bytes.return_value = iter([sse_bytes])
    mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    error_obj = ErrorEventError(type="tokens", code="rate_limit_exceeded", message="throttled")
    mock_config.transform_streaming_response.return_value = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR, sequence_number=0, error=error_obj
    )

    iterator = SyncResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    with pytest.raises(MidStreamFallbackError) as exc_info:
        for _ in iterator:
            pass
    assert exc_info.value.status_code == 429
    assert isinstance(exc_info.value.original_exception, litellm.RateLimitError)


def test_every_openai_sdk_response_error_code_has_explicit_status_mapping():
    from typing import get_args

    from openai.types.responses.response_error import ResponseError

    sdk_codes = set(get_args(ResponseError.model_fields["code"].annotation))
    unmapped = sdk_codes - set(_ERROR_CODE_HTTP_STATUS)
    assert unmapped == set(), (
        f"OpenAI SDK ResponseError codes missing from _ERROR_CODE_HTTP_STATUS: {sorted(unmapped)}; "
        "classify each new code with an explicit HTTP status instead of letting it default to 500"
    )


@pytest.mark.parametrize(
    "code,expected_status",
    [
        ("server_error", 500),
        ("rate_limit_exceeded", 429),
        ("insufficient_quota", 429),
        ("vector_store_timeout", 504),
        ("invalid_prompt", 400),
        ("invalid_image", 400),
        ("invalid_image_format", 400),
        ("invalid_base64_image", 400),
        ("invalid_image_url", 400),
        ("image_too_large", 400),
        ("image_too_small", 400),
        ("image_parse_error", 400),
        ("image_content_policy_violation", 400),
        ("invalid_image_mode", 400),
        ("image_file_too_large", 400),
        ("unsupported_image_media_type", 400),
        ("empty_image_file", 400),
        ("failed_to_download_image", 400),
        ("image_file_not_found", 400),
        ("totally_unknown_future_code", 500),
        ("429", 429),
        ("503", 503),
        ("200", 500),
    ],
)
def test_status_code_for_documented_response_error_codes(code: str, expected_status: int):
    assert _status_code_for_error_fields(None, code) == expected_status


def test_specific_error_code_wins_over_generic_error_type():
    assert _status_code_for_error_fields("server_error", "invalid_image") == 400


def test_maybe_raise_for_response_failed_event_maps_image_code_to_400():
    iterator = _make_iterator()
    mock_response_obj = Mock()
    mock_response_obj.error = {"code": "image_content_policy_violation", "message": "image rejected"}
    chunk = Mock()
    chunk.type = "response.failed"
    chunk.response = mock_response_obj
    with pytest.raises(litellm.BadRequestError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 400
    assert not isinstance(exc_info.value, MidStreamFallbackError)


def test_maybe_raise_for_error_event_maps_vector_store_timeout_to_retriable_504():
    iterator = _make_iterator()
    chunk = _make_error_chunk("server_error", "vector_store_timeout", "vector store timed out")
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 504


@pytest.mark.parametrize(
    ("code", "status_code", "exception_type"),
    [
        ("rate_limit_exceeded", 429, litellm.RateLimitError),
        ("server_error", 500, litellm.InternalServerError),
        (429, 429, litellm.RateLimitError),
    ],
)
def test_openai_top_level_error_event_keeps_the_provider_code_and_message(
    code: str | int, status_code: int, exception_type: type[Exception]
):
    chunk = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR, sequence_number=3, code=code, message=f"upstream said {code}"
    )
    with pytest.raises(MidStreamFallbackError) as exc_info:
        _make_iterator()._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == status_code
    assert type(exc_info.value.original_exception) is exception_type
    assert f"upstream said {code}" in str(exc_info.value.original_exception)


_AZURE_RESPONSES_API_BASE: Final = "https://responses-mid-stream.openai.azure.com"
_AZURE_RESPONSES_REQUEST_URL: Final = f"{_AZURE_RESPONSES_API_BASE}/openai/responses?api-version=2025-04-01-preview"
_AZURE_RESPONSES_FAILED_EVENT: Final = {
    "type": "response.failed",
    "sequence_number": 2,
    "response": {
        "id": "resp_mid_stream",
        "object": "response",
        "created_at": 1790000000,
        "status": "failed",
        "model": "gpt-5.4-mini",
        "output": [],
        "error": {"code": "server_error", "message": "upstream failed mid-stream"},
    },
}
_AZURE_RESPONSES_WS_URL: Final = "wss://responses-mid-stream.openai.azure.com/openai/v1/responses?api-version=preview"


def _azure_responses_logging_obj(logged_api_base: str, routed_api_base: str | None) -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="gpt-5.4-mini",
        messages=[],
        stream=True,
        call_type="aresponses",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="responses-mid-stream",
        function_id="responses-mid-stream",
    )
    logging_obj.update_environment_variables(
        model="gpt-5.4-mini",
        optional_params={},
        litellm_params=get_litellm_params(
            api_base=logged_api_base,
            custom_llm_provider="azure",
            metadata={
                "model_group": "gpt-5.4-mini-azure",
                "deployment": "azure/gpt-5.4-mini",
                "api_base": routed_api_base,
            },
        ),
    )
    return logging_obj


def _assert_names_the_azure_deployment(exception: BaseException, api_base: str = _AZURE_RESPONSES_API_BASE) -> None:
    debug_info: Final = getattr(exception, "litellm_debug_info", None)
    assert isinstance(debug_info, str)
    assert f"API Base: `{api_base}`" in debug_info
    assert "model_group: `gpt-5.4-mini-azure`" in debug_info
    assert "deployment: `azure/gpt-5.4-mini`" in debug_info
    assert "api.openai.com" not in debug_info


@pytest.mark.asyncio
async def test_response_failed_after_output_reports_the_deployment_it_was_sent_to():
    delta: Final = {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": "msg_mid_stream",
        "output_index": 0,
        "content_index": 0,
        "delta": "1",
    }
    body: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in (delta, _AZURE_RESPONSES_FAILED_EVENT))
    iterator: Final = ResponsesAPIStreamingIterator(
        response=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode()),
        model="gpt-5.4-mini",
        responses_api_provider_config=ProviderConfigManager.get_provider_responses_api_config(
            provider="azure", model="gpt-5.4-mini"
        ),
        logging_obj=_azure_responses_logging_obj(_AZURE_RESPONSES_REQUEST_URL, _AZURE_RESPONSES_API_BASE),
        custom_llm_provider="azure",
    )
    stream: Final = aiter(iterator)
    assert getattr(await anext(stream), "delta", None) == "1"

    with pytest.raises(MidStreamFallbackError) as exc_info:
        await anext(stream)
    assert exc_info.value.is_pre_first_chunk is False
    _assert_names_the_azure_deployment(exc_info.value.original_exception)


class _ScriptedAzureSocket:
    def __init__(self, events: tuple[str, ...]) -> None:
        self._events = iter(events)
        self._drained = asyncio.Event()

    async def recv(self, decode: bool = False) -> str:
        event: Final = next(self._events, None)
        if event is None:
            self._drained.set()
            raise ConnectionError("backend closed")
        return event

    async def send(self, message: str) -> None:
        return None

    async def close(self) -> None:
        return None

    async def send_text(self, data: str) -> None:
        return None

    async def receive_text(self) -> str:
        await self._drained.wait()
        raise ConnectionError("client closed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("routed_api_base", "reported_api_base"),
    [
        (_AZURE_RESPONSES_API_BASE, _AZURE_RESPONSES_API_BASE),
        (None, "wss://responses-mid-stream.openai.azure.com/openai/v1/responses"),
    ],
    ids=["router", "no_router"],
)
async def test_responses_websocket_turn_failure_reports_the_deployment_it_was_sent_to(
    routed_api_base: str | None, reported_api_base: str
):
    connection_logging_obj: Final = _azure_responses_logging_obj(_AZURE_RESPONSES_WS_URL, routed_api_base)
    connection_logging_obj.pre_call(input=None, api_key="", additional_args={"api_base": _AZURE_RESPONSES_WS_URL})
    socket: Final = _ScriptedAzureSocket((json.dumps(_AZURE_RESPONSES_FAILED_EVENT),))
    streaming: Final = ResponsesWebSocketStreaming(
        websocket=socket,
        backend_ws=socket,
        logging_obj=connection_logging_obj,
        authorized_model="gpt-5.4-mini",
        custom_llm_provider="azure",
    )

    failure: Final = await streaming.bidirectional_forward()

    assert failure is not None
    _assert_names_the_azure_deployment(failure, reported_api_base)
