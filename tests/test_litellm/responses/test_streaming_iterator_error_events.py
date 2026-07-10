"""
Regression: in-stream error events (type="error", type="response.failed") must
raise litellm.APIError so callers see an exception rather than a benign chunk.
Previously they were returned as-is, silently bypassing router cooldown logic.

Also covers: ErrorEventError.param must accept dict payloads without raising a
Pydantic ValidationError (previously typed as Optional[str]).
"""

import json
import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.types.llms.openai import (
    ErrorEvent,
    ErrorEventError,
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


def _make_error_chunk(code: str, message: str = "err") -> ErrorEvent:
    error_obj = ErrorEventError(
        type="rate_limit_error" if code.startswith("rate_limit") else "invalid_request_error",
        code=code,
        message=message,
    )
    return ErrorEvent(type=ResponsesAPIStreamEvents.ERROR, sequence_number=0, error=error_obj)


def test_maybe_raise_for_error_event_raises_on_error_type():
    iterator = _make_iterator()
    chunk = _make_error_chunk("internal_error", "something went wrong")
    with pytest.raises(litellm.APIError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 500


def test_maybe_raise_for_error_event_maps_rate_limit_to_429():
    iterator = _make_iterator()
    chunk = _make_error_chunk("rate_limit_exceeded", "Too many requests")
    with pytest.raises(litellm.APIError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 429


def test_maybe_raise_for_error_event_maps_invalid_request_to_400():
    iterator = _make_iterator()
    chunk = _make_error_chunk("invalid_request_error", "bad request")
    with pytest.raises(litellm.APIError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 400


def test_maybe_raise_for_error_event_maps_context_length_to_400():
    iterator = _make_iterator()
    chunk = _make_error_chunk("context_length_exceeded", "too long")
    with pytest.raises(litellm.APIError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 400


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


@pytest.mark.asyncio
async def test_async_iterator_raises_api_error_on_error_event():
    error_payload = {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
            "message": "rate limited",
        },
    }
    sse_bytes = f"data: {json.dumps(error_payload)}\n\n".encode()

    async def mock_aiter_bytes():
        yield sse_bytes

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    error_obj = ErrorEventError(
        type="rate_limit_error", code="rate_limit_exceeded", message="rate limited"
    )
    mock_config.transform_streaming_response.return_value = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR, sequence_number=0, error=error_obj
    )

    iterator = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    with pytest.raises(litellm.APIError) as exc_info:
        async for _ in iterator:
            pass
    assert exc_info.value.status_code == 429


def test_maybe_raise_for_response_failed_event_with_dict_error():
    """response.failed chunks carry a dict error on .response.error; covers dict branch."""
    iterator = _make_iterator()
    mock_response_obj = Mock()
    mock_response_obj.error = {"type": "rate_limit_error", "code": "rate_limit_exceeded", "message": "throttled"}
    chunk = Mock()
    chunk.type = "response.failed"
    chunk.response = mock_response_obj
    with pytest.raises(litellm.APIError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 429


def test_maybe_raise_for_error_event_null_error_obj():
    """error chunk with no error field: message and code default; raises 500."""
    iterator = _make_iterator()
    chunk = Mock()
    chunk.type = "error"
    chunk.error = None
    with pytest.raises(litellm.APIError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 500
    assert "Response API in-stream error" in str(exc_info.value)


def test_sync_iterator_raises_api_error_on_error_event():
    """SyncResponsesAPIStreamingIterator must raise APIError on error events."""
    error_payload = {
        "type": "error",
        "error": {"type": "rate_limit_error", "code": "rate_limit_exceeded", "message": "throttled"},
    }
    sse_bytes = f"data: {json.dumps(error_payload)}\n\n".encode()

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.iter_bytes.return_value = iter([sse_bytes])
    mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    error_obj = ErrorEventError(
        type="rate_limit_error", code="rate_limit_exceeded", message="throttled"
    )
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

    with pytest.raises(litellm.APIError) as exc_info:
        for _ in iterator:
            pass
    assert exc_info.value.status_code == 429
