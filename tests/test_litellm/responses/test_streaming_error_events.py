import json
from unittest.mock import Mock, patch

import httpx
import pytest

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


def _mock_logging_obj() -> Mock:
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj.model_call_details = {"litellm_params": {}}
    logging_obj.async_failure_handler = Mock()
    logging_obj.failure_handler = Mock()
    return logging_obj


@pytest.mark.asyncio
async def test_async_responses_stream_error_event_raises_litellm_exception():
    error_chunk = {
        "type": "error",
        "sequence_number": 2,
        "error": {
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "message": "Input exceeds the model context window.",
            "param": "input",
        },
    }

    async def mock_aiter_bytes():
        yield f"data: {json.dumps(error_chunk)}\n\n".encode("utf-8")

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = _mock_logging_obj()
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_config.transform_streaming_response.return_value = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=2,
        error=ErrorEventError(
            type="invalid_request_error",
            code="context_length_exceeded",
            message="Input exceeds the model context window.",
            param="input",
        ),
    )
    iterator = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5.4-mini",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    with (
        pytest.raises(litellm.ContextWindowExceededError),
        patch(
            "litellm.responses.streaming_iterator.run_async_function"
        ) as mock_run_async,
        patch("litellm.responses.streaming_iterator.executor") as mock_executor,
    ):
        await iterator.__anext__()

    assert iterator.finished is True
    mock_run_async.assert_called_once()
    mock_executor.submit.assert_called_once()


def test_sync_responses_stream_error_event_raises_litellm_exception():
    error_chunk = {
        "type": "error",
        "sequence_number": 2,
        "error": {
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
            "message": "Too many requests.",
            "param": None,
        },
    }

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.iter_bytes.return_value = [
        f"data: {json.dumps(error_chunk)}\n\n".encode("utf-8")
    ]
    mock_logging_obj = _mock_logging_obj()
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_config.transform_streaming_response.return_value = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=2,
        error=ErrorEventError(
            type="rate_limit_error",
            code="rate_limit_exceeded",
            message="Too many requests.",
            param=None,
        ),
    )
    iterator = SyncResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5.4-mini",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    with (
        pytest.raises(litellm.RateLimitError),
        patch(
            "litellm.responses.streaming_iterator.run_async_function"
        ) as mock_run_async,
        patch("litellm.responses.streaming_iterator.executor") as mock_executor,
    ):
        next(iterator)

    assert iterator.finished is True
    mock_run_async.assert_called_once()
    mock_executor.submit.assert_called_once()


def test_responses_stream_error_event_exception_mapping_fallbacks():
    iterator = BaseResponsesAPIStreamingIterator(
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.example.test")
        ),
        model="gpt-5.4-mini",
        responses_api_provider_config=Mock(spec=BaseResponsesAPIConfig),
        logging_obj=_mock_logging_obj(),
        custom_llm_provider="openai",
    )

    auth_event = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=1,
        error=ErrorEventError(
            type="authentication_error",
            code="invalid_api_key",
            message="Invalid API key.",
            param=None,
        ),
    )
    default_event = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=2,
        error=ErrorEventError(
            type="invalid_request_error",
            code="bad_request",
            message="Bad request.",
            param="input",
        ),
    )

    assert isinstance(
        iterator._exception_from_error_event(auth_event), litellm.AuthenticationError
    )
    assert isinstance(
        iterator._exception_from_error_event(default_event), litellm.BadRequestError
    )
