"""
Regression tests for https://github.com/BerriAI/litellm/issues/46557.

A chat-completions call on a responses-mode deployment wraps the
BaseResponsesAPIStreamingIterator in a CustomStreamWrapper, and both layers
share one Logging object. When the stream fails mid-flight the inner iterator
dispatches the failure handlers before re-raising, and the outer wrapper
catches the same exception and dispatches them again. The
Logging.mark_logging_complete stream carve-out must stay restricted to
success events: a stream is one request, so its failure must be logged exactly
once, and the has_logged_async_failure marker is what stops the second
dispatch. Before the logging split this marker was skipped for stream=True
calls entirely and every CustomLogger.async_log_failure_event fired twice
for the same exception object, which also misattributed the duplicate to
whatever deployment the logging object described when the fire-and-forget
outer task eventually ran.
"""

import asyncio
import json
from datetime import datetime
from typing import Any
from unittest.mock import Mock

from litellm.completion_extras.litellm_responses_transformation.transformation import (
    OpenAiResponsesToChatCompletionStreamIterator,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ErrorEvent,
    ErrorEventError,
    ResponsesAPIStreamEvents,
)

_MID_STREAM_ERROR_EVENTS: list[dict[str, Any]] = [
    {"type": "response.output_text.delta", "delta": "partial "},
    {
        "type": "error",
        "error": {"type": "server_error", "code": "internal_error", "message": "boom"},
    },
]

_PRE_FIRST_CHUNK_ERROR_EVENTS: list[dict[str, Any]] = [
    {
        "type": "error",
        "error": {"type": "server_error", "code": "internal_error", "message": "boom"},
    },
]


class _FailureEventRecorder(CustomLogger):
    """Records every async_log_failure_event invocation with the kwargs it saw."""

    def __init__(self) -> None:
        super().__init__()
        self.failure_events: list[dict[str, Any]] = []

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        self.failure_events.append(
            {
                "exception": kwargs.get("exception"),
                "start_time": start_time,
                "end_time": end_time,
            }
        )


def _make_bridge_stream(
    logging_obj: LiteLLMLoggingObj,
    events: list[dict[str, Any]],
) -> CustomStreamWrapper:
    """Rebuild the chat-completions streaming stack for a responses-mode
    deployment the way the bridge wires it: the SSE responses iterator,
    wrapped in the responses-to-chat translation iterator, wrapped in the
    CustomStreamWrapper, all sharing one Logging object."""
    sse_payload: bytes = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)

    async def mock_aiter_bytes() -> Any:
        yield sse_payload

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    def transform(model: str, parsed_chunk: dict[str, Any], logging_obj: Any) -> Any:
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

    responses_iterator = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5.4",
        responses_api_provider_config=mock_config,
        logging_obj=logging_obj,
        custom_llm_provider="openai",
    )
    bridge_iterator = OpenAiResponsesToChatCompletionStreamIterator(
        responses_iterator,
        sync_stream=False,
        json_mode=False,
    )
    return CustomStreamWrapper(
        completion_stream=bridge_iterator,
        model="gpt-5.4",
        custom_llm_provider="openai",
        logging_obj=logging_obj,
    )


async def _drain_stream_and_settle_logging(stream: CustomStreamWrapper) -> None:
    try:
        async for _ in stream:
            pass
    except Exception:
        pass
    # Both failure dispatches are fire-and-forget asyncio tasks: the inner
    # iterator's spawned logging task and the wrapper's created task. Give the
    # loop enough turns to run whichever tasks exist before counting.
    for _ in range(10):
        await asyncio.sleep(0)


def _make_logging_obj(recorder: _FailureEventRecorder) -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="test-call-46557",
        function_id="test-function-46557",
        dynamic_async_failure_callbacks=[recorder],
    )


async def test_mid_stream_failure_logs_exactly_one_async_failure_event() -> None:
    """A failure after streamed output must reach async_log_failure_event
    exactly once, from whichever layer dispatches first, and the logged
    exception must describe the mid-stream error the stream raised."""
    recorder = _FailureEventRecorder()
    logging_obj = _make_logging_obj(recorder)
    stream = _make_bridge_stream(logging_obj, _MID_STREAM_ERROR_EVENTS)

    await _drain_stream_and_settle_logging(stream)

    assert len(recorder.failure_events) == 1, (
        f"a mid-stream failure must produce exactly 1 async failure event, "
        f"got {len(recorder.failure_events)}: {recorder.failure_events}"
    )
    assert logging_obj.model_call_details.get("has_logged_async_failure") is True
    logged_exception = recorder.failure_events[0]["exception"]
    assert logged_exception is not None
    # The stream surfaced the mapped provider error, so the logged failure
    # must describe that same failure, not a different or wrapped one.
    assert "boom" in str(logged_exception)


async def test_pre_first_chunk_failure_logs_exactly_one_async_failure_event() -> None:
    """A failure before any output started must also log exactly one async
    failure event; the dedup must not depend on chunks having been yielded."""
    recorder = _FailureEventRecorder()
    logging_obj = _make_logging_obj(recorder)
    stream = _make_bridge_stream(logging_obj, _PRE_FIRST_CHUNK_ERROR_EVENTS)

    await _drain_stream_and_settle_logging(stream)

    assert len(recorder.failure_events) == 1, (
        f"a pre-first-chunk failure must produce exactly 1 async failure event, "
        f"got {len(recorder.failure_events)}: {recorder.failure_events}"
    )
    assert logging_obj.model_call_details.get("has_logged_async_failure") is True
