"""Regression tests for LIT-4185 — /v1/responses streaming must stamp
completion_start_time on the first chunk so downstream TTFT consumers
(Prometheus, OTEL, SpendLogs completionStartTime) do not fall back to
completion_start_time = end_time."""

import asyncio
import json
from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, Mock

import anyio
import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


def _sse_event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _mock_config() -> Mock:
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
    mock_responses_api_response.id = "resp_ttft"

    def _transform(model, parsed_chunk, logging_obj):
        evt_type = parsed_chunk.get("type")
        if evt_type == "response.completed":
            completed = Mock(spec=ResponseCompletedEvent)
            completed.type = ResponsesAPIStreamEvents.RESPONSE_COMPLETED
            completed.response = mock_responses_api_response
            return completed
        stub = Mock()
        stub.type = evt_type
        return stub

    mock_config.transform_streaming_response.side_effect = _transform
    return mock_config


def _make_iterator(
    *,
    sse_events: list[bytes],
    logging_obj: LiteLLMLoggingObj,
    trailing_error: Optional[Exception] = None,
) -> ResponsesAPIStreamingIterator:
    async def aiter_bytes():
        for evt in sse_events:
            yield evt
        if trailing_error is not None:
            raise trailing_error

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = aiter_bytes

    return ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-4o-mini",
        responses_api_provider_config=_mock_config(),
        logging_obj=logging_obj,
        litellm_metadata={},
        custom_llm_provider="openai",
    )


def _make_sync_iterator(
    *,
    sse_events: list[bytes],
    logging_obj: LiteLLMLoggingObj,
    trailing_error: Optional[Exception] = None,
) -> SyncResponsesAPIStreamingIterator:
    def iter_bytes():
        for evt in sse_events:
            yield evt
        if trailing_error is not None:
            raise trailing_error

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.iter_bytes = iter_bytes

    return SyncResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-4o-mini",
        responses_api_provider_config=_mock_config(),
        logging_obj=logging_obj,
        litellm_metadata={},
        custom_llm_provider="openai",
    )


def _logging_obj_stub() -> Mock:
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj.completion_start_time = None
    logging_obj.model_call_details = {"litellm_params": {}}
    return logging_obj


@pytest.mark.asyncio
async def test_responses_streaming_aclose_releases_response():
    iterator = _make_iterator(
        sse_events=_COMPLETE_STREAM_EVENTS,
        logging_obj=_logging_obj_stub(),
    )
    iterator.response.aclose = AsyncMock()

    await iterator.aclose()

    iterator.response.aclose.assert_awaited_once()
    assert iterator.finished is True


@pytest.mark.asyncio
async def test_responses_streaming_aclose_suppresses_genuine_cleanup_failure():
    """A non-RuntimeError failure in ``response.aclose()`` does not hit the sync
    fallback, so this exercises the real cleanup-failure path: the exception is
    swallowed (aclose does not raise) and ``finished`` is still set."""
    iterator = _make_iterator(
        sse_events=_COMPLETE_STREAM_EVENTS,
        logging_obj=_logging_obj_stub(),
    )
    iterator.response.aclose = AsyncMock(side_effect=ValueError("cleanup boom"))

    await iterator.aclose()

    iterator.response.aclose.assert_awaited_once()
    assert iterator.finished is True


@pytest.mark.asyncio
async def test_responses_streaming_aclose_finishes_cleanup_under_native_cancellation():
    """A native ``asyncio.Task.cancel()`` mid-close must not abandon the httpx
    connection: the inner ``response.aclose()`` still runs to completion, the
    cancellation is honored (re-raised, not swallowed), and ``finished`` is set."""
    iterator = _make_iterator(
        sse_events=_COMPLETE_STREAM_EVENTS,
        logging_obj=_logging_obj_stub(),
    )
    started = asyncio.Event()
    release = asyncio.Event()
    inner_completed = False

    async def blocking_aclose() -> None:
        nonlocal inner_completed
        started.set()
        await release.wait()
        inner_completed = True

    iterator.response.aclose = blocking_aclose

    task = asyncio.ensure_future(iterator.aclose())
    await started.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert inner_completed is True
    assert iterator.finished is True


@pytest.mark.asyncio
async def test_responses_streaming_aclose_is_idempotent():
    iterator = _make_iterator(
        sse_events=_COMPLETE_STREAM_EVENTS,
        logging_obj=_logging_obj_stub(),
    )
    iterator.response.aclose = AsyncMock()

    await iterator.aclose()
    await iterator.aclose()

    assert iterator.finished is True


@pytest.mark.asyncio
async def test_responses_streaming_aclose_falls_back_to_sync_close_on_runtime_error():
    """httpx raises ``RuntimeError`` when ``aclose()`` hits a sync stream; the
    iterator must fall back to the sync ``close()`` so the connection is released."""
    iterator = _make_iterator(
        sse_events=_COMPLETE_STREAM_EVENTS,
        logging_obj=_logging_obj_stub(),
    )
    iterator.response.aclose = AsyncMock(
        side_effect=RuntimeError("Attempted to call an async close on an sync stream.")
    )
    iterator.response.close = Mock()

    await iterator.aclose()

    iterator.response.close.assert_called_once()
    assert iterator.finished is True


@pytest.mark.asyncio
async def test_responses_streaming_iteration_after_aclose_stops_without_failure():
    """Iterating a closed stream must end cleanly with ``StopAsyncIteration``
    instead of surfacing ``httpx.StreamClosed`` through the failure handler."""
    iterator = _make_iterator(
        sse_events=_COMPLETE_STREAM_EVENTS,
        logging_obj=_logging_obj_stub(),
    )
    iterator.response.aclose = AsyncMock()

    await iterator.aclose()

    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()

    assert iterator._failure_handled is False


def test_sync_responses_streaming_close_releases_response():
    iterator = _make_sync_iterator(
        sse_events=_COMPLETE_STREAM_EVENTS,
        logging_obj=_logging_obj_stub(),
    )
    iterator.response.close = Mock()

    iterator.close()

    iterator.response.close.assert_called_once()
    assert iterator.finished is True


def _base_iterator_without_response() -> BaseResponsesAPIStreamingIterator:
    iterator = object.__new__(BaseResponsesAPIStreamingIterator)
    iterator.finished = False
    return iterator


@pytest.mark.asyncio
async def test_aclose_without_response_attribute_marks_finished():
    """Subclasses that skip the base ``__init__`` never set ``self.response``;
    the inherited ``aclose`` must not raise ``AttributeError`` for them."""
    iterator = _base_iterator_without_response()

    await iterator.aclose()

    assert iterator.finished is True


def test_close_without_response_attribute_marks_finished():
    iterator = _base_iterator_without_response()

    iterator.close()

    assert iterator.finished is True


@pytest.mark.asyncio
async def test_aclose_marks_finished_before_awaiting_response_close():
    """``finished`` must flip True at ``aclose`` entry so a concurrent
    ``__anext__`` stops immediately instead of iterating (and possibly
    failure-logging) the stream that is being deliberately closed."""
    iterator = _make_iterator(
        sse_events=_COMPLETE_STREAM_EVENTS,
        logging_obj=_logging_obj_stub(),
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_aclose() -> None:
        entered.set()
        await release.wait()

    iterator.response.aclose = blocking_aclose

    close_task = asyncio.ensure_future(iterator.aclose())
    await entered.wait()

    assert iterator.finished is True
    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()
    assert iterator._failure_handled is False

    release.set()
    await close_task
    assert iterator.finished is True


@pytest.mark.asyncio
async def test_aclose_completes_cleanup_and_preserves_persistent_anyio_cancellation():
    """Under a persistently cancelled anyio scope the shield lets cleanup run to
    completion (no spin, no hang) and returns normally, yet the cancellation is
    not lost: the caller's next checkpoint still raises ``CancelledError``."""
    iterator = _base_iterator_without_response()
    cleanup_done = asyncio.Event()

    async def slow_aclose() -> None:
        await asyncio.sleep(0.2)
        cleanup_done.set()

    response = Mock()
    response.aclose = slow_aclose
    iterator.response = response

    async def scenario() -> None:
        async with anyio.create_task_group() as tg:

            async def worker() -> None:
                tg.cancel_scope.cancel()
                await iterator.aclose()
                assert cleanup_done.is_set() is True
                assert iterator.finished is True
                with pytest.raises(asyncio.CancelledError):
                    await anyio.lowlevel.checkpoint()

            tg.start_soon(worker)

    await asyncio.wait_for(scenario(), timeout=5)


@pytest.mark.asyncio
async def test_responses_streaming_stamps_completion_start_time_on_first_chunk():
    """Without the fix, `logging_obj.completion_start_time` stays None across the
    entire stream and _success_handler_helper_fn falls back to end_time — collapsing
    the reported TTFT to full generation time."""
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj.completion_start_time = None
    logging_obj.model_call_details = {"litellm_params": {}}
    stamped: list[datetime] = []

    def _update(*, completion_start_time):
        stamped.append(completion_start_time)
        logging_obj.completion_start_time = completion_start_time
        logging_obj.model_call_details["completion_start_time"] = completion_start_time

    logging_obj._update_completion_start_time.side_effect = _update

    iterator = _make_iterator(
        sse_events=[
            _sse_event({"type": "response.created"}),
            _sse_event({"type": "response.output_text.delta", "delta": "hi"}),
            _sse_event({"type": "response.completed"}),
        ],
        logging_obj=logging_obj,
    )

    async for _ in iterator:
        pass

    assert len(stamped) == 1, (
        f"Expected exactly one first-chunk stamp; got {len(stamped)}. "
        "Later chunks must not re-stamp completion_start_time."
    )
    assert isinstance(stamped[0], datetime)


@pytest.mark.asyncio
async def test_responses_streaming_does_not_reset_prior_completion_start_time():
    """If `completion_start_time` is already set (e.g. by an outer wrapper), the
    iterator must not overwrite it — otherwise TTFT would collapse to
    time-to-last-chunk under contention."""
    prior = datetime(2020, 1, 1, 0, 0, 0)
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj.completion_start_time = prior
    logging_obj.model_call_details = {"litellm_params": {}}

    iterator = _make_iterator(
        sse_events=[
            _sse_event({"type": "response.created"}),
            _sse_event({"type": "response.completed"}),
        ],
        logging_obj=logging_obj,
    )

    async for _ in iterator:
        pass

    logging_obj._update_completion_start_time.assert_not_called()
    assert logging_obj.completion_start_time == prior


_COMPLETE_STREAM_EVENTS = [
    _sse_event({"type": "response.created"}),
    _sse_event({"type": "response.output_text.delta", "delta": "hi"}),
    _sse_event({"type": "response.completed"}),
]

_TRAILING_ERRORS = [
    httpx.ReadError("Response payload is not completed"),
    httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("trailing_error", _TRAILING_ERRORS, ids=type)
async def test_transport_error_after_completed_event_ends_stream_cleanly(trailing_error):
    """A sloppy connection close after `response.completed` must not turn a
    complete stream into an error (regression guard for the transport no longer
    swallowing ClientPayloadError/TransferEncodingError)."""
    iterator = _make_iterator(
        sse_events=_COMPLETE_STREAM_EVENTS,
        logging_obj=_logging_obj_stub(),
        trailing_error=trailing_error,
    )

    seen = [event.type async for event in iterator]

    assert ResponsesAPIStreamEvents.RESPONSE_COMPLETED in seen


@pytest.mark.asyncio
async def test_transport_error_before_completed_event_raises():
    """A connection lost before any terminal event is a real failure and must
    surface, not end the stream as if it completed."""
    iterator = _make_iterator(
        sse_events=_COMPLETE_STREAM_EVENTS[:-1],
        logging_obj=_logging_obj_stub(),
        trailing_error=httpx.ReadError("Response payload is not completed"),
    )

    with pytest.raises(httpx.ReadError):
        async for _ in iterator:
            pass


@pytest.mark.parametrize("trailing_error", _TRAILING_ERRORS, ids=type)
def test_sync_transport_error_after_completed_event_ends_stream_cleanly(trailing_error):
    iterator = _make_sync_iterator(
        sse_events=_COMPLETE_STREAM_EVENTS,
        logging_obj=_logging_obj_stub(),
        trailing_error=trailing_error,
    )

    seen = [event.type for event in iterator]

    assert ResponsesAPIStreamEvents.RESPONSE_COMPLETED in seen


def test_sync_transport_error_before_completed_event_raises():
    iterator = _make_sync_iterator(
        sse_events=_COMPLETE_STREAM_EVENTS[:-1],
        logging_obj=_logging_obj_stub(),
        trailing_error=httpx.ReadError("Response payload is not completed"),
    )

    with pytest.raises(httpx.ReadError):
        for _ in iterator:
            pass
