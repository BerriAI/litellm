"""Regression tests for LIT-4185 — /v1/responses streaming must stamp
completion_start_time on the first chunk so downstream TTFT consumers
(Prometheus, OTEL, SpendLogs completionStartTime) do not fall back to
completion_start_time = end_time."""

import asyncio
import importlib
import json
from contextlib import suppress
from types import SimpleNamespace
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Final, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from pydantic_core import PydanticSerializationError

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import(
    CachedResponsesAPIStreamingIterator,
    MockResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    ResponsesWebSocketStreaming,
    SyncResponsesAPIStreamingIterator,
    _estimate_usage_from_text,
)
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.responses import streaming_iterator as streaming_module
from litellm.types.utils import CallTypes, Usage
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


def _sse_event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _mock_config() -> Mock:
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_responses_api_response = ResponsesAPIResponse(
        id="resp_ttft",
        created_at=0,
        status="completed",
        model="gpt-4o-mini",
        object="response",
        output=[],
        usage=ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )

    def _transform(model, parsed_chunk, logging_obj):
        evt_type = parsed_chunk.get("type")
        if evt_type == "response.completed":
            return ResponseCompletedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=mock_responses_api_response,
            )
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
    config: Mock | None = None,
    request_data: dict | None = None,
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
        responses_api_provider_config=config or _mock_config(),
        logging_obj=logging_obj,
        litellm_metadata={},
        custom_llm_provider="openai",
        request_data=request_data,
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

    logging_obj.update_completion_start_time.side_effect = _update

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

    logging_obj.update_completion_start_time.assert_not_called()
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


_DONE_MARKER: Final = b"data: [DONE]\n\n"
_CREATED_EVENT: Final = _sse_event({"type": "response.created"})
_IN_PROGRESS_EVENT: Final = _sse_event({"type": "response.in_progress"})
_PARTIAL_OUTPUT_EVENTS: Final = _COMPLETE_STREAM_EVENTS[:-1]
_PRE_OUTPUT_PREFIXES: Final = [
    pytest.param([], True, id="nothing-yielded"),
    pytest.param([_CREATED_EVENT], False, id="created"),
    pytest.param([_CREATED_EVENT, _IN_PROGRESS_EVENT], False, id="created-and-in-progress"),
]


def _failure_tracking_logging_obj() -> Mock:
    logging_obj: Final = _logging_obj_stub()
    logging_obj.async_failure_handler = AsyncMock()
    return logging_obj


def _assert_failure_logged_once(logging_obj: Mock, exception: Exception) -> None:
    assert logging_obj.async_failure_handler.await_count == 1
    assert logging_obj.async_failure_handler.await_args.kwargs["exception"] is exception


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix, pre_first_chunk", _PRE_OUTPUT_PREFIXES)
@pytest.mark.parametrize("trailing_error", _TRAILING_ERRORS, ids=type)
async def test_transport_error_before_any_output_raises_fallback_error(prefix, pre_first_chunk, trailing_error):
    """A connection lost while only lifecycle events (response.created / response.in_progress)
    have streamed is fallback-eligible, so it must surface as the MidStreamFallbackError the
    router re-routes, carrying the raw transport error and no generated content."""
    logging_obj: Final = _failure_tracking_logging_obj()
    iterator: Final = _make_iterator(sse_events=prefix, logging_obj=logging_obj, trailing_error=trailing_error)

    with pytest.raises(MidStreamFallbackError) as exc_info:
        async for _ in iterator:
            pass

    assert exc_info.value.original_exception is trailing_error
    assert exc_info.value.is_pre_first_chunk is pre_first_chunk
    assert exc_info.value.generated_content == ""
    _assert_failure_logged_once(logging_obj, trailing_error)


@pytest.mark.asyncio
async def test_transport_error_after_output_started_is_not_fallback_eligible():
    logging_obj: Final = _failure_tracking_logging_obj()
    trailing_error: Final = httpx.ReadError("Response payload is not completed")
    iterator: Final = _make_iterator(
        sse_events=_PARTIAL_OUTPUT_EVENTS, logging_obj=logging_obj, trailing_error=trailing_error
    )

    with pytest.raises(httpx.ReadError) as exc_info:
        async for _ in iterator:
            pass

    assert exc_info.value is trailing_error
    _assert_failure_logged_once(logging_obj, trailing_error)


@pytest.mark.asyncio
@pytest.mark.parametrize("trailer", [[], [_DONE_MARKER]], ids=["eof", "done-marker"])
async def test_stream_ending_after_partial_output_without_terminal_event_raises(trailer):
    """A clean EOF or `[DONE]` after output text but with no response.completed /
    response.incomplete / response.failed is a truncated answer: the partial events still
    reach the caller, then an explicit error follows instead of a normal end of stream."""
    logging_obj: Final = _failure_tracking_logging_obj()
    iterator: Final = _make_iterator(sse_events=[*_PARTIAL_OUTPUT_EVENTS, *trailer], logging_obj=logging_obj)

    created: Final = await iterator.__anext__()
    delta: Final = await iterator.__anext__()
    with pytest.raises(litellm.APIConnectionError) as exc_info:
        await iterator.__anext__()

    assert (created.type, delta.type) == ("response.created", "response.output_text.delta")
    assert not isinstance(exc_info.value, MidStreamFallbackError)
    assert exc_info.value.llm_provider == "openai"
    _assert_failure_logged_once(logging_obj, exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix, pre_first_chunk", _PRE_OUTPUT_PREFIXES)
@pytest.mark.parametrize("trailer", [[], [_DONE_MARKER]], ids=["eof", "done-marker"])
async def test_stream_ending_before_any_output_raises_fallback_error(prefix, pre_first_chunk, trailer):
    logging_obj: Final = _failure_tracking_logging_obj()
    iterator: Final = _make_iterator(sse_events=[*prefix, *trailer], logging_obj=logging_obj)

    with pytest.raises(MidStreamFallbackError) as exc_info:
        async for _ in iterator:
            pass

    assert isinstance(exc_info.value.original_exception, litellm.APIConnectionError)
    assert exc_info.value.is_pre_first_chunk is pre_first_chunk
    assert exc_info.value.generated_content == ""
    _assert_failure_logged_once(logging_obj, exc_info.value.original_exception)


@pytest.mark.asyncio
@pytest.mark.parametrize("trailer", [[], [_DONE_MARKER]], ids=["eof", "done-marker"])
async def test_complete_stream_still_ends_normally(trailer):
    logging_obj: Final = _failure_tracking_logging_obj()
    iterator: Final = _make_iterator(sse_events=[*_COMPLETE_STREAM_EVENTS, *trailer], logging_obj=logging_obj)

    seen: Final = [event.type async for event in iterator]

    assert seen[-1] == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    assert logging_obj.async_failure_handler.await_count == 0


@pytest.mark.parametrize("trailing_error", _TRAILING_ERRORS, ids=type)
def test_sync_transport_error_before_any_output_raises_fallback_error(trailing_error):
    logging_obj: Final = _failure_tracking_logging_obj()
    iterator: Final = _make_sync_iterator(
        sse_events=[_CREATED_EVENT, _IN_PROGRESS_EVENT],
        logging_obj=logging_obj,
        trailing_error=trailing_error,
    )

    with pytest.raises(MidStreamFallbackError) as exc_info:
        for _ in iterator:
            pass

    assert exc_info.value.original_exception is trailing_error
    assert exc_info.value.is_pre_first_chunk is False
    assert exc_info.value.generated_content == ""
    _assert_failure_logged_once(logging_obj, trailing_error)


@pytest.mark.parametrize("trailer", [[], [_DONE_MARKER]], ids=["eof", "done-marker"])
def test_sync_stream_ending_after_partial_output_without_terminal_event_raises(trailer):
    logging_obj: Final = _failure_tracking_logging_obj()
    iterator: Final = _make_sync_iterator(sse_events=[*_PARTIAL_OUTPUT_EVENTS, *trailer], logging_obj=logging_obj)

    created: Final = next(iterator)
    delta: Final = next(iterator)
    with pytest.raises(litellm.APIConnectionError) as exc_info:
        next(iterator)

    assert (created.type, delta.type) == ("response.created", "response.output_text.delta")
    assert not isinstance(exc_info.value, MidStreamFallbackError)
    _assert_failure_logged_once(logging_obj, exc_info.value)


def test_sync_stream_ending_before_any_output_raises_fallback_error():
    logging_obj: Final = _failure_tracking_logging_obj()
    iterator: Final = _make_sync_iterator(sse_events=[_CREATED_EVENT], logging_obj=logging_obj)

    with pytest.raises(MidStreamFallbackError) as exc_info:
        for _ in iterator:
            pass

    assert isinstance(exc_info.value.original_exception, litellm.APIConnectionError)
    assert exc_info.value.is_pre_first_chunk is False
    _assert_failure_logged_once(logging_obj, exc_info.value.original_exception)


@pytest.mark.parametrize("trailer", [[], [_DONE_MARKER]], ids=["eof", "done-marker"])
def test_sync_complete_stream_still_ends_normally(trailer):
    logging_obj: Final = _failure_tracking_logging_obj()
    iterator: Final = _make_sync_iterator(sse_events=[*_COMPLETE_STREAM_EVENTS, *trailer], logging_obj=logging_obj)

    seen: Final = [event.type for event in iterator]

    assert seen[-1] == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    assert logging_obj.async_failure_handler.await_count == 0


class _LoopRecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.failure_loop: asyncio.AbstractEventLoop | None = None
        self.failure_deployment_id: str | None = None
        self.failure_finished = False
        self.hook_loop: asyncio.AbstractEventLoop | None = None
        self.hook_finished = False

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.failure_loop = asyncio.get_running_loop()
        self.failure_deployment_id = kwargs["litellm_params"].get("model_info", {}).get("id")
        await asyncio.sleep(0.05)
        self.failure_finished = True

    async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
        self.hook_loop = asyncio.get_running_loop()
        await asyncio.sleep(0.05)
        self.hook_finished = True
        return None


class _SyncOnlyRecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.sync_failure_finished = False

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.sync_failure_finished = True


class _OrderRecordingSyncLogger(CustomLogger):
    def __init__(self, async_recorder: _LoopRecordingLogger) -> None:
        super().__init__()
        self._async_recorder: Final = async_recorder
        self.async_failure_finished_first: bool | None = None

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.async_failure_finished_first = self._async_recorder.failure_finished


def _real_logging_obj(
    *, call_type: str = "aresponses", litellm_params: dict[str, object] | None = None
) -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type=call_type,
        start_time=datetime.now(),
        litellm_call_id="lit-8678-test",
        function_id="lit-8678-test",
    )
    logging_obj.model_call_details["litellm_params"] = (
        dict(litellm_params) if litellm_params is not None else {"aresponses": True}
    )
    return logging_obj


async def _wait_until(condition: Callable[[], bool]) -> None:
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


@pytest.mark.asyncio
async def test_transport_error_failure_logging_runs_on_the_iterating_loop(monkeypatch):
    """LIT-8678: a stream failure used to run async_failure_handler on a helper loop in a
    worker thread and block the iterating loop until it finished, so a callback waiting on
    state bound to that loop (a batch logger's flush lock) stalled the whole proxy."""
    recorder: Final = _LoopRecordingLogger()
    monkeypatch.setattr(litellm, "_async_failure_callback", [recorder])
    monkeypatch.setattr(litellm, "failure_callback", [])
    iterator: Final = _make_iterator(
        sse_events=_PARTIAL_OUTPUT_EVENTS,
        logging_obj=_real_logging_obj(),
        trailing_error=httpx.ReadError("Response payload is not completed"),
    )

    with pytest.raises(httpx.ReadError):
        async for _ in iterator:
            pass

    assert recorder.failure_finished is True
    assert recorder.failure_loop is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_failure_logging_finishes_before_the_error_reaches_the_consumer(monkeypatch):
    """The router's mid-stream fallback re-enters the same logging object for the next
    deployment as soon as it catches the error, so failure logging that still runs after
    the raise reads the fallback deployment's params and cools down the wrong deployment."""
    recorder: Final = _LoopRecordingLogger()
    monkeypatch.setattr(litellm, "_async_failure_callback", [recorder])
    monkeypatch.setattr(litellm, "failure_callback", [])
    logging_obj: Final = _real_logging_obj(
        litellm_params={"aresponses": True, "model_info": {"id": "primary-deployment"}}
    )
    iterator: Final = _make_iterator(
        sse_events=[],
        logging_obj=logging_obj,
        trailing_error=httpx.ReadError("Response payload is not completed"),
    )

    with pytest.raises(MidStreamFallbackError):
        async for _ in iterator:
            pass
    logging_obj.model_call_details["litellm_params"]["model_info"] = {"id": "fallback-deployment"}
    await _wait_until(lambda: recorder.failure_finished)

    assert recorder.failure_deployment_id == "primary-deployment"


@pytest.mark.asyncio
async def test_sync_stream_failure_inside_a_running_loop_still_runs_sync_only_callbacks(monkeypatch):
    recorder: Final = _SyncOnlyRecordingLogger()
    monkeypatch.setattr(litellm, "failure_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    iterator: Final = _make_sync_iterator(
        sse_events=_PARTIAL_OUTPUT_EVENTS,
        logging_obj=_real_logging_obj(call_type="responses", litellm_params={}),
        trailing_error=httpx.ReadError("Response payload is not completed"),
    )

    with pytest.raises(httpx.ReadError):
        for _ in iterator:
            pass

    await _wait_until(lambda: recorder.sync_failure_finished)


@pytest.mark.asyncio
async def test_sync_failure_callbacks_run_after_async_failure_logging_finishes(monkeypatch):
    """Both handlers read the same logging object, so the sync one must not start while the
    async one is still running, which is the ordering the blocking dispatch used to give."""
    async_recorder: Final = _LoopRecordingLogger()
    sync_recorder: Final = _OrderRecordingSyncLogger(async_recorder)
    monkeypatch.setattr(litellm, "_async_failure_callback", [async_recorder])
    monkeypatch.setattr(litellm, "failure_callback", [sync_recorder])
    iterator: Final = _make_sync_iterator(
        sse_events=_PARTIAL_OUTPUT_EVENTS,
        logging_obj=_real_logging_obj(call_type="responses", litellm_params={}),
        trailing_error=httpx.ReadError("Response payload is not completed"),
    )

    with pytest.raises(httpx.ReadError):
        for _ in iterator:
            pass

    await _wait_until(lambda: sync_recorder.async_failure_finished_first is not None)

    assert sync_recorder.async_failure_finished_first is True


@pytest.mark.asyncio
async def test_completed_stream_success_deployment_hook_runs_on_the_iterating_loop(monkeypatch):
    recorder: Final = _LoopRecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    iterator: Final = _make_iterator(sse_events=_COMPLETE_STREAM_EVENTS, logging_obj=_logging_obj_stub())

    async for _ in iterator:
        pass

    assert recorder.hook_finished is True
    assert recorder.hook_loop is asyncio.get_running_loop()


def test_stream_cache_write_completes_when_asyncio_run_closes_the_loop(monkeypatch):
    """
    Regression test for LIT-6184 on the /v1/responses streaming surface: the
    completed-stream cache write was dispatched as a bare fire-and-forget task,
    so asyncio.run cancelled it at loop close before the write landed. The
    write must survive loop shutdown just like the chat-completions one.
    """
    import asyncio
    from types import SimpleNamespace

    import litellm
    from litellm.types.utils import CallTypes

    writes = []

    class _SlowWriteCache:
        async def async_add_cache(self, result, dynamic_cache_object=None, **kwargs):
            await asyncio.sleep(0.2)
            writes.append(result)

        def add_cache(self, *args, **kwargs):
            raise AssertionError("sync write must not run on the async path")

    caching_handler = SimpleNamespace(
        request_kwargs={
            "model": "test-model",
            "input": "hello",
            "stream": True,
            "caching": True,
            "metadata": None,
            "custom_llm_provider": "openai",
        },
        preset_cache_key="responses-stream-cache-key",
        original_function=litellm.aresponses,
        dual_cache=None,
        _should_store_result_in_cache=lambda original_function, kwargs: True,
    )
    logging_obj = SimpleNamespace(
        model_call_details={"litellm_params": {}},
        llm_caching_handler=caching_handler,
    )
    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=Mock(spec=BaseResponsesAPIConfig),
        logging_obj=logging_obj,
        request_data=caching_handler.request_kwargs,
        call_type=CallTypes.aresponses.value,
    )
    iterator.completed_response = ResponseCompletedEvent(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=ResponsesAPIResponse(
            id="resp_lit6184",
            created_at=int(datetime.now().timestamp()),
            status="completed",
            model="test-model",
            object="response",
            output=[
                {
                    "type": "message",
                    "id": "msg_lit6184",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "cached", "annotations": []}],
                }
            ],
        ),
    )
    monkeypatch.setattr(litellm, "cache", _SlowWriteCache())

    async def _short_lived_script():
        iterator._persist_completed_response_to_cache(is_async=True)

    asyncio.run(_short_lived_script())

    assert len(writes) == 1


def test_run_post_success_hooks_does_not_report_generation_time_as_overhead():
    """LIT-5466: the provider call is timed to first byte, so at stream completion the total minus
    that duration is token generation, not LiteLLM overhead."""
    logging_obj = _logging_obj_stub()
    logging_obj.model_call_details = {"litellm_params": {}, "llm_api_duration_ms": 200.0}
    logging_obj.caching_details = None

    class _CompletedEvent:
        def __init__(self) -> None:
            self._hidden_params: dict = {}

    iterator = _make_iterator(sse_events=[], logging_obj=logging_obj)
    iterator.completed_response = _CompletedEvent()
    iterator.start_time = datetime(2025, 1, 1, 0, 0, 0)

    iterator._run_post_success_hooks(datetime(2025, 1, 1, 0, 0, 10))

    assert iterator.completed_response._hidden_params["_response_ms"] == 10000.0
    assert "litellm_overhead_time_ms" not in iterator.completed_response._hidden_params


def _mock_config_with_completed_response(response: ResponsesAPIResponse) -> Mock:
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    def _transform(model, parsed_chunk, logging_obj):
        evt_type = parsed_chunk.get("type")
        if evt_type == "response.completed":
            return ResponseCompletedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=response,
            )
        stub = Mock()
        stub.type = evt_type
        if "delta" in parsed_chunk:
            stub.delta = parsed_chunk.get("delta")
        if "item" in parsed_chunk:
            stub.item = parsed_chunk.get("item")
        return stub

    mock_config.transform_streaming_response.side_effect = _transform
    return mock_config


def _responses_api_response_without_usage() -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_no_usage",
        created_at=int(datetime(2025, 1, 1).timestamp()),
        status="completed",
        model="gpt-4o-mini",
        object="response",
        output=[],
        usage=None,
    )


@pytest.mark.asyncio
async def test_completed_event_without_usage_gets_text_estimate():
    """A response.completed event carrying usage: null still bills: the
    iterator estimates usage from the request input and generated text."""
    response = _responses_api_response_without_usage()
    iterator = _make_iterator(
        sse_events=[
            _sse_event({"type": "response.output_text.delta", "delta": "hello world"}),
            _sse_event({"type": "response.completed", "response": {}}),
        ],
        logging_obj=_logging_obj_stub(),
        config=_mock_config_with_completed_response(response),
        request_data={"input": "count these input tokens please"},
    )

    async for _ in iterator:
        pass

    usage = iterator.completed_response.response.usage
    assert usage is not None
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens


@pytest.mark.asyncio
async def test_completed_event_with_usage_is_left_untouched():
    """Provider-reported usage on response.completed wins over the estimate."""
    response = _responses_api_response_with_usage()
    iterator = _make_iterator(
        sse_events=[
            _sse_event({"type": "response.output_text.delta", "delta": "hello world"}),
            _sse_event({"type": "response.completed", "response": {}}),
        ],
        logging_obj=_logging_obj_stub(),
        config=_mock_config_with_completed_response(response),
        request_data={"input": "count these input tokens please"},
    )

    async for _ in iterator:
        pass

    usage = iterator.completed_response.response.usage
    assert usage.input_tokens == 20
    assert usage.output_tokens == 60
    assert usage.total_tokens == 80


def _responses_api_response_with_usage() -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_lit6427",
        created_at=int(datetime(2025, 1, 1).timestamp()),
        status="completed",
        model="mantle-claude",
        object="response",
        output=[],
        usage=ResponseAPIUsage(input_tokens=20, output_tokens=60, total_tokens=80),
    )


def test_stamp_responses_usage_cost_stamps_computed_cost():
    from litellm.responses.streaming_iterator import _stamp_responses_usage_cost

    response = _responses_api_response_with_usage()
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj.response_cost_calculator.return_value = 0.000704

    _stamp_responses_usage_cost(response, logging_obj)

    assert getattr(response.usage, "cost", None) == pytest.approx(0.000704)
    logging_obj.response_cost_calculator.assert_called_once_with(result=response)


def test_stamp_responses_usage_cost_keeps_provider_reported_cost():
    from litellm.responses.streaming_iterator import _stamp_responses_usage_cost

    response = _responses_api_response_with_usage()
    setattr(response.usage, "cost", 0.5)
    logging_obj = Mock(spec=LiteLLMLoggingObj)

    _stamp_responses_usage_cost(response, logging_obj)

    assert getattr(response.usage, "cost", None) == pytest.approx(0.5)
    logging_obj.response_cost_calculator.assert_not_called()


def _unvalidated_response_with_dict_usage(usage: dict) -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_construct(
        id="resp_lit7391",
        created_at=int(datetime(2025, 1, 1).timestamp()),
        status="completed",
        model="perplexity/deepseek-v4-flash-0731",
        object="response",
        output=[],
        truncation="",
        usage=usage,
    )


def test_stamp_responses_usage_cost_keeps_provider_cost_from_dict_usage():
    from litellm.responses.streaming_iterator import _stamp_responses_usage_cost

    response = _unvalidated_response_with_dict_usage(
        {
            "input_tokens": 29,
            "output_tokens": 120,
            "output_tokens_details": {"reasoning_tokens": 117},
            "total_tokens": 149,
            "cost": {"currency": "USD", "input_cost": 0, "output_cost": 3e-05, "total_cost": 3e-05},
        }
    )
    logging_obj = Mock(spec=LiteLLMLoggingObj)

    _stamp_responses_usage_cost(response, logging_obj)

    assert isinstance(response.usage, ResponseAPIUsage)
    assert response.usage.cost == pytest.approx(3e-05)
    assert response.usage.output_tokens_details.reasoning_tokens == 117
    logging_obj.response_cost_calculator.assert_not_called()


def test_stamp_responses_usage_cost_computes_cost_for_dict_usage_without_cost():
    from litellm.responses.streaming_iterator import _stamp_responses_usage_cost

    response = _unvalidated_response_with_dict_usage({"input_tokens": 29, "output_tokens": 120, "total_tokens": 149})
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj.response_cost_calculator.return_value = 0.000704

    _stamp_responses_usage_cost(response, logging_obj)

    assert isinstance(response.usage, ResponseAPIUsage)
    assert response.usage.cost == pytest.approx(0.000704)
    logging_obj.response_cost_calculator.assert_called_once_with(result=response)


def test_stamp_responses_usage_cost_survives_calculator_failure():
    from litellm.responses.streaming_iterator import _stamp_responses_usage_cost

    response = _responses_api_response_with_usage()
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj.response_cost_calculator.side_effect = RuntimeError("cost map unavailable")

    _stamp_responses_usage_cost(response, logging_obj)

    assert getattr(response.usage, "cost", None) is None


def _capture_dispatch(logged: list):
    """Record the object handed to the success handlers.

    ``Mock(spec=LiteLLMLoggingObj).dispatch_success_handlers`` is an AsyncMock whose side effect
    only runs when the coroutine is awaited, so capture with a plain function instead.
    """

    async def _noop() -> None:
        return None

    def _dispatch(result, **kwargs):
        logged.append(result)
        return _noop()

    return _dispatch


def _headers_config(*, transform_hidden_params: Optional[dict] = None) -> Mock:
    """Config whose completed event carries a real ResponsesAPIResponse, so the logging copy
    performs a genuine model_dump/model_validate round trip."""
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    def _transform(model, parsed_chunk, logging_obj):
        evt_type = parsed_chunk.get("type")
        if evt_type != "response.completed":
            stub = Mock()
            stub.type = evt_type
            return stub
        response = ResponsesAPIResponse(
            id="resp_headers",
            created_at=1,
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        if transform_hidden_params is not None:
            response._hidden_params.update(transform_hidden_params)
        return ResponseCompletedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
            response=response,
        )

    mock_config.transform_streaming_response.side_effect = _transform
    return mock_config


def _make_header_iterator(
    *,
    headers: dict,
    config: Mock,
    logging_obj: LiteLLMLoggingObj,
) -> ResponsesAPIStreamingIterator:
    async def aiter_bytes():
        yield _sse_event({"type": "response.completed"})

    mock_response = Mock()
    mock_response.headers = headers
    mock_response.aiter_bytes = aiter_bytes

    return ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-4o-mini",
        responses_api_provider_config=config,
        logging_obj=logging_obj,
        litellm_metadata={},
        custom_llm_provider="azure",
    )


@pytest.mark.asyncio
async def test_streaming_logging_response_carries_provider_response_headers():
    """LIT-6055: the provider headers the iterator captured must reach the logged response, so
    custom loggers can read Azure's apim-request-id from the callback payload."""
    logging_obj = _logging_obj_stub()
    logged: list[object] = []
    logging_obj.dispatch_success_handlers = _capture_dispatch(logged)

    logging_obj._on_deferred_stream_complete = None

    iterator = _make_header_iterator(
        headers={"apim-request-id": "azure-correlation-1", "x-ms-region": "East US 2"},
        config=_headers_config(),
        logging_obj=logging_obj,
    )
    async for _ in iterator:
        pass

    assert len(logged) == 1
    hidden_params = logged[0].response._hidden_params
    assert hidden_params["additional_headers"]["llm_provider-apim-request-id"] == "azure-correlation-1"
    assert hidden_params["additional_headers"]["llm_provider-x-ms-region"] == "East US 2"
    assert hidden_params["headers"]["apim-request-id"] == "azure-correlation-1"
    # the proxy builds the client's response headers from the iterator's own dict, so the logged
    # response must hold copies rather than alias it
    assert hidden_params["additional_headers"] is not iterator._hidden_params["additional_headers"]
    assert hidden_params["headers"] is not iterator._raw_response_headers


@pytest.mark.asyncio
async def test_streaming_logging_copy_preserves_transform_hidden_params():
    """LIT-6055: model_validate(model_dump()) drops pydantic private attributes, so headers a
    provider transform already set on the response (fake_stream) must be re-applied."""
    logging_obj = _logging_obj_stub()
    logged: list[object] = []
    logging_obj.dispatch_success_handlers = _capture_dispatch(logged)

    logging_obj._on_deferred_stream_complete = None

    iterator = _make_header_iterator(
        headers={},
        config=_headers_config(
            transform_hidden_params={
                "additional_headers": {"llm_provider-apim-request-id": "from-transform"},
                "headers": {"apim-request-id": "from-transform"},
                "response_cost": 0.5,
            }
        ),
        logging_obj=logging_obj,
    )
    async for _ in iterator:
        pass

    assert len(logged) == 1
    hidden_params = logged[0].response._hidden_params
    assert hidden_params["additional_headers"]["llm_provider-apim-request-id"] == "from-transform"
    assert hidden_params["headers"]["apim-request-id"] == "from-transform"
    assert iterator.completed_response is not logged[0]
    # only the header keys travel: response_cost would short-circuit the cost calculator
    assert "response_cost" not in hidden_params


@pytest.mark.asyncio
async def test_streaming_logging_copy_fallback_leaves_caller_event_untouched():
    """LIT-6055: when the logging copy falls back to the original event, the header restore must
    not stamp logging-only state onto the object the caller is iterating."""
    logging_obj = _logging_obj_stub()
    logged: list[object] = []
    logging_obj.dispatch_success_handlers = _capture_dispatch(logged)
    logging_obj._on_deferred_stream_complete = None

    iterator = _make_header_iterator(
        headers={"apim-request-id": "azure-correlation-1"},
        config=_headers_config(),
        logging_obj=logging_obj,
    )
    async for _ in iterator:
        pass

    assert len(logged) == 1
    iterator._completed_response_logged = False
    logged.clear()
    with patch.object(type(iterator.completed_response), "model_dump", side_effect=ValueError("cannot serialize")):
        iterator._log_completed_response(is_async=True)

    assert len(logged) == 1
    assert logged[0] is not iterator.completed_response
    assert logged[0].response is not iterator.completed_response.response
    assert logged[0].response._hidden_params["headers"]["apim-request-id"] == "azure-correlation-1"
    assert iterator.completed_response.response._hidden_params == {}


def _unvalidated_completed_config() -> Mock:
    """Config whose completed event carries a Perplexity-style response that fails validation
    (``truncation: ""``) and already holds the stamped ``ResponseAPIUsage``."""
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    def _transform(model, parsed_chunk, logging_obj):
        response = _unvalidated_response_with_dict_usage(
            ResponseAPIUsage(input_tokens=29, output_tokens=373, total_tokens=402, cost={"total_cost": 0.0001})
        )
        return ResponseCompletedEvent(type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=response)

    mock_config.transform_streaming_response.side_effect = _transform
    return mock_config


@pytest.mark.asyncio
async def test_streaming_logging_copy_keeps_client_usage_when_response_fails_validation():
    """LIT-7391: the logging copy cannot round-trip a response that fails validation, and logging
    rewrites the assembled response's usage to chat shape in place, so the event handed to logging
    must never be the one the caller receives."""
    logging_obj = _logging_obj_stub()
    logging_obj.stream = True
    logged: list[object] = []
    logging_obj.dispatch_success_handlers = _capture_dispatch(logged)
    logging_obj._on_deferred_stream_complete = None

    iterator = _make_header_iterator(headers={}, config=_unvalidated_completed_config(), logging_obj=logging_obj)
    events = [event async for event in iterator]

    assert len(logged) == 1
    now = datetime.now()
    LiteLLMLoggingObj._get_assembled_streaming_response(
        logging_obj, logged[0], start_time=now, end_time=now, is_async=True, streaming_chunks=[]
    )
    assert logged[0].response.usage["prompt_tokens"] == 29

    client_usage = events[-1].response.usage
    assert isinstance(client_usage, ResponseAPIUsage)
    assert client_usage.input_tokens == 29
    assert client_usage.cost == pytest.approx(0.0001)


@pytest.mark.asyncio
async def test_completed_event_without_usage_counts_tool_call_arguments():
    """A function-call-only stream still bills output tokens: streamed
    function_call_arguments deltas feed the text estimate."""
    response = _responses_api_response_without_usage()
    iterator = _make_iterator(
        sse_events=[
            _sse_event(
                {
                    "type": "response.output_item.added",
                    "item": {"type": "function_call", "name": "get_weather", "call_id": "call_1"},
                }
            ),
            _sse_event(
                {
                    "type": "response.function_call_arguments.delta",
                    "delta": '{"location": "San Francisco", "unit": "celsius"}',
                }
            ),
            _sse_event({"type": "response.completed", "response": {}}),
        ],
        logging_obj=_logging_obj_stub(),
        config=_mock_config_with_completed_response(response),
        request_data={"input": "what is the weather in san francisco"},
    )

    async for _ in iterator:
        pass

    usage = iterator.completed_response.response.usage
    assert usage is not None
    assert usage.output_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens


@pytest.mark.asyncio
async def test_completed_event_without_usage_counts_multimodal_input_as_messages():
    """Multimodal request input is counted as chat messages, not as a JSON blob:
    a huge base64 image must not inflate the estimated input tokens."""
    image_input: Final = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is in this image"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + "A" * 4000,
                },
            ],
        }
    ]
    json_count: Final = litellm.token_counter(model="gpt-4o-mini", text=json.dumps(image_input))
    response = _responses_api_response_without_usage()
    iterator = _make_iterator(
        sse_events=[
            _sse_event({"type": "response.output_text.delta", "delta": "it is a cat"}),
            _sse_event({"type": "response.completed", "response": {}}),
        ],
        logging_obj=_logging_obj_stub(),
        config=_mock_config_with_completed_response(response),
        request_data={"input": image_input},
    )

    async for _ in iterator:
        pass

    usage = iterator.completed_response.response.usage
    assert usage is not None
    assert usage.input_tokens < json_count / 2


@pytest.mark.asyncio
async def test_completed_event_survives_a_failing_usage_estimate():
    """A malformed request input that makes the message transformer raise must not
    break a stream that previously completed: the estimate is best-effort and
    falls back to usage None."""
    malformed_input: Final = [{"type": "message", "role": "user", "content": 42}]
    with pytest.raises(ValueError, match="Invalid content type"):
        _estimate_usage_from_text("gpt-4o-mini", malformed_input, {"input": malformed_input}, "hello world")

    response = _responses_api_response_without_usage()
    iterator = _make_iterator(
        sse_events=[
            _sse_event({"type": "response.output_text.delta", "delta": "hello world"}),
            _sse_event({"type": "response.completed", "response": {}}),
        ],
        logging_obj=_logging_obj_stub(),
        config=_mock_config_with_completed_response(response),
        request_data={"input": malformed_input},
    )

    yielded: list = []
    async for chunk in iterator:
        yielded.append(chunk)

    assert yielded
    assert iterator.completed_response.response.usage is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_delta_event_type",
    ["response.custom_tool_call_input.delta", "response.mcp_call_arguments.delta"],
)
async def test_completed_event_without_usage_counts_tool_input_deltas(tool_delta_event_type):
    """Custom-tool and MCP argument deltas feed the streamed usage fallback the
    same way function_call_arguments deltas do."""
    response = _responses_api_response_without_usage()
    iterator = _make_iterator(
        sse_events=[
            _sse_event({"type": tool_delta_event_type, "delta": '{"query": "weather in sf"}'}),
            _sse_event({"type": "response.completed", "response": {}}),
        ],
        logging_obj=_logging_obj_stub(),
        config=_mock_config_with_completed_response(response),
        request_data={"input": "what is the weather in san francisco"},
    )

    async for _ in iterator:
        pass

    usage = iterator.completed_response.response.usage
    assert usage is not None
    assert usage.output_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens


@pytest.mark.asyncio
async def test_completed_event_with_a_dict_response_is_typed_and_billed():
    """transform_streaming_response can model_construct a terminal event whose
    response stays a plain dict; the iterator must type it so the estimated
    usage reaches the cost stamping path."""
    dict_response: Final = {
        "id": "resp_dict",
        "model": "gpt-4o-mini",
        "object": "response",
        "output": [],
        "usage": None,
    }

    def _transform(model, parsed_chunk, logging_obj):
        if parsed_chunk.get("type") == "response.completed":
            return ResponseCompletedEvent.model_construct(type="response.completed", response=dict_response)
        stub: Final = Mock()
        stub.type = parsed_chunk.get("type")
        if "delta" in parsed_chunk:
            stub.delta = parsed_chunk.get("delta")
        return stub

    config: Final = Mock(spec=BaseResponsesAPIConfig)
    config.transform_streaming_response.side_effect = _transform
    logging_obj: Final = _logging_obj_stub()
    logging_obj.response_cost_calculator.return_value = 0.000704
    iterator: Final = _make_iterator(
        sse_events=[
            _sse_event({"type": "response.output_text.delta", "delta": "hello world"}),
            _sse_event({"type": "response.completed", "response": {}}),
        ],
        logging_obj=logging_obj,
        config=config,
        request_data={"input": "count these input tokens please"},
    )

    yielded: Final = [chunk async for chunk in iterator]

    terminal_event: Final = iterator.completed_response
    assert yielded[-1] is terminal_event
    completed_response: Final = terminal_event.response
    assert isinstance(completed_response, ResponsesAPIResponse)
    usage: Final = completed_response.usage
    assert usage is not None
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.cost == pytest.approx(0.000704)
    logging_obj.response_cost_calculator.assert_any_call(result=completed_response)


def test_billed_terminal_response_keeps_a_response_that_already_has_usage():
    from litellm.responses.streaming_iterator import _billed_terminal_response

    response: Final = _responses_api_response_with_usage()

    assert _billed_terminal_response(response, None) is response


def test_billed_terminal_response_copies_when_estimating_and_leaves_the_original_untouched():
    from litellm.responses.streaming_iterator import _billed_terminal_response

    response: Final = _responses_api_response_without_usage()
    estimated: Final = ResponseAPIUsage(input_tokens=3, output_tokens=4, total_tokens=7)

    billed: Final = _billed_terminal_response(response, lambda: estimated)

    assert billed is not response
    assert billed.usage is estimated
    assert response.usage is None


def test_persist_completed_response_to_cache_skips_a_response_without_output(monkeypatch):
    logging_obj: Final = _logging_obj_stub()
    caching_handler: Final = Mock()
    caching_handler.request_kwargs = {"stream": True}
    logging_obj.llm_caching_handler = caching_handler
    iterator: Final = _make_iterator(sse_events=[], logging_obj=logging_obj)
    iterator.completed_response = ResponseCompletedEvent.model_construct(
        type="response.completed",
        response=ResponsesAPIResponse.model_construct(id="resp_empty", output=[], usage=None),
    )
    cache: Final = Mock()
    monkeypatch.setattr(litellm, "cache", cache)

    iterator._persist_completed_response_to_cache(is_async=False)

    cache.add_cache.assert_not_called()
    caching_handler.should_store_result_in_cache.assert_not_called()


def test_persist_completed_response_to_cache_survives_an_unserializable_response(monkeypatch):
    bad_response: Final = ResponsesAPIResponse.model_construct(id="r", output=[object()], usage=None)
    with pytest.raises(PydanticSerializationError):
        bad_response.model_dump_json()

    logging_obj: Final = _logging_obj_stub()
    caching_handler: Final = Mock()
    caching_handler.request_kwargs = {"stream": True}
    logging_obj.llm_caching_handler = caching_handler
    iterator: Final = _make_iterator(sse_events=[], logging_obj=logging_obj)
    iterator.completed_response = ResponseCompletedEvent.model_construct(
        type="response.completed", response=bad_response
    )
    cache: Final = Mock()
    monkeypatch.setattr(litellm, "cache", cache)

    iterator._persist_completed_response_to_cache(is_async=False)

    cache.add_cache.assert_not_called()


class _NativeWebSocketRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.successes = 0
        self.failures = 0
        self.observed: Final = asyncio.Event()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.successes += 1
        self.observed.set()

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.failures += 1
        self.observed.set()


def _native_ws_logging(recorder: _NativeWebSocketRecorder) -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="openai/ws-interruption-test",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type=CallTypes.aresponses_websocket.value,
        start_time=datetime(2026, 1, 1),
        litellm_call_id="ws-interruption-test",
        function_id="ws-interruption-test",
        dynamic_async_success_callbacks=[recorder],
        dynamic_async_failure_callbacks=[recorder],
    )
    logging_obj.update_environment_variables(
        litellm_params={"aresponses": True, "custom_llm_provider": "openai"}, optional_params={}
    )
    return logging_obj


def _native_ws_created(response_id: str = "resp_partial") -> str:
    return json.dumps(
        {"type": "response.created", "response": {"id": response_id, "status": "in_progress", "output": []}}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("close_code", [1000, 1011])
async def test_native_ws_upstream_close_during_a_response_logs_failure(close_code: int) -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.frames import Close

    recorder: Final = _NativeWebSocketRecorder()
    client: Final = Mock(send_text=AsyncMock())
    backend: Final = Mock(
        recv=AsyncMock(side_effect=[_native_ws_created(), ConnectionClosed(Close(close_code, "peer end"), None)])
    )
    handler: Final = ResponsesWebSocketStreaming(
        websocket=client,
        backend_ws=backend,
        logging_obj=_native_ws_logging(recorder),
        custom_llm_provider="openai",
    )

    await handler.backend_to_client()
    await asyncio.wait_for(recorder.observed.wait(), 1)

    assert (recorder.successes, recorder.failures) == (0, 1)
    assert json.loads(client.send_text.await_args_list[-1].args[0])["type"] == "error"


@pytest.mark.asyncio
async def test_native_ws_upstream_close_finishes_the_relay_without_a_client_message() -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.frames import Close

    recorder: Final = _NativeWebSocketRecorder()
    waiting_client: Final = asyncio.Event()
    client: Final = Mock(
        send_text=AsyncMock(), receive_text=AsyncMock(side_effect=waiting_client.wait), close=AsyncMock()
    )
    backend: Final = Mock(
        recv=AsyncMock(side_effect=[_native_ws_created(), ConnectionClosed(Close(1000, "peer end"), None)]),
        close=AsyncMock(),
    )
    handler: Final = ResponsesWebSocketStreaming(
        websocket=client, backend_ws=backend, logging_obj=_native_ws_logging(recorder), custom_llm_provider="openai"
    )

    failure: Final = await asyncio.wait_for(handler.bidirectional_forward(), 1)
    await asyncio.wait_for(recorder.observed.wait(), 1)

    assert failure is not None
    assert (recorder.successes, recorder.failures) == (0, 1)
    client.close.assert_awaited_once()
    backend.close.assert_awaited_once()


def _native_ws_terminal(event_type: str, response_id: str = "resp_partial") -> str:
    if event_type == "error":
        return json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": "provider error"}})
    return json.dumps(
        {
            "type": event_type,
            "response": {
                "id": response_id,
                "status": event_type.removeprefix("response."),
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                **(
                    {"error": {"type": "server_error", "message": "provider failure"}}
                    if event_type == "response.failed"
                    else {}
                ),
            },
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("close_code", [1000, 1011])
@pytest.mark.parametrize(
    "terminal,expected_counts",
    [("response.completed", (1, 0)), ("response.incomplete", (1, 0)), ("response.failed", (0, 1)), ("error", (0, 1))],
)
async def test_native_ws_close_after_a_terminal_event_keeps_its_logging_outcome(
    close_code: int, terminal: str, expected_counts: tuple[int, int]
) -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.frames import Close

    recorder: Final = _NativeWebSocketRecorder()
    waiting_client: Final = asyncio.Event()
    client: Final = Mock(
        send_text=AsyncMock(), receive_text=AsyncMock(side_effect=waiting_client.wait), close=AsyncMock()
    )
    backend: Final = Mock(
        recv=AsyncMock(
            side_effect=[
                _native_ws_created(),
                _native_ws_terminal(terminal),
                ConnectionClosed(Close(close_code, "peer end"), None),
            ]
        ),
        close=AsyncMock(),
    )
    handler: Final = ResponsesWebSocketStreaming(
        websocket=client, backend_ws=backend, logging_obj=_native_ws_logging(recorder), custom_llm_provider="openai"
    )

    failure: Final = await asyncio.wait_for(handler.bidirectional_forward(), 1)
    await asyncio.wait_for(recorder.observed.wait(), 1)

    assert (recorder.successes, recorder.failures) == expected_counts
    assert (failure is not None) == (expected_counts[1] == 1)
    assert [json.loads(call.args[0])["type"] for call in client.send_text.await_args_list] == [
        "response.created",
        terminal,
    ]
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_ws_interruption_preserves_completed_turn_usage_without_inventing_partial_usage() -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.frames import Close

    litellm.register_model(
        {
            "openai/ws-interruption-test": {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )
    recorder: Final = _NativeWebSocketRecorder()
    logging_obj: Final = _native_ws_logging(recorder)
    client: Final = Mock(send_text=AsyncMock())
    backend: Final = Mock(
        recv=AsyncMock(
            side_effect=[
                _native_ws_created("resp_complete"),
                _native_ws_terminal("response.completed", "resp_complete"),
                _native_ws_created(),
                json.dumps({"type": "response.output_text.delta", "delta": "unbilled partial text"}),
                ConnectionClosed(Close(1000, "peer end"), None),
            ]
        )
    )
    handler: Final = ResponsesWebSocketStreaming(
        websocket=client, backend_ws=backend, logging_obj=logging_obj, custom_llm_provider="openai"
    )

    await handler.backend_to_client()
    await asyncio.wait_for(recorder.observed.wait(), 1)

    usage: Final = logging_obj.model_call_details["combined_usage_object"]
    assert isinstance(usage, Usage)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (10, 5, 15)
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(10 * 0.001 + 5 * 0.002)
    assert (recorder.successes, recorder.failures) == (0, 1)


@pytest.mark.asyncio
async def test_native_ws_close_before_the_first_provider_event_fails_an_accepted_request() -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.frames import Close

    request_sent: Final = asyncio.Event()

    async def recv(decode: bool = False) -> str:
        await request_sent.wait()
        raise ConnectionClosed(Close(1000, "peer end"), None)

    async def send(message: str) -> None:
        request_sent.set()

    recorder: Final = _NativeWebSocketRecorder()
    waiting_client: Final = asyncio.Event()
    client: Final = Mock(
        send_text=AsyncMock(), receive_text=AsyncMock(side_effect=waiting_client.wait), close=AsyncMock()
    )
    backend: Final = Mock(recv=recv, send=send, close=AsyncMock())
    handler: Final = ResponsesWebSocketStreaming(
        websocket=client,
        backend_ws=backend,
        logging_obj=_native_ws_logging(recorder),
        custom_llm_provider="openai",
        first_message=json.dumps({"type": "response.create", "input": "hi"}),
    )

    failure: Final = await asyncio.wait_for(handler.bidirectional_forward(), 1)
    await asyncio.wait_for(recorder.observed.wait(), 1)

    assert failure is not None
    assert [json.loads(call.args[0])["type"] for call in client.send_text.await_args_list] == ["error"]
    assert (recorder.successes, recorder.failures) == (0, 1)
    assert client.receive_text.await_count == 1
    client.close.assert_awaited_once()
    backend.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_ws_client_disconnect_cancels_the_backend_and_records_interruption() -> None:
    from starlette.websockets import WebSocketDisconnect

    created_sent: Final = asyncio.Event()
    backend_cancelled: Final = asyncio.Event()
    never_finishes: Final = asyncio.Event()
    frames: Final = iter((_native_ws_created(),))

    async def recv(decode: bool = False) -> str:
        frame: Final = next(frames, None)
        if frame is not None:
            return frame
        try:
            await never_finishes.wait()
        finally:
            backend_cancelled.set()
        raise AssertionError("backend was not cancelled")

    async def send_text(data: str) -> None:
        if json.loads(data)["type"] == "response.created":
            created_sent.set()
            return
        raise WebSocketDisconnect()

    async def receive_text() -> str:
        await created_sent.wait()
        raise WebSocketDisconnect()

    recorder: Final = _NativeWebSocketRecorder()
    client: Final = Mock(
        send_text=send_text, receive_text=receive_text, close=AsyncMock(side_effect=WebSocketDisconnect())
    )
    backend: Final = Mock(recv=recv, close=AsyncMock())
    handler: Final = ResponsesWebSocketStreaming(
        websocket=client, backend_ws=backend, logging_obj=_native_ws_logging(recorder), custom_llm_provider="openai"
    )

    failure: Final = await asyncio.wait_for(handler.bidirectional_forward(), 1)
    await asyncio.wait_for(recorder.observed.wait(), 1)

    assert failure is not None
    assert backend_cancelled.is_set()
    assert (recorder.successes, recorder.failures) == (0, 1)
    backend.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_ws_cancelled_notification_still_logs_failure_and_closes_both_peers() -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.frames import Close

    notifying_client: Final = asyncio.Event()
    never_finishes: Final = asyncio.Event()

    async def send_text(data: str) -> None:
        if json.loads(data)["type"] == "error":
            notifying_client.set()
            await never_finishes.wait()

    recorder: Final = _NativeWebSocketRecorder()
    logging_obj: Final = _native_ws_logging(recorder)
    client: Final = Mock(
        send_text=send_text, receive_text=AsyncMock(side_effect=never_finishes.wait), close=AsyncMock()
    )
    backend: Final = Mock(
        recv=AsyncMock(side_effect=[_native_ws_created(), ConnectionClosed(Close(1000, "peer end"), None)]),
        close=AsyncMock(),
    )
    handler: Final = ResponsesWebSocketStreaming(
        websocket=client, backend_ws=backend, logging_obj=logging_obj, custom_llm_provider="openai"
    )

    relay_task: Final = asyncio.create_task(handler.bidirectional_forward())
    await asyncio.wait_for(notifying_client.wait(), 1)
    relay_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await relay_task
    assert logging_obj.model_call_details.get("combined_usage_object") is not None
    await asyncio.wait_for(recorder.observed.wait(), 1)
    assert (recorder.successes, recorder.failures) == (0, 1)
    client.close.assert_awaited_once()
    backend.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_ws_already_closed_upstream_fails_the_pre_read_request() -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.frames import Close

    recorder: Final = _NativeWebSocketRecorder()
    waiting_client: Final = asyncio.Event()
    client: Final = Mock(
        send_text=AsyncMock(), receive_text=AsyncMock(side_effect=waiting_client.wait), close=AsyncMock()
    )
    backend: Final = Mock(
        recv=AsyncMock(side_effect=ConnectionClosed(Close(1000, "peer end"), None)),
        send=AsyncMock(side_effect=ConnectionClosed(Close(1000, "peer end"), None)),
        close=AsyncMock(),
    )
    handler: Final = ResponsesWebSocketStreaming(
        websocket=client,
        backend_ws=backend,
        logging_obj=_native_ws_logging(recorder),
        custom_llm_provider="openai",
        first_message=json.dumps({"type": "response.create", "input": "hi"}),
    )

    failure: Final = await asyncio.wait_for(handler.bidirectional_forward(), 1)

    assert failure is not None
    await asyncio.wait_for(recorder.observed.wait(), 1)
    assert (recorder.successes, recorder.failures) == (0, 1)
    assert [json.loads(call.args[0])["type"] for call in client.send_text.await_args_list] == ["error"]


@pytest.mark.asyncio
async def test_native_ws_ready_next_request_after_upstream_close_has_one_final_logging_outcome() -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.frames import Close

    recorder: Final = _NativeWebSocketRecorder()
    client: Final = Mock(
        send_text=AsyncMock(),
        receive_text=AsyncMock(return_value=json.dumps({"type": "response.create", "input": "next turn"})),
        close=AsyncMock(),
    )
    backend: Final = Mock(
        recv=AsyncMock(
            side_effect=[
                _native_ws_created(),
                _native_ws_terminal("response.completed"),
                ConnectionClosed(Close(1000, "peer end"), None),
            ]
        ),
        send=AsyncMock(side_effect=ConnectionClosed(Close(1000, "peer end"), None)),
        close=AsyncMock(),
    )
    handler: Final = ResponsesWebSocketStreaming(
        websocket=client, backend_ws=backend, logging_obj=_native_ws_logging(recorder), custom_llm_provider="openai"
    )

    failure: Final = await asyncio.wait_for(handler.bidirectional_forward(), 1)
    await asyncio.wait_for(recorder.observed.wait(), 1)

    assert failure is not None
    assert (recorder.successes, recorder.failures) == (0, 1)


@pytest.mark.asyncio
async def test_native_ws_stalled_error_notification_does_not_keep_the_relay_open() -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.frames import Close

    never_finishes: Final = asyncio.Event()

    async def send_text(data: str) -> None:
        if json.loads(data)["type"] == "error":
            await never_finishes.wait()

    recorder: Final = _NativeWebSocketRecorder()
    client: Final = Mock(
        send_text=send_text, receive_text=AsyncMock(side_effect=never_finishes.wait), close=AsyncMock()
    )
    backend: Final = Mock(
        recv=AsyncMock(side_effect=[_native_ws_created(), ConnectionClosed(Close(1000, "peer end"), None)]),
        close=AsyncMock(),
    )
    handler: Final = ResponsesWebSocketStreaming(
        websocket=client, backend_ws=backend, logging_obj=_native_ws_logging(recorder), custom_llm_provider="openai"
    )

    failure: Final = await asyncio.wait_for(handler.bidirectional_forward(), 2)
    await asyncio.wait_for(recorder.observed.wait(), 1)

    assert failure is not None
    assert (recorder.successes, recorder.failures) == (0, 1)
    client.close.assert_awaited_once()
    backend.close.assert_awaited_once()

@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="function")
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    importlib.reload(litellm)
    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            importlib.reload(litellm.proxy.proxy_server)
    except Exception:
        pass
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)

class _FakeLoggingObj:
    def __init__(self):
        self.success_calls = 0
        self.async_success_calls = 0
        self.failure_calls = 0
        self.async_failure_calls = 0
        self.last_success_kwargs = None
        self.last_async_success_kwargs = None
        self.start_time = datetime.now()
        self.completion_start_time = None
        self.model_call_details = {"litellm_params": {}}

    # Signature alignment with Logging handlers
    async def dispatch_success_handlers(self, *args, **kwargs):
        kwargs.pop("prefer_async_handlers", None)
        await self.async_success_handler(*args, **kwargs)
        self.success_handler(*args, **kwargs)

    def success_handler(self, *args, **kwargs):
        self.success_calls += 1
        self.last_success_kwargs = kwargs

    async def async_success_handler(self, *args, **kwargs):
        self.async_success_calls += 1
        self.last_async_success_kwargs = kwargs

    def failure_handler(self, *args, **kwargs):
        self.failure_calls += 1

    async def async_failure_handler(self, *args, **kwargs):
        self.async_failure_calls += 1

    def update_completion_start_time(self, completion_start_time):
        self.completion_start_time = completion_start_time
        self.model_call_details["completion_start_time"] = completion_start_time

def _make_completed_response(response_id: str = "resp_test") -> ResponseCompletedEvent:
    return ResponseCompletedEvent(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=ResponsesAPIResponse(
            id=response_id,
            created_at=int(datetime.now().timestamp()),
            status="completed",
            model="test-model",
            object="response",
            output=[
                {
                    "type": "message",
                    "id": f"msg_{response_id}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "cached streamed response",
                            "annotations": [],
                        }
                    ],
                }
            ],
        ),
    )

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_log_background_task_failure_logs_task_exceptions(monkeypatch):
    error_logger = MagicMock()
    monkeypatch.setattr(streaming_module.verbose_logger, "error", error_logger)

    async def _boom():
        raise RuntimeError("boom")

    task = asyncio.create_task(_boom())
    with suppress(RuntimeError):
        await task

    streaming_module._log_background_task_failure(task, task_name="cache write")

    error_logger.assert_called_once()
    assert error_logger.call_args.args == (
        "%s failed: %s",
        "cache write",
        task.exception(),
    )

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_log_background_task_failure_ignores_cancelled_tasks(monkeypatch):
    error_logger = MagicMock()
    monkeypatch.setattr(streaming_module.verbose_logger, "error", error_logger)

    task = asyncio.create_task(asyncio.sleep(1))
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    streaming_module._log_background_task_failure(task, task_name="cache write")

    error_logger.assert_not_called()

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_content_part_done_event_supports_refusal_and_reasoning_text():
    refusal_event = streaming_module._build_content_part_done_event(
        item_id="msg_1",
        output_index=0,
        content_index=0,
        part_payload={"type": "refusal", "refusal": "no"},
    )
    reasoning_event = streaming_module._build_content_part_done_event(
        item_id="msg_1",
        output_index=0,
        content_index=1,
        part_payload={"type": "reasoning_text", "reasoning": "because"},
    )
    unsupported_event = streaming_module._build_content_part_done_event(
        item_id="msg_1",
        output_index=0,
        content_index=2,
        part_payload={"type": "image"},
    )

    assert refusal_event.part.type == "refusal"
    assert refusal_event.part.refusal == "no"
    assert reasoning_event.part.type == "reasoning_text"
    assert reasoning_event.part.reasoning == "because"
    assert unsupported_event is None

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_dump_response_object_handles_model_and_unknown_values():
    response = ResponsesAPIResponse(
        id="resp_dump",
        created_at=int(datetime.now().timestamp()),
        status="completed",
        model="gpt-4.1-mini",
        object="response",
        output=[],
    )

    assert streaming_module._dump_response_object(response)["id"] == "resp_dump"
    assert streaming_module._dump_response_object({"type": "message"}) == {"type": "message"}
    assert streaming_module._dump_response_object(object()) == {}

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_responses_streaming_triggers_hooks(monkeypatch):
    """
    Ensure streaming iterator fires success + post-call hooks for responses API.
    """
    hook_calls = {"post_call": 0, "metadata": 0}
    seen = {}

    async def fake_post_call(request_data, response, call_type):
        hook_calls["post_call"] += 1
        seen["request_data"] = request_data
        seen["call_type"] = call_type

    def fake_update_metadata(**kwargs):
        hook_calls["metadata"] += 1

    monkeypatch.setattr(
        streaming_module,
        "async_post_call_success_deployment_hook",
        fake_post_call,
    )
    monkeypatch.setattr(
        streaming_module,
        "update_response_metadata",
        fake_update_metadata,
    )

    logging_obj = _FakeLoggingObj()

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=SimpleNamespace(),  # not used in this test
        logging_obj=logging_obj,
        request_data={"foo": "bar", "litellm_params": {}},
        call_type=CallTypes.responses.value,
    )

    # Simulate completed streaming event
    iterator.completed_response = SimpleNamespace(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=SimpleNamespace()
    )

    iterator._handle_logging_completed_response()
    await asyncio.sleep(0.2)  # allow async tasks to run

    assert logging_obj.success_calls == 1
    assert logging_obj.async_success_calls == 1
    assert hook_calls["post_call"] == 1
    assert hook_calls["metadata"] == 1
    assert seen["request_data"]["foo"] == "bar"
    assert seen["request_data"].get("litellm_params") is not None
    assert seen["call_type"] == CallTypes.responses

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_responses_streaming_calls_post_streaming_deployment_hook(monkeypatch):
    """
    Ensure per-chunk streaming deployment hook can modify chunks.
    """

    class _HookLogger(CustomLogger):
        async def async_post_call_streaming_deployment_hook(self, request_data, response_chunk, call_type):
            response_chunk.tagged = True
            return response_chunk

    # Set callbacks to our fake hook
    original_callbacks = litellm.callbacks
    litellm.callbacks = [_HookLogger()]

    logging_obj = _FakeLoggingObj()

    class _StubConfig:
        def transform_streaming_response(self, **kwargs):
            return SimpleNamespace(type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, response=None)

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_StubConfig(),
        logging_obj=logging_obj,
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )

    # Call hook helper directly to verify chunk is modified/flagged
    chunk = SimpleNamespace(type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, response=None)
    chunk = await streaming_module.call_post_streaming_hooks_for_testing(iterator, chunk)
    assert getattr(chunk, "_post_streaming_hooks_ran", False) is True
    assert getattr(chunk, "tagged", False) is True

    # reset callbacks
    litellm.callbacks = original_callbacks

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_responses_streaming_failure_triggers_failure_handlers():
    """
    If transform raises, failure handlers should be called.
    """

    class _FailConfig:
        def transform_streaming_response(self, **kwargs):
            raise ValueError("boom")

    logging_obj = _FakeLoggingObj()

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_FailConfig(),
        logging_obj=logging_obj,
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )

    with pytest.raises(ValueError, match="boom"):
        iterator._process_chunk('{"delta": "chunk"}')

    # allow failure callbacks to run
    await asyncio.sleep(0.2)
    assert logging_obj.failure_calls >= 1
    assert logging_obj.async_failure_calls >= 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_process_chunk_requires_provider_config():
    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=None,
        logging_obj=_FakeLoggingObj(),
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )

    with pytest.raises(ValueError, match="responses_api_provider_config is required"):
        iterator._process_chunk(json.dumps({"type": "response.completed"}))

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_process_chunk_wraps_encrypted_content_with_model_id():
    openai_types = streaming_module._get_openai_response_types()

    class _EncryptedConfig:
        def transform_streaming_response(self, **kwargs):
            return openai_types.OutputItemAddedEvent(
                type=openai_types.ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=openai_types.BaseLiteLLMOpenAIResponseObject(
                    id="rs_123",
                    type="reasoning",
                    encrypted_content="ciphertext",
                ),
            )

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_EncryptedConfig(),
        logging_obj=_FakeLoggingObj(),
        litellm_metadata={
            "encrypted_content_affinity_enabled": True,
            "model_info": {"id": "model-123"},
        },
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )

    event = iterator._process_chunk(json.dumps({"type": "response.output_item.added"}))

    assert event.item.encrypted_content.startswith("litellm_enc:")
    assert event.item.encrypted_content.endswith(";ciphertext")

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_process_chunk_completed_response_updates_id_and_usage_cost(monkeypatch):
    original_include_cost = litellm.include_cost_in_streaming_usage
    litellm.include_cost_in_streaming_usage = True
    openai_types = streaming_module._get_openai_response_types()

    class _CompletedConfig:
        def transform_streaming_response(self, **kwargs):
            return openai_types.ResponseCompletedEvent(
                type=openai_types.ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=ResponsesAPIResponse(
                    id="resp_live",
                    created_at=int(datetime.now().timestamp()),
                    status="completed",
                    model="test-model",
                    object="response",
                    output=[],
                    usage=openai_types.ResponseAPIUsage(
                        input_tokens=1,
                        output_tokens=2,
                        total_tokens=3,
                    ),
                ),
            )

    logging_obj = _FakeLoggingObj()
    logging_obj.response_cost_calculator = MagicMock(return_value=1.23)
    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_CompletedConfig(),
        logging_obj=logging_obj,
        litellm_metadata={"model_info": {"id": "model-123"}},
        custom_llm_provider="openai",
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )
    completion_handler = MagicMock()
    monkeypatch.setattr(iterator, "_handle_logging_completed_response", completion_handler)

    try:
        # Chunk must include a top-level "response" key so BaseResponsesAPIStreamingIterator
        # runs update_responses_api_response_id_with_model_id (see streaming_iterator.py).
        event = iterator._process_chunk(json.dumps({"type": "response.completed", "response": {"id": "resp_live"}}))
    finally:
        litellm.include_cost_in_streaming_usage = original_include_cost

    assert iterator.completed_response is event
    assert event.response.id != "resp_live"
    assert event.response.id.startswith("resp_")
    assert event.response.usage.cost == 1.23
    completion_handler.assert_called_once()

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_process_chunk_failed_response_triggers_failure_logging(monkeypatch):
    openai_types = streaming_module._get_openai_response_types()

    class _FailedConfig:
        def transform_streaming_response(self, **kwargs):
            return openai_types.ResponseFailedEvent(
                type=openai_types.ResponsesAPIStreamEvents.RESPONSE_FAILED,
                response=ResponsesAPIResponse(
                    id="resp_failed",
                    created_at=int(datetime.now().timestamp()),
                    status="failed",
                    model="test-model",
                    object="response",
                    output=[],
                    error={"message": "provider failed"},
                ),
            )

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_FailedConfig(),
        logging_obj=_FakeLoggingObj(),
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )
    failure_handler = MagicMock()
    monkeypatch.setattr(iterator, "_handle_logging_failed_response", failure_handler)

    event = iterator._process_chunk(json.dumps({"type": "response.failed"}))

    assert iterator.completed_response is event
    failure_handler.assert_called_once()

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_handle_logging_failed_response_uses_response_error_message():
    openai_types = streaming_module._get_openai_response_types()
    logging_obj = _FakeLoggingObj()
    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=SimpleNamespace(),
        logging_obj=logging_obj,
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )
    iterator.completed_response = openai_types.ResponseFailedEvent(
        type=openai_types.ResponsesAPIStreamEvents.RESPONSE_FAILED,
        response=ResponsesAPIResponse(
            id="resp_failed_real",
            created_at=int(datetime.now().timestamp()),
            status="failed",
            model="test-model",
            object="response",
            output=[],
            error={"message": "provider failed"},
        ),
    )

    iterator._handle_logging_failed_response()
    await asyncio.sleep(0.2)

    assert logging_obj.failure_calls == 1
    assert logging_obj.async_failure_calls == 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_process_chunk_returns_none_for_invalid_json_and_non_dict_payload():
    class _NoopConfig:
        def transform_streaming_response(self, **kwargs):
            raise AssertionError("should not be called")

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_NoopConfig(),
        logging_obj=_FakeLoggingObj(),
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )

    assert iterator._process_chunk("not-json") is None
    assert iterator._process_chunk(json.dumps(["not", "a", "dict"])) is None

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_process_chunk_cost_annotation_failure_is_nonfatal(monkeypatch):
    original_include_cost = litellm.include_cost_in_streaming_usage
    litellm.include_cost_in_streaming_usage = True
    openai_types = streaming_module._get_openai_response_types()

    class _CompletedConfig:
        def transform_streaming_response(self, **kwargs):
            return openai_types.ResponseCompletedEvent(
                type=openai_types.ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=ResponsesAPIResponse(
                    id="resp_cost_failure",
                    created_at=int(datetime.now().timestamp()),
                    status="completed",
                    model="test-model",
                    object="response",
                    output=[],
                    usage=openai_types.ResponseAPIUsage(
                        input_tokens=1,
                        output_tokens=2,
                        total_tokens=3,
                    ),
                ),
            )

    logging_obj = _FakeLoggingObj()
    logging_obj.response_cost_calculator = MagicMock(side_effect=RuntimeError("boom"))
    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_CompletedConfig(),
        logging_obj=logging_obj,
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )
    completion_handler = MagicMock()
    monkeypatch.setattr(iterator, "_handle_logging_completed_response", completion_handler)

    try:
        event = iterator._process_chunk(json.dumps({"type": "response.completed"}))
    finally:
        litellm.include_cost_in_streaming_usage = original_include_cost

    assert iterator.completed_response is event
    assert event.response.usage.cost is None
    completion_handler.assert_called_once()

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_get_completed_response_object_accepts_direct_response():
    logging_obj = _FakeLoggingObj()
    iterator = SyncResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=SimpleNamespace(),
        logging_obj=logging_obj,
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )
    direct_response = _make_completed_response("resp_direct").response
    iterator.completed_response = direct_response

    assert iterator._get_completed_response_object() is direct_response

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_responses_streaming_completed_event_persists_async_cache():
    logging_obj = _FakeLoggingObj()
    original_cache = litellm.cache
    litellm.cache = SimpleNamespace(
        async_add_cache=AsyncMock(),
        add_cache=MagicMock(),
    )
    caching_handler = SimpleNamespace(
        request_kwargs={
            "model": "test-model",
            "input": "hello",
            "stream": True,
            "caching": True,
            "cache_key": "stale-request-cache-key",
            "metadata": None,
            "custom_llm_provider": "openai",
        },
        preset_cache_key="responses-stream-cache-key",
        original_function=litellm.aresponses,
        async_set_cache=AsyncMock(),
        _should_store_result_in_cache=lambda original_function, kwargs: True,
    )
    logging_obj.llm_caching_handler = caching_handler

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=SimpleNamespace(),
        logging_obj=logging_obj,
        request_data=caching_handler.request_kwargs,
        call_type=CallTypes.aresponses.value,
    )
    iterator.completed_response = _make_completed_response()

    iterator._handle_logging_completed_response()
    await asyncio.sleep(0.2)

    litellm.cache.async_add_cache.assert_called_once()
    assert litellm.cache.async_add_cache.call_args.kwargs["stream"] is True
    assert litellm.cache.async_add_cache.call_args.kwargs["cache_key"] == "responses-stream-cache-key"
    assert "metadata" not in litellm.cache.async_add_cache.call_args.kwargs
    assert "custom_llm_provider" not in litellm.cache.async_add_cache.call_args.kwargs
    assert json.loads(litellm.cache.async_add_cache.call_args.args[0])["id"] == iterator.completed_response.response.id
    litellm.cache = original_cache

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_responses_streaming_completed_event_persists_sync_cache():
    logging_obj = _FakeLoggingObj()
    original_cache = litellm.cache
    litellm.cache = SimpleNamespace(
        async_add_cache=AsyncMock(),
        add_cache=MagicMock(),
    )
    caching_handler = SimpleNamespace(
        request_kwargs={
            "model": "test-model",
            "input": "hello",
            "stream": True,
            "caching": True,
            "cache_key": "stale-request-cache-key",
            "metadata": None,
            "custom_llm_provider": "openai",
        },
        preset_cache_key="responses-stream-cache-key",
        original_function=litellm.responses,
        sync_set_cache=MagicMock(),
        _should_store_result_in_cache=lambda original_function, kwargs: True,
    )
    logging_obj.llm_caching_handler = caching_handler

    iterator = SyncResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=SimpleNamespace(),
        logging_obj=logging_obj,
        request_data=caching_handler.request_kwargs,
        call_type=CallTypes.responses.value,
    )
    iterator.completed_response = _make_completed_response("resp_sync")

    iterator._handle_logging_completed_response()

    litellm.cache.add_cache.assert_called_once()
    assert litellm.cache.add_cache.call_args.kwargs["stream"] is True
    assert litellm.cache.add_cache.call_args.kwargs["cache_key"] == "responses-stream-cache-key"
    assert "metadata" not in litellm.cache.add_cache.call_args.kwargs
    assert "custom_llm_provider" not in litellm.cache.add_cache.call_args.kwargs
    assert json.loads(litellm.cache.add_cache.call_args.args[0])["id"] == iterator.completed_response.response.id
    litellm.cache = original_cache

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_log_completed_response_sync_direct_path(monkeypatch):
    hook_calls = {"post_call": 0, "metadata": 0}

    async def fake_post_call(request_data, response, call_type):
        hook_calls["post_call"] += 1

    def fake_update_metadata(**kwargs):
        hook_calls["metadata"] += 1

    monkeypatch.setattr(
        streaming_module,
        "async_post_call_success_deployment_hook",
        fake_post_call,
    )
    monkeypatch.setattr(
        streaming_module,
        "update_response_metadata",
        fake_update_metadata,
    )

    logging_obj = _FakeLoggingObj()
    iterator = SyncResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=SimpleNamespace(),
        logging_obj=logging_obj,
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )
    iterator._persist_completed_response_before_logging = False
    iterator.completed_response = _make_completed_response("resp_log_sync")

    iterator._log_completed_response(is_async=False)
    asyncio.run(asyncio.sleep(0.2))

    assert logging_obj.success_calls == 1
    assert logging_obj.async_success_calls == 1
    assert hook_calls["post_call"] == 1
    assert hook_calls["metadata"] == 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_log_completed_response_falls_back_when_model_validate_fails(monkeypatch):
    class _BadSerializableResponse:
        @classmethod
        def model_validate(cls, value):
            raise RuntimeError("nope")

        def model_dump(self):
            return {"id": "bad"}

    logging_obj = _FakeLoggingObj()
    iterator = SyncResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=SimpleNamespace(),
        logging_obj=logging_obj,
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )
    iterator._persist_completed_response_before_logging = False
    iterator.completed_response = _BadSerializableResponse()
    monkeypatch.setattr(iterator, "_run_post_success_hooks", MagicMock())

    iterator._log_completed_response(is_async=False)
    asyncio.run(asyncio.sleep(0.2))

    assert logging_obj.success_calls == 1
    assert logging_obj.async_success_calls == 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.parametrize(
    "scenario",
    [
        "already_cached",
        "not_completed",
        "missing_caching_handler",
        "not_streaming",
        "store_disabled",
        "missing_cache_backend",
    ],
)
def test_persist_completed_response_to_cache_guard_branches(monkeypatch, scenario):
    logging_obj = _FakeLoggingObj()
    iterator = SyncResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=SimpleNamespace(),
        logging_obj=logging_obj,
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )
    openai_types = streaming_module._get_openai_response_types()
    completed_event = _make_completed_response("resp_guard")
    iterator.completed_response = completed_event

    if scenario == "already_cached":
        iterator._completed_response_cached = True
    elif scenario == "not_completed":
        iterator.completed_response = openai_types.ResponseIncompleteEvent(
            type=openai_types.ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE,
            response=completed_event.response,
        )
    elif scenario == "missing_caching_handler":
        logging_obj.llm_caching_handler = None
    else:
        logging_obj.llm_caching_handler = SimpleNamespace(
            request_kwargs={
                "model": "test-model",
                "input": "hello",
                "stream": scenario != "not_streaming",
                "cache_key": "request-cache-key",
                "metadata": None,
                "custom_llm_provider": "openai",
            },
            preset_cache_key=None,
            original_function=litellm.responses,
            dual_cache=None,
            _should_store_result_in_cache=lambda original_function, kwargs: scenario != "store_disabled",
        )
        if scenario == "missing_cache_backend":
            monkeypatch.setattr(streaming_module.litellm, "cache", None)
        else:
            monkeypatch.setattr(
                streaming_module.litellm,
                "cache",
                SimpleNamespace(add_cache=MagicMock(), async_add_cache=AsyncMock()),
            )

    iterator._persist_completed_response_to_cache(is_async=False)

    expected_cached_flag = scenario == "already_cached"
    assert iterator._completed_response_cached is expected_cached_flag

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_build_synthetic_response_events_covers_annotations_function_calls_and_refusals():
    original_include_cost = litellm.include_cost_in_streaming_usage
    litellm.include_cost_in_streaming_usage = True
    logging_obj = _FakeLoggingObj()
    logging_obj.response_cost_calculator = MagicMock(side_effect=RuntimeError("boom"))
    transformed = ResponsesAPIResponse(
        id="resp_events",
        created_at=int(datetime.now().timestamp()),
        status="completed",
        model="gpt-4.1-mini",
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg_events",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "hello world",
                        "annotations": [{"type": "file_citation", "file_id": "file_1"}],
                    },
                    {
                        "type": "refusal",
                        "refusal": "no thanks",
                    },
                ],
            },
            {
                "type": "function_call",
                "id": "fc_events",
                "call_id": "call_123",
                "name": "lookup",
                "arguments": '{"id":1}',
            },
        ],
    )

    try:
        events = streaming_module.build_synthetic_response_events(
            transformed=transformed,
            logging_obj=logging_obj,
            chunk_size=5,
        )
    finally:
        litellm.include_cost_in_streaming_usage = original_include_cost

    event_types = [event.type.value if hasattr(event.type, "value") else str(event.type) for event in events]

    assert "response.output_text.annotation.added" in event_types
    assert "response.refusal.delta" in event_types
    assert "response.refusal.done" in event_types
    assert "response.function_call_arguments.delta" in event_types
    assert "response.function_call_arguments.done" in event_types
    assert event_types[-1] == "response.completed"

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_mock_responses_streaming_iterator_async_iteration_logs_completion(
    monkeypatch,
):
    hook_calls = {"post_call": 0, "metadata": 0}

    async def fake_post_call(request_data, response, call_type):
        hook_calls["post_call"] += 1

    def fake_update_metadata(**kwargs):
        hook_calls["metadata"] += 1

    monkeypatch.setattr(
        streaming_module,
        "async_post_call_success_deployment_hook",
        fake_post_call,
    )
    monkeypatch.setattr(
        streaming_module,
        "update_response_metadata",
        fake_update_metadata,
    )

    class _MockTransformConfig:
        def transform_response_api_response(self, **kwargs):
            return _make_completed_response("resp_mock").response

    logging_obj = _FakeLoggingObj()

    iterator = MockResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_MockTransformConfig(),
        logging_obj=logging_obj,
        request_data={"model": "test-model", "stream": True},
        call_type=CallTypes.responses.value,
    )

    streamed_events = [event async for event in iterator]
    await asyncio.sleep(0.2)

    assert streamed_events[0].type == ResponsesAPIStreamEvents.RESPONSE_CREATED
    assert streamed_events[-1].type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    assert logging_obj.success_calls == 1
    assert logging_obj.async_success_calls == 1
    assert hook_calls["post_call"] == 1
    assert hook_calls["metadata"] == 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_mock_responses_streaming_iterator_sync_iteration_logs_completion(monkeypatch):
    hook_calls = {"post_call": 0, "metadata": 0}

    async def fake_post_call(request_data, response, call_type):
        hook_calls["post_call"] += 1

    def fake_update_metadata(**kwargs):
        hook_calls["metadata"] += 1

    monkeypatch.setattr(
        streaming_module,
        "async_post_call_success_deployment_hook",
        fake_post_call,
    )
    monkeypatch.setattr(
        streaming_module,
        "update_response_metadata",
        fake_update_metadata,
    )

    class _MockTransformConfig:
        def transform_response_api_response(self, **kwargs):
            return _make_completed_response("resp_mock_sync").response

    logging_obj = _FakeLoggingObj()
    iterator = MockResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_MockTransformConfig(),
        logging_obj=logging_obj,
        request_data={"model": "test-model", "stream": True},
        call_type=CallTypes.responses.value,
    )

    streamed_events = list(iterator)
    asyncio.run(asyncio.sleep(0.2))

    assert streamed_events[0].type == ResponsesAPIStreamEvents.RESPONSE_CREATED
    assert streamed_events[-1].type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    assert logging_obj.success_calls == 1
    assert logging_obj.async_success_calls == 1
    assert hook_calls["post_call"] == 1
    assert hook_calls["metadata"] == 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_cached_responses_stream_async_hit_triggers_success_callbacks(
    monkeypatch,
):
    hook_calls = {"post_call": 0, "metadata": 0}

    async def fake_post_call(request_data, response, call_type):
        hook_calls["post_call"] += 1

    def fake_update_metadata(**kwargs):
        hook_calls["metadata"] += 1

    monkeypatch.setattr(
        streaming_module,
        "async_post_call_success_deployment_hook",
        fake_post_call,
    )
    monkeypatch.setattr(
        streaming_module,
        "update_response_metadata",
        fake_update_metadata,
    )

    logging_obj = _FakeLoggingObj()
    original_cache = litellm.cache
    litellm.cache = SimpleNamespace(
        async_add_cache=AsyncMock(),
        add_cache=MagicMock(),
    )
    logging_obj.llm_caching_handler = SimpleNamespace(
        request_kwargs={"model": "test-model", "input": "hello", "stream": True},
        preset_cache_key="responses-stream-cache-key",
        original_function=litellm.aresponses,
        _should_store_result_in_cache=lambda original_function, kwargs: True,
    )

    iterator = CachedResponsesAPIStreamingIterator(
        response=_make_completed_response("resp_cached_async").response,
        logging_obj=logging_obj,
        request_data={"model": "test-model", "input": "hello", "stream": True},
        call_type=CallTypes.aresponses.value,
    )

    streamed_events = [event async for event in iterator]
    await asyncio.sleep(0.2)

    assert streamed_events[-1].type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    assert logging_obj.success_calls == 1
    assert logging_obj.async_success_calls == 1
    assert logging_obj.last_success_kwargs["cache_hit"] is True
    assert logging_obj.last_async_success_kwargs["cache_hit"] is True
    assert hook_calls["post_call"] == 1
    assert hook_calls["metadata"] == 1
    litellm.cache.async_add_cache.assert_not_called()
    litellm.cache.add_cache.assert_not_called()
    litellm.cache = original_cache

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_cached_responses_stream_sync_hit_triggers_success_callbacks(monkeypatch):
    hook_calls = {"post_call": 0, "metadata": 0}

    async def fake_post_call(request_data, response, call_type):
        hook_calls["post_call"] += 1

    def fake_update_metadata(**kwargs):
        hook_calls["metadata"] += 1

    monkeypatch.setattr(
        streaming_module,
        "async_post_call_success_deployment_hook",
        fake_post_call,
    )
    monkeypatch.setattr(
        streaming_module,
        "update_response_metadata",
        fake_update_metadata,
    )

    logging_obj = _FakeLoggingObj()
    original_cache = litellm.cache
    litellm.cache = SimpleNamespace(
        async_add_cache=AsyncMock(),
        add_cache=MagicMock(),
    )
    logging_obj.llm_caching_handler = SimpleNamespace(
        request_kwargs={"model": "test-model", "input": "hello", "stream": True},
        preset_cache_key="responses-stream-cache-key",
        original_function=litellm.responses,
        _should_store_result_in_cache=lambda original_function, kwargs: True,
    )

    iterator = CachedResponsesAPIStreamingIterator(
        response=_make_completed_response("resp_cached_sync").response,
        logging_obj=logging_obj,
        request_data={"model": "test-model", "input": "hello", "stream": True},
        call_type=CallTypes.responses.value,
    )

    streamed_events = list(iterator)
    asyncio.run(asyncio.sleep(0.2))

    assert streamed_events[-1].type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    assert logging_obj.success_calls == 1
    assert logging_obj.async_success_calls == 1
    assert logging_obj.last_success_kwargs["cache_hit"] is True
    assert logging_obj.last_async_success_kwargs["cache_hit"] is True
    assert hook_calls["post_call"] == 1
    assert hook_calls["metadata"] == 1
    litellm.cache.async_add_cache.assert_not_called()
    litellm.cache.add_cache.assert_not_called()
    litellm.cache = original_cache
