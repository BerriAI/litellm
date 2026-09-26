"""Regression tests for LIT-4185 — /v1/responses streaming must stamp
completion_start_time on the first chunk so downstream TTFT consumers
(Prometheus, OTEL, SpendLogs completionStartTime) do not fall back to
completion_start_time = end_time."""

import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from typing import Final, Optional
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from pydantic_core import PydanticSerializationError

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import (
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
    _estimate_usage_from_text,
)
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


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
    logging_obj.dispatch_failure_handlers = AsyncMock()
    return logging_obj


def _assert_failure_logged_once(logging_obj: Mock, exception: Exception) -> None:
    assert logging_obj.async_failure_handler.await_count == 1
    assert logging_obj.async_failure_handler.await_args.kwargs["exception"] is exception


async def _assert_failure_dispatched_once(logging_obj: Mock, exception: Exception) -> None:
    await asyncio.sleep(0)
    assert logging_obj.dispatch_failure_handlers.await_count == 1
    assert logging_obj.dispatch_failure_handlers.await_args.args[0] is exception


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
    await _assert_failure_dispatched_once(logging_obj, trailing_error)


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
    await _assert_failure_dispatched_once(logging_obj, trailing_error)


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
    await _assert_failure_dispatched_once(logging_obj, exc_info.value)


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
    await _assert_failure_dispatched_once(logging_obj, exc_info.value.original_exception)


@pytest.mark.asyncio
@pytest.mark.parametrize("trailer", [[], [_DONE_MARKER]], ids=["eof", "done-marker"])
async def test_complete_stream_still_ends_normally(trailer):
    logging_obj: Final = _failure_tracking_logging_obj()
    iterator: Final = _make_iterator(sse_events=[*_COMPLETE_STREAM_EVENTS, *trailer], logging_obj=logging_obj)

    seen: Final = [event.type async for event in iterator]

    assert seen[-1] == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    assert logging_obj.dispatch_failure_handlers.await_count == 0
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
        self.failure_finished = False
        self.hook_loop: asyncio.AbstractEventLoop | None = None
        self.hook_finished = False

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.failure_loop = asyncio.get_running_loop()
        await asyncio.sleep(0.05)
        self.failure_finished = True

    async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
        self.hook_loop = asyncio.get_running_loop()
        await asyncio.sleep(0.05)
        self.hook_finished = True
        return None


def _real_logging_obj() -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="aresponses",
        start_time=datetime.now(),
        litellm_call_id="lit-8678-test",
        function_id="lit-8678-test",
    )
    logging_obj.model_call_details["litellm_params"] = {"aresponses": True}
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

    assert recorder.failure_finished is False
    await _wait_until(lambda: recorder.failure_finished)
    assert recorder.failure_loop is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_completed_stream_success_deployment_hook_runs_on_the_iterating_loop(monkeypatch):
    recorder: Final = _LoopRecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    iterator: Final = _make_iterator(sse_events=_COMPLETE_STREAM_EVENTS, logging_obj=_logging_obj_stub())

    async for _ in iterator:
        pass

    assert recorder.hook_finished is False
    await _wait_until(lambda: recorder.hook_finished)
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
        _llm_caching_handler=caching_handler,
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
            output=[],
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
    logging_obj._response_cost_calculator.return_value = 0.000704

    _stamp_responses_usage_cost(response, logging_obj)

    assert getattr(response.usage, "cost", None) == pytest.approx(0.000704)
    logging_obj._response_cost_calculator.assert_called_once_with(result=response)


def test_stamp_responses_usage_cost_keeps_provider_reported_cost():
    from litellm.responses.streaming_iterator import _stamp_responses_usage_cost

    response = _responses_api_response_with_usage()
    setattr(response.usage, "cost", 0.5)
    logging_obj = Mock(spec=LiteLLMLoggingObj)

    _stamp_responses_usage_cost(response, logging_obj)

    assert getattr(response.usage, "cost", None) == pytest.approx(0.5)
    logging_obj._response_cost_calculator.assert_not_called()


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
    logging_obj._response_cost_calculator.assert_not_called()


def test_stamp_responses_usage_cost_computes_cost_for_dict_usage_without_cost():
    from litellm.responses.streaming_iterator import _stamp_responses_usage_cost
    response = _unvalidated_response_with_dict_usage({"input_tokens": 29, "output_tokens": 120, "total_tokens": 149})
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj._response_cost_calculator.return_value = 0.000704

    _stamp_responses_usage_cost(response, logging_obj)

    assert isinstance(response.usage, ResponseAPIUsage)
    assert response.usage.cost == pytest.approx(0.000704)
    logging_obj._response_cost_calculator.assert_called_once_with(result=response)


def test_stamp_responses_usage_cost_survives_calculator_failure():
    from litellm.responses.streaming_iterator import _stamp_responses_usage_cost

    response = _responses_api_response_with_usage()
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj._response_cost_calculator.side_effect = RuntimeError("cost map unavailable")

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
    logging_obj._response_cost_calculator.return_value = 0.000704
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
    logging_obj._response_cost_calculator.assert_any_call(result=completed_response)


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


def test_persist_completed_response_to_cache_survives_an_unserializable_response(monkeypatch):
    bad_response: Final = ResponsesAPIResponse.model_construct(id="r", output=[object()], usage=None)
    with pytest.raises(PydanticSerializationError):
        bad_response.model_dump_json()

    logging_obj: Final = _logging_obj_stub()
    caching_handler: Final = Mock()
    caching_handler.request_kwargs = {"stream": True}
    logging_obj._llm_caching_handler = caching_handler
    iterator: Final = _make_iterator(sse_events=[], logging_obj=logging_obj)
    iterator.completed_response = ResponseCompletedEvent.model_construct(
        type="response.completed", response=bad_response
    )
    cache: Final = Mock()
    monkeypatch.setattr(litellm, "cache", cache)

    iterator._persist_completed_response_to_cache(is_async=False)

    cache.add_cache.assert_not_called()
