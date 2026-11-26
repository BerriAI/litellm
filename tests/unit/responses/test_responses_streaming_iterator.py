"""
Regression tests for LIT-4210: the streaming iterators must never run the sync
success_handler on the thread-pool executor concurrently with
async_success_handler. Concurrent mutation of the shared response object /
model_call_details from two threads segfaults pydantic-core (customer pods
crashed with exit 139 whenever any CustomLogger was registered).
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from typing import Final, cast

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils import thread_pool_executor as thread_pool_executor_module
from litellm.responses import streaming_iterator as responses_streaming_iterator_module
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import StandardLoggingPayload


class RecordingCustomLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.async_hook_started: float | None = None
        self.async_hook_finished: float | None = None

    async def _record(self):
        self.async_hook_started = time.monotonic()
        await asyncio.sleep(0.2)
        self.async_hook_finished = time.monotonic()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._record()

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        await self._record()


class RecordingExecutor:
    def __init__(self, inner):
        self._inner = inner
        self.submits: list = []

    def submit(self, fn, *args, **kwargs):
        self.submits.append((time.monotonic(), fn))
        return self._inner.submit(fn, *args, **kwargs)

    def submit_times_for(self, logging_obj) -> list:
        return [t for t, fn in self.submits if getattr(fn, "__self__", None) is logging_obj]


@pytest.fixture(autouse=True)
def _isolate_callbacks():
    saved = (
        litellm.callbacks,
        litellm.success_callback,
        litellm._async_success_callback,
        litellm.failure_callback,
        litellm._async_failure_callback,
    )
    yield
    (
        litellm.callbacks,
        litellm.success_callback,
        litellm._async_success_callback,
        litellm.failure_callback,
        litellm._async_failure_callback,
    ) = saved


@pytest.fixture
def recording_executor(monkeypatch):
    recording = RecordingExecutor(thread_pool_executor_module.executor)
    monkeypatch.setattr(thread_pool_executor_module, "executor", recording)
    monkeypatch.setattr(responses_streaming_iterator_module, "executor", recording)
    return recording


def _make_logging_obj() -> LitellmLogging:
    logging_obj = LitellmLogging(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="aresponses",
        start_time=time.time(),
        litellm_call_id="lit-4210-test",
        function_id="lit-4210-test",
    )
    logging_obj.model_call_details["litellm_params"] = {"aresponses": True}
    return logging_obj


def _make_iterator(logging_obj: LitellmLogging) -> ResponsesAPIStreamingIterator:
    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="gpt-5.4-nano",
        responses_api_provider_config=None,
        logging_obj=logging_obj,
    )
    iterator.completed_response = ResponsesAPIResponse(
        id="resp_lit4210",
        created_at=1700000000.0,
        model="gpt-5.4-nano",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata={},
        temperature=1.0,
        top_p=1.0,
    )
    return iterator


async def _await_tasks_spawned_since(existing: set["asyncio.Task[object]"]) -> None:
    spawned = asyncio.all_tasks() - existing
    await asyncio.wait_for(asyncio.gather(*spawned, return_exceptions=True), timeout=5)


@pytest.mark.asyncio
async def test_custom_logger_only_never_submits_sync_success_handler(recording_executor):
    recorder = RecordingCustomLogger()
    litellm.success_callback = [recorder]
    litellm._async_success_callback = [recorder]

    logging_obj = _make_logging_obj()
    iterator = _make_iterator(logging_obj)

    existing_tasks = asyncio.all_tasks()
    iterator._log_completed_response(is_async=True)
    await _await_tasks_spawned_since(existing_tasks)

    assert recorder.async_hook_started is not None
    assert recording_executor.submit_times_for(logging_obj) == []


@pytest.mark.asyncio
async def test_sync_callbacks_run_only_after_async_handler_completes(recording_executor):
    recorder = RecordingCustomLogger()
    sync_events: list = []

    def sync_callback(kwargs, response_obj, start_time, end_time):
        sync_events.append(time.monotonic())

    litellm.success_callback = [recorder, sync_callback]
    litellm._async_success_callback = [recorder]

    logging_obj = _make_logging_obj()
    iterator = _make_iterator(logging_obj)

    existing_tasks = asyncio.all_tasks()
    iterator._log_completed_response(is_async=True)
    await _await_tasks_spawned_since(existing_tasks)

    assert recorder.async_hook_finished is not None
    submit_times = recording_executor.submit_times_for(logging_obj)
    assert len(submit_times) == 1
    assert submit_times[0] >= recorder.async_hook_finished


_CREATED_RESPONSE_WITH_PRIORITY_TIER: Final[Mapping[str, object]] = {
    "id": "resp_partial",
    "object": "response",
    "created_at": 1700000000.0,
    "model": "gpt-5.4-nano",
    "status": "in_progress",
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "service_tier": "priority",
}
_CREATED_WITH_PRIORITY_TIER: Final[Mapping[str, object]] = {
    "type": "response.created",
    "sequence_number": 0,
    "response": _CREATED_RESPONSE_WITH_PRIORITY_TIER,
}


def _text_delta(text: str, sequence_number: int) -> Mapping[str, object]:
    return {
        "type": "response.output_text.delta",
        "sequence_number": sequence_number,
        "item_id": "msg_partial",
        "output_index": 0,
        "content_index": 0,
        "delta": text,
    }


def _completed_after(text: str) -> Mapping[str, object]:
    return {
        "type": "response.completed",
        "sequence_number": 9,
        "response": {
            **_CREATED_RESPONSE_WITH_PRIORITY_TIER,
            "status": "completed",
            "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
        },
    }


def _live_iterator(logging_obj: LitellmLogging) -> ResponsesAPIStreamingIterator:
    return ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="gpt-5.4-nano",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=logging_obj,
        request_data={"model": "gpt-5.4-nano", "input": "tell me a story"},
        custom_llm_provider="openai",
    )


def test_partial_response_on_disconnect_bills_the_relayed_text_at_the_served_service_tier():
    iterator: Final = _live_iterator(_make_logging_obj())
    for event in (_CREATED_WITH_PRIORITY_TIER, _text_delta("Once upon a time ", 1), _text_delta("a fox slept.", 2)):
        iterator._process_chunk(json.dumps(event))

    incomplete_event: Final = iterator.partial_response_on_disconnect()

    assert incomplete_event is not None
    assert incomplete_event.type == "response.incomplete"
    partial: Final = incomplete_event.response
    assert partial.status == "incomplete"
    assert getattr(partial, "service_tier") == "priority"
    assert partial.output[0].content[0].text == "Once upon a time a fox slept."
    assert partial.usage.output_tokens == litellm.token_counter(
        model="gpt-5.4-nano", text="Once upon a time a fox slept."
    )
    assert partial.usage.input_tokens == litellm.token_counter(
        model="gpt-5.4-nano", messages=[{"role": "user", "content": "tell me a story"}]
    )
    assert partial.usage.total_tokens == partial.usage.input_tokens + partial.usage.output_tokens


@pytest.mark.parametrize(
    "events",
    [
        (),
        (_CREATED_WITH_PRIORITY_TIER,),
        (_CREATED_WITH_PRIORITY_TIER, _text_delta("Once upon a time", 1), _completed_after("Once upon a time")),
    ],
    ids=["nothing_relayed", "created_only", "already_completed"],
)
@pytest.mark.asyncio
async def test_partial_response_on_disconnect_is_none_when_there_is_nothing_to_bill_or_the_stream_billed_itself(
    events: tuple[Mapping[str, object], ...],
):
    iterator: Final = _live_iterator(_make_logging_obj())
    for event in events:
        iterator._process_chunk(json.dumps(event))

    assert iterator.partial_response_on_disconnect() is None
    await iterator._await_pending_logging()


_PROVIDER_USAGE_AFTER_THE_CLIENT_LEFT: Final[Mapping[str, object]] = {
    "input_tokens": 1200,
    "input_tokens_details": {"cached_tokens": 1024},
    "output_tokens": 900,
    "output_tokens_details": {"reasoning_tokens": 832},
    "total_tokens": 2100,
}


class _UpstreamThatReasonsUntilReleased:
    """Sends response.created, then nothing until released, then a reasoning item and the provider's terminal usage."""

    headers: dict[str, str] = {}

    def __init__(self, fails_after_release: bool = False) -> None:
        self.released: Final = asyncio.Event()
        self.closed = False
        self._fails_after_release: Final = fails_after_release

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        yield f"data: {json.dumps(_CREATED_WITH_PRIORITY_TIER)}\n\n".encode()
        await self.released.wait()
        if self._fails_after_release:
            raise httpx.ReadError("connection reset by peer")
        completed: Final = {
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                **_CREATED_RESPONSE_WITH_PRIORITY_TIER,
                "status": "completed",
                "usage": _PROVIDER_USAGE_AFTER_THE_CLIENT_LEFT,
            },
        }
        yield f"data: {json.dumps(completed)}\n\n".encode()

    async def aclose(self) -> None:
        self.closed = True


class _PayloadRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.failure_payloads: tuple[StandardLoggingPayload, ...] = ()
        self.success_payloads: tuple[StandardLoggingPayload, ...] = ()

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.failure_payloads = (
            *self.failure_payloads,
            cast(StandardLoggingPayload, kwargs["standard_logging_object"]),
        )

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.success_payloads = (
            *self.success_payloads,
            cast(StandardLoggingPayload, kwargs["standard_logging_object"]),
        )


def _recorded_stream(
    monkeypatch: pytest.MonkeyPatch, upstream: _UpstreamThatReasonsUntilReleased
) -> tuple[ResponsesAPIStreamingIterator, _PayloadRecorder]:
    recorder: Final = _PayloadRecorder()
    monkeypatch.setattr(litellm, "failure_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_failure_callback", [recorder])
    monkeypatch.setattr(litellm, "success_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    logging_obj: Final = _make_logging_obj()
    logging_obj.update_environment_variables(
        model="gpt-5.4-nano",
        optional_params={"stream": True},
        litellm_params={"custom_llm_provider": "openai", "aresponses": True},
    )
    iterator: Final = ResponsesAPIStreamingIterator(
        response=cast(httpx.Response, upstream),
        model="gpt-5.4-nano",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=logging_obj,
        request_data={"model": "gpt-5.4-nano", "input": "tell me a story"},
        custom_llm_provider="openai",
    )
    return iterator, recorder


async def _cancel_the_consumer_while_the_provider_is_reasoning(iterator: ResponsesAPIStreamingIterator) -> None:
    """What a client disconnect does: the read waiting on the provider is cancelled, then the wrappers close the stream."""
    assert (await iterator.__anext__()).type == "response.created"
    waiting_on_the_provider: Final = asyncio.ensure_future(iterator.__anext__())
    await asyncio.sleep(0)
    waiting_on_the_provider.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting_on_the_provider
    await iterator.aclose()


@pytest.mark.asyncio
async def test_a_stream_cut_before_any_output_bills_the_usage_the_provider_reports_once_it_is_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream: Final = _UpstreamThatReasonsUntilReleased()
    iterator, recorder = _recorded_stream(monkeypatch, upstream)
    iterator.outlive_consumer_until_settled()
    existing_tasks: Final = asyncio.all_tasks()

    await _cancel_the_consumer_while_the_provider_is_reasoning(iterator)
    assert upstream.closed is False, "closing here loses the provider's terminal usage"
    upstream.released.set()
    drained: Final = await iterator.settle_after_consumer_left(drain=True)
    await iterator._await_pending_logging()
    await _await_tasks_spawned_since(existing_tasks)

    assert drained == "success"
    assert upstream.closed is True
    (success_payload,) = recorder.success_payloads
    assert recorder.failure_payloads == ()
    assert (
        success_payload["prompt_tokens"],
        success_payload["completion_tokens"],
        success_payload["total_tokens"],
    ) == (1200, 900, 2100)
    usage: Final = cast(dict[str, dict[str, object]], cast(dict[str, object], success_payload["response"])["usage"])
    assert usage["prompt_tokens_details"]["cached_tokens"] == 1024
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 832
    assert iterator.partial_response_on_disconnect() is None, "a drained stream must not be billed a second time"


@pytest.mark.parametrize("loop_turns_before_the_cancel", range(6))
@pytest.mark.asyncio
async def test_a_consumer_cancelled_just_as_the_terminal_event_arrives_still_bills_the_usage_the_provider_reports(
    monkeypatch: pytest.MonkeyPatch, loop_turns_before_the_cancel: int
) -> None:
    upstream: Final = _UpstreamThatReasonsUntilReleased()
    iterator, recorder = _recorded_stream(monkeypatch, upstream)
    iterator.outlive_consumer_until_settled()
    existing_tasks: Final = asyncio.all_tasks()
    assert (await iterator.__anext__()).type == "response.created"
    upstream.released.set()
    reading_the_terminal_event: Final = asyncio.ensure_future(iterator.__anext__())
    for _ in range(loop_turns_before_the_cancel):
        await asyncio.sleep(0)
    reading_the_terminal_event.cancel()
    await asyncio.gather(reading_the_terminal_event, return_exceptions=True)
    await iterator.aclose()

    await iterator.settle_after_consumer_left(drain=True)
    await iterator._await_pending_logging()
    await _await_tasks_spawned_since(existing_tasks)

    (success_payload,) = recorder.success_payloads
    assert recorder.failure_payloads == ()
    assert (
        success_payload["prompt_tokens"],
        success_payload["completion_tokens"],
        success_payload["total_tokens"],
    ) == (1200, 900, 2100), "an event the provider already sent must not be dropped by the cancel"


@pytest.mark.asyncio
async def test_a_stream_cut_after_its_terminal_event_was_logged_reports_that_the_success_event_owns_the_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream: Final = _UpstreamThatReasonsUntilReleased()
    iterator, recorder = _recorded_stream(monkeypatch, upstream)
    iterator.outlive_consumer_until_settled()
    existing_tasks: Final = asyncio.all_tasks()
    upstream.released.set()
    assert [(await iterator.__anext__()).type for _ in range(2)] == ["response.created", "response.completed"]

    billed_by_the_success_event: Final = await iterator.settle_after_consumer_left(drain=True)
    await iterator._await_pending_logging()
    await _await_tasks_spawned_since(existing_tasks)

    assert billed_by_the_success_event == "success", "None has the proxy release the parallel-request slot a second time"
    assert len(recorder.success_payloads) == 1
    assert upstream.closed is True


@pytest.mark.parametrize("outlives_its_consumer", [True, False], ids=["told_to_close", "never_kept_open"])
@pytest.mark.asyncio
async def test_a_cut_stream_that_is_not_drained_closes_the_provider_connection_and_bills_no_success(
    monkeypatch: pytest.MonkeyPatch, outlives_its_consumer: bool
) -> None:
    upstream: Final = _UpstreamThatReasonsUntilReleased()
    iterator, recorder = _recorded_stream(monkeypatch, upstream)
    if outlives_its_consumer:
        iterator.outlive_consumer_until_settled()

    await _cancel_the_consumer_while_the_provider_is_reasoning(iterator)
    assert upstream.closed is not outlives_its_consumer
    drained: Final = await iterator.settle_after_consumer_left(drain=outlives_its_consumer is False)

    assert drained is None
    assert upstream.closed is True
    assert recorder.success_payloads == ()


@pytest.mark.asyncio
async def test_a_kept_open_stream_nobody_settles_closes_the_provider_connection_after_the_grace_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(responses_streaming_iterator_module, "RESPONSES_API_UNCLAIMED_STREAM_CLOSE_GRACE_SECONDS", 0)
    upstream: Final = _UpstreamThatReasonsUntilReleased()
    iterator, _ = _recorded_stream(monkeypatch, upstream)
    iterator.outlive_consumer_until_settled()
    existing_tasks: Final = asyncio.all_tasks()

    await _cancel_the_consumer_while_the_provider_is_reasoning(iterator)
    await _await_tasks_spawned_since(existing_tasks)

    assert upstream.closed is True


@pytest.mark.asyncio
async def test_a_provider_failure_while_draining_a_cut_stream_is_logged_as_a_failure_not_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream: Final = _UpstreamThatReasonsUntilReleased(fails_after_release=True)
    iterator, recorder = _recorded_stream(monkeypatch, upstream)
    iterator.outlive_consumer_until_settled()

    await _cancel_the_consumer_while_the_provider_is_reasoning(iterator)
    upstream.released.set()
    drained: Final = await iterator.settle_after_consumer_left(drain=True)
    await iterator._await_pending_logging()

    assert drained == "failure", "the failure callback released the slot, so the proxy must not release it again"
    assert upstream.closed is True
    assert recorder.success_payloads == ()
    assert len(recorder.failure_payloads) == 1
