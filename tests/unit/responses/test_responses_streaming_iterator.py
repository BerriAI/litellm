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
from litellm.exceptions import MidStreamFallbackError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils import thread_pool_executor as thread_pool_executor_module
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses import streaming_iterator as responses_streaming_iterator_module
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIResponse, ResponsesAPIStreamingResponse
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


class FailureRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.failure_payloads: tuple[StandardLoggingPayload, ...] = ()
        self.success_payloads: tuple[StandardLoggingPayload, ...] = ()

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.failure_payloads = (*self.failure_payloads, cast(StandardLoggingPayload, kwargs["standard_logging_object"]))

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.success_payloads = (*self.success_payloads, cast(StandardLoggingPayload, kwargs["standard_logging_object"]))


class _ChunkRecorder:
    def __init__(self) -> None:
        self.chunks: tuple[ResponsesAPIStreamingResponse, ...] = ()

    async def drain(self, iterator: ResponsesAPIStreamingIterator) -> None:
        async for chunk in iterator:
            self.chunks = (*self.chunks, chunk)


_SEA_DELTAS = ("The sea ", "is wide ", "and deep.")
_SEA_PROMPT = "Write a 300 word story about the sea."
_UNREACHABLE_IMAGE_INPUT: list[dict[str, object]] = [
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Describe this picture."},
            {"type": "input_image", "image_url": "http://images.invalid/photo.png", "detail": "high"},
        ],
    }
]


_TOOL_ARGUMENT_DELTAS = ('{"city": ', '"Lisbon", ', '"unit": "celsius"}')


class _UpstreamThatTimesOutAfterThreeDeltas:
    headers: dict[str, str] = {}

    def __init__(self, event_type: str = "response.output_text.delta", deltas: tuple[str, ...] = _SEA_DELTAS) -> None:
        self.event_type = event_type
        self.deltas = deltas

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for sequence_number, delta in enumerate(self.deltas):
            event = {
                "type": self.event_type,
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "delta": delta,
                "sequence_number": sequence_number,
            }
            yield f"data: {json.dumps(event)}\n\n".encode()
        raise httpx.ReadTimeout("Timeout on reading data from socket")


class _UpstreamThatFailsWithUsageAfterThreeDeltas:
    headers: dict[str, str] = {}

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for sequence_number, delta in enumerate(_SEA_DELTAS):
            event = {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": delta,
                "sequence_number": sequence_number,
            }
            yield f"data: {json.dumps(event)}\n\n".encode()
        failed = {
            "type": "response.failed",
            "sequence_number": len(_SEA_DELTAS),
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 1700000000,
                "status": "failed",
                "model": "gpt-5.4-nano",
                "output": [],
                "error": {"code": "server_error", "message": "The model had an internal error"},
                "usage": {"input_tokens": 400, "output_tokens": 9, "total_tokens": 409},
            },
        }
        yield f"data: {json.dumps(failed)}\n\n".encode()


async def _failure_payload_after_mid_stream_timeout(
    monkeypatch: pytest.MonkeyPatch,
    request_input: str | list[dict[str, object]],
    upstream: _UpstreamThatTimesOutAfterThreeDeltas | None = None,
    background_fields: Mapping[str, object] | None = None,
) -> tuple[FailureRecorder, StandardLoggingPayload]:
    upstream_response: Final = upstream or _UpstreamThatTimesOutAfterThreeDeltas()
    recorder = FailureRecorder()
    monkeypatch.setattr(litellm, "failure_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_failure_callback", [recorder])
    monkeypatch.setattr(litellm, "success_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    logging_obj = _make_logging_obj()
    logging_obj.update_environment_variables(
        model="gpt-5.4-nano",
        optional_params={"stream": True},
        litellm_params={"custom_llm_provider": "openai", "aresponses": True},
    )
    iterator = ResponsesAPIStreamingIterator(
        response=cast(httpx.Response, upstream_response),
        model="gpt-5.4-nano",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=logging_obj,
        custom_llm_provider="openai",
        request_data={"model": "gpt-5.4-nano", "input": request_input, "stream": True, **(background_fields or {})},
    )
    received = _ChunkRecorder()

    with pytest.raises(litellm.Timeout, match="Timeout on reading data from socket"):
        await received.drain(iterator)
    assert [getattr(chunk, "delta", None) for chunk in received.chunks] == list(upstream_response.deltas)

    await iterator._await_pending_logging()
    (payload,) = recorder.failure_payloads
    return recorder, payload


@pytest.mark.asyncio
async def test_mid_stream_read_timeout_logs_failure_with_partial_usage_and_no_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder, payload = await _failure_payload_after_mid_stream_timeout(monkeypatch, _SEA_PROMPT)

    assert payload["status"] == "failure"
    assert payload["prompt_tokens"] == litellm.token_counter(
        model="gpt-5.4-nano", messages=[{"role": "user", "content": _SEA_PROMPT}]
    )
    assert payload["completion_tokens"] == litellm.token_counter(
        model="gpt-5.4-nano", text="".join(_SEA_DELTAS), count_response_tokens=True
    )
    assert payload["response_cost"] > 0
    assert recorder.success_payloads == ()


@pytest.mark.asyncio
async def test_mid_stream_read_timeout_bills_the_tool_call_arguments_already_streamed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream: Final = _UpstreamThatTimesOutAfterThreeDeltas(
        event_type="response.function_call_arguments.delta", deltas=_TOOL_ARGUMENT_DELTAS
    )
    _, payload = await _failure_payload_after_mid_stream_timeout(monkeypatch, _SEA_PROMPT, upstream)

    assert payload["status"] == "failure"
    assert payload["completion_tokens"] == litellm.token_counter(
        model="gpt-5.4-nano", text="".join(_TOOL_ARGUMENT_DELTAS), count_response_tokens=True
    )
    assert payload["response_cost"] > 0


@pytest.mark.parametrize(
    "background_fields",
    [{"background": True}, {"extra_body": {"background": True}}],
    ids=["top_level", "extra_body"],
)
@pytest.mark.asyncio
async def test_mid_stream_failure_of_a_background_response_bills_nothing_since_retrieving_it_bills_the_whole_run(
    monkeypatch: pytest.MonkeyPatch, background_fields: Mapping[str, object]
) -> None:
    _, payload = await _failure_payload_after_mid_stream_timeout(
        monkeypatch, _SEA_PROMPT, background_fields=background_fields
    )

    assert payload["status"] == "failure"
    assert payload["completion_tokens"] == 0
    assert payload["response_cost"] == 0


@pytest.mark.asyncio
async def test_mid_stream_failure_usage_estimate_never_fetches_image_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, payload = await _failure_payload_after_mid_stream_timeout(monkeypatch, _UNREACHABLE_IMAGE_INPUT)

    assert payload["status"] == "failure"
    assert payload["prompt_tokens"] == litellm.token_counter(
        model="gpt-5.4-nano",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this picture."},
                    {"type": "image_url", "image_url": {"url": "http://images.invalid/photo.png", "detail": "high"}},
                ],
            }
        ],
        use_default_image_token_count=True,
    )
    assert payload["completion_tokens"] > 0


@pytest.mark.asyncio
async def test_mid_stream_failure_keeps_the_usage_the_provider_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = FailureRecorder()
    monkeypatch.setattr(litellm, "failure_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_failure_callback", [recorder])
    logging_obj = _make_logging_obj()
    logging_obj.update_environment_variables(
        model="gpt-5.4-nano",
        optional_params={"stream": True},
        litellm_params={"custom_llm_provider": "openai", "aresponses": True},
    )
    iterator = ResponsesAPIStreamingIterator(
        response=cast(httpx.Response, _UpstreamThatFailsWithUsageAfterThreeDeltas()),
        model="gpt-5.4-nano",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=logging_obj,
        custom_llm_provider="openai",
        request_data={"model": "gpt-5.4-nano", "input": _SEA_PROMPT, "stream": True},
    )

    with pytest.raises(MidStreamFallbackError, match="The model had an internal error"):
        await _ChunkRecorder().drain(iterator)
    await iterator._await_pending_logging()

    (payload,) = recorder.failure_payloads
    assert (payload["prompt_tokens"], payload["completion_tokens"]) == (400, 9)


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


def _live_iterator(
    logging_obj: LitellmLogging, upstream: object = None, background: bool = False
) -> ResponsesAPIStreamingIterator:
    return ResponsesAPIStreamingIterator(
        response=cast(httpx.Response, upstream or httpx.Response(200)),
        model="gpt-5.4-nano",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=logging_obj,
        request_data={"model": "gpt-5.4-nano", "input": "tell me a story", "background": background},
        custom_llm_provider="openai",
    )


class _UpstreamThatTimesOutAfterRelayingText:
    headers: dict[str, str] = {}

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for event in (_CREATED_WITH_PRIORITY_TIER, _text_delta("Once upon a time", 1)):
            yield f"data: {json.dumps(event)}\n\n".encode()
        raise httpx.ReadTimeout("Timeout on reading data from socket")


class _UpstreamServedAtTierThatTimesOut:
    headers: dict[str, str] = {}

    def __init__(self, served_tier: str) -> None:
        self.served_tier = served_tier

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        created: Final = {
            **_CREATED_WITH_PRIORITY_TIER,
            "response": {**_CREATED_RESPONSE_WITH_PRIORITY_TIER, "service_tier": self.served_tier},
        }
        for event in (created, _text_delta("Once upon a time", 1)):
            yield f"data: {json.dumps(event)}\n\n".encode()
        raise httpx.ReadTimeout("Timeout on reading data from socket")


async def _mid_stream_failure_cost(
    monkeypatch: pytest.MonkeyPatch, requested_tier: str | None, served_tier: str
) -> float:
    recorder: Final = FailureRecorder()
    monkeypatch.setattr(litellm, "failure_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_failure_callback", [recorder])
    logging_obj: Final = _make_logging_obj()
    logging_obj.update_environment_variables(
        model="gpt-5.4-nano",
        optional_params={"stream": True} if requested_tier is None else {"stream": True, "service_tier": requested_tier},
        litellm_params={"custom_llm_provider": "openai", "aresponses": True},
    )
    iterator: Final = _live_iterator(logging_obj, upstream=_UpstreamServedAtTierThatTimesOut(served_tier))

    with pytest.raises(litellm.Timeout):
        await _ChunkRecorder().drain(iterator)
    await iterator._await_pending_logging()

    (payload,) = recorder.failure_payloads
    return payload["response_cost"]


@pytest.mark.asyncio
async def test_mid_stream_failure_bills_the_relayed_text_at_the_tier_the_provider_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flex_requested_default_served: Final = await _mid_stream_failure_cost(monkeypatch, "flex", "default")
    nothing_requested_default_served: Final = await _mid_stream_failure_cost(monkeypatch, None, "default")

    assert nothing_requested_default_served > 0
    assert flex_requested_default_served == nothing_requested_default_served


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


@pytest.mark.asyncio
async def test_partial_response_on_disconnect_is_none_once_the_failure_log_billed_the_relayed_text():
    iterator: Final = _live_iterator(_make_logging_obj(), upstream=_UpstreamThatTimesOutAfterRelayingText())

    with pytest.raises(litellm.Timeout):
        await _ChunkRecorder().drain(iterator)

    assert iterator.partial_response_on_disconnect() is None
    await iterator._await_pending_logging()


def test_partial_response_on_disconnect_reports_the_relayed_text_of_a_background_response():
    iterator: Final = _live_iterator(_make_logging_obj(), background=True)
    for event in (_CREATED_WITH_PRIORITY_TIER, _text_delta("Once upon a time", 1)):
        iterator._process_chunk(json.dumps(event))

    incomplete_event: Final = iterator.partial_response_on_disconnect()

    assert incomplete_event is not None
    assert incomplete_event.response.output[0].content[0].text == "Once upon a time"
