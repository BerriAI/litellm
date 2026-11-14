"""
Regression test for LIT-4210: completing an A2A stream must not run the sync
success_handler on the thread-pool executor concurrently with
async_success_handler (cross-thread pydantic mutation segfaults pydantic-core).
"""

import asyncio
import importlib.util
import time
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import datetime
from typing import Final
from types import SimpleNamespace

import pytest

import litellm
from litellm.a2a_protocol import streaming_iterator as a2a_streaming_iterator_module
from litellm.a2a_protocol.streaming_iterator import A2AStreamingIterator
from litellm.a2a_protocol.utils import A2ARequestUtils
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils import thread_pool_executor as thread_pool_executor_module
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging


class RecordingCustomLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.async_hook_fired = False

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.async_hook_fired = True

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        self.async_hook_fired = True


class RecordingExecutor:
    def __init__(self, inner):
        self._inner = inner
        self.submits: list = []

    def submit(self, fn, *args, **kwargs):
        self.submits.append(fn)
        return self._inner.submit(fn, *args, **kwargs)

    def submitted_for(self, logging_obj) -> list:
        return [fn for fn in self.submits if getattr(fn, "__self__", None) is logging_obj]


@pytest.mark.asyncio
async def test_custom_logger_only_never_submits_sync_success_handler(monkeypatch):
    recording_executor = RecordingExecutor(thread_pool_executor_module.executor)
    monkeypatch.setattr(thread_pool_executor_module, "executor", recording_executor)
    monkeypatch.setattr(a2a_streaming_iterator_module, "executor", recording_executor, raising=False)

    recorder = RecordingCustomLogger()
    monkeypatch.setattr(litellm, "success_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])

    logging_obj = LitellmLogging(
        model="a2a/test-agent",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="a2a_send_message_streaming",
        start_time=time.time(),
        litellm_call_id="lit-4210-test",
        function_id="lit-4210-test",
    )

    async def _empty_stream():
        return
        yield

    iterator = A2AStreamingIterator(
        stream=_empty_stream(),
        request=SimpleNamespace(
            params=SimpleNamespace(message={"role": "user", "parts": [{"kind": "text", "text": "hi"}]})
        ),
        logging_obj=logging_obj,
        agent_name="test-agent",
    )

    await iterator._handle_stream_complete()
    await asyncio.sleep(0.5)

    assert recorder.async_hook_fired is True
    assert recording_executor.submitted_for(logging_obj) == []


class _AgentChunk:
    def __init__(self, text: str):
        self._text = text

    def model_dump(self, mode: str, exclude_none: bool) -> dict:
        return {"result": {"kind": "message", "role": "agent", "parts": [{"kind": "text", "text": self._text}]}}


@pytest.mark.asyncio
async def test_stream_completion_counts_tokens_off_the_event_loop(monkeypatch):
    from tests.large_text import text
    from tests.unit.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        warm_tokenizer,
    )

    warm_tokenizer("gpt-5.6-luna")
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    logging_obj = LitellmLogging(
        model="a2a/test-agent",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="a2a_send_message_streaming",
        start_time=time.time(),
        litellm_call_id="lit-7190-test",
        function_id="lit-7190-test",
    )

    async def _stream():
        yield _AgentChunk(text * 100)

    iterator = A2AStreamingIterator(
        stream=_stream(),
        request=SimpleNamespace(
            params=SimpleNamespace(message={"role": "user", "parts": [{"kind": "text", "text": text * 100}]})
        ),
        logging_obj=logging_obj,
        agent_name="test-agent",
    )

    async def drain() -> int:
        return len([chunk async for chunk in iterator])

    yielded, took, lags = await timed_with_loop_lags(drain)

    assert yielded == 1
    usage = logging_obj.model_call_details["usage"]
    assert usage.prompt_tokens > 100_000
    assert usage.completion_tokens > 100_000
    assert_loop_stayed_free(took, lags)


class _SuccessRecorder(CustomLogger):
    def __init__(self):
        super().__init__()
        self.logged: list[tuple[datetime, object]] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.logged.append((start_time, kwargs.get("completion_start_time")))

    async def async_log_stream_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.logged.append((start_time, kwargs.get("completion_start_time")))


@pytest.mark.asyncio
async def test_stream_logs_its_first_chunk_time_measured_from_the_call_start():
    call_start: Final = datetime(2026, 10, 9, 12, 0, 0)
    recorder: Final = _SuccessRecorder()
    logging_obj: Final = LitellmLogging(
        model="a2a/test-agent",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="a2a_send_message_streaming",
        start_time=call_start,
        litellm_call_id="a2a-first-chunk",
        function_id="a2a-first-chunk",
        dynamic_async_success_callbacks=[recorder],
    )

    async def _stream():
        yield _AgentChunk("Hel")
        yield _AgentChunk("Hello")

    iterator: Final = A2AStreamingIterator(
        stream=_stream(),
        request=SimpleNamespace(
            params=SimpleNamespace(message={"role": "user", "parts": [{"kind": "text", "text": "hi"}]})
        ),
        logging_obj=logging_obj,
        agent_name="test-agent",
    )
    tasks_before: Final = asyncio.all_tasks()

    _ = await iterator.__anext__()
    first_chunk_at: Final = logging_obj.completion_start_time
    _ = [chunk async for chunk in iterator]
    await asyncio.gather(*(asyncio.all_tasks() - tasks_before))

    assert first_chunk_at is not None
    assert recorder.logged == [(call_start, first_chunk_at)]


_REPLY: Final = (
    "Paris is the capital of France. It sits on the Seine, holds about two million people, "
    "and has been the seat of the French government for most of the last thousand years."
)


def _artifact_event(text: str, append: bool | None, artifact_id: str = "a1") -> Mapping[str, object]:
    result: Final = {
        "kind": "artifact-update",
        "taskId": "t1",
        "contextId": "c1",
        "artifact": {"artifactId": artifact_id, "parts": [{"kind": "text", "text": text}]},
    }
    return {"jsonrpc": "2.0", "id": "r1", "result": result if append is None else {**result, "append": append}}


def _status_event(state: str, text: str | None = None) -> Mapping[str, object]:
    message: Final = {"kind": "message", "role": "agent", "messageId": "m1", "parts": [{"kind": "text", "text": text}]}
    status: Final = {"state": state} if text is None else {"state": state, "message": message}
    return {
        "jsonrpc": "2.0",
        "id": "r1",
        "result": {
            "kind": "status-update",
            "taskId": "t1",
            "contextId": "c1",
            "final": state == "completed",
            "status": status,
        },
    }


_FIRST_ARTIFACT: Final = "Paris is the capital of France and sits on the Seine."
_SECOND_ARTIFACT: Final = "It has been the seat of the French government for most of the last thousand years."


async def _snapshot_stream() -> AsyncIterator[Mapping[str, object]]:
    for end in range(40, len(_REPLY) + 40, 40):
        yield _artifact_event(_REPLY[:end], append=None)


async def _watsonx_stream() -> AsyncIterator[Mapping[str, object]]:
    from litellm.a2a_protocol.providers.watsonx_orchestrate.transformation import (
        WatsonxOrchestrateTransformation,
    )

    async for event in WatsonxOrchestrateTransformation.fake_streaming_from_text(_REPLY, "r1", delay_ms=0):
        yield event


async def _two_artifact_stream() -> AsyncIterator[Mapping[str, object]]:
    yield _artifact_event(_FIRST_ARTIFACT, append=None, artifact_id="a1")
    yield _artifact_event(_SECOND_ARTIFACT, append=None, artifact_id="a2")
    yield _status_event("completed")


async def _status_message_stream() -> AsyncIterator[Mapping[str, object]]:
    yield _status_event("working", "Looking that up")
    yield _status_event("working", _REPLY)
    yield _status_event("completed")


async def _artifact_without_id_stream() -> AsyncIterator[Mapping[str, object]]:
    yield {
        "jsonrpc": "2.0",
        "id": "r1",
        "result": {
            "kind": "artifact-update",
            "taskId": "t1",
            "contextId": "c1",
            "artifact": {"parts": [{"kind": "text", "text": _REPLY}]},
        },
    }
    yield _status_event("completed")


async def _url_agent_sdk_stream() -> AsyncIterator[object]:
    from a2a.compat.v0_3.types import SendStreamingMessageResponse

    for event in (
        _status_event("working", "Looking that up"),
        _artifact_event(_REPLY, append=False),
        _status_event("completed"),
    ):
        yield SendStreamingMessageResponse.model_validate(event)


@pytest.mark.parametrize(
    ("agent_stream", "reply"),
    (
        pytest.param(_snapshot_stream, _REPLY, id="replacing_snapshots"),
        pytest.param(_watsonx_stream, _REPLY, id="watsonx_chunks"),
        pytest.param(_two_artifact_stream, f"{_FIRST_ARTIFACT} {_SECOND_ARTIFACT}", id="two_artifacts"),
        pytest.param(_status_message_stream, _REPLY, id="status_message_reply"),
        pytest.param(_artifact_without_id_stream, _REPLY, id="artifact_without_id"),
        pytest.param(
            _url_agent_sdk_stream,
            _REPLY,
            id="url_agent_sdk_events",
            marks=pytest.mark.skipif(importlib.util.find_spec("a2a") is None, reason="needs the optional a2a-sdk"),
        ),
    ),
)
@pytest.mark.asyncio
async def test_stream_counts_the_whole_reply_once(
    agent_stream: Callable[[], AsyncIterator[object]], reply: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    logging_obj: Final = LitellmLogging(
        model="a2a_agent/trip-planner",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="asend_message_streaming",
        start_time=time.time(),
        litellm_call_id="artifact-count",
        function_id="artifact-count",
    )
    iterator: Final = A2AStreamingIterator(
        stream=agent_stream(),
        request=SimpleNamespace(params=SimpleNamespace(message={"role": "user", "parts": []})),
        logging_obj=logging_obj,
        agent_name="trip-planner",
    )

    _ = [chunk async for chunk in iterator]

    assert logging_obj.model_call_details["usage"].completion_tokens == A2ARequestUtils.count_tokens(reply)
