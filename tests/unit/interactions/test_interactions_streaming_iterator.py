"""
Regression test for LIT-4210: completing an async Interactions API stream must
not run the sync success_handler on the thread-pool executor concurrently with
async_success_handler (cross-thread pydantic mutation segfaults pydantic-core).
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Final

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.interactions import streaming_iterator as interactions_streaming_iterator_module
from litellm.interactions.streaming_iterator import InteractionsAPIStreamingIterator
from litellm.litellm_core_utils import thread_pool_executor as thread_pool_executor_module
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.llms.gemini.interactions.transformation import GoogleAIStudioInteractionsConfig
from litellm.types.interactions import InteractionsAPIStreamingResponse


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


@pytest.mark.asyncio
async def test_custom_logger_only_never_submits_sync_success_handler(monkeypatch):
    recording_executor = RecordingExecutor(thread_pool_executor_module.executor)
    monkeypatch.setattr(thread_pool_executor_module, "executor", recording_executor)
    monkeypatch.setattr(interactions_streaming_iterator_module, "executor", recording_executor)

    recorder = RecordingCustomLogger()
    litellm.success_callback = [recorder]
    litellm._async_success_callback = [recorder]

    logging_obj = LitellmLogging(
        model="gemini/gemini-3-pro-preview",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="ainteraction",
        start_time=time.time(),
        litellm_call_id="lit-4210-test",
        function_id="lit-4210-test",
    )
    iterator = InteractionsAPIStreamingIterator(
        response=httpx.Response(200),
        model="gemini/gemini-3-pro-preview",
        interactions_api_config=None,
        logging_obj=logging_obj,
    )
    iterator.completed_response = InteractionsAPIStreamingResponse()

    existing_tasks = asyncio.all_tasks()
    iterator._handle_logging_completed_response()
    spawned_tasks = asyncio.all_tasks() - existing_tasks
    await asyncio.wait_for(asyncio.gather(*spawned_tasks, return_exceptions=True), timeout=5)

    assert recorder.async_hook_fired is True
    assert recording_executor.submitted_for(logging_obj) == []


class _SuccessRecorder(CustomLogger):
    def __init__(self):
        super().__init__()
        self.logged: list[tuple[datetime, object]] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.logged.append((start_time, kwargs.get("completion_start_time")))


@pytest.mark.asyncio
async def test_stream_logs_its_first_event_time_measured_from_the_call_start():
    call_start: Final = datetime(2026, 10, 9, 12, 0, 0)
    recorder: Final = _SuccessRecorder()
    logging_obj: Final = LitellmLogging(
        model="gemini/gemini-3-pro-preview",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="ainteraction",
        start_time=call_start,
        litellm_call_id="interactions-stream",
        function_id="interactions-stream",
        dynamic_async_success_callbacks=[recorder],
    )
    events: Final = (
        {"event_type": "step.delta", "id": "interaction-1"},
        {"event_type": "interaction.completed", "id": "interaction-1", "status": "completed"},
    )
    response: Final = httpx.Response(200, content="".join(f"data: {json.dumps(e)}\n\n" for e in events).encode())
    iterator: Final = InteractionsAPIStreamingIterator(
        response=response,
        model="gemini/gemini-3-pro-preview",
        interactions_api_config=GoogleAIStudioInteractionsConfig(),
        logging_obj=logging_obj,
    )
    tasks_before: Final = asyncio.all_tasks()

    _ = await iterator.__anext__()
    first_event_at: Final = logging_obj.completion_start_time
    _ = [event async for event in iterator]
    await asyncio.gather(*(asyncio.all_tasks() - tasks_before))

    assert first_event_at is not None
    assert recorder.logged == [(call_start, first_event_at)]
