"""
Regression test for LIT-4210: completing an A2A stream must not run the sync
success_handler on the thread-pool executor concurrently with
async_success_handler (cross-thread pydantic mutation segfaults pydantic-core).
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

import litellm
from litellm.a2a_protocol import streaming_iterator as a2a_streaming_iterator_module
from litellm.a2a_protocol.streaming_iterator import A2AStreamingIterator
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
    monkeypatch.setattr(a2a_streaming_iterator_module, "executor", recording_executor, raising=False)

    recorder = RecordingCustomLogger()
    litellm.success_callback = [recorder]
    litellm._async_success_callback = [recorder]

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
    from tests.test_litellm.litellm_core_utils.event_loop_lag import (
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
