import asyncio
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.responses import streaming_iterator as streaming_module
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import CallTypes


class _FakeLoggingObj:
    def __init__(self):
        self.success_calls = 0
        self.async_success_calls = 0
        self.failure_calls = 0
        self.async_failure_calls = 0
        self.start_time = datetime.now()
        self.model_call_details = {"litellm_params": {}}

    # Signature alignment with Logging handlers
    def success_handler(self, *args, **kwargs):
        self.success_calls += 1

    async def async_success_handler(self, *args, **kwargs):
        self.async_success_calls += 1

    def failure_handler(self, *args, **kwargs):
        self.failure_calls += 1

    async def async_failure_handler(self, *args, **kwargs):
        self.async_failure_calls += 1


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


@pytest.mark.asyncio
async def test_responses_streaming_calls_post_streaming_deployment_hook(monkeypatch):
    """
    Ensure per-chunk streaming deployment hook can modify chunks.
    """

    class _HookLogger(CustomLogger):
        async def async_post_call_streaming_deployment_hook(
            self, request_data, response_chunk, call_type
        ):
            response_chunk.tagged = True
            return response_chunk

    # Set callbacks to our fake hook
    original_callbacks = litellm.callbacks
    litellm.callbacks = [_HookLogger()]

    logging_obj = _FakeLoggingObj()

    class _StubConfig:
        def transform_streaming_response(self, **kwargs):
            return SimpleNamespace(
                type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, response=None
            )

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

    with pytest.raises(ValueError):
        iterator._process_chunk('{"delta": "chunk"}')

    # allow failure callbacks to run
    await asyncio.sleep(0.2)
    assert logging_obj.failure_calls >= 1
    assert logging_obj.async_failure_calls >= 1
