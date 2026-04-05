import asyncio
from datetime import datetime
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.responses import streaming_iterator as streaming_module
from litellm.responses.streaming_iterator import (
    CachedResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import CallTypes


class _FakeLoggingObj:
    def __init__(self):
        self.success_calls = 0
        self.async_success_calls = 0
        self.failure_calls = 0
        self.async_failure_calls = 0
        self.last_success_kwargs = None
        self.last_async_success_kwargs = None
        self.start_time = datetime.now()
        self.model_call_details = {"litellm_params": {}}

    # Signature alignment with Logging handlers
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
    chunk = SimpleNamespace(
        type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, response=None
    )
    chunk = await streaming_module.call_post_streaming_hooks_for_testing(
        iterator, chunk
    )
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
    logging_obj._llm_caching_handler = caching_handler

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
    assert (
        litellm.cache.async_add_cache.call_args.kwargs["cache_key"]
        == "responses-stream-cache-key"
    )
    assert "metadata" not in litellm.cache.async_add_cache.call_args.kwargs
    assert "custom_llm_provider" not in litellm.cache.async_add_cache.call_args.kwargs
    assert (
        json.loads(litellm.cache.async_add_cache.call_args.args[0])["id"]
        == iterator.completed_response.response.id
    )
    litellm.cache = original_cache


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
    logging_obj._llm_caching_handler = caching_handler

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
    assert (
        litellm.cache.add_cache.call_args.kwargs["cache_key"]
        == "responses-stream-cache-key"
    )
    assert "metadata" not in litellm.cache.add_cache.call_args.kwargs
    assert "custom_llm_provider" not in litellm.cache.add_cache.call_args.kwargs
    assert (
        json.loads(litellm.cache.add_cache.call_args.args[0])["id"]
        == iterator.completed_response.response.id
    )
    litellm.cache = original_cache


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
    logging_obj._llm_caching_handler = SimpleNamespace(
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
    logging_obj._llm_caching_handler = SimpleNamespace(
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
