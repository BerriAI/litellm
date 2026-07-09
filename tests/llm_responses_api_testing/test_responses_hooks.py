import asyncio
from contextlib import suppress
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
    MockResponsesAPIStreamingIterator,
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

    def _update_completion_start_time(self, completion_start_time):
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
    assert streaming_module._dump_response_object({"type": "message"}) == {
        "type": "message"
    }
    assert streaming_module._dump_response_object(object()) == {}


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


def test_process_chunk_recovers_output_from_stream_events_before_logging(monkeypatch):
    openai_types = streaming_module._get_openai_response_types()

    class _RecoveringConfig:
        def transform_streaming_response(self, **kwargs):
            parsed_chunk = kwargs["parsed_chunk"]
            event_type = parsed_chunk.get("type")
            if event_type == openai_types.ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE:
                return openai_types.OutputTextDoneEvent(
                    type=openai_types.ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
                    item_id="msg_1",
                    output_index=0,
                    content_index=0,
                    text="bridge-ok",
                )
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
                        input_tokens=10,
                        output_tokens=3,
                        total_tokens=13,
                    ),
                ),
            )

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_RecoveringConfig(),
        logging_obj=_FakeLoggingObj(),
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )
    completion_handler = MagicMock()
    monkeypatch.setattr(
        iterator, "_handle_logging_completed_response", completion_handler
    )

    iterator._process_chunk(
        json.dumps(
            {
                "type": "response.output_text.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": "bridge-ok",
            }
        )
    )
    event = iterator._process_chunk(
        json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_live",
                    "output": [],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "total_tokens": 13,
                    },
                },
            }
        )
    )

    assert event.response.output != []
    output_item = event.response.output[0]
    assert output_item["content"][0]["text"] == "bridge-ok"
    assert iterator.completed_response is event
    completion_handler.assert_called_once()


def test_record_recoverable_output_chunk_handles_output_item_done(monkeypatch):
    """OUTPUT_ITEM_DONE chunks populate _recovered_output_items (takes precedence
    over any synthetic text-only recovery at the same output_index)."""

    class _NoopConfig:
        def transform_streaming_response(self, **kwargs):
            return None

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_NoopConfig(),
        logging_obj=_FakeLoggingObj(),
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )

    real_item = {
        "type": "message",
        "id": "msg_real",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": "real-item-text",
                "annotations": [],
            }
        ],
    }
    iterator._process_chunk(
        json.dumps(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": real_item,
            }
        )
    )

    assert iterator._recovered_output_items[0] == real_item
    # Pre-seed a synthetic text-only item at the same index to verify the
    # OUTPUT_ITEM_DONE branch overwrites it on the next pass.
    iterator._recovered_text_only_items[0] = {
        "type": "message",
        "id": "msg_synthetic",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "synthetic", "annotations": []}],
    }
    iterator._process_chunk(
        json.dumps(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": real_item,
            }
        )
    )
    assert iterator._recovered_output_items[0] == real_item


def test_recover_output_on_completed_response_skips_when_existing_output_present():
    """If the terminal response already carries output, recovery is a no-op."""
    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=SimpleNamespace(),
        logging_obj=_FakeLoggingObj(),
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )
    # Seed the recovery buffers — these should be ignored when the response
    # object already has output.
    iterator._recovered_output_items[0] = {
        "type": "message",
        "id": "msg_real",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "real", "annotations": []}],
    }

    response_obj = ResponsesAPIResponse(
        id="resp_skip",
        created_at=int(datetime.now().timestamp()),
        status="completed",
        model="test-model",
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg_provider",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "provider-supplied",
                        "annotations": [],
                    }
                ],
            }
        ],
    )

    iterator._recover_output_on_completed_response(response_obj)

    # Existing output is untouched.
    assert len(response_obj.output) == 1
    assert response_obj.output[0].id == "msg_provider"


def test_recover_output_on_completed_response_noop_when_buffers_empty():
    """No recoverable items + empty response.output → no mutation, no crash."""
    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=SimpleNamespace(),
        logging_obj=_FakeLoggingObj(),
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )

    response_obj = ResponsesAPIResponse(
        id="resp_empty",
        created_at=int(datetime.now().timestamp()),
        status="completed",
        model="test-model",
        object="response",
        output=[],
    )

    iterator._recover_output_on_completed_response(response_obj)

    assert response_obj.output == []


def test_process_chunk_recovers_output_for_incomplete_terminal_event(monkeypatch):
    """RESPONSE_INCOMPLETE with empty output should also trigger recovery
    (the iterator recovery path runs before logging for incomplete/failed
    terminal events, not just RESPONSE_COMPLETED)."""
    openai_types = streaming_module._get_openai_response_types()

    class _IncompleteConfig:
        def transform_streaming_response(self, **kwargs):
            parsed_chunk = kwargs["parsed_chunk"]
            event_type = parsed_chunk.get("type")
            if event_type == openai_types.ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE:
                return openai_types.OutputTextDoneEvent(
                    type=openai_types.ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
                    item_id="msg_inc",
                    output_index=0,
                    content_index=0,
                    text="partial",
                )
            return openai_types.ResponseIncompleteEvent(
                type=openai_types.ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE,
                response=ResponsesAPIResponse(
                    id="resp_inc",
                    created_at=int(datetime.now().timestamp()),
                    status="incomplete",
                    model="test-model",
                    object="response",
                    output=[],
                    incomplete_details={"reason": "max_output_tokens"},
                ),
            )

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_IncompleteConfig(),
        logging_obj=_FakeLoggingObj(),
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )
    completion_handler = MagicMock()
    monkeypatch.setattr(
        iterator, "_handle_logging_completed_response", completion_handler
    )

    iterator._process_chunk(
        json.dumps(
            {
                "type": "response.output_text.done",
                "item_id": "msg_inc",
                "output_index": 0,
                "content_index": 0,
                "text": "partial",
            }
        )
    )
    event = iterator._process_chunk(
        json.dumps(
            {
                "type": "response.incomplete",
                "response": {"id": "resp_inc", "output": []},
            }
        )
    )

    assert event.response.output != []
    assert event.response.output[0]["content"][0]["text"] == "partial"
    assert iterator.completed_response is event
    completion_handler.assert_called_once()


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
    logging_obj._response_cost_calculator = MagicMock(return_value=1.23)
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
    monkeypatch.setattr(
        iterator, "_handle_logging_completed_response", completion_handler
    )

    try:
        # Chunk must include a top-level "response" key so BaseResponsesAPIStreamingIterator
        # runs _update_responses_api_response_id_with_model_id (see streaming_iterator.py).
        event = iterator._process_chunk(
            json.dumps({"type": "response.completed", "response": {"id": "resp_live"}})
        )
    finally:
        litellm.include_cost_in_streaming_usage = original_include_cost

    assert iterator.completed_response is event
    assert event.response.id != "resp_live"
    assert event.response.id.startswith("resp_")
    assert event.response.usage.cost == 1.23
    completion_handler.assert_called_once()


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
    logging_obj._response_cost_calculator = MagicMock(side_effect=RuntimeError("boom"))
    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="test-model",
        responses_api_provider_config=_CompletedConfig(),
        logging_obj=logging_obj,
        request_data={"foo": "bar"},
        call_type=CallTypes.responses.value,
    )
    completion_handler = MagicMock()
    monkeypatch.setattr(
        iterator, "_handle_logging_completed_response", completion_handler
    )

    try:
        event = iterator._process_chunk(json.dumps({"type": "response.completed"}))
    finally:
        litellm.include_cost_in_streaming_usage = original_include_cost

    assert iterator.completed_response is event
    assert event.response.usage.cost is None
    completion_handler.assert_called_once()


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
        logging_obj._llm_caching_handler = None
    else:
        logging_obj._llm_caching_handler = SimpleNamespace(
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
            _should_store_result_in_cache=lambda original_function, kwargs: (
                scenario != "store_disabled"
            ),
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


def test_build_synthetic_response_events_covers_annotations_function_calls_and_refusals():
    original_include_cost = litellm.include_cost_in_streaming_usage
    litellm.include_cost_in_streaming_usage = True
    logging_obj = _FakeLoggingObj()
    logging_obj._response_cost_calculator = MagicMock(side_effect=RuntimeError("boom"))
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
        events = streaming_module._build_synthetic_response_events(
            transformed=transformed,
            logging_obj=logging_obj,
            chunk_size=5,
        )
    finally:
        litellm.include_cost_in_streaming_usage = original_include_cost

    event_types = [
        event.type.value if hasattr(event.type, "value") else str(event.type)
        for event in events
    ]

    assert "response.output_text.annotation.added" in event_types
    assert "response.refusal.delta" in event_types
    assert "response.refusal.done" in event_types
    assert "response.function_call_arguments.delta" in event_types
    assert "response.function_call_arguments.done" in event_types
    assert event_types[-1] == "response.completed"


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
