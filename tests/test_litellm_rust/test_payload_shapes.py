import asyncio
import json
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm import diagnostics
from litellm.litellm_core_utils.payload_shapes import record_shape
from litellm.rust_bridge.streams import Stream, SyncStream
from tests.test_litellm_rust.support.cache import chunk_bytes
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service
from tests.test_litellm_rust.support.requests import MESSAGES_EVENTS, MESSAGES_MODEL

pytestmark = pytest.mark.requires_rust_extension
_BODY: Final = TypeAdapter(dict[str, JsonValue])
_EVENTS: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])


@pytest.mark.parametrize("asynchronous", (False, True))
def test_real_python_sdk_boundaries_export_shapes_only(monkeypatch, asynchronous):
    monkeypatch.setenv("LITELLM_RUST", "0")
    with recording_service() as collector, recording_service() as provider:
        collector.default_response = ResponseSpec(body={"status": 1})
        provider.default_response = ResponseSpec(body={
            "id": "private-response-id", "object": "chat.completion", "created": 0, "model": "diagnostic-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "private-answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        try:
            assert diagnostics.configure({
                "enabled": True, "payload_shapes": True,
                "policy": {"target_prefixes": ["litellm_payload_shapes", "litellm_inference_chat"]},
                "destinations": [{"transport": "posthog", "name": "shapes", "api_key": "phc_test", "endpoint": collector.base_url}],
            })
            kwargs: Final = {"messages": [{"role": "user", "content": "private-prompt"}], "api_key": "private-key", "api_base": provider.base_url}
            response: Final = asyncio.run(litellm.acompletion("openai/diagnostic-test", **kwargs)) if asynchronous else litellm.completion("openai/diagnostic-test", **kwargs)
            assert response.choices[0].message.content == "private-answer"
            assert diagnostics.force_flush()
        finally:
            assert diagnostics.shutdown()
        events: Final = _EVENTS.validate_python(_BODY.validate_json(collector.requests[0].raw_body)["batch"])
        shape_events: Final = tuple(event for event in events if event["event"] == "llm.payload.shape")
        fields: Final = tuple(_BODY.validate_python(_BODY.validate_python(event["properties"])["fields"]) for event in shape_events)
        native_events: Final = tuple(event for event in events if _BODY.validate_python(event["properties"])["target"] != "litellm_payload_shapes")
        assert native_events == ()
        stages: Final = tuple(field["payload.stage"] for field in fields)
        assert stages == ("litellm.request.received", "provider.request.transformed", "provider.request.sent", "provider.response.received", "litellm.response.normalized")
        assert len({field["trace_id"] for field in fields}) == 1
        assert "$['model']" in fields[0]["payload.field_paths"]
        assert "$['messages'][*]['content']" in fields[2]["payload.field_paths"]
        assert "$['choices'][*]['message']['content']" in fields[3]["payload.field_paths"]
        encoded: Final = json.dumps(shape_events)
        assert all(value not in encoded for value in ("private-answer", "private-prompt", "private-response-id", "private-key", provider.base_url))
        assert provider.requests[0].body["messages"][0]["content"] == "private-prompt"


@pytest.mark.parametrize("asynchronous", (False, True))
def test_admitted_native_messages_share_sdk_capture_without_duplicate_stages(monkeypatch, asynchronous):
    monkeypatch.setenv("LITELLM_RUST", "1")
    with recording_service() as collector, recording_service() as provider:
        collector.default_response = ResponseSpec(body={"status": 1})
        provider.default_response = ResponseSpec(body={
            "id": "private-response-id", "type": "message", "role": "assistant", "model": "diagnostic-test",
            "content": [{"type": "text", "text": "private-answer"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
        try:
            assert diagnostics.configure({
                "enabled": True, "payload_shapes": True,
                "policy": {"target_prefixes": ["litellm_payload_shapes", "litellm_inference_messages"]},
                "destinations": [{"transport": "posthog", "name": "shapes", "api_key": "phc_test", "endpoint": collector.base_url}],
            })
            kwargs: Final = {"model": "anthropic/diagnostic-test", "messages": [{"role": "user", "content": "private-prompt"}], "max_tokens": 16, "api_key": "private-key", "api_base": provider.base_url}
            response: Final = asyncio.run(litellm.anthropic.messages.acreate(**kwargs)) if asynchronous else litellm.anthropic.messages.create(**kwargs)
            assert _BODY.validate_python(response)["content"] == [{"type": "text", "text": "private-answer"}]
            assert diagnostics.force_flush()
        finally:
            assert diagnostics.shutdown()
        events: Final = _EVENTS.validate_python(_BODY.validate_json(collector.requests[0].raw_body)["batch"])
        shape_events: Final = tuple(event for event in events if event["event"] == "llm.payload.shape")
        fields: Final = tuple(_BODY.validate_python(_BODY.validate_python(event["properties"])["fields"]) for event in shape_events)
        assert any(_BODY.validate_python(event["properties"])["target"] == "litellm_inference_messages" for event in events)
        assert tuple(field["payload.stage"] for field in fields) == ("litellm.request.received", "provider.request.transformed", "provider.request.sent", "provider.response.received", "litellm.response.normalized")
        assert len({field["trace_id"] for field in fields}) == 1
        assert "$['content'][*]['text']" in fields[-1]["payload.field_paths"]
        encoded: Final = json.dumps(shape_events)
        assert all(value not in encoded for value in ("private-answer", "private-prompt", "private-response-id", "private-key", provider.base_url))


async def _consume_native_messages(arguments: dict[str, object], cancel: bool) -> bytes:
    stream: Final = await litellm.anthropic.messages.acreate(**arguments)
    assert isinstance(stream, Stream)
    if cancel:
        first: Final = chunk_bytes(await anext(stream))
        await stream.aclose()
        return first
    return b"".join([chunk_bytes(chunk) async for chunk in stream])


def _consume_sync_native_messages(arguments: dict[str, object], cancel: bool) -> bytes:
    stream: Final = litellm.anthropic.messages.create(**arguments)
    assert isinstance(stream, SyncStream)
    if cancel:
        first: Final = chunk_bytes(next(stream))
        stream.close()
        return first
    return b"".join(chunk_bytes(chunk) for chunk in stream)


@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("cancel", (False, True))
def test_native_sdk_stream_summaries_survive_delivery_and_close(monkeypatch, asynchronous, cancel):
    monkeypatch.setenv("LITELLM_RUST", "1")
    with recording_service() as collector, recording_service() as provider:
        collector.default_response = ResponseSpec(body={"status": 1})
        provider.default_response = ResponseSpec(body=None, events=MESSAGES_EVENTS)
        try:
            assert diagnostics.configure({
                "enabled": True, "payload_shapes": True,
                "policy": {"target_prefixes": ["litellm_payload_shapes", "litellm_inference_messages"]},
                "destinations": [{"transport": "posthog", "name": "shapes", "api_key": "phc_test", "endpoint": collector.base_url}],
            })
            arguments: Final = {"model": MESSAGES_MODEL, "messages": [{"role": "user", "content": "private-prompt"}], "max_tokens": 16, "stream": True, "api_key": "private-key", "api_base": provider.base_url}
            body: Final = asyncio.run(_consume_native_messages(arguments, cancel)) if asynchronous else _consume_sync_native_messages(arguments, cancel)
            assert b"message_start" in body
            if not cancel:
                assert b"Hello from native Messages" in body
                assert b"message_stop" in body
            assert diagnostics.force_flush()
        finally:
            assert diagnostics.shutdown()
        events: Final = _EVENTS.validate_python(_BODY.validate_json(collector.requests[0].raw_body)["batch"])
        shape_events: Final = tuple(event for event in events if event["event"] == "llm.payload.shape")
        fields: Final = tuple(_BODY.validate_python(_BODY.validate_python(event["properties"])["fields"]) for event in shape_events)
        assert any(_BODY.validate_python(event["properties"])["target"] == "litellm_inference_messages" for event in events)
        assert len(fields) == 5
        assert tuple(field["payload.stage"] for field in fields[:3]) == ("litellm.request.received", "provider.request.transformed", "provider.request.sent")
        assert {field["payload.stage"] for field in fields[3:]} == {"provider.response.received", "litellm.response.normalized"}
        assert len({field["trace_id"] for field in fields}) == 1
        outcomes: Final = ("cancelled", "cancelled") if cancel else ("success", "success")
        assert tuple(field["payload.outcome"] for field in fields[-2:]) == outcomes
        if not cancel:
            assert "$['delta']['text']" in fields[-1]["payload.field_paths"]
            assert "$['usage']['output_tokens']" in fields[-1]["payload.field_paths"]
        assert all(field["payload.shape_truncated"] is False for field in fields[-2:])
        encoded: Final = json.dumps(shape_events)
        assert all(value not in encoded for value in ("Hello from native Messages", "private-prompt", "private-key", provider.base_url))


@pytest.mark.parametrize("enabled,sample_rate", ((False, 1), (True, 0)))
def test_capture_opt_in_and_destination_sampling_are_independent(enabled, sample_rate):
    with recording_service() as collector:
        collector.expected_requests = 0
        try:
            assert diagnostics.configure({"enabled": True, "payload_shapes": enabled, "policy": {"sample_rate": sample_rate}, "destinations": [{"transport": "posthog", "name": "shapes", "api_key": "phc_test", "endpoint": collector.base_url}]})
            record_shape("litellm.request.received", {"input": "private"}, "call-id")
            assert diagnostics.force_flush()
        finally:
            assert diagnostics.shutdown()
        assert collector.requests == []
