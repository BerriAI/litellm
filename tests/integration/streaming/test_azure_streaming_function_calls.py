from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from typing import Final

from integration._support.client import JSON_OBJECT, Gateway, eventually, list_value, object_value
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue
from redis import Redis

_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}
_MESSAGES: Final = [{"role": "user", "content": "Weather in Boston?"}]


def _frame(stream_id: str, delta: JsonValue, finish_reason: str | None = None) -> bytes:
    chunk: Final = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _frames(stream_id: str) -> tuple[bytes, ...]:
    return (
        _frame(
            stream_id,
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-weather",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location":"Bos',
                        },
                    }
                ],
            },
        ),
        _frame(stream_id, {"tool_calls": [{"index": 0, "function": {"arguments": 'ton","unit":"fahrenheit"}'}}]}),
        _frame(stream_id, {}, finish_reason="tool_calls"),
        b"data: [DONE]\n\n",
    )


def _responder(stream_id: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2024-02-15-preview"
        assert request.headers["api-key"] == "synthetic-azure-key"
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-4.1-mini"
        assert body["messages"] == _MESSAGES
        assert body["stream"] is True
        assert body["tools"] == [_TOOL]
        assert body["tool_choice"] == "auto"
        return Reply(content_type="text/event-stream", chunks=_frames(stream_id))

    return respond


def _stored(stream_id: str) -> bool:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as store:
        return any(
            stream_id.encode() in (store.get(key) or b"")
            for key in store.scan_iter(count=1000)
            if store.type(key) == b"string"
        )


def _tool_call_stream(gateway: Gateway, model: str) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    with gateway.client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": _MESSAGES,
            "tools": [_TOOL],
            "tool_choice": "auto",
            "stream": True,
        },
        headers={"Authorization": f"Bearer {gateway.key}"},
    ) as response:
        assert response.status_code == 200, response.read().decode()
        events: Final = tuple(
            JSON_OBJECT.validate_json(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )
    choices: Final = tuple(object_value(list_value(event["choices"])[0]) for event in events)
    calls: Final = tuple(
        object_value(call)
        for choice in choices
        for call in list_value(object_value(choice["delta"]).get("tool_calls") or [])
    )
    functions: Final = tuple(object_value(call["function"]) for call in calls)
    return (
        tuple(str(function["name"]) for function in functions if isinstance(function.get("name"), str)),
        "".join(str(function["arguments"]) for function in functions if isinstance(function.get("arguments"), str)),
        tuple(str(choice["finish_reason"]) for choice in choices if isinstance(choice.get("finish_reason"), str)),
    )


def test_azure_astreaming_and_function_calling(gateway: Gateway) -> None:
    stream_id: Final = f"azure-stream-{uuid.uuid4().hex}"
    with wire_server(_responder(stream_id)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-4.1-mini",
            api_base=wire.url,
            api_key="synthetic-azure-key",
            api_version="2024-02-15-preview",
        )
        first: Final = _tool_call_stream(gateway, model)
        eventually(lambda: _stored(stream_id), bool)
        cached: Final = _tool_call_stream(gateway, model)
        received: Final = wire.drain()

    expected: Final = (
        ("get_current_weather",),
        '{"location":"Boston","unit":"fahrenheit"}',
        ("tool_calls",),
    )
    assert first == expected
    assert cached == expected
    assert len(received) == 1
