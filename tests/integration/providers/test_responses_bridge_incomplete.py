import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue


_WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _message_item(item_id: str, text: str) -> dict[str, JsonValue]:
    return {
        "type": "message",
        "id": item_id,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


_WEATHER_CALL: Final[dict[str, JsonValue]] = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "get_weather",
    "arguments": '{"city":"Paris"}',
    "status": "completed",
}

_TWO_MESSAGES_THEN_CALL: Final = (
    _message_item("msg_first", "First message."),
    _message_item("msg_second", "Second message."),
    _WEATHER_CALL,
)


def test_chat_over_responses_deployment_folds_two_messages_and_function_call_into_one_choice(
    gateway: Gateway,
) -> None:
    identity: Final = "responses-bridge-two-messages-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_two_messages",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "output": list(_TWO_MESSAGES_THEN_CALL),
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "First message.Second message.",
                    "tool_calls": [
                        {
                            "id": "fc_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            "index": 0,
                        }
                    ],
                },
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]


def _responses_frame(event: str, payload: dict[str, JsonValue]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _message_item_frames(output_index: int, item: dict[str, JsonValue]) -> tuple[bytes, ...]:
    item_id: Final = item["id"]
    text: Final = object_value(list_value(item["content"])[0])["text"]
    return (
        _responses_frame(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {**item, "status": "in_progress", "content": []},
            },
        ),
        _responses_frame(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            },
        ),
        _responses_frame(
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": output_index, "item": item},
        ),
    )


def _function_call_frames(output_index: int, item: dict[str, JsonValue]) -> tuple[bytes, ...]:
    return (
        _responses_frame(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {**item, "status": "in_progress", "arguments": ""},
            },
        ),
        _responses_frame(
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"],
                "output_index": output_index,
                "delta": item["arguments"],
            },
        ),
        _responses_frame(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"],
                "output_index": output_index,
                "arguments": item["arguments"],
            },
        ),
        _responses_frame(
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": output_index, "item": item},
        ),
    )


def _sse_payloads(lines: list[str]) -> list[dict[str, JsonValue]]:
    return [json.loads(line[6:]) for line in lines if line.startswith("data: ") and line != "data: [DONE]"]


def test_chat_stream_over_responses_deployment_folds_two_messages_and_function_call_into_one_choice(
    gateway: Gateway,
) -> None:
    identity: Final = "responses-bridge-two-messages-stream-" + uuid.uuid4().hex
    response_payload: Final[dict[str, JsonValue]] = {
        "id": "resp_two_messages_stream",
        "object": "response",
        "created_at": 1789788253,
        "model": "gpt-6-sol",
        "status": "in_progress",
        "output": [],
    }

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/responses", request.target
        assert json.loads(request.body)["stream"] is True, request.body
        return Reply(
            content_type="text/event-stream",
            chunks=(
                _responses_frame("response.created", {"type": "response.created", "response": response_payload}),
                *_message_item_frames(0, _TWO_MESSAGES_THEN_CALL[0]),
                *_message_item_frames(1, _TWO_MESSAGES_THEN_CALL[1]),
                *_function_call_frames(2, _TWO_MESSAGES_THEN_CALL[2]),
                _responses_frame(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": {
                            **response_payload,
                            "status": "completed",
                            "output": list(_TWO_MESSAGES_THEN_CALL),
                            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                        },
                    },
                ),
            ),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-6-sol", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [_WEATHER_TOOL],
                "stream": True,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        chunks: Final = _sse_payloads(list(response.iter_lines()))
        choices: Final = [object_value(choice) for chunk in chunks for choice in list_value(chunk["choices"])]
        assert [choice["index"] for choice in choices] == [0] * len(choices), chunks
        deltas: Final = [object_value(choice["delta"]) for choice in choices]
        assert (
            "".join(str(delta["content"]) for delta in deltas if delta.get("content"))
            == "First message.Second message."
        )
        tool_calls: Final = [
            object_value(call)
            for delta in deltas
            if delta.get("tool_calls")
            for call in list_value(delta["tool_calls"])
        ]
        assert [
            object_value(call["function"])["name"] for call in tool_calls if object_value(call["function"]).get("name")
        ] == ["get_weather"], chunks
        assert (
            "".join(str(object_value(call["function"]).get("arguments") or "") for call in tool_calls)
            == '{"city":"Paris"}'
        ), chunks
        assert [choice["finish_reason"] for choice in choices if choice.get("finish_reason")] == ["tool_calls"], chunks
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
