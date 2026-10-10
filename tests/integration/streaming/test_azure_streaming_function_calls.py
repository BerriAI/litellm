from __future__ import annotations

import json
from typing import Final

from integration._support.client import Gateway, JSON_OBJECT, list_value, object_value
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

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


def _frame(delta: JsonValue, finish_reason: str | None = None) -> bytes:
    chunk: Final = {
        "id": "azure-stream-tool-call",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def test_azure_astreaming_and_function_calling(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target.startswith(
            "/openai/deployments/gpt-4.1-mini/chat/completions?api-version="
        )
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-4.1-mini"
        assert body["stream"] is True
        assert body["tools"] == [_TOOL]
        return Reply(
            content_type="text/event-stream",
            chunks=(
                _frame(
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
                    }
                ),
                _frame(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": 'ton","unit":"fahrenheit"}'},
                            }
                        ]
                    }
                ),
                _frame({}, finish_reason="tool_calls"),
                b"data: [DONE]\n\n",
            ),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-4.1-mini",
            api_base=wire.url,
            api_key="synthetic-azure-key",
            api_version="2024-02-15-preview",
        )
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Weather in Boston?"}],
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

    choices: Final = tuple(
        object_value(list_value(event["choices"])[0]) for event in events
    )
    deltas: Final = tuple(object_value(choice["delta"]) for choice in choices)
    tool_calls: Final = tuple(
        object_value(call)
        for delta in deltas
        for call in list_value(delta.get("tool_calls", []))
    )
    functions: Final = tuple(object_value(call["function"]) for call in tool_calls)
    arguments: Final = "".join(
        value["arguments"] for value in functions if isinstance(value.get("arguments"), str)
    )
    tool_names: Final = tuple(
        value["name"] for value in functions if isinstance(value.get("name"), str)
    )
    finish_reasons: Final = tuple(
        choice["finish_reason"]
        for choice in choices
        if isinstance(choice.get("finish_reason"), str)
    )

    assert tool_names == ("get_current_weather",)
    assert arguments == '{"location":"Boston","unit":"fahrenheit"}'
    assert finish_reasons == ("tool_calls",)
