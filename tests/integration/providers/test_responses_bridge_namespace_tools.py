import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
JSON_LIST: Final = TypeAdapter(list[dict[str, JsonValue]])
NAMESPACE: Final = "mcp__everything"
TOOL_NAME: Final = "get_sum"
FLATTENED_NAME: Final = f"{NAMESPACE}__{TOOL_NAME}"
CALL_ID: Final = "call_synthetic_get_sum"
ARGUMENTS: Final = json.dumps({"a": 2, "b": 3})
PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "required": ["a", "b"],
    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
}
NAMESPACE_TOOL: Final[dict[str, JsonValue]] = {
    "type": "namespace",
    "name": NAMESPACE,
    "description": "Tools exposed by the everything MCP server",
    "tools": [
        {
            "type": "function",
            "name": TOOL_NAME,
            "description": "Adds two numbers",
            "strict": False,
            "parameters": PARAMETERS,
        }
    ],
}
EXPECTED_CHAT_TOOLS: Final[list[JsonValue]] = [
    {
        "type": "function",
        "function": {
            "name": FLATTENED_NAME,
            "description": "Tools exposed by the everything MCP server\n\nAdds two numbers",
            "parameters": PARAMETERS,
            "strict": False,
        },
    }
]


def tool_call_completion(marker: str) -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{marker}",
            "object": "chat.completion",
            "created": 1789788253,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": CALL_ID,
                                "type": "function",
                                "function": {"name": FLATTENED_NAME, "arguments": ARGUMENTS},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        }
    ).encode()


def text_completion(marker: str) -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{marker}-final",
            "object": "chat.completion",
            "created": 1789788254,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "The sum is 5"},
                }
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45},
        }
    ).encode()


@pytest.mark.covers("other.provider_wire.responses_bridge.codex_namespace_tools_reach_chat_upstream_and_round_trip")
def test_codex_namespace_tool_is_flattened_for_chat_upstream_and_restored_in_responses_output(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    prompt: Final = f"add 2 and 3 {marker}"

    def chat_peer(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert body["tools"] == EXPECTED_CHAT_TOOLS, body
        messages: Final = JSON_LIST.validate_python(body["messages"])
        if len(messages) == 1:
            return Reply(body=tool_call_completion(marker))
        assert messages[1]["role"] == "assistant", messages
        history_calls: Final = JSON_LIST.validate_python(messages[1]["tool_calls"])
        assert [(call["id"], call["function"]) for call in history_calls] == [
            (CALL_ID, {"name": FLATTENED_NAME, "arguments": ARGUMENTS})
        ], messages
        assert messages[2] == {"role": "tool", "tool_call_id": CALL_ID, "content": "5"}, messages
        return Reply(body=text_completion(marker))

    with wire_server(chat_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="deepseek/gpt-4o-mini", api_base=wire.url + "/v1")
        first: Final = gateway.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": prompt, "tools": [NAMESPACE_TOOL], "store": False},
        )
        assert first.status_code == 200, first.text
        first_output: Final = JSON_LIST.validate_python(JSON_OBJECT.validate_json(first.content)["output"])
        calls: Final = tuple(item for item in first_output if item["type"] == "function_call")
        assert len(calls) == 1, first.text
        assert calls[0]["name"] == TOOL_NAME, first.text
        assert calls[0]["namespace"] == NAMESPACE, first.text
        assert calls[0]["call_id"] == CALL_ID, first.text
        assert calls[0]["arguments"] == ARGUMENTS, first.text

        second: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": [
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]},
                    {
                        "type": "function_call",
                        "call_id": CALL_ID,
                        "name": TOOL_NAME,
                        "namespace": NAMESPACE,
                        "arguments": ARGUMENTS,
                    },
                    {"type": "function_call_output", "call_id": CALL_ID, "output": "5"},
                ],
                "tools": [NAMESPACE_TOOL],
                "store": False,
            },
        )
        assert second.status_code == 200, second.text
        second_output: Final = JSON_LIST.validate_python(JSON_OBJECT.validate_json(second.content)["output"])
        assert [item["type"] for item in second_output] == ["message"], second.text
        assert JSON_LIST.validate_python(second_output[0]["content"])[0]["text"] == "The sum is 5", second.text
        assert len(wire.drain()) == 2
