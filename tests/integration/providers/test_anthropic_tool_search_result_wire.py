import json
import uuid
from collections.abc import Sequence
from typing import Final

from integration._support.anthropic_thinking import sse_chunks
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "claude-opus-5-5"
_API_KEY: Final = "synthetic-anthropic-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SEARCH_ID: Final = "srvtoolu_01ScriptedSearch"
_WEATHER_ID: Final = "toolu_01ScriptedWeather"
_SEARCH_INPUT: Final = {"query": "weather"}
_SEARCH_RESULT: Final = {
    "type": "tool_search_tool_result",
    "tool_use_id": _SEARCH_ID,
    "content": {
        "type": "tool_search_tool_search_result",
        "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
    },
}
_TOOLS: Final = [
    {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
        "defer_loading": True,
    },
]
_USAGE: Final = {"input_tokens": 40, "output_tokens": 12}


def _search_turn_blocks() -> list[JsonValue]:
    return [
        {"type": "server_tool_use", "id": _SEARCH_ID, "name": "tool_search_tool_bm25", "input": _SEARCH_INPUT},
        _SEARCH_RESULT,
        {"type": "tool_use", "id": _WEATHER_ID, "name": "get_weather", "input": {"city": "Paris"}},
    ]


def _message(identity: str, content: Sequence[JsonValue], stop_reason: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": _MODEL,
        "content": list(content),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _USAGE,
    }


def _search_turn_events(identity: str) -> tuple[dict[str, JsonValue], ...]:
    return (
        {"type": "message_start", "message": {**_message(identity, [], "tool_use"), "stop_reason": None}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": _SEARCH_ID,
                "name": "tool_search_tool_bm25",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(_SEARCH_INPUT)},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": _SEARCH_RESULT},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": _WEATHER_ID, "name": "get_weather", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"city": "Paris"}'},
        },
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": _USAGE},
        {"type": "message_stop"},
    )


def _assistant_turn_on_the_wire(body: dict[str, JsonValue]) -> list[JsonValue]:
    messages: Final = body["messages"]
    assert isinstance(messages, list), body
    assistant: Final = messages[1]
    assert isinstance(assistant, dict) and assistant["role"] == "assistant", body
    content: Final = assistant["content"]
    assert isinstance(content, list), body
    return content


def _second_turn_messages(assistant: dict[str, JsonValue]) -> list[JsonValue]:
    return [
        {"role": "user", "content": "What is the weather in Paris?"},
        assistant,
        {"role": "tool", "tool_call_id": _WEATHER_ID, "content": "18C and cloudy"},
    ]


def _respond(identity: str, stream: bool, captured: list[dict[str, JsonValue]]):
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        captured.append(body)  # mutable-ok: the scripted upstream records each request it served
        if len(captured) == 2:
            return Reply(body=json.dumps(_message(identity, [{"type": "text", "text": "18C"}], "end_turn")).encode())
        if stream:
            return Reply(content_type="text/event-stream", chunks=sse_chunks(_search_turn_events(identity)))
        return Reply(body=json.dumps(_message(identity, _search_turn_blocks(), "tool_use")).encode())

    return respond


def _streamed_assistant(lines: Sequence[str]) -> dict[str, JsonValue]:
    deltas: Final = tuple(
        chunk["choices"][0]["delta"]
        for line in lines
        if line.startswith("data: ") and line != "data: [DONE]"
        for chunk in (json.loads(line.removeprefix("data: ")),)
        if chunk["choices"]
    )
    tool_calls: Final = {}  # mutable-ok: stream deltas are folded into one tool call per index
    for delta in deltas:
        for call in delta.get("tool_calls") or ():
            entry = tool_calls.setdefault(call["index"], {"id": None, "type": "function", "name": "", "arguments": ""})
            entry["id"] = call.get("id") or entry["id"]
            entry["name"] += call["function"].get("name") or ""
            entry["arguments"] += call["function"].get("arguments") or ""
    fields: Final = next(
        (
            delta["provider_specific_fields"]
            for delta in reversed(deltas)
            if (delta.get("provider_specific_fields") or {}).get("tool_results")
        ),
        {},
    )
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": entry["id"],
                "type": "function",
                "function": {"name": entry["name"], "arguments": entry["arguments"]},
            }
            for entry in tool_calls.values()
        ],
        "provider_specific_fields": fields,
    }


def _round_trip(gateway: Gateway, stream: bool) -> tuple[dict[str, JsonValue], list[JsonValue]]:
    identity: Final = f"anthropic-tool-search-{uuid.uuid4().hex}"
    captured: Final[list[dict[str, JsonValue]]] = []
    with wire_server(_respond(identity, stream, captured)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        first_request: Final = {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": _TOOLS,
            "stream": stream,
        }
        first: Final = gateway.request("POST", "/v1/chat/completions", first_request)
        assert first.status_code == 200, first.text
        assistant: Final = (
            _streamed_assistant(first.text.splitlines()) if stream else first.json()["choices"][0]["message"]
        )
        second: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {**first_request, "stream": False, "messages": _second_turn_messages(assistant)},
        )
        assert second.status_code == 200, second.text
        assert len(wire.drain()) == 2
    return assistant, _assistant_turn_on_the_wire(captured[1])


def _assert_search_replayed_in_place(assistant: dict[str, JsonValue], wire_content: list[JsonValue]) -> None:
    fields: Final = assistant["provider_specific_fields"]
    assert isinstance(fields, dict) and fields.get("tool_results") == [_SEARCH_RESULT], assistant
    assert wire_content == [
        {"type": "server_tool_use", "id": _SEARCH_ID, "name": "tool_search_tool_bm25", "input": _SEARCH_INPUT},
        _SEARCH_RESULT,
        {"type": "tool_use", "id": _WEATHER_ID, "name": "get_weather", "input": {"city": "Paris"}},
    ], wire_content


def test_chat_completions_replays_a_tool_search_result_right_after_its_search_call(gateway: Gateway) -> None:
    assistant, wire_content = _round_trip(gateway, stream=False)
    _assert_search_replayed_in_place(assistant, wire_content)


def test_streamed_chat_completions_replays_a_tool_search_result_right_after_its_search_call(gateway: Gateway) -> None:
    assistant, wire_content = _round_trip(gateway, stream=True)
    _assert_search_replayed_in_place(assistant, wire_content)
