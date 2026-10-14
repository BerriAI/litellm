import json
import uuid
from typing import Final

import pytest
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, string_value
from integration._support.openai_wire import answering_model_discovery, responses_reply
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(120)

_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_TOOL_SEARCH: Final[dict[str, JsonValue]] = {"type": "tool_search"}


def _function(name: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"name": name, "description": f"Look up {name}", "parameters": _PARAMETERS, **extra}


def _wire_function(name: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"type": "function", "strict": None, **_function(name), **extra}


@pytest.mark.parametrize(
    "deferred_tool",
    [
        {"type": "function", "function": _function("get_weather"), "defer_loading": True},
        {"type": "function", "function": _function("get_weather", defer_loading=True)},
    ],
    ids=["tool-level", "function-level"],
)
def test_chat_bridge_forwards_defer_loading_so_tool_search_has_a_deferred_tool(
    gateway: Gateway, deferred_tool: dict[str, JsonValue]
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(pcb.respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": pcb.prompt(marker)}],
                "tools": [
                    _TOOL_SEARCH,
                    deferred_tool,
                    {"type": "function", "function": _function("get_time")},
                    {"type": "function", "function": _function("get_news", defer_loading=False)},
                ],
                **pcb.NO_CACHE,
            },
        )
        assert response.status_code == 200, response.text
        message: Final = rv.JSON_OBJECT.validate_python(
            rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(response.text)["choices"])[0]["message"]
        )
        assert message["content"] == rv.answer(marker), response.text
        assert pcb.body_of(pcb.posted(wire, marker))["tools"] == [
            _TOOL_SEARCH,
            _wire_function("get_weather", defer_loading=True),
            _wire_function("get_time"),
            _wire_function("get_news", defer_loading=False),
        ]


_CALL_ID: Final = "call_weather"
_TOOL_SEARCH_CALL: Final[dict[str, JsonValue]] = {
    "id": "tsc_weather",
    "type": "tool_search_call",
    "arguments": {"query": "weather"},
    "execution": "server",
    "status": "completed",
}
_TOOL_SEARCH_OUTPUT: Final[dict[str, JsonValue]] = {
    "id": "tso_weather",
    "type": "tool_search_output",
    "execution": "server",
    "status": "completed",
    "tools": [_wire_function("get_weather", defer_loading=True)],
}
_WEATHER_CALL: Final[dict[str, JsonValue]] = {
    "type": "function_call",
    "call_id": _CALL_ID,
    "name": "get_weather",
    "arguments": '{"city": "Paris"}',
    "namespace": "get_weather",
}


def _searches_then_answers(request: Request) -> Reply:
    body: Final = pcb.body_of(request)
    marker: Final = rv.newest_marker(request.body.decode())
    identity: Final = f"resp_{marker}-{uuid.uuid4().hex}"
    if any(item.get("type") == "function_call_output" for item in pcb.input_items(request)):
        return responses_reply(identity, string_value(body["model"]), rv.answer(marker), stream=False)
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": body["model"],
                "output": [
                    _TOOL_SEARCH_CALL,
                    _TOOL_SEARCH_OUTPUT,
                    {**_WEATHER_CALL, "id": "fc_weather", "status": "completed"},
                ],
                "usage": rv.USAGE,
            }
        ).encode()
    )


def test_chat_bridge_replays_tool_search_items_on_the_next_turn(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    tools: Final = [_TOOL_SEARCH, {"type": "function", "function": _function("get_weather"), "defer_loading": True}]
    user_turn: Final[dict[str, JsonValue]] = {"role": "user", "content": pcb.prompt(marker)}
    with wire_server(answering_model_discovery(_searches_then_answers)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        first: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [user_turn], "tools": tools, **pcb.NO_CACHE}
        )
        assert first.status_code == 200, first.text
        assistant_turn: Final = rv.JSON_OBJECT.validate_python(
            rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(first.text)["choices"])[0]["message"]
        )
        assert rv.JSON_OBJECT.validate_python(assistant_turn["provider_specific_fields"])["tool_search_items"] == [
            _TOOL_SEARCH_CALL,
            _TOOL_SEARCH_OUTPUT,
        ], first.text
        pcb.posted(wire, marker)

        second: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [user_turn, assistant_turn, {"role": "tool", "tool_call_id": _CALL_ID, "content": "sunny"}],
                "tools": tools,
                **pcb.NO_CACHE,
            },
        )
        assert second.status_code == 200, second.text
        replayed: Final = [
            item
            for item in pcb.input_items(pcb.posted(wire, marker))
            if item["type"] not in ("message", "function_call_output")
        ]
        assert replayed == [_TOOL_SEARCH_CALL, _TOOL_SEARCH_OUTPUT, _WEATHER_CALL]
