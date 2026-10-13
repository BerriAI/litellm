import asyncio
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import unquote

import httpx
import pytest
from _pytest.mark.structures import ParameterSet
from integration._support import claude_code as cc
from integration._support import responses_vendor as rv
from integration._support.anthropic_thinking import BEDROCK_MODEL, Event, sse_chunks
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_CLAUDE: Final = cc.OPUS
_CLAUDE_WITHOUT_TOOL_SEARCH: Final = "claude-scripted-unlisted"
_GEMINI: Final = "gemini-2.5-flash"
_MISTRAL: Final = "mistral-large-latest"
_REGEX_TOOL_TYPE: Final = "tool_search_tool_regex_20251119"
_REGEX_TOOL_NAME: Final = "tool_search_tool_regex"
_ALL_DEFERRED_ERROR: Final = "At least one tool must have defer_loading=false"
_DEFERRED_NAMES: Final = ("get_weather", "finance__get_stock_price")
_CACHE_BUST: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ITEMS: Final = TypeAdapter(list[dict[str, JsonValue]])
_WEATHER_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_TICKER_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"ticker": {"type": "string"}},
    "required": ["ticker"],
}
_EMAIL_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"to": {"type": "string"}},
    "required": ["to"],
}

Route = Literal["gemini", "bedrock_converse", "mistral", "unlisted_claude", "anthropic"]
Search = Literal["absent", "dropped"]


@dataclass(frozen=True, slots=True)
class _Call:
    route: Route
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


def _prompt(marker: str) -> str:
    return f"Weather in Paris and the ACME price marker-{marker}"


def _tool_search(**fields: JsonValue) -> dict[str, JsonValue]:
    return {"type": "tool_search", **fields}


def _function_tool(defer_loading: JsonValue = True) -> dict[str, JsonValue]:
    return {
        "type": "function",
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": _WEATHER_PARAMETERS,
        "defer_loading": defer_loading,
    }


def _loaded_tool() -> dict[str, JsonValue]:
    return {"type": "function", "name": "send_email", "description": "Send an email", "parameters": _EMAIL_PARAMETERS}


def _namespace_tool() -> dict[str, JsonValue]:
    return {
        "type": "namespace",
        "name": "finance",
        "description": "Finance tools",
        "tools": [
            {
                "type": "function",
                "name": "get_stock_price",
                "description": "Price of a ticker",
                "parameters": _TICKER_PARAMETERS,
                "defer_loading": True,
            }
        ],
    }


def _search_tools(route: Route, search: Search) -> tuple[dict[str, JsonValue], ...]:
    if search == "absent":
        return ()
    return (_tool_search(execution="client"),) if route == "anthropic" else (_tool_search(),)


def _gemini_frame(marker: str) -> dict[str, JsonValue]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": f"loaded marker-{marker}"}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
        "modelVersion": _GEMINI,
    }


def _chat_completion(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"cmpl-{marker}",
        "object": "chat.completion",
        "created": 1,
        "model": _MISTRAL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"loaded marker-{marker}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _chat_chunks(marker: str) -> tuple[bytes, ...]:
    head: Final[dict[str, JsonValue]] = {
        "id": f"cmpl-{marker}",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _MISTRAL,
    }
    return (
        rv.chat_sse(
            {
                **head,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": f"loaded marker-{marker}"},
                        "finish_reason": None,
                    }
                ],
            }
        ),
        rv.chat_sse(
            {
                **head,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }
        ),
        b"data: [DONE]\n\n",
    )


def _claude_message(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{marker}",
        "type": "message",
        "role": "assistant",
        "model": _CLAUDE,
        "content": [{"type": "text", "text": f"loaded marker-{marker}"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 7},
    }


def _claude_events(marker: str) -> tuple[Event, ...]:
    opened: Final = {**_claude_message(marker), "content": [], "stop_reason": None}
    return (
        {"type": "message_start", "message": opened},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": f"loaded marker-{marker}"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    )


def _anthropic_error(status: int, message: str) -> Reply:
    body: Final = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return Reply(status=status, body=json.dumps(body).encode())


def _request_tools(body: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return _ITEMS.validate_python(body.get("tools") or [])


def _gemini(request: Request) -> Reply:
    marker: Final = rv.newest_marker(request.body.decode()) or "health"
    if f"/models/{_GEMINI}:streamGenerateContent" in request.target:
        return Reply(content_type="text/event-stream", chunks=(rv.chat_sse(_gemini_frame(marker)),))
    if f"/models/{_GEMINI}:generateContent" in request.target:
        return Reply(body=json.dumps(_gemini_frame(marker)).encode())
    return rv.error(404, f"unexpected target {request.target}", "not_found")


def _converse(request: Request) -> Reply:
    marker: Final = rv.newest_marker(request.body.decode()) or "health"
    if unquote(request.target) != f"/model/{BEDROCK_MODEL}/converse":
        return rv.error(404, f"unexpected target {request.target}", "not_found")
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": f"converse marker-{marker}"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def _mistral(request: Request) -> Reply:
    marker: Final = rv.newest_marker(request.body.decode()) or "health"
    if request.target != "/v1/chat/completions":
        return rv.error(404, f"unexpected target {request.target}", "not_found")
    body: Final = _JSON_OBJECT.validate_json(request.body)
    deferred: Final = [index for index, tool in enumerate(_request_tools(body)) if "defer_loading" in tool]
    if deferred:
        return rv.error(
            422, f"tools.{deferred[0]}.defer_loading: Extra inputs are not permitted marker-{marker}", "extra_forbidden"
        )
    if body.get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=_chat_chunks(marker))
    return Reply(body=json.dumps(_chat_completion(marker)).encode())


def _claude(request: Request) -> Reply:
    if request.method != "POST":
        return Reply()
    marker: Final = rv.newest_marker(request.body.decode()) or "health"
    body: Final = _JSON_OBJECT.validate_json(request.body)
    tools: Final = _request_tools(body)
    regex_tools: Final = [tool for tool in tools if tool.get("type") == _REGEX_TOOL_TYPE]
    function_tools: Final = [tool for tool in tools if tool.get("type") != _REGEX_TOOL_TYPE]
    if not regex_tools and function_tools and all(tool.get("defer_loading") is True for tool in function_tools):
        return _anthropic_error(400, f"tools: {_ALL_DEFERRED_ERROR} marker-{marker}")
    if body.get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=sse_chunks(_claude_events(marker)))
    return Reply(body=json.dumps(_claude_message(marker)).encode())


def _vendor(route: Route) -> Callable[[Request], Reply]:
    match route:
        case "gemini":
            return _gemini
        case "bedrock_converse":
            return _converse
        case "mistral":
            return _mistral
        case "unlisted_claude" | "anthropic":
            return _claude


def _deployment(scenario: Scenario, route: Route, wire_url: str) -> str:
    match route:
        case "gemini":
            return scenario.model(model=f"gemini/{_GEMINI}", api_base=wire_url, api_key="synthetic-gemini-key")
        case "bedrock_converse":
            return scenario.model(
                model=f"bedrock/converse/{BEDROCK_MODEL}",
                aws_access_key_id="AKIASCRIPTEDPROVIDER",
                aws_secret_access_key="scripted-secret",
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=wire_url,
            )
        case "mistral":
            return scenario.model(model=f"mistral/{_MISTRAL}", api_base=wire_url, api_key="synthetic-mistral-key")
        case "unlisted_claude":
            return scenario.model(
                model=f"anthropic/{_CLAUDE_WITHOUT_TOOL_SEARCH}", api_base=wire_url, api_key=cc.ANTHROPIC_API_KEY
            )
        case "anthropic":
            return scenario.model(model=f"anthropic/{_CLAUDE}", api_base=wire_url, api_key=cc.ANTHROPIC_API_KEY)


def _responses_body(
    model: str, marker: str, tools: Sequence[Mapping[str, JsonValue]], stream: bool
) -> dict[str, JsonValue]:
    return {
        "model": model,
        "input": _prompt(marker),
        "tools": [dict(tool) for tool in tools],
        "stream": stream,
        **_CACHE_BUST,
    }


def _raw(gateway: Gateway, body: Mapping[str, JsonValue]) -> httpx.Response:
    with httpx.Client(base_url=str(gateway.client.base_url).rstrip("/"), trust_env=False, timeout=60) as client:
        headers: Final = {"Authorization": f"Bearer {gateway.key}"}
        with client.stream("POST", "/v1/responses", json=dict(body), headers=headers) as response:
            response.read()
            return response


def _assert_served(response: httpx.Response, marker: str, stream: bool) -> None:
    assert response.status_code == 200, response.text
    assert f"marker-{marker}" in response.text, response.text
    if stream:
        assert "response.completed" in response.text, response.text
        return
    types: Final = [
        item["type"] for item in _ITEMS.validate_python(_JSON_OBJECT.validate_json(response.content)["output"])
    ]
    assert "message" in types, response.text


def _only_request(wire: Wire) -> tuple[Request, dict[str, JsonValue]]:
    received: Final = wire.drain()
    assert len(received) == 1, [(request.method, request.target) for request in received]
    return received[0], _JSON_OBJECT.validate_json(received[0].body)


def _assert_target(route: Route, stream: bool, request: Request, body: Mapping[str, JsonValue]) -> None:
    match route:
        case "gemini":
            method: Final = "streamGenerateContent" if stream else "generateContent"
            assert f"/models/{_GEMINI}:{method}" in request.target, request.target
        case "bedrock_converse":
            assert unquote(request.target) == f"/model/{BEDROCK_MODEL}/converse", request.target
        case "mistral":
            assert request.target == "/v1/chat/completions", request.target
            assert body.get("stream", False) is stream, body
        case "unlisted_claude" | "anthropic":
            assert request.target == "/v1/messages", request.target
            assert body.get("stream", False) is stream, body


def _wire_tool_names(route: Route, body: Mapping[str, JsonValue]) -> list[str]:
    match route:
        case "gemini":
            declarations: Final = _ITEMS.validate_python(
                _ITEMS.validate_python(body["tools"])[0]["function_declarations"]
            )
            return [str(declaration["name"]) for declaration in declarations]
        case "bedrock_converse":
            specs: Final = _ITEMS.validate_python(_JSON_OBJECT.validate_python(body["toolConfig"])["tools"])
            return [str(_JSON_OBJECT.validate_python(spec["toolSpec"])["name"]) for spec in specs]
        case "mistral":
            return [
                str(_JSON_OBJECT.validate_python(tool["function"])["name"])
                for tool in _ITEMS.validate_python(body["tools"])
            ]
        case "unlisted_claude" | "anthropic":
            return [str(tool["name"]) for tool in _ITEMS.validate_python(body["tools"])]


def _assert_undeferred_wire(call: _Call, request: Request, body: Mapping[str, JsonValue]) -> None:
    _assert_target(call.route, call.stream, request, body)
    assert _wire_tool_names(call.route, body) == list(_DEFERRED_NAMES), body
    assert "defer_loading" not in request.body.decode(), body
    assert "tool_search" not in request.body.decode(), body


def _modes(route: Route) -> tuple[bool, ...]:
    return (False,) if route == "bedrock_converse" else (False, True)


def _cells() -> tuple[ParameterSet, ...]:
    routes: Final[tuple[Route, ...]] = ("gemini", "bedrock_converse", "mistral", "unlisted_claude", "anthropic")
    searches: Final[tuple[Search, ...]] = ("absent", "dropped")
    combos: Final = [
        (route, search, stream) for route in routes for search in searches for stream in _modes(route)
    ]  # comprehension-ok: a parametrize grid
    return tuple(
        pytest.param(route, search, stream, id=f"{route}-{search}-{'stream' if stream else 'sync'}")
        for route, search, stream in combos
    )


@pytest.mark.parametrize(("route", "search", "stream"), _cells())
def test_without_a_search_tool_on_the_wire_every_tool_arrives_without_defer_loading(
    gateway: Gateway, route: Route, search: Search, stream: bool
) -> None:
    call: Final = _Call(route, stream, uuid.uuid4().hex)
    with wire_server(_vendor(route)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, route, wire.url)
        tools: Final = (*_search_tools(route, search), _function_tool(), _namespace_tool())
        response: Final = _raw(gateway, _responses_body(model, call.marker, tools, stream))
        _assert_served(response, call.marker, stream)
        request, body = _only_request(wire)
        _assert_undeferred_wire(call, request, body)


def test_a_false_defer_loading_never_reaches_an_openai_compatible_wire_as_a_key(gateway: Gateway) -> None:
    call: Final = _Call("mistral", False, uuid.uuid4().hex)
    with wire_server(_mistral) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "mistral", wire.url)
        tools: Final = (_function_tool(defer_loading=False), _namespace_tool())
        response: Final = _raw(gateway, _responses_body(model, call.marker, tools, False))
        _assert_served(response, call.marker, False)
        request, body = _only_request(wire)
        _assert_undeferred_wire(call, request, body)


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
def test_the_supported_anthropic_route_keeps_each_tools_own_defer_loading_with_the_search_tool(
    gateway: Gateway, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_claude) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url)
        tools: Final = (_tool_search(), _function_tool(), _namespace_tool(), _loaded_tool())
        response: Final = _raw(gateway, _responses_body(model, marker, tools, stream))
        _assert_served(response, marker, stream)
        request, body = _only_request(wire)
        _assert_target("anthropic", stream, request, body)
        sent: Final = _ITEMS.validate_python(body["tools"])
        assert sent[0] == {"type": _REGEX_TOOL_TYPE, "name": _REGEX_TOOL_NAME}, sent
        assert [(tool["name"], tool.get("defer_loading")) for tool in sent[1:]] == [
            ("get_weather", True),
            ("finance__get_stock_price", True),
            ("send_email", None),
        ], sent
        assert "defer_loading" not in json.dumps(sent[3]), sent


@pytest.mark.parametrize("route", ["gemini", "mistral"])
def test_a_non_boolean_defer_loading_is_refused_with_400_on_a_route_without_tool_search(
    gateway: Gateway, route: Route
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_vendor(route)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, route, wire.url)
        refused: Final = _raw(
            gateway, _responses_body(model, marker, (_function_tool("true"), _namespace_tool()), False)
        )
        assert refused.status_code == 400 and "defer_loading must be a boolean" in refused.text, refused.text
        assert wire.drain() == (), refused.text
        follow_up: Final = _Call(route, False, uuid.uuid4().hex)
        served: Final = _raw(
            gateway, _responses_body(model, follow_up.marker, (_function_tool(), _namespace_tool()), False)
        )
        _assert_served(served, follow_up.marker, False)
        request, body = _only_request(wire)
        _assert_undeferred_wire(follow_up, request, body)


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    body: Final = _responses_body(model, call.marker, (_function_tool(), _namespace_tool()), call.stream)
    async with client.stream(
        "POST", "/v1/responses", json=body, headers={"Authorization": f"Bearer {key}"}
    ) as response:
        raw: Final = await response.aread()
    return _Served(call, response.status_code, raw.decode())


async def _burst(base_url: str, key: str, models: Mapping[Route, str], calls: Sequence[_Call]) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        return tuple(await asyncio.gather(*(_send(client, key, models[call.route], call) for call in calls)))


def _burst_calls() -> tuple[_Call, ...]:
    routes: Final[tuple[Route, ...]] = ("gemini", "bedrock_converse", "mistral", "unlisted_claude")
    return tuple(
        _Call(route, stream, uuid.uuid4().hex) for route in routes for stream in _modes(route)
    )  # comprehension-ok: a two-axis burst grid


def _marked(received: Sequence[Request], marker: str) -> tuple[Request, dict[str, JsonValue]]:
    (request,) = [request for request in received if marker in request.body.decode()]
    return request, _JSON_OBJECT.validate_json(request.body)


def test_a_burst_across_four_routes_strips_every_deferral(gateway: Gateway) -> None:
    calls: Final = _burst_calls()
    with (
        wire_server(_gemini) as gemini_wire,
        wire_server(_converse) as converse_wire,
        wire_server(_mistral) as mistral_wire,
        wire_server(_claude) as claude_wire,
        gateway.scenario() as scenario,
    ):
        wires: Final[dict[Route, Wire]] = {
            "gemini": gemini_wire,
            "bedrock_converse": converse_wire,
            "mistral": mistral_wire,
            "unlisted_claude": claude_wire,
        }
        models: Final = {route: _deployment(scenario, route, wire.url) for route, wire in wires.items()}
        served: Final = asyncio.run(_burst(str(gateway.client.base_url).rstrip("/"), gateway.key, models, calls))
        assert [item.status for item in served] == [200] * len(calls), [
            (item.status, item.text[:200]) for item in served
        ]
        received: Final = {route: wire.drain() for route, wire in wires.items()}
        for item in served:
            assert f"marker-{item.call.marker}" in item.text, item.text
            request, body = _marked(received[item.call.route], item.call.marker)
            _assert_undeferred_wire(item.call, request, body)
