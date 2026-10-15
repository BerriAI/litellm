import asyncio
import itertools
import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import EllipsisType, MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import unquote

import anthropic
import httpx
import openai
import pytest
from _pytest.mark.structures import ParameterSet
from anthropic.types.beta import BetaToolUnionParam
from integration._support import claude_code as cc
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support.anthropic_thinking import BEDROCK_MODEL, EVENT_STREAM, MODEL, Event, aws_chunks, sse_chunks
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, responses_reply
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.test_bedrock_auth_wire import RESPONSE as CONVERSE_RESPONSE
from openai.types.responses import ResponseToolSearchCall
from pydantic import JsonValue, TypeAdapter
from redis import Redis

_CLAUDE: Final = cc.OPUS
_CLAUDE_WITHOUT_TOOL_SEARCH: Final = "claude-scripted-unlisted"
_GEMINI: Final = "gemini-2.5-flash"
_REGEX_TOOL_TYPE: Final = "tool_search_tool_regex_20251119"
_REGEX_TOOL_NAME: Final = "tool_search_tool_regex"
_REGEX_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType({"type": _REGEX_TOOL_TYPE, "name": _REGEX_TOOL_NAME})
_BEDROCK_REGEX_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": _REGEX_TOOL_NAME, "name": _REGEX_TOOL_NAME}
)
_REGEX_TOOL_TYPES: Final = frozenset({_REGEX_TOOL_TYPE, _REGEX_TOOL_NAME})
_TOOL_SEARCH_BETA_PREFIXES: Final = ("advanced-tool-use", "tool-search-tool")
_TOOL_SEARCH_CALL_KEYS: Final = frozenset({"id", "type", "call_id", "arguments", "execution", "status"})
_SEARCH_ARGUMENTS: Final = '{"pattern": "weather|stock"}'
_TRUNCATED_ARGUMENTS: Final = '{"pattern": '
_ALL_DEFERRED_ERROR: Final = "At least one tool must have defer_loading=false"
_DEFERRED_NAMES: Final = ("get_weather", "finance__get_stock_price")
_CACHE_BUST: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ITEMS: Final = TypeAdapter(list[dict[str, JsonValue]])
_STRINGS: Final = TypeAdapter(list[JsonValue])
_VERTEX_PROJECT: Final = "scripted-project"
_VERTEX_LOCATION: Final = "us-east5"
_VERTEX_MODEL_PATH: Final = (
    f"/v1/projects/{_VERTEX_PROJECT}/locations/{_VERTEX_LOCATION}/publishers/anthropic/models/{MODEL}"
)
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
_GEMINI_REPLY: Final[dict[str, JsonValue]] = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "gemini control"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
    "modelVersion": _GEMINI,
}

_ORDER_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}},
    "required": ["order_id"],
}
_MALFORMED_FLAGS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"defer_loading": "true", "allowed_callers": "direct"}
)
_WELL_FORMED_FLAGS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"defer_loading": True, "allowed_callers": ["direct"]}
)
_FLAG_REFUSALS: Final[Mapping[str, str]] = MappingProxyType(
    {"defer_loading": "defer_loading must be a boolean", "allowed_callers": "allowed_callers must be a list of strings"}
)

_Client = Literal["sdk", "async_sdk", "raw"]
Provider = Literal["anthropic", "bedrock_invoke", "vertex_ai"]
FlaggedKind: TypeAlias = Literal["function", "custom", "namespace_member"]
ToolFlag: TypeAlias = Literal["defer_loading", "allowed_callers"]
DropSource: TypeAlias = Literal["off", "request", "deployment", "null", "request_string"]


@dataclass(frozen=True, slots=True)
class _Answer:
    payload: dict[str, JsonValue]
    events: tuple[dict[str, JsonValue], ...]


def _tool_search(**fields: JsonValue) -> dict[str, JsonValue]:
    return {"type": "tool_search", **fields}


def _function_tool(defer_loading: JsonValue | EllipsisType = True) -> dict[str, JsonValue]:
    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": _WEATHER_PARAMETERS,
    }
    if isinstance(defer_loading, EllipsisType):
        return tool
    return {**tool, "defer_loading": defer_loading}


def _namespace_tool(**member_fields: JsonValue) -> dict[str, JsonValue]:
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
                **member_fields,
            }
        ],
    }


def _chat_function_tool(defer_loading: JsonValue = True) -> dict[str, JsonValue]:
    return {
        "type": "function",
        "function": {"name": "get_weather", "description": "Weather for a city", "parameters": _WEATHER_PARAMETERS},
        "defer_loading": defer_loading,
    }


def _messages_function_tool() -> dict[str, JsonValue]:
    return {
        "name": "get_weather",
        "description": "Weather for a city",
        "input_schema": _WEATHER_PARAMETERS,
        "defer_loading": True,
    }


def _prompt(marker: str) -> str:
    return f"Weather in Paris and the ACME price marker-{marker}"


def _search_result(marker: str) -> dict[str, JsonValue]:
    return {
        "type": "tool_search_tool_result",
        "tool_use_id": f"srvtoolu_{marker}",
        "content": {
            "type": "tool_search_tool_search_result",
            "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
        },
    }


def _search_blocks(marker: str) -> tuple[dict[str, JsonValue], ...]:
    return (
        {
            "type": "server_tool_use",
            "id": f"srvtoolu_{marker}",
            "name": _REGEX_TOOL_NAME,
            "input": json.loads(_SEARCH_ARGUMENTS),
        },
        _search_result(marker),
    )


def _message(marker: str, searched: bool, tool_use_name: str) -> dict[str, JsonValue]:
    search_blocks: Final = _search_blocks(marker) if searched else ()
    return {
        "id": f"msg_{marker}",
        "type": "message",
        "role": "assistant",
        "model": _CLAUDE,
        "content": [
            {"type": "text", "text": f"searching marker-{marker}"},
            *search_blocks,
            {"type": "tool_use", "id": f"toolu_{marker}", "name": tool_use_name, "input": {"city": "Paris"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 7, "server_tool_use": {"tool_search_requests": 1}},
    }


def _search_events(marker: str, search_arguments: str) -> tuple[Event, ...]:
    half: Final = len(search_arguments) // 2
    opened: Final[Event] = {
        "type": "server_tool_use",
        "id": f"srvtoolu_{marker}",
        "name": _REGEX_TOOL_NAME,
        "input": {},
    }
    return (
        {"type": "content_block_start", "index": 1, "content_block": opened},
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": search_arguments[:half]},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": search_arguments[half:]},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": _search_result(marker)},
        {"type": "content_block_stop", "index": 2},
    )


def _events(marker: str, search_arguments: str, searched: bool, tool_use_name: str) -> tuple[Event, ...]:
    opened: Final = {**_message(marker, searched, tool_use_name), "content": [], "stop_reason": None}
    search_events: Final = _search_events(marker, search_arguments) if searched else ()
    tool_index: Final = 3 if searched else 1
    tool_use: Final[Event] = {"type": "tool_use", "id": f"toolu_{marker}", "name": tool_use_name, "input": {}}
    return (
        {"type": "message_start", "message": opened},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": f"searching marker-{marker}"},
        },
        {"type": "content_block_stop", "index": 0},
        *search_events,
        {"type": "content_block_start", "index": tool_index, "content_block": tool_use},
        {
            "type": "content_block_delta",
            "index": tool_index,
            "delta": {"type": "input_json_delta", "partial_json": '{"city": "Paris"}'},
        },
        {"type": "content_block_stop", "index": tool_index},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    )


def _anthropic_error(status: int, message: str) -> Reply:
    body: Final = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return Reply(status=status, body=json.dumps(body).encode())


def _streams(request: Request, body: Mapping[str, JsonValue]) -> bool:
    return body.get("stream") is True or request.target.endswith(":streamRawPredict")


@dataclass(frozen=True, slots=True)
class _Vendor:
    search_arguments: str = _SEARCH_ARGUMENTS
    pause: float = 0
    searched: bool | None = None
    tool_use_name: str = "get_weather"
    error: tuple[int, str] | None = None

    def respond(self, request: Request) -> Reply:
        if request.method != "POST":
            return Reply()
        body: Final = _JSON_OBJECT.validate_json(request.body)
        marker: Final = rv.newest_marker(request.body.decode()) or "health"
        if self.error is not None:
            return _anthropic_error(self.error[0], f"{self.error[1]} marker-{marker}")
        tools: Final = _ITEMS.validate_python(body.get("tools") or [])
        regex_tools: Final = [tool for tool in tools if tool.get("type") in _REGEX_TOOL_TYPES]
        function_tools: Final = [tool for tool in tools if tool.get("type") not in _REGEX_TOOL_TYPES]
        if len(regex_tools) > 1:
            return _anthropic_error(400, f"tools: duplicate tool name {_REGEX_TOOL_NAME} marker-{marker}")
        if not regex_tools and function_tools and all(tool.get("defer_loading") is True for tool in function_tools):
            return _anthropic_error(400, f"tools: {_ALL_DEFERRED_ERROR} marker-{marker}")
        searched: Final = bool(regex_tools) if self.searched is None else self.searched
        if request.target.endswith("/invoke-with-response-stream"):
            events: Final = _events(marker, self.search_arguments, searched, self.tool_use_name)
            return Reply(content_type=EVENT_STREAM, chunks=aws_chunks(events), pause_between_chunks=self.pause)
        if _streams(request, body):
            return Reply(
                content_type="text/event-stream",
                chunks=sse_chunks(_events(marker, self.search_arguments, searched, self.tool_use_name)),
                pause_between_chunks=self.pause,
            )
        return Reply(body=json.dumps(_message(marker, searched, self.tool_use_name)).encode())


def _base_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _sdk(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{_base_url(gateway)}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _async_sdk(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{_base_url(gateway)}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=60),
    )


def _claude_sdk(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=_base_url(gateway),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _raw(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    with httpx.Client(base_url=_base_url(gateway), trust_env=False, timeout=60) as client:
        headers: Final = {"Authorization": f"Bearer {gateway.key}", "anthropic-version": "2023-06-01"}
        with client.stream("POST", path, json=dict(body), headers=headers) as response:
            response.read()
            return response


def _frames(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(_JSON_OBJECT.validate_json(line[6:]) for line in text.splitlines() if line.startswith("data: {"))


def _completed_of(events: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    completed: Final = [event for event in events if event.get("type") == "response.completed"]
    assert len(completed) == 1, [event.get("type") for event in events]
    return _JSON_OBJECT.validate_python(completed[0]["response"])


def _responses_body(
    model: str, marker: str, tools: Sequence[Mapping[str, JsonValue]], stream: bool
) -> dict[str, JsonValue]:
    return {"model": model, "input": _prompt(marker), "tools": [dict(tool) for tool in tools], "stream": stream}


def _sdk_answer(client: openai.OpenAI, body: Mapping[str, JsonValue]) -> _Answer:
    tools: Final = _ITEMS.validate_python(body["tools"])
    if body["stream"] is not True:
        created: Final = client.responses.create(
            model=str(body["model"]), input=str(body["input"]), tools=tools, extra_body=dict(_CACHE_BUST)
        )
        return _Answer(created.model_dump(), ())
    events: Final = tuple(
        event.model_dump()
        for event in client.responses.create(
            model=str(body["model"]), input=str(body["input"]), tools=tools, stream=True, extra_body=dict(_CACHE_BUST)
        )
    )
    return _Answer(_completed_of(events), events)


async def _async_sdk_answer(client: openai.AsyncOpenAI, body: Mapping[str, JsonValue]) -> _Answer:
    tools: Final = _ITEMS.validate_python(body["tools"])
    if body["stream"] is not True:
        created: Final = await client.responses.create(
            model=str(body["model"]), input=str(body["input"]), tools=tools, extra_body=dict(_CACHE_BUST)
        )
        return _Answer(created.model_dump(), ())
    stream: Final = await client.responses.create(
        model=str(body["model"]), input=str(body["input"]), tools=tools, stream=True, extra_body=dict(_CACHE_BUST)
    )
    events: Final = tuple([event.model_dump() async for event in stream])
    return _Answer(_completed_of(events), events)


def _raw_answer(gateway: Gateway, body: Mapping[str, JsonValue]) -> _Answer:
    response: Final = _raw(gateway, "/v1/responses", {**body, **_CACHE_BUST})
    assert response.status_code == 200, response.text
    if body["stream"] is not True:
        return _Answer(_JSON_OBJECT.validate_json(response.content), ())
    events: Final = _frames(response.text)
    return _Answer(_completed_of(events), events)


def _answer(gateway: Gateway, client: _Client, body: Mapping[str, JsonValue]) -> _Answer:
    match client:
        case "sdk":
            return _sdk_answer(_sdk(gateway), body)
        case "async_sdk":
            return asyncio.run(_async_sdk_answer(_async_sdk(gateway), body))
        case "raw":
            return _raw_answer(gateway, body)


def _output_types(payload: Mapping[str, JsonValue]) -> list[JsonValue]:
    return [item["type"] for item in _ITEMS.validate_python(payload["output"])]


def _tool_items(payload: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return [item for item in _ITEMS.validate_python(payload["output"]) if item["type"] not in ("message", "reasoning")]


def _search_call_fields(item: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: item[key] for key in ("id", "call_id", "arguments", "execution", "status")}


def _assert_search_call_output(payload: Mapping[str, JsonValue], marker: str) -> None:
    items: Final = _tool_items(payload)
    assert len(items) == 2, payload
    search, function = items
    assert search["type"] == "tool_search_call", payload
    assert _search_call_fields(search) == {
        "id": f"tsc_srvtoolu_{marker}",
        "call_id": f"srvtoolu_{marker}",
        "arguments": json.loads(_SEARCH_ARGUMENTS),
        "execution": "server",
        "status": "completed",
    }, search
    assert (function["type"], function["name"], function["call_id"]) == (
        "function_call",
        "get_weather",
        f"toolu_{marker}",
    ), function
    assert json.loads(str(function["arguments"])) == {"city": "Paris"}, function


def _item_of(event: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_python(event["item"])


def _item_events(events: Sequence[Mapping[str, JsonValue]], item_type: str) -> list[dict[str, JsonValue]]:
    return [
        dict(event)
        for event in events
        if event["type"] in ("response.output_item.added", "response.output_item.done")
        and _item_of(event)["type"] == item_type
    ]


def _assert_search_call_events(events: Sequence[Mapping[str, JsonValue]], marker: str, final_status: str) -> None:
    search_events: Final = _item_events(events, "tool_search_call")
    assert [event["type"] for event in search_events] == [
        "response.output_item.added",
        "response.output_item.done",
    ], [event["type"] for event in events]
    added, done = (_item_of(event) for event in search_events)
    assert (added["id"], added["status"], added["arguments"]) == (f"tsc_srvtoolu_{marker}", "in_progress", {}), added
    assert (done["id"], done["status"]) == (f"tsc_srvtoolu_{marker}", final_status), done
    assert done["arguments"] == (json.loads(_SEARCH_ARGUMENTS) if final_status == "completed" else {}), done
    search_deltas: Final = [
        event
        for event in events
        if event["type"] == "response.function_call_arguments.delta" and event["item_id"] == f"tsc_srvtoolu_{marker}"
    ]
    assert search_deltas == [], search_deltas
    function_events: Final = _item_events(events, "function_call")
    assert [event["type"] for event in function_events] == [
        "response.output_item.added",
        "response.output_item.done",
    ], [event["type"] for event in events]
    assert {event["output_index"] for event in search_events}.isdisjoint(
        {event["output_index"] for event in function_events}
    ), (search_events, function_events)


def _only_request(wire: Wire) -> tuple[Request, dict[str, JsonValue]]:
    received: Final = wire.drain()
    assert len(received) == 1, [(request.method, request.target) for request in received]
    return received[0], _JSON_OBJECT.validate_json(received[0].body)


def _header_betas(request: Request) -> tuple[str, ...]:
    return tuple(beta.strip() for beta in request.headers.get("anthropic-beta", "").split(",") if beta.strip())


def _target(provider: Provider, stream: bool) -> str:
    match provider:
        case "anthropic":
            return "/v1/messages"
        case "bedrock_invoke":
            return f"/model/{BEDROCK_MODEL}/invoke-with-response-stream" if stream else f"/model/{BEDROCK_MODEL}/invoke"
        case "vertex_ai":
            return f"{_VERTEX_MODEL_PATH}:streamRawPredict?alt=sse" if stream else f"{_VERTEX_MODEL_PATH}:rawPredict"


def _body_betas(body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    return tuple(str(beta) for beta in _STRINGS.validate_python(body.get("anthropic_beta") or []))


def _enables_tool_search(betas: Sequence[str]) -> bool:
    return any(beta.startswith(_TOOL_SEARCH_BETA_PREFIXES) for beta in betas)


def _assert_tool_search_beta(provider: Provider, request: Request, body: Mapping[str, JsonValue]) -> None:
    match provider:
        case "anthropic":
            assert _enables_tool_search(_header_betas(request)), dict(request.headers)
        case "vertex_ai":
            assert _enables_tool_search(_header_betas(request)), dict(request.headers)
            assert _enables_tool_search(_body_betas(body)), body
        case "bedrock_invoke":
            assert _enables_tool_search(_body_betas(body)), body


def _assert_claude_wire(
    provider: Provider, stream: bool, request: Request, body: Mapping[str, JsonValue], deferred: Sequence[str]
) -> None:
    assert unquote(request.target) == _target(provider, stream), request.target
    _assert_tool_search_beta(provider, request, body)
    tools: Final = _ITEMS.validate_python(body["tools"])
    assert tools[0] == dict(_BEDROCK_REGEX_TOOL if provider == "bedrock_invoke" else _REGEX_TOOL), tools
    assert [tool["name"] for tool in tools[1:]] == list(deferred), tools
    assert [tool.get("defer_loading") for tool in tools[1:]] == [True] * len(deferred), tools
    assert "no-cache" not in request.body.decode(), request.body


def _assert_anthropic_wire(request: Request, body: Mapping[str, JsonValue], deferred: Sequence[str]) -> None:
    _assert_claude_wire("anthropic", False, request, body, deferred)


def _assert_dropped_on_wire(body: Mapping[str, JsonValue], names: Sequence[str]) -> None:
    tools: Final = _ITEMS.validate_python(body.get("tools") or [])
    assert [tool["name"] for tool in tools] == list(names), tools
    assert _REGEX_TOOL_TYPE not in json.dumps(body), body


def _spend_rows(model: str, expected: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT request_id, status, cache_hit FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)
        ),
        lambda found: len(found) >= expected,
        seconds=70,
    )


def _assert_spend_rows(model: str, response_ids: Sequence[str]) -> None:
    rows: Final = _spend_rows(model, len(response_ids))
    logged: Final = {str(row["request_id"]): str(row["status"]) for row in rows}
    assert len(logged) == len(rows) == len(response_ids), rows
    for response_id in response_ids:
        (match,) = [logged_id for logged_id in logged if rv.same_response(logged_id, response_id)]
        assert logged[match] == "success", rows


def _deployment(scenario: Scenario, provider: Provider, wire_url: str, upstream_url: str) -> str:
    match provider:
        case "anthropic":
            return scenario.model(model=f"anthropic/{_CLAUDE}", api_base=wire_url, api_key=cc.ANTHROPIC_API_KEY)
        case "bedrock_invoke":
            return scenario.model(
                model=f"bedrock/invoke/{BEDROCK_MODEL}",
                api_base=wire_url,
                aws_access_key_id="AKIASCRIPTEDPROVIDER",
                aws_secret_access_key="scripted-secret",
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=wire_url,
            )
        case "vertex_ai":
            return scenario.model(
                model=f"vertex_ai/{MODEL}",
                api_base=f"{wire_url}{_VERTEX_MODEL_PATH}",
                api_key=None,
                vertex_project=_VERTEX_PROJECT,
                vertex_location=_VERTEX_LOCATION,
                vertex_credentials=service_account_json(_VERTEX_PROJECT, upstream_url.rstrip("/")),
            )


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("client", ["sdk", "async_sdk", "raw"])
def test_tool_search_reaches_claude_with_deferred_tools_and_returns_a_tool_search_call(
    gateway: Gateway, client: _Client, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(), _function_tool(), _namespace_tool())
        answer: Final = _answer(gateway, client, _responses_body(model, marker, tools, stream))
        _assert_search_call_output(answer.payload, marker)
        if stream:
            _assert_search_call_events(answer.events, marker, "completed")
        request, body = _only_request(wire)
        _assert_anthropic_wire(request, body, _DEFERRED_NAMES)
        assert body.get("stream", False) is stream, body
        _assert_spend_rows(model, (str(answer.payload["id"]),))


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("provider", ["bedrock_invoke", "vertex_ai"])
def test_bedrock_invoke_and_vertex_claude_keep_the_tool_search_tool(
    gateway: Gateway, provider: Provider, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, provider, wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(), _function_tool(), _namespace_tool())
        answer: Final = _raw_answer(gateway, _responses_body(model, marker, tools, stream))
        _assert_search_call_output(answer.payload, marker)
        if stream:
            _assert_search_call_events(answer.events, marker, "completed")
        request, body = _only_request(wire)
        _assert_claude_wire(provider, stream, request, body, _DEFERRED_NAMES)
        _assert_spend_rows(model, (str(answer.payload["id"]),))


@pytest.mark.parametrize("provider", ["anthropic", "bedrock_invoke"])
def test_a_non_streaming_tool_search_call_item_carries_only_its_own_keys(gateway: Gateway, provider: Provider) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, provider, wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(), _function_tool(), _namespace_tool())
        response: Final = _raw(
            gateway, "/v1/responses", {**_responses_body(model, marker, tools, False), **_CACHE_BUST}
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        _assert_search_call_output(payload, marker)
        (search,) = [item for item in _tool_items(payload) if item["type"] == "tool_search_call"]
        assert frozenset(search) == _TOOL_SEARCH_CALL_KEYS, search
        request, body = _only_request(wire)
        _assert_claude_wire(provider, False, request, body, _DEFERRED_NAMES)


@pytest.mark.parametrize("provider", ["anthropic", "bedrock_invoke"])
def test_a_streamed_tool_search_call_item_carries_only_its_own_keys(gateway: Gateway, provider: Provider) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, provider, wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(), _function_tool(), _namespace_tool())
        answer: Final = _raw_answer(gateway, _responses_body(model, marker, tools, True))
        _assert_search_call_output(answer.payload, marker)
        _assert_search_call_events(answer.events, marker, "completed")
        streamed: Final = [_item_of(event) for event in _item_events(answer.events, "tool_search_call")]
        (completed,) = [item for item in _tool_items(answer.payload) if item["type"] == "tool_search_call"]
        assert [frozenset(item) for item in (*streamed, completed)] == [_TOOL_SEARCH_CALL_KEYS] * 3, streamed
        request, body = _only_request(wire)
        _assert_claude_wire(provider, True, request, body, _DEFERRED_NAMES)


def _schema_properties(schemas: Mapping[str, JsonValue], name: str) -> frozenset[str]:
    return frozenset(object_value(object_value(schemas[name])["properties"]))


def test_the_openapi_schema_keeps_the_sdk_tool_search_call_beside_the_proxy_output_item(gateway: Gateway) -> None:
    schemas: Final = object_value(object_value(gateway.get("/openapi.json")["components"])["schemas"])
    proxy_item: Final = _schema_properties(schemas, "OutputToolSearchCall")
    sdk_item: Final = _schema_properties(schemas, "ResponseToolSearchCall")
    assert proxy_item == _TOOL_SEARCH_CALL_KEYS, schemas.get("OutputToolSearchCall")
    assert sdk_item == frozenset(ResponseToolSearchCall.model_fields), schemas.get("ResponseToolSearchCall")
    assert [name for name in schemas if name.endswith("ToolSearchCall") and "__" in name] == [], sorted(schemas)


def test_search_arguments_that_never_parse_stream_an_incomplete_tool_search_call(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor(search_arguments=_TRUNCATED_ARGUMENTS).respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(), _function_tool())
        answer: Final = _raw_answer(gateway, _responses_body(model, marker, tools, True))
        _assert_search_call_events(answer.events, marker, "incomplete")
        search, function = _tool_items(answer.payload)
        assert (search["type"], search["status"], search["arguments"]) == ("tool_search_call", "incomplete", {}), search
        assert (function["type"], function["name"]) == ("function_call", "get_weather"), function
        request, body = _only_request(wire)
        _assert_anthropic_wire(request, body, ("get_weather",))
        _assert_spend_rows(model, (str(answer.payload["id"]),))


def test_tool_search_alone_reaches_claude_as_the_only_tool(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        answer: Final = _raw_answer(gateway, _responses_body(model, marker, (_tool_search(),), False))
        assert [item["type"] for item in _tool_items(answer.payload)] == ["tool_search_call", "function_call"], answer
        request, body = _only_request(wire)
        _assert_anthropic_wire(request, body, ())


@pytest.mark.parametrize("execution", [..., None, "server"], ids=["missing", "null", "server"])
def test_a_missing_null_or_server_execution_is_a_server_tool_search(gateway: Gateway, execution: object) -> None:
    marker: Final = uuid.uuid4().hex
    tool: Final = (
        _tool_search()
        if execution is ...
        else _tool_search(execution=TypeAdapter(JsonValue).validate_python(execution))
    )
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        answer: Final = _raw_answer(gateway, _responses_body(model, marker, (tool, _function_tool()), False))
        _assert_search_call_output(answer.payload, marker)
        request, body = _only_request(wire)
        _assert_anthropic_wire(request, body, ("get_weather",))


@pytest.mark.parametrize("defer_loading", [None, False, ...], ids=["null", "false", "missing"])
def test_a_null_false_or_missing_defer_loading_leaves_the_tool_loaded(
    gateway: Gateway, defer_loading: JsonValue | EllipsisType
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(), _function_tool(defer_loading))
        answer: Final = _raw_answer(gateway, _responses_body(model, marker, tools, False))
        _assert_search_call_output(answer.payload, marker)
        request, body = _only_request(wire)
        assert request.target == "/v1/messages", request.target
        regex_tool, weather = _ITEMS.validate_python(body["tools"])
        assert regex_tool == dict(_REGEX_TOOL), body
        assert weather["name"] == "get_weather" and "defer_loading" not in weather, weather


def _await_cached(needle: str) -> None:
    store: Final = Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))

    def cached() -> bool:
        return any(
            needle.encode() in (store.get(key) or b"")
            for key in store.scan_iter(count=1000)
            if store.type(key) == b"string"
        )

    eventually(cached, lambda found: found, seconds=30)


def test_a_cache_hit_still_carries_the_tool_search_call(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        body: Final = _responses_body(model, marker, (_tool_search(), _function_tool()), False)
        first: Final = _raw(gateway, "/v1/responses", body)
        assert first.status_code == 200, first.text
        _assert_search_call_output(_JSON_OBJECT.validate_json(first.content), marker)
        assert len(wire.drain()) == 1
        _await_cached(f"srvtoolu_{marker}")
        hit: Final = _raw(gateway, "/v1/responses", body)
        assert hit.status_code == 200, hit.text
        _assert_search_call_output(_JSON_OBJECT.validate_json(hit.content), marker)
        assert wire.drain() == (), hit.text
        rows: Final = _spend_rows(model, 2)
        assert len(rows) == 2 and all(str(row["status"]) == "success" for row in rows), rows
        assert sorted(str(row["cache_hit"]) == "True" for row in rows) == [False, True], rows


def test_a_cache_hit_tool_search_call_item_carries_only_its_own_keys(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        body: Final = _responses_body(model, marker, (_tool_search(), _function_tool()), False)
        first: Final = _raw(gateway, "/v1/responses", body)
        assert first.status_code == 200, first.text
        assert len(wire.drain()) == 1
        _await_cached(f"srvtoolu_{marker}")
        hit: Final = _raw(gateway, "/v1/responses", body)
        assert hit.status_code == 200, hit.text
        assert wire.drain() == (), hit.text
        payload: Final = _JSON_OBJECT.validate_json(hit.content)
        _assert_search_call_output(payload, marker)
        (search,) = [item for item in _tool_items(payload) if item["type"] == "tool_search_call"]
        assert frozenset(search) == _TOOL_SEARCH_CALL_KEYS, search


def test_model_info_reports_tool_search_support_for_the_claude_deployment(gateway: Gateway) -> None:
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        response: Final = gateway.request("GET", "/model/info", None)
        assert response.status_code == 200, response.text
        (entry,) = [
            item
            for item in _ITEMS.validate_python(_JSON_OBJECT.validate_json(response.content)["data"])
            if item["model_name"] == model
        ]
        assert _JSON_OBJECT.validate_python(entry["model_info"]).get("supports_tool_search") is True, entry


def test_gemini_keeps_dropping_tool_search_with_a_warning(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex

    def gemini(request: Request) -> Reply:
        assert request.target.endswith(f"/models/{_GEMINI}:generateContent"), request.target
        return Reply(body=json.dumps(_GEMINI_REPLY).encode())

    with wire_server(gemini) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{_GEMINI}", api_base=wire.url, api_key="synthetic-gemini-key")
        tools: Final = (_tool_search(), _function_tool(), _namespace_tool())
        answer: Final = _raw_answer(gateway, _responses_body(model, marker, tools, False))
        assert _output_types(answer.payload) == ["message"], answer
        _, body = _only_request(wire)
        declarations: Final = _ITEMS.validate_python(_ITEMS.validate_python(body["tools"])[0]["function_declarations"])
        assert [declaration["name"] for declaration in declarations] == list(_DEFERRED_NAMES), body
        assert "tool_search" not in json.dumps(body), body


def test_bedrock_converse_keeps_dropping_the_tool_search_tool(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex

    def converse(request: Request) -> Reply:
        assert unquote(request.target) == f"/model/{BEDROCK_MODEL}/converse", request.target
        return Reply(body=CONVERSE_RESPONSE)

    with wire_server(converse) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/converse/{BEDROCK_MODEL}",
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
        )
        tools: Final = (_tool_search(), _function_tool(), _namespace_tool())
        answer: Final = _raw_answer(gateway, _responses_body(model, marker, tools, False))
        assert _output_types(answer.payload) == ["message"], answer
        _, body = _only_request(wire)
        specs: Final = _ITEMS.validate_python(_JSON_OBJECT.validate_python(body["toolConfig"])["tools"])
        assert [_JSON_OBJECT.validate_python(spec["toolSpec"])["name"] for spec in specs] == list(_DEFERRED_NAMES), body
        assert "tool_search" not in json.dumps(body), body


def test_a_claude_the_cost_map_does_not_list_gets_the_tool_dropped_on_the_anthropic_wire(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_CLAUDE_WITHOUT_TOOL_SEARCH}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY
        )
        tools: Final = (_tool_search(), _function_tool(defer_loading=False), _namespace_tool())
        answer: Final = _raw_answer(gateway, _responses_body(model, marker, tools, False))
        assert _output_types(answer.payload) == ["message", "function_call"], answer
        request, body = _only_request(wire)
        assert request.target == "/v1/messages", request.target
        _assert_dropped_on_wire(body, _DEFERRED_NAMES)


def test_client_execution_drops_the_tool_search_tool(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(execution="client"), _function_tool(defer_loading=False), _namespace_tool())
        answer: Final = _raw_answer(gateway, _responses_body(model, marker, tools, False))
        assert _output_types(answer.payload) == ["message", "function_call"], answer
        _, body = _only_request(wire)
        _assert_dropped_on_wire(body, _DEFERRED_NAMES)


@pytest.mark.parametrize("drop_params", [False, True], ids=["refused", "dropped"])
@pytest.mark.parametrize(
    "execution", ["", "x" * 5000, 5, ["server"]], ids=["empty", "five-kilobyte-string", "int", "list"]
)
def test_a_malformed_execution_is_refused_with_400_unless_drop_params_leaves_a_server_search(
    gateway: Gateway, execution: JsonValue, drop_params: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(execution=execution), _function_tool())
        response: Final = _raw(
            gateway,
            "/v1/responses",
            {**_responses_body(model, marker, tools, False), **_CACHE_BUST, "drop_params": drop_params},
        )
        if not drop_params:
            assert response.status_code == 400, response.text
            assert "tool_search execution must be 'server' or 'client'" in response.text, response.text
            assert wire.drain() == (), response.text
            return
        assert response.status_code == 200, response.text
        _assert_search_call_output(_JSON_OBJECT.validate_json(response.content), marker)
        request, body = _only_request(wire)
        _assert_anthropic_wire(request, body, ("get_weather",))


def _tool_results(messages: Sequence[Mapping[str, JsonValue]]) -> Iterator[dict[str, JsonValue]]:
    for turn in messages:
        if turn["role"] != "user" or not isinstance(turn["content"], list):
            continue
        yield from (block for block in _ITEMS.validate_python(turn["content"]) if block["type"] == "tool_result")


def _assert_only_the_function_call_is_replayed(messages: Sequence[Mapping[str, JsonValue]], marker: str) -> None:
    assert "tool_search" not in json.dumps(messages) and "srvtoolu_" not in json.dumps(messages), messages
    (assistant,) = [turn for turn in messages if turn["role"] == "assistant"]
    blocks: Final = _ITEMS.validate_python(assistant["content"])
    tool_uses: Final = [(block["type"], block.get("id")) for block in blocks if block["type"] == "tool_use"]
    assert tool_uses == [("tool_use", f"toolu_{marker}")], blocks
    assert [block["tool_use_id"] for block in _tool_results(messages)] == [f"toolu_{marker}"], messages


def test_tool_search_items_are_not_replayed_to_claude(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": _prompt(marker)},
        {
            "type": "tool_search_call",
            "id": f"tsc_srvtoolu_{marker}",
            "call_id": f"srvtoolu_{marker}",
            "arguments": json.loads(_SEARCH_ARGUMENTS),
            "execution": "server",
            "status": "completed",
        },
        {
            "type": "tool_search_output",
            "id": f"tso_{marker}",
            "call_id": f"srvtoolu_{marker}",
            "tools": [{"type": "function", "name": "get_weather"}],
            "status": "completed",
        },
        {
            "type": "function_call",
            "id": f"fc_{marker}",
            "call_id": f"toolu_{marker}",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
            "status": "completed",
        },
        {"type": "function_call_output", "call_id": f"toolu_{marker}", "output": "22C"},
        {"role": "user", "content": f"And tomorrow? marker-{marker}"},
    ]
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "input": history,
            "tools": [_tool_search(), _function_tool()],
            **_CACHE_BUST,
        }
        response: Final = _raw(gateway, "/v1/responses", body)
        assert response.status_code == 200, response.text
        _assert_search_call_output(_JSON_OBJECT.validate_json(response.content), marker)
        request, sent = _only_request(wire)
        _assert_anthropic_wire(request, sent, ("get_weather",))
        _assert_only_the_function_call_is_replayed(_ITEMS.validate_python(sent["messages"]), marker)


def test_chat_completions_drop_a_replayed_search_call_that_has_no_result(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "messages": [
                {"role": "user", "content": _prompt(marker)},
                {
                    "role": "assistant",
                    "content": f"searching marker-{marker}",
                    "tool_calls": [
                        {
                            "id": f"srvtoolu_{marker}",
                            "type": "function",
                            "function": {"name": _REGEX_TOOL_NAME, "arguments": _SEARCH_ARGUMENTS},
                        },
                        {
                            "id": f"toolu_{marker}",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": f"toolu_{marker}", "content": "22C"},
                {"role": "user", "content": f"And tomorrow? marker-{marker}"},
            ],
            "tools": [dict(_REGEX_TOOL), _chat_function_tool()],
            **_CACHE_BUST,
        }
        response: Final = _raw(gateway, "/v1/chat/completions", body)
        assert response.status_code == 200, response.text
        request, sent = _only_request(wire)
        _assert_anthropic_wire(request, sent, ("get_weather",))
        _assert_only_the_function_call_is_replayed(_ITEMS.validate_python(sent["messages"]), marker)


def test_a_callers_function_named_like_the_search_tool_stays_a_function_call(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    own_tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "name": _REGEX_TOOL_NAME,
        "description": "The caller's own tool search",
        "parameters": _WEATHER_PARAMETERS,
    }
    with (
        wire_server(_Vendor(searched=False, tool_use_name=_REGEX_TOOL_NAME).respond) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        answer: Final = _raw_answer(gateway, _responses_body(model, marker, (own_tool,), False))
        (call,) = _tool_items(answer.payload)
        assert (call["type"], call["name"], call["call_id"]) == (
            "function_call",
            _REGEX_TOOL_NAME,
            f"toolu_{marker}",
        ), call
        _, body = _only_request(wire)
        assert [tool["name"] for tool in _ITEMS.validate_python(body["tools"])] == [_REGEX_TOOL_NAME], body


def test_an_unrequested_server_tool_use_from_the_vendor_stays_a_function_call(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor(searched=True).respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        answer: Final = _raw_answer(
            gateway, _responses_body(model, marker, (_function_tool(defer_loading=False),), False)
        )
        assert "tool_search_call" not in _output_types(answer.payload), answer
        calls: Final = {str(item["call_id"]): str(item["name"]) for item in _tool_items(answer.payload)}
        assert calls == {f"srvtoolu_{marker}": _REGEX_TOOL_NAME, f"toolu_{marker}": "get_weather"}, answer
        _, body = _only_request(wire)
        _assert_dropped_on_wire(body, ("get_weather",))


@pytest.mark.parametrize(
    "defer_loading",
    ["true", 1, "", "x" * 5000, [True]],
    ids=["string", "int", "empty-string", "five-kilobyte-string", "list"],
)
def test_a_non_boolean_defer_loading_is_refused_with_400_and_the_proxy_keeps_serving(
    gateway: Gateway, defer_loading: JsonValue
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(), _function_tool(defer_loading))
        refused: Final = _raw(gateway, "/v1/responses", {**_responses_body(model, marker, tools, False), **_CACHE_BUST})
        assert refused.status_code == 400 and "defer_loading must be a boolean" in refused.text, refused.text
        assert wire.drain() == (), refused.text
        follow_up: Final = uuid.uuid4().hex
        answer: Final = _raw_answer(
            gateway, _responses_body(model, follow_up, (_tool_search(), _function_tool()), False)
        )
        _assert_search_call_output(answer.payload, follow_up)
        assert len(wire.drain()) == 1


def test_a_non_boolean_defer_loading_on_a_namespace_member_is_refused_with_400(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(), _function_tool(), _namespace_tool(defer_loading="true"))
        refused: Final = _raw(gateway, "/v1/responses", {**_responses_body(model, marker, tools, False), **_CACHE_BUST})
        assert refused.status_code == 400 and "defer_loading must be a boolean" in refused.text, refused.text
        assert wire.drain() == (), refused.text


@pytest.mark.parametrize("allowed_callers", ["code_execution", 5, [1]], ids=["string", "int", "list-of-ints"])
def test_a_malformed_allowed_callers_on_a_namespace_member_is_refused_with_400(
    gateway: Gateway, allowed_callers: JsonValue
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(), _function_tool(), _namespace_tool(allowed_callers=allowed_callers))
        refused: Final = _raw(gateway, "/v1/responses", {**_responses_body(model, marker, tools, False), **_CACHE_BUST})
        assert refused.status_code == 400 and "allowed_callers must be a list of strings" in refused.text, refused.text
        assert wire.drain() == (), refused.text


def _tool_flags(flag: ToolFlag) -> Mapping[str, JsonValue]:
    return {key: _MALFORMED_FLAGS[key] if key == flag else value for key, value in _WELL_FORMED_FLAGS.items()}


def _flagged_tool(kind: FlaggedKind, flag: ToolFlag) -> dict[str, JsonValue]:
    match kind:
        case "function":
            return {
                "type": "function",
                "name": "lookup_order",
                "description": "Look up an order",
                "parameters": _ORDER_PARAMETERS,
                **_tool_flags(flag),
            }
        case "custom":
            return {"type": "custom", "name": "run_sql", "description": "Run a SQL query", **_tool_flags(flag)}
        case "namespace_member":
            return _namespace_tool(**_tool_flags(flag))


def _flagged_wire_name(kind: FlaggedKind) -> str:
    match kind:
        case "function":
            return "lookup_order"
        case "custom":
            return "run_sql"
        case "namespace_member":
            return "finance__get_stock_price"


def _request_drop_params(source: DropSource) -> Mapping[str, JsonValue]:
    match source:
        case "off" | "deployment":
            return {}
        case "request":
            return {"drop_params": True}
        case "null":
            return {"drop_params": None}
        case "request_string":
            return {"drop_params": "true"}


def _drop_params_deployment(scenario: Scenario, source: DropSource, wire_url: str) -> str:
    if source == "deployment":
        return scenario.model(
            model=f"anthropic/{_CLAUDE}", api_base=wire_url, api_key=cc.ANTHROPIC_API_KEY, drop_params=True
        )
    return scenario.model(model=f"anthropic/{_CLAUDE}", api_base=wire_url, api_key=cc.ANTHROPIC_API_KEY)


def _drop_params_cells() -> tuple[ParameterSet, ...]:
    kinds: Final[tuple[FlaggedKind, ...]] = ("function", "custom", "namespace_member")
    flags: Final[tuple[ToolFlag, ...]] = ("defer_loading", "allowed_callers")
    sources: Final[tuple[DropSource, ...]] = ("off", "request", "deployment")
    grid: Final = tuple(
        pytest.param(kind, flag, source, id=f"{kind}-{flag}-{source}")
        for kind, flag, source in itertools.product(kinds, flags, sources)
    )
    edges: Final = tuple(
        pytest.param("function", "defer_loading", source, id=f"function-defer_loading-{source}")
        for source in ("null", "request_string")
    )
    return (*grid, *edges)


def _assert_flag_dropped_on_wire(body: Mapping[str, JsonValue], kind: FlaggedKind, flag: ToolFlag) -> None:
    tools: Final = _ITEMS.validate_python(body["tools"])
    assert tools[0] == dict(_REGEX_TOOL), tools
    (flagged,) = [tool for tool in tools if tool.get("name") == _flagged_wire_name(kind)]
    assert flag not in flagged, flagged
    kept: Final = "allowed_callers" if flag == "defer_loading" else "defer_loading"
    assert flagged[kept] == _WELL_FORMED_FLAGS[kept], flagged
    (weather,) = [tool for tool in tools if tool.get("name") == "get_weather"]
    assert weather.get("defer_loading") is True, weather


@pytest.mark.parametrize(("kind", "flag", "source"), _drop_params_cells())
def test_drop_params_drops_a_malformed_tool_flag_and_without_it_the_request_is_refused_with_400(
    gateway: Gateway, kind: FlaggedKind, flag: ToolFlag, source: DropSource
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _drop_params_deployment(scenario, source, wire.url)
        tools: Final = (_tool_search(), _function_tool(), _flagged_tool(kind, flag))
        response: Final = _raw(
            gateway,
            "/v1/responses",
            {**_responses_body(model, marker, tools, False), **_CACHE_BUST, **_request_drop_params(source)},
        )
        if source in ("off", "null"):
            assert response.status_code == 400, response.text
            assert "litellm.BadRequestError" in response.text, response.text
            assert _FLAG_REFUSALS[flag] in response.text, response.text
            assert wire.drain() == (), response.text
            follow_up: Final = uuid.uuid4().hex
            served: Final = _raw_answer(
                gateway, _responses_body(model, follow_up, (_tool_search(), _function_tool()), False)
            )
            _assert_search_call_output(served.payload, follow_up)
            assert len(wire.drain()) == 1
            return
        assert response.status_code == 200, response.text
        _assert_search_call_output(_JSON_OBJECT.validate_json(response.content), marker)
        request, body = _only_request(wire)
        assert request.target == "/v1/messages", request.target
        _assert_flag_dropped_on_wire(body, kind, flag)


def test_duplicate_tool_search_tools_reach_claude_and_its_refusal_reaches_the_caller(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(), _tool_search(), _function_tool())
        response: Final = _raw(
            gateway, "/v1/responses", {**_responses_body(model, marker, tools, False), **_CACHE_BUST}
        )
        assert response.status_code == 400, response.text
        assert f"duplicate tool name {_REGEX_TOOL_NAME} marker-{marker}" in response.text, response.text
        _, body = _only_request(wire)
        regex_tools: Final = [
            tool for tool in _ITEMS.validate_python(body["tools"]) if tool["type"] == _REGEX_TOOL_TYPE
        ]
        assert len(regex_tools) == 2, body


@pytest.mark.parametrize("status", [401, 404])
def test_a_scripted_provider_error_reaches_the_caller_and_lands_as_a_failed_spend_row(
    gateway: Gateway, status: int
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor(error=(status, f"scripted {status}")).respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (_tool_search(), _function_tool())
        response: Final = _raw(
            gateway, "/v1/responses", {**_responses_body(model, marker, tools, False), **_CACHE_BUST}
        )
        assert response.status_code == status, response.text
        assert f"scripted {status} marker-{marker}" in response.text, response.text
        request, body = _only_request(wire)
        _assert_anthropic_wire(request, body, ("get_weather",))
        call_id: Final = response.headers["x-litellm-call-id"]
        rows: Final = eventually(
            lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
            lambda found: len(found) == 1,
            seconds=70,
        )
        assert str(rows[0]["status"]) == "failure", rows


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
def test_chat_completions_keep_forwarding_the_regex_tool_and_defer_loading(gateway: Gateway, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "messages": [{"role": "user", "content": _prompt(marker)}],
            "tools": [dict(_REGEX_TOOL), _chat_function_tool()],
            "stream": stream,
            **_CACHE_BUST,
        }
        response: Final = _raw(gateway, "/v1/chat/completions", body)
        assert response.status_code == 200, response.text
        if stream:
            assert "data: [DONE]" in response.text and "get_weather" in response.text, response.text
        else:
            calls: Final = _ITEMS.validate_python(
                _JSON_OBJECT.validate_python(
                    _ITEMS.validate_python(_JSON_OBJECT.validate_json(response.content)["choices"])[0]["message"]
                )["tool_calls"]
            )
            assert [_JSON_OBJECT.validate_python(call["function"])["name"] for call in calls] == [
                _REGEX_TOOL_NAME,
                "get_weather",
            ], calls
        request, sent = _only_request(wire)
        _assert_anthropic_wire(request, sent, ("get_weather",))


def test_messages_endpoint_keeps_forwarding_the_regex_tool_and_defer_loading(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        created: Final = _claude_sdk(gateway).beta.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": _prompt(marker)}],
            tools=TypeAdapter(list[BetaToolUnionParam]).validate_python([dict(_REGEX_TOOL), _messages_function_tool()]),
            extra_body=dict(_CACHE_BUST),
        )
        assert created.id == f"msg_{marker}", created.model_dump_json()
        assert "get_weather" in created.model_dump_json(), created.model_dump_json()
        request, sent = _only_request(wire)
        _assert_anthropic_wire(request, sent, ("get_weather",))
        streamed: Final = _raw(
            gateway,
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": _prompt(marker)}],
                "tools": [dict(_REGEX_TOOL), _messages_function_tool()],
                "stream": True,
                **_CACHE_BUST,
            },
        )
        assert streamed.status_code == 200, streamed.text
        assert "event: message_stop" in streamed.text and "get_weather" in streamed.text, streamed.text
        stream_request, stream_sent = _only_request(wire)
        _assert_anthropic_wire(stream_request, stream_sent, ("get_weather",))
        assert stream_sent.get("stream") is True, stream_sent


_DEFER_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_PLAIN_TOOL_SEARCH: Final[dict[str, JsonValue]] = {"type": "tool_search"}


def _defer_function(name: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"name": name, "description": f"Look up {name}", "parameters": _DEFER_PARAMETERS, **extra}


def _defer_wire_function(name: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"type": "function", "strict": None, **_defer_function(name), **extra}


@pytest.mark.parametrize(
    "deferred_tool",
    [
        {"type": "function", "function": _defer_function("get_weather"), "defer_loading": True},
        {"type": "function", "function": _defer_function("get_weather", defer_loading=True)},
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
                    _PLAIN_TOOL_SEARCH,
                    deferred_tool,
                    {"type": "function", "function": _defer_function("get_time")},
                    {"type": "function", "function": _defer_function("get_news", defer_loading=False)},
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
            _PLAIN_TOOL_SEARCH,
            _defer_wire_function("get_weather", defer_loading=True),
            _defer_wire_function("get_time"),
            _defer_wire_function("get_news", defer_loading=False),
        ]


_OURS_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_OURS_TOOL_SEARCH: Final[dict[str, JsonValue]] = {"type": "tool_search"}


def _ours_function(name: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"name": name, "description": f"Look up {name}", "parameters": _OURS_PARAMETERS, **extra}


def _ours_wire_function(name: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"type": "function", "strict": None, **_ours_function(name), **extra}


_OURS_CALL_ID: Final = "call_weather"
_OURS_TOOL_SEARCH_CALL: Final[dict[str, JsonValue]] = {
    "id": "tsc_weather",
    "type": "tool_search_call",
    "arguments": {"query": "weather"},
    "execution": "server",
    "status": "completed",
}
_OURS_TOOL_SEARCH_OUTPUT: Final[dict[str, JsonValue]] = {
    "id": "tso_weather",
    "type": "tool_search_output",
    "execution": "server",
    "status": "completed",
    "tools": [_ours_wire_function("get_weather", defer_loading=True)],
}
_OURS_WEATHER_CALL: Final[dict[str, JsonValue]] = {
    "type": "function_call",
    "call_id": _OURS_CALL_ID,
    "name": "get_weather",
    "arguments": '{"city": "Paris"}',
    "namespace": "get_weather",
}


def _ours_searches_then_answers(request: Request) -> Reply:
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
                    _OURS_TOOL_SEARCH_CALL,
                    _OURS_TOOL_SEARCH_OUTPUT,
                    {**_OURS_WEATHER_CALL, "id": "fc_weather", "status": "completed"},
                ],
                "usage": rv.USAGE,
            }
        ).encode()
    )


def test_chat_bridge_replays_tool_search_items_on_the_next_turn(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    tools: Final = [
        _OURS_TOOL_SEARCH,
        {"type": "function", "function": _ours_function("get_weather"), "defer_loading": True},
    ]
    user_turn: Final[dict[str, JsonValue]] = {"role": "user", "content": pcb.prompt(marker)}
    with wire_server(answering_model_discovery(_ours_searches_then_answers)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        first: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [user_turn], "tools": tools, **pcb.NO_CACHE}
        )
        assert first.status_code == 200, first.text
        assistant_turn: Final = rv.JSON_OBJECT.validate_python(
            rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(first.text)["choices"])[0]["message"]
        )
        assert rv.JSON_OBJECT.validate_python(assistant_turn["provider_specific_fields"])["tool_search_items"] == [
            _OURS_TOOL_SEARCH_CALL,
            _OURS_TOOL_SEARCH_OUTPUT,
        ], first.text
        pcb.posted(wire, marker)

        second: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    user_turn,
                    assistant_turn,
                    {"role": "tool", "tool_call_id": _OURS_CALL_ID, "content": "sunny"},
                ],
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
        assert replayed == [_OURS_TOOL_SEARCH_CALL, _OURS_TOOL_SEARCH_OUTPUT, _OURS_WEATHER_CALL]
