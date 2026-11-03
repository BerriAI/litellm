import asyncio
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import unquote

import httpx
import openai
import pytest
from integration._support import claude_code as cc
from integration._support import responses_vendor as rv
from integration._support.anthropic_thinking import BEDROCK_MODEL, EVENT_STREAM, MODEL, Event, aws_chunks, sse_chunks
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.openai_wire import chat_reply
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.test_bedrock_auth_wire import RESPONSE as CONVERSE_RESPONSE
from pydantic import JsonValue, TypeAdapter

_CLAUDE: Final = cc.OPUS
UNLISTED_CLAUDE: Final = "claude-scripted-unlisted"
_NOVA: Final = "amazon.nova-pro-v1:0"
_REGEX_TOOL_TYPE: Final = "tool_search_tool_regex_20251119"
_REGEX_TOOL_NAME: Final = "tool_search_tool_regex"
_BM25_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
)
_REGEX_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType({"type": _REGEX_TOOL_TYPE, "name": _REGEX_TOOL_NAME})
_SEARCH_TOOL_TYPES: Final = frozenset(
    {_REGEX_TOOL_TYPE, _REGEX_TOOL_NAME, "tool_search_tool_bm25_20251119", "tool_search_tool_bm25"}
)
_TOOL_SEARCH_BETA_PREFIXES: Final = ("advanced-tool-use", "tool-search-tool")
_SEARCH_ARGUMENTS: Final = '{"pattern": "weather"}'
_ALL_DEFERRED_ERROR: Final = "At least one tool must have defer_loading=false"
_MODEL_INFO_HINT: Final = "supports_tool_search: true"
_INVOKE_HINT: Final = "bedrock/invoke/<model>"
DEFERRED_NAMES: Final = ("get_weather", "get_stock_price", "lookup_customer")
_CACHE_BUST: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
_VERTEX_PROJECT: Final = "scripted-project"
_VERTEX_LOCATION: Final = "us-east5"
_VERTEX_MODEL_PATH: Final = (
    f"/v1/projects/{_VERTEX_PROJECT}/locations/{_VERTEX_LOCATION}/publishers/anthropic/models/{MODEL}"
)
_BETAS: Final = TypeAdapter(list[JsonValue])
_CITY_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

Client: TypeAlias = Literal["sdk", "async_sdk", "raw"]
Provider: TypeAlias = Literal["anthropic", "bedrock_invoke", "vertex_ai", "azure_ai", "claude_platform"]
DropSource: TypeAlias = Literal["off", "request", "deployment", "null", "request_string"]


@dataclass(frozen=True, slots=True)
class Answer:
    status: int
    payload: dict[str, JsonValue]
    frames: tuple[dict[str, JsonValue], ...]
    text: str


def _prompt(marker: str) -> str:
    return f"Weather in Paris marker-{marker}"


def tool_search(**fields: JsonValue) -> dict[str, JsonValue]:
    return {"type": "tool_search", **fields}


def _top_level_deferred() -> dict[str, JsonValue]:
    return {
        "type": "function",
        "function": {"name": "get_weather", "description": "Weather for a city", "parameters": _CITY_PARAMETERS},
        "defer_loading": True,
    }


def _function_level_deferred() -> dict[str, JsonValue]:
    return {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Price of a ticker",
            "parameters": _CITY_PARAMETERS,
            "defer_loading": True,
        },
    }


def _namespace() -> dict[str, JsonValue]:
    return {
        "type": "namespace",
        "name": "crm",
        "description": "CRM tools",
        "tools": [
            {
                "type": "function",
                "name": "lookup_customer",
                "description": "Look up a customer",
                "parameters": _CITY_PARAMETERS,
                "defer_loading": True,
            }
        ],
    }


def _loaded_function() -> dict[str, JsonValue]:
    return {
        "type": "function",
        "function": {"name": "echo", "description": "Echo a city", "parameters": _CITY_PARAMETERS},
    }


def deferred_tools() -> tuple[dict[str, JsonValue], ...]:
    return (_top_level_deferred(), _function_level_deferred(), _namespace())


def _search_result(marker: str) -> dict[str, JsonValue]:
    return {
        "type": "tool_search_tool_result",
        "tool_use_id": f"srvtoolu_{marker}",
        "content": {
            "type": "tool_search_tool_search_result",
            "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
        },
    }


def _message(marker: str, searched: bool) -> dict[str, JsonValue]:
    search_blocks: Final[tuple[dict[str, JsonValue], ...]] = (
        (
            {
                "type": "server_tool_use",
                "id": f"srvtoolu_{marker}",
                "name": _REGEX_TOOL_NAME,
                "input": json.loads(_SEARCH_ARGUMENTS),
            },
            _search_result(marker),
        )
        if searched
        else ()
    )
    return {
        "id": f"msg_{marker}",
        "type": "message",
        "role": "assistant",
        "model": _CLAUDE,
        "content": [
            {"type": "text", "text": f"searching marker-{marker}"},
            *search_blocks,
            {"type": "tool_use", "id": f"toolu_{marker}", "name": "get_weather", "input": {"city": "Paris"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 20,
            "output_tokens": 7,
            "server_tool_use": {"tool_search_requests": 1 if searched else 0},
        },
    }


def _search_events(marker: str) -> tuple[Event, ...]:
    return (
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "server_tool_use",
                "id": f"srvtoolu_{marker}",
                "name": _REGEX_TOOL_NAME,
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": _SEARCH_ARGUMENTS},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": _search_result(marker)},
        {"type": "content_block_stop", "index": 2},
    )


def _events(marker: str, searched: bool) -> tuple[Event, ...]:
    opened: Final = {**_message(marker, searched), "content": [], "stop_reason": None}
    tool_index: Final = 3 if searched else 1
    return (
        {"type": "message_start", "message": opened},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": f"searching marker-{marker}"},
        },
        {"type": "content_block_stop", "index": 0},
        *(_search_events(marker) if searched else ()),
        {
            "type": "content_block_start",
            "index": tool_index,
            "content_block": {"type": "tool_use", "id": f"toolu_{marker}", "name": "get_weather", "input": {}},
        },
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


def claude_vendor(request: Request) -> Reply:
    if request.method != "POST":
        return Reply()
    body: Final = rv.JSON_OBJECT.validate_json(request.body)
    marker: Final = rv.newest_marker(request.body.decode()) or "health"
    tools: Final = rv.ITEMS.validate_python(body.get("tools") or [])
    search_tools: Final = [tool for tool in tools if tool.get("type") in _SEARCH_TOOL_TYPES]
    function_tools: Final = [tool for tool in tools if tool.get("type") not in _SEARCH_TOOL_TYPES]
    if len(search_tools) > 1:
        return _anthropic_error(400, f"tools: Tool names must be unique marker-{marker}")
    if not search_tools and function_tools and any(tool.get("defer_loading") is True for tool in function_tools):
        return _anthropic_error(400, f"tools: {_ALL_DEFERRED_ERROR} marker-{marker}")
    searched: Final = bool(search_tools)
    if request.target.endswith("/invoke-with-response-stream"):
        return Reply(content_type=EVENT_STREAM, chunks=aws_chunks(_events(marker, searched)))
    if body.get("stream") is True or ":streamRawPredict" in request.target:
        return Reply(content_type="text/event-stream", chunks=sse_chunks(_events(marker, searched)))
    return Reply(body=json.dumps(_message(marker, searched)).encode())


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


def raw_chat(gateway: Gateway, body: Mapping[str, JsonValue]) -> httpx.Response:
    with httpx.Client(base_url=_base_url(gateway), trust_env=False, timeout=60) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={**body, **_CACHE_BUST},
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            response.read()
            return response


def chat_frames(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(rv.JSON_OBJECT.validate_json(line[6:]) for line in text.splitlines() if line.startswith("data: {"))


def chat_body(
    model: str, marker: str, tools: Sequence[Mapping[str, JsonValue]], stream: bool, **extra: JsonValue
) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": _prompt(marker)}],
        "tools": [dict(tool) for tool in tools],
        "stream": stream,
        "num_retries": 0,
        **extra,
    }


def _sdk_answer(client: openai.OpenAI, body: Mapping[str, JsonValue]) -> Answer:
    tools: Final = rv.ITEMS.validate_python(body["tools"])
    messages: Final = rv.ITEMS.validate_python(body["messages"])
    if body["stream"] is not True:
        created: Final = client.chat.completions.create(
            model=str(body["model"]), messages=messages, tools=tools, extra_body=dict(_CACHE_BUST)
        )
        return Answer(200, created.model_dump(), (), created.model_dump_json())
    frames: Final = tuple(
        chunk.model_dump()
        for chunk in client.chat.completions.create(
            model=str(body["model"]), messages=messages, tools=tools, stream=True, extra_body=dict(_CACHE_BUST)
        )
    )
    return Answer(200, {}, frames, json.dumps(frames))


async def _async_sdk_answer(client: openai.AsyncOpenAI, body: Mapping[str, JsonValue]) -> Answer:
    tools: Final = rv.ITEMS.validate_python(body["tools"])
    messages: Final = rv.ITEMS.validate_python(body["messages"])
    if body["stream"] is not True:
        created: Final = await client.chat.completions.create(
            model=str(body["model"]), messages=messages, tools=tools, extra_body=dict(_CACHE_BUST)
        )
        return Answer(200, created.model_dump(), (), created.model_dump_json())
    stream: Final = await client.chat.completions.create(
        model=str(body["model"]), messages=messages, tools=tools, stream=True, extra_body=dict(_CACHE_BUST)
    )
    frames: Final = tuple([chunk.model_dump() async for chunk in stream])
    return Answer(200, {}, frames, json.dumps(frames))


def raw_answer(gateway: Gateway, body: Mapping[str, JsonValue]) -> Answer:
    response: Final = raw_chat(gateway, body)
    if response.status_code != 200 or body.get("stream") is True:
        return Answer(response.status_code, {}, chat_frames(response.text), response.text)
    return Answer(200, rv.JSON_OBJECT.validate_json(response.content), (), response.text)


def _answer(gateway: Gateway, client: Client, body: Mapping[str, JsonValue]) -> Answer:
    match client:
        case "sdk":
            return _sdk_answer(_sdk(gateway), body)
        case "async_sdk":
            return asyncio.run(_async_sdk_answer(_async_sdk(gateway), body))
        case "raw":
            return raw_answer(gateway, body)


def _first_choice(payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return rv.ITEMS.validate_python(payload["choices"])[0]


def _streamed_tool_calls(frames: Sequence[Mapping[str, JsonValue]]) -> list[tuple[str, str]]:
    deltas: Final = [
        rv.JSON_OBJECT.validate_python(_first_choice(frame)["delta"]) for frame in frames if frame.get("choices")
    ]
    opened: Final = rv.ITEMS.validate_python(
        list(chain.from_iterable(rv.ITEMS.validate_python(delta.get("tool_calls") or []) for delta in deltas))
    )
    return [
        (str(call["id"]), str(rv.JSON_OBJECT.validate_python(call["function"])["name"]))
        for call in opened
        if call.get("id")
    ]


def answered_tool_calls(answer: Answer) -> list[tuple[str, str]]:
    if answer.frames:
        return _streamed_tool_calls(answer.frames)
    message: Final = rv.JSON_OBJECT.validate_python(_first_choice(answer.payload)["message"])
    return [
        (str(call["id"]), str(rv.JSON_OBJECT.validate_python(call["function"])["name"]))
        for call in rv.ITEMS.validate_python(message.get("tool_calls") or [])
    ]


def response_id(answer: Answer) -> str:
    return str((answer.frames[0] if answer.frames else answer.payload)["id"])


def assert_search_call_answered(answer: Answer, marker: str) -> None:
    assert answer.status == 200, answer.text
    assert answered_tool_calls(answer) == [
        (f"srvtoolu_{marker}", _REGEX_TOOL_NAME),
        (f"toolu_{marker}", "get_weather"),
    ], answer.text
    if answer.frames:
        return
    usage: Final = rv.JSON_OBJECT.validate_python(answer.payload["usage"])
    assert rv.JSON_OBJECT.validate_python(usage["server_tool_use"])["tool_search_requests"] == 1, usage


def only_request(wire: Wire) -> tuple[Request, dict[str, JsonValue]]:
    received: Final = tuple(request for request in wire.drain() if request.method == "POST")
    assert len(received) == 1, [request.target for request in received]
    return received[0], rv.JSON_OBJECT.validate_json(received[0].body)


def _target(provider: Provider, stream: bool) -> str:
    match provider:
        case "anthropic" | "claude_platform":
            return "/v1/messages"
        case "azure_ai":
            return "/anthropic/v1/messages"
        case "bedrock_invoke":
            return f"/model/{BEDROCK_MODEL}/invoke-with-response-stream" if stream else f"/model/{BEDROCK_MODEL}/invoke"
        case "vertex_ai":
            return f"{_VERTEX_MODEL_PATH}:streamRawPredict?alt=sse" if stream else f"{_VERTEX_MODEL_PATH}:rawPredict"


def _betas(request: Request, body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    header: Final = tuple(beta.strip() for beta in request.headers.get("anthropic-beta", "").split(",") if beta.strip())
    in_body: Final = tuple(str(beta) for beta in _BETAS.validate_python(body.get("anthropic_beta") or []))
    return (*header, *in_body)


def assert_search_tool_on_the_wire(
    provider: Provider, stream: bool, request: Request, body: Mapping[str, JsonValue], tools_before: int = 0
) -> list[dict[str, JsonValue]]:
    assert unquote(request.target) == _target(provider, stream), request.target
    assert provider == "azure_ai" or any(
        beta.startswith(_TOOL_SEARCH_BETA_PREFIXES) for beta in _betas(request, body)
    ), (dict(request.headers), body.get("anthropic_beta"))
    assert "no-cache" not in request.body.decode() and '"tool_search"' not in request.body.decode(), body
    tools: Final = rv.ITEMS.validate_python(body["tools"])
    search_tools: Final = [index for index, tool in enumerate(tools) if tool.get("type") in _SEARCH_TOOL_TYPES]
    assert search_tools == [tools_before], tools
    return tools


def assert_deferred_tools_kept(tools: Sequence[Mapping[str, JsonValue]]) -> None:
    deferred: Final = [
        (tool["name"], tool.get("defer_loading")) for tool in tools if tool.get("type") not in _SEARCH_TOOL_TYPES
    ]
    assert deferred == [(name, True) for name in DEFERRED_NAMES], tools


def assert_no_search_and_no_deferral(body: Mapping[str, JsonValue], names: Sequence[str]) -> None:
    sent: Final = json.dumps(body)
    assert "defer_loading" not in sent and "tool_search" not in sent, body
    assert [tool["name"] for tool in rv.ITEMS.validate_python(body["tools"])] == list(names), body


def _spend_rows(model: str, expected: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=70,
    )


def assert_lands_once_as(model: str, identities: Sequence[str], status: str) -> None:
    rows: Final = _spend_rows(model, len(identities))
    logged: Final = {str(row["request_id"]): str(row["status"]) for row in rows}
    assert len(logged) == len(rows) == len(identities), rows
    matched: Final = [
        [logged_id for logged_id in logged if rv.same_response(logged_id, identity)] for identity in identities
    ]
    assert all(len(ids) == 1 for ids in matched), (identities, rows)
    assert {logged[ids[0]] for ids in matched} == {status}, rows


def claude_deployment(scenario: Scenario, provider: Provider, wire_url: str, upstream_url: str) -> str:
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
        case "azure_ai":
            return scenario.model(model=f"azure_ai/{MODEL}", api_base=wire_url, api_key="scripted-azure-key")
        case "claude_platform":
            return scenario.model(
                model=f"bedrock/claude_platform/{MODEL}",
                api_base=wire_url,
                api_key="scripted-platform-key",
                aws_region_name="us-east-1",
                workspace_id="scripted-workspace",
            )


def _unlisted(scenario: Scenario, wire_url: str, **extra: JsonValue) -> str:
    return scenario.model(
        model=f"anthropic/{UNLISTED_CLAUDE}", api_base=wire_url, api_key=cc.ANTHROPIC_API_KEY, **extra
    )


def _assert_refused_before_the_wire(answer: Answer, wire: Wire, *, hint: str | None, model: str) -> None:
    assert answer.status == 400, answer.text
    assert f"{model} on " in answer.text and "cannot run the OpenAI `tool_search` tool as sent" in answer.text, (
        answer.text
    )
    assert "drop_params" in answer.text, answer.text
    assert (hint is not None and hint in answer.text) or (
        hint is None and _MODEL_INFO_HINT not in answer.text and _INVOKE_HINT not in answer.text
    ), answer.text
    assert [request for request in wire.drain() if request.method == "POST"] == [], answer.text


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("client", ["sdk", "async_sdk", "raw"])
def test_a_server_tool_search_reaches_claude_as_its_regex_tool_with_every_deferral_kept(
    gateway: Gateway, client: Client, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(claude_vendor) as wire, gateway.scenario() as scenario:
        model: Final = claude_deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (tool_search(execution="server"), *deferred_tools())
        answer: Final = _answer(gateway, client, chat_body(model, marker, tools, stream))
        assert_search_call_answered(answer, marker)
        request, body = only_request(wire)
        assert_deferred_tools_kept(assert_search_tool_on_the_wire("anthropic", stream, request, body))
        assert body.get("stream", False) is stream, body
        assert_lands_once_as(model, (response_id(answer),), "success")


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("provider", ["bedrock_invoke", "vertex_ai", "azure_ai", "claude_platform"])
def test_every_claude_route_that_runs_tool_search_gets_the_regex_tool(
    gateway: Gateway, provider: Provider, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(claude_vendor) as wire, gateway.scenario() as scenario:
        model: Final = claude_deployment(scenario, provider, wire.url, gateway.upstream_url)
        answer: Final = raw_answer(gateway, chat_body(model, marker, (tool_search(), *deferred_tools()), stream))
        assert_search_call_answered(answer, marker)
        request, body = only_request(wire)
        assert_deferred_tools_kept(assert_search_tool_on_the_wire(provider, stream, request, body))
        assert_lands_once_as(model, (response_id(answer),), "success")


def test_several_tool_search_items_go_out_as_one_regex_tool_at_the_first_items_position(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(claude_vendor) as wire, gateway.scenario() as scenario:
        model: Final = claude_deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (
            _loaded_function(),
            tool_search(),
            _top_level_deferred(),
            tool_search(execution="server"),
            _function_level_deferred(),
        )
        answer: Final = raw_answer(gateway, chat_body(model, marker, tools, False))
        assert_search_call_answered(answer, marker)
        request, body = only_request(wire)
        sent: Final = assert_search_tool_on_the_wire("anthropic", False, request, body, tools_before=1)
        assert sent[1] == dict(_REGEX_TOOL), sent
        assert [(tool.get("name"), tool.get("defer_loading")) for tool in sent] == [
            ("echo", None),
            (_REGEX_TOOL_NAME, None),
            ("get_weather", True),
            ("get_stock_price", True),
        ], sent


@pytest.mark.parametrize("native", [_REGEX_TOOL, _BM25_TOOL], ids=["regex", "bm25"])
def test_a_native_search_tool_the_caller_sent_goes_out_alone(gateway: Gateway, native: Mapping[str, JsonValue]) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(claude_vendor) as wire, gateway.scenario() as scenario:
        model: Final = claude_deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (dict(native), tool_search(execution="client"), tool_search(), _top_level_deferred())
        answer: Final = raw_answer(gateway, chat_body(model, marker, tools, False))
        assert_search_call_answered(answer, marker)
        request, body = only_request(wire)
        sent: Final = assert_search_tool_on_the_wire("anthropic", False, request, body)
        assert sent[0] == dict(native), sent
        assert [(tool["name"], tool.get("defer_loading")) for tool in sent[1:]] == [("get_weather", True)], sent


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
def test_a_claude_the_cost_map_does_not_list_is_refused_with_400_naming_model_info(
    gateway: Gateway, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(claude_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _unlisted(scenario, wire.url)
        refused: Final = raw_answer(gateway, chat_body(model, marker, (tool_search(), *deferred_tools()), stream))
        _assert_refused_before_the_wire(refused, wire, hint=_MODEL_INFO_HINT, model=UNLISTED_CLAUDE)
        follow_up: Final = uuid.uuid4().hex
        served: Final = raw_answer(gateway, chat_body(model, follow_up, (_loaded_function(),), stream))
        assert served.status == 200 and f"marker-{follow_up}" in served.text, served.text
        _, body = only_request(wire)
        assert [tool["name"] for tool in rv.ITEMS.validate_python(body["tools"])] == ["echo"], body


def _drop_params_fields(source: DropSource) -> Mapping[str, JsonValue]:
    match source:
        case "off" | "deployment":
            return {}
        case "request":
            return {"drop_params": True}
        case "null":
            return {"drop_params": None}
        case "request_string":
            return {"drop_params": "true"}


@pytest.mark.parametrize("source", ["off", "request", "deployment", "null", "request_string"])
def test_drop_params_drops_the_search_and_every_deferral_and_without_it_the_call_is_refused(
    gateway: Gateway, source: DropSource
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(claude_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _unlisted(scenario, wire.url, **({"drop_params": True} if source == "deployment" else {}))
        tools: Final = (tool_search(), *deferred_tools())
        answer: Final = raw_answer(gateway, chat_body(model, marker, tools, False, **_drop_params_fields(source)))
        if source in ("off", "null"):
            _assert_refused_before_the_wire(answer, wire, hint=_MODEL_INFO_HINT, model=UNLISTED_CLAUDE)
            return
        assert answer.status == 200, answer.text
        assert answered_tool_calls(answer) == [(f"toolu_{marker}", "get_weather")], answer.text
        request, body = only_request(wire)
        assert request.target == "/v1/messages", request.target
        assert_no_search_and_no_deferral(body, DEFERRED_NAMES)
        assert_lands_once_as(model, (response_id(answer),), "success")


@pytest.mark.parametrize("provider", ["anthropic", "azure_ai"])
def test_model_info_supports_tool_search_turns_the_search_on_for_a_name_the_cost_map_does_not_know(
    gateway: Gateway, provider: Literal["anthropic", "azure_ai"]
) -> None:
    unknown_name: Final = f"claude-scripted-flagged-{uuid.uuid4().hex[:12]}"
    markers: Final = tuple(uuid.uuid4().hex for _ in range(6))
    with wire_server(claude_vendor) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model_info={"supports_tool_search": True},
            model=f"{provider}/{unknown_name}",
            api_base=wire.url,
            api_key="scripted-flagged-key",
        )
        answers: Final = tuple(
            raw_answer(gateway, chat_body(model, marker, (tool_search(), *deferred_tools()), False))
            for marker in markers
        )
        for answer, marker in zip(answers, markers, strict=True):
            assert_search_call_answered(answer, marker)
        received: Final = tuple(request for request in wire.drain() if request.method == "POST")
        assert len(received) == len(markers), [request.target for request in received]
        for request in received:
            assert_deferred_tools_kept(
                assert_search_tool_on_the_wire(provider, False, request, rv.JSON_OBJECT.validate_json(request.body))
            )
        assert_lands_once_as(model, tuple(response_id(answer) for answer in answers), "success")


@pytest.mark.parametrize(
    "search_tools",
    [
        (tool_search(execution="client"),),
        (tool_search(execution="server"), tool_search(execution="client")),
        (tool_search(execution=5),),
    ],
    ids=["client", "server-and-client", "int"],
)
def test_a_client_executed_or_malformed_search_is_refused_with_400_without_the_model_info_hint(
    gateway: Gateway, search_tools: tuple[dict[str, JsonValue], ...]
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(claude_vendor) as wire, gateway.scenario() as scenario:
        model: Final = claude_deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        refused: Final = raw_answer(gateway, chat_body(model, marker, (*search_tools, *deferred_tools()), False))
        _assert_refused_before_the_wire(refused, wire, hint=None, model=_CLAUDE)


def test_a_client_executed_search_with_drop_params_is_dropped_with_every_deferral(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(claude_vendor) as wire, gateway.scenario() as scenario:
        model: Final = claude_deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        tools: Final = (tool_search(execution="client"), *deferred_tools())
        answer: Final = raw_answer(gateway, chat_body(model, marker, tools, False, drop_params=True))
        assert answer.status == 200, answer.text
        _, body = only_request(wire)
        assert_no_search_and_no_deferral(body, DEFERRED_NAMES)


def _converse_vendor(request: Request) -> Reply:
    assert request.method == "POST" and unquote(request.target).endswith("/converse"), request.target
    return Reply(body=CONVERSE_RESPONSE)


def _converse(scenario: Scenario, wire_url: str, model: str, **extra: JsonValue) -> str:
    return scenario.model(
        model=model,
        api_base=wire_url,
        aws_access_key_id="AKIASCRIPTEDPROVIDER",
        aws_secret_access_key="scripted-secret",
        aws_region_name="us-east-1",
        aws_bedrock_runtime_endpoint=wire_url,
        **extra,
    )


def _converse_tools() -> tuple[dict[str, JsonValue], ...]:
    return (tool_search(execution="server"), _top_level_deferred(), _function_level_deferred())


@pytest.mark.parametrize(
    ("route_model", "hint"),
    [
        (f"bedrock/converse/{BEDROCK_MODEL}", _INVOKE_HINT),
        (f"bedrock/us.{BEDROCK_MODEL}", _INVOKE_HINT),
        (f"bedrock/converse/{_NOVA}", None),
    ],
    ids=["claude-converse-prefix", "claude-no-route-prefix", "nova"],
)
def test_bedrock_converse_refuses_tool_search_with_400(gateway: Gateway, route_model: str, hint: str | None) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_converse_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire.url, route_model)
        refused: Final = raw_answer(gateway, chat_body(model, marker, _converse_tools(), False))
        _assert_refused_before_the_wire(refused, wire, hint=hint, model=route_model.split("/")[-1])
        assert " on bedrock " in refused.text, refused.text


@pytest.mark.parametrize("source", ["request", "deployment"])
def test_bedrock_converse_with_drop_params_drops_the_search_and_every_deferral(
    gateway: Gateway, source: Literal["request", "deployment"]
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_converse_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _converse(
            scenario,
            wire.url,
            f"bedrock/converse/{BEDROCK_MODEL}",
            **({"drop_params": True} if source == "deployment" else {}),
        )
        extra: Final[Mapping[str, JsonValue]] = {"drop_params": True} if source == "request" else {}
        answer: Final = raw_answer(gateway, chat_body(model, marker, _converse_tools(), False, **extra))
        assert answer.status == 200 and "bedrock wire control" in answer.text, answer.text
        request, body = only_request(wire)
        assert unquote(request.target) == f"/model/{BEDROCK_MODEL}/converse", request.target
        sent: Final = json.dumps(body)
        assert "tool_search" not in sent and "defer_loading" not in sent, body
        specs: Final = rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_python(body["toolConfig"])["tools"])
        assert [rv.JSON_OBJECT.validate_python(spec["toolSpec"])["name"] for spec in specs] == [
            "get_weather",
            "get_stock_price",
        ], body
        assert_lands_once_as(model, (response_id(answer),), "success")


def _databricks_vendor(request: Request) -> Reply:
    assert request.method == "POST" and request.target.endswith("/chat/completions"), request.target
    marker: Final = rv.newest_marker(request.body.decode()) or "health"
    return chat_reply(f"chatcmpl-{marker}", "databricks-claude", f"answer marker-{marker}", stream=False)


@pytest.mark.parametrize("drop_params", [False, True], ids=["refused", "dropped"])
def test_databricks_claude_answers_400_or_drops_the_search_under_drop_params(
    gateway: Gateway, drop_params: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_databricks_vendor) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"databricks/databricks-{_CLAUDE}", api_base=f"{wire.url}/serving-endpoints", api_key="scripted-dbx"
        )
        answer: Final = raw_answer(
            gateway,
            chat_body(model, marker, _converse_tools(), False, **({"drop_params": True} if drop_params else {})),
        )
        if not drop_params:
            _assert_refused_before_the_wire(answer, wire, hint=None, model=f"databricks-{_CLAUDE}")
            assert " on databricks " in answer.text, answer.text
            return
        assert answer.status == 200 and f"answer marker-{marker}" in answer.text, answer.text
        _, body = only_request(wire)
        sent: Final = json.dumps(body)
        assert "tool_search" not in sent and "defer_loading" not in sent, body
        names: Final = [
            rv.JSON_OBJECT.validate_python(tool["function"])["name"] for tool in rv.ITEMS.validate_python(body["tools"])
        ]
        assert names == ["get_weather", "get_stock_price"], body


def test_databricks_non_claude_gets_the_tools_as_sent(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_databricks_vendor) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="databricks/databricks-meta-llama-4-maverick",
            api_base=f"{wire.url}/serving-endpoints",
            api_key="scripted-dbx",
        )
        tools: Final = (tool_search(), _top_level_deferred())
        answer: Final = raw_answer(gateway, chat_body(model, marker, tools, False))
        assert answer.status == 200 and f"answer marker-{marker}" in answer.text, answer.text
        _, body = only_request(wire)
        assert rv.ITEMS.validate_python(body["tools"]) == [dict(tool) for tool in tools], body


def test_a_fallback_from_openai_to_claude_carries_the_tool_search_to_claude(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex

    def overloaded_openai(request: Request) -> Reply:
        return Reply(status=429, body=json.dumps({"error": {"message": "scripted 429", "type": "rate_limit"}}).encode())

    with (
        wire_server(overloaded_openai) as openai_wire,
        wire_server(claude_vendor) as claude_wire,
        gateway.scenario() as scenario,
    ):
        primary: Final = scenario.model(model="openai/gpt-5.6", api_base=f"{openai_wire.url}/v1", api_key="scripted")
        claude: Final = claude_deployment(scenario, "anthropic", claude_wire.url, gateway.upstream_url)
        answer: Final = raw_answer(
            gateway, chat_body(primary, marker, (tool_search(), *deferred_tools()), False, fallbacks=[claude])
        )
        assert_search_call_answered(answer, marker)
        posted: Final = [request for request in openai_wire.drain() if request.method == "POST"]
        assert len(posted) == 1 and f"marker-{marker}" in posted[0].body.decode(), posted
        request, body = only_request(claude_wire)
        assert_deferred_tools_kept(assert_search_tool_on_the_wire("anthropic", False, request, body))
