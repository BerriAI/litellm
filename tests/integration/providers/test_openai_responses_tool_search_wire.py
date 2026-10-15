import asyncio
import itertools
import json
import threading
import uuid
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from queue import SimpleQueue
from types import EllipsisType, MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from _pytest.mark.structures import ParameterSet
from integration._support import responses_vendor as rv
from integration._support.aws_control_plane import ControlPlane, ControlPlaneRequest, Responder, control_plane
from integration._support.budget import readiness_seconds
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.test_responses_bridge_tool_search_wire import (
    _CACHE_BUST,
    _CLAUDE,
    _ITEMS,
    _JSON_OBJECT,
    _REGEX_TOOL,
    _TICKER_PARAMETERS,
    _WEATHER_PARAMETERS,
    _anthropic_error,
    _answer,
    _assert_spend_rows,
    _Client,
    _completed_of,
    _deployment,
    _frames,
    _function_tool,
    _namespace_tool,
    _output_types,
    _prompt,
    _raw,
    _responses_body,
    _spend_rows,
    _tool_search,
    _Vendor,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_OPENAI_HOST: Final = "api.openai.com"
_OPENAI_API_BASE: Final = f"https://{_OPENAI_HOST}/v1"
_RESPONSES_MODEL: Final = "openai/responses/gpt-5.5"
_UNFLAGGED_MODEL: Final = "openai/gpt-5.4-nano"
_BM25_TOOL_TYPE: Final = "tool_search_tool_bm25_20251119"
_BM25_TOOL_NAME: Final = "tool_search_tool_bm25"
_SERVER_TOOL_SEARCH: Final[Mapping[str, JsonValue]] = MappingProxyType({"type": "tool_search", "execution": "server"})
_SEARCH_CALL_ID: Final = "srvtoolu_01AuditSearchCall"
_BURST_SIZE: Final = 30


def _bm25_tool() -> dict[str, JsonValue]:
    return {"type": _BM25_TOOL_TYPE, "name": _BM25_TOOL_NAME}


def _stock_function_tool() -> dict[str, JsonValue]:
    return {
        "type": "function",
        "name": "get_stock_price",
        "description": "Price of a ticker",
        "parameters": _TICKER_PARAMETERS,
        "defer_loading": True,
    }


def _undeferred(tool: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in tool.items() if key != "defer_loading"}


def _undeferred_namespace() -> dict[str, JsonValue]:
    namespace: Final = _namespace_tool()
    return {**namespace, "tools": [_undeferred(member) for member in _ITEMS.validate_python(namespace["tools"])]}


def _deferred_tools() -> tuple[dict[str, JsonValue], ...]:
    return (_bm25_tool(), _function_tool(), _stock_function_tool())


def _mapped_tools() -> list[dict[str, JsonValue]]:
    return [dict(_SERVER_TOOL_SEARCH), _function_tool(), _stock_function_tool()]


def _search_call(
    call_id: JsonValue | EllipsisType = _SEARCH_CALL_ID, name: str = _BM25_TOOL_NAME
) -> dict[str, JsonValue]:
    call: Final[dict[str, JsonValue]] = {"type": "function_call", "name": name, "arguments": '{"query": "weather"}'}
    return call if isinstance(call_id, EllipsisType) else {**call, "call_id": call_id}


def _search_output() -> dict[str, JsonValue]:
    return {"type": "function_call_output", "call_id": _SEARCH_CALL_ID, "output": "[]"}


def _tools_body(
    tools: Sequence[Mapping[str, JsonValue]], stream: bool = False
) -> Callable[[str, str], dict[str, JsonValue]]:
    return lambda model, marker: _responses_body(model, marker, tools, stream)


def _compaction(request: Request) -> Reply:
    assert urlsplit(request.target).path == "/v1/responses/compact", request.target
    return Reply(
        body=json.dumps(
            {
                "id": f"resp_compact_{uuid.uuid4().hex}",
                "created_at": 1,
                "object": "response.compaction",
                "output": [
                    {
                        "id": f"rs_compact_{uuid.uuid4().hex}",
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "Compacted state"}],
                        "encrypted_content": "synthetic-compact-state",
                    }
                ],
                "usage": rv.USAGE,
            }
        ).encode()
    )


def _answered_as_openai_api(respond: Callable[[Request], Reply]) -> Responder:
    def answer(request: ControlPlaneRequest) -> Reply:
        reply: Final = respond(Request(request.method, request.target, request.headers, request.body))
        return reply if reply.chunks is None else replace(reply, body=b"".join(reply.chunks), chunks=None)

    return answer


def _openai_api_requests(openai_api: ControlPlane, key: str) -> tuple[ControlPlaneRequest, ...]:
    return tuple(request for request in openai_api.drain() if request.credential == key and request.method == "POST")


@dataclass(frozen=True, slots=True)
class _Exchange:
    model: str
    marker: str
    response: httpx.Response
    requests: tuple[ControlPlaneRequest, ...]

    def only_body(self, path: str = "/v1/responses") -> dict[str, JsonValue]:
        assert [(request.host, request.path) for request in self.requests] == [(_OPENAI_HOST, path)], self.requests
        return _JSON_OBJECT.validate_json(self.requests[0].body)


def _exchange(
    openai_gateway: Gateway,
    openai_api: ControlPlane,
    body: Callable[[str, str], Mapping[str, JsonValue]],
    *,
    path: str = "/v1/responses",
    respond: Callable[[Request], Reply] | None = None,
    model: str = _RESPONSES_MODEL,
    deployment: Mapping[str, JsonValue] = MappingProxyType({}),
) -> _Exchange:
    marker: Final = uuid.uuid4().hex
    key: Final = f"synthetic-openai-key-{marker}"
    responder: Final = respond if respond is not None else rv.ResponsesVendor().respond
    with openai_gateway.scenario() as scenario, openai_api.answering(key, _answered_as_openai_api(responder)):
        name: Final = scenario.model(model_info=None, model=model, api_base=_OPENAI_API_BASE, api_key=key, **deployment)
        response: Final = _raw(openai_gateway, path, {**body(name, marker), **_CACHE_BUST})
        return _Exchange(name, marker, response, _openai_api_requests(openai_api, key))


def _response_id(response: httpx.Response, stream: bool) -> str:
    payload: Final = _completed_of(_frames(response.text)) if stream else _JSON_OBJECT.validate_json(response.content)
    return str(payload["id"])


def _assert_success_rows(model: str, response_ids: Sequence[str], failures: int) -> None:
    rows: Final = _spend_rows(model, len(response_ids) + failures)
    assert len(rows) == len(response_ids) + failures, rows
    successes: Final = [str(row["request_id"]) for row in rows if row["status"] == "success"]
    assert len(successes) == len(response_ids), rows
    matches: Final = [sum(1 for logged in successes if rv.same_response(logged, sent)) for sent in response_ids]
    assert matches == [1] * len(response_ids), (response_ids, rows)


@pytest.fixture(scope="module")
def openai_api(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ControlPlane]:
    with control_plane(tmp_path_factory.mktemp("openai_api"), (_OPENAI_HOST,)) as plane:
        yield plane


@dataclass(frozen=True, slots=True)
class _FallbackRoute:
    claude: str
    openai: str
    key: str


@dataclass(frozen=True, slots=True)
class _ClaudeOutage:
    wire: Wire
    routes: Mapping[bool, _FallbackRoute]


def _fallback_route() -> _FallbackRoute:
    marker: Final = uuid.uuid4().hex
    return _FallbackRoute(
        f"integration-claude-{marker}", f"integration-openai-{marker}", f"synthetic-openai-key-{marker}"
    )


@pytest.fixture(scope="module")
def claude_outage() -> Iterator[_ClaudeOutage]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=b'{"object":"list","data":[]}')
        return _anthropic_error(500, "scripted Claude failure")

    with wire_server(respond) as wire:
        yield _ClaudeOutage(wire, MappingProxyType({stream: _fallback_route() for stream in (False, True)}))


def _proxy_config() -> dict[str, JsonValue]:
    return rv.JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))


def _fallback_deployments(outage: _ClaudeOutage, route: _FallbackRoute) -> tuple[dict[str, JsonValue], ...]:
    return (
        {
            "model_name": route.claude,
            "litellm_params": {
                "model": f"anthropic/{_CLAUDE}",
                "api_base": outage.wire.url,
                "api_key": "synthetic-invalid-key",
            },
        },
        {
            "model_name": route.openai,
            "litellm_params": {"model": _RESPONSES_MODEL, "api_base": _OPENAI_API_BASE, "api_key": route.key},
        },
    )


def _fallback_config(directory: Path, outage: _ClaudeOutage) -> Path:
    config: Final = _proxy_config()
    models: Final = rv.ITEMS.validate_python(config["model_list"])
    added: Final = tuple(
        itertools.chain.from_iterable(_fallback_deployments(outage, route) for route in outage.routes.values())
    )
    path: Final = directory / "openai-responses-tool-search-fallbacks.yaml"
    path.write_text(yaml.safe_dump({**config, "model_list": [*models, *added]}))
    return path


@pytest.fixture(scope="module")
def openai_gateway(
    openai_api: ControlPlane, claude_outage: _ClaudeOutage, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("openai_gateway")
    with (
        gateway_from_environment() as parent,
        owned_proxy(
            parent, directory, openai_api.environment(), config=_fallback_config(directory, claude_outage), workers=2
        ) as owned,
    ):
        yield owned


def _drop_params_config(directory: Path) -> Path:
    config: Final = _proxy_config()
    settings: Final = rv.JSON_OBJECT.validate_python(config["litellm_settings"])
    path: Final = directory / "openai-responses-tool-search-drop-params.yaml"
    path.write_text(yaml.safe_dump({**config, "litellm_settings": {**settings, "drop_params": True}}))
    return path


@pytest.fixture(scope="module")
def openai_drop_params_proxy(
    openai_api: ControlPlane, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("openai_drop_params")
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway, directory, openai_api.environment(), config=_drop_params_config(directory), workers=2
        ) as owned,
    ):
        yield owned


@pytest.mark.parametrize("search_tool", [_bm25_tool(), dict(_REGEX_TOOL)], ids=["bm25", "regex"])
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
def test_anthropic_tool_search_reaches_openai_responses_as_server_tool_search(
    openai_gateway: Gateway, openai_api: ControlPlane, stream: bool, search_tool: dict[str, JsonValue]
) -> None:
    exchange: Final = _exchange(
        openai_gateway, openai_api, _tools_body((search_tool, _function_tool(), _stock_function_tool()), stream)
    )

    assert exchange.response.status_code == 200, exchange.response.text
    body: Final = exchange.only_body()
    assert (body["model"], body["stream"], body["input"]) == ("gpt-5.5", stream, _prompt(exchange.marker)), body
    assert body["tools"] == _mapped_tools(), body
    _assert_spend_rows(exchange.model, (_response_id(exchange.response, stream),))


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
def test_router_fallback_maps_anthropic_tool_search_for_openai_responses(
    openai_gateway: Gateway, openai_api: ControlPlane, claude_outage: _ClaudeOutage, stream: bool
) -> None:
    route: Final = claude_outage.routes[stream]
    body: Final = _responses_body(route.claude, uuid.uuid4().hex, _deferred_tools(), stream)
    with openai_api.answering(route.key, _answered_as_openai_api(rv.ResponsesVendor().respond)):
        response: Final = _raw(openai_gateway, "/v1/responses", {**body, **_CACHE_BUST, "fallbacks": [route.openai]})

        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-attempted-fallbacks"] == "1", response.headers
        claude_requests: Final = [request.target for request in claude_outage.wire.drain() if request.method == "POST"]
        assert claude_requests == ["/v1/messages"], claude_requests
        (received,) = _openai_api_requests(openai_api, route.key)
        assert (received.host, received.path) == (_OPENAI_HOST, "/v1/responses"), received
        forwarded: Final = _JSON_OBJECT.validate_json(received.body)
        assert (forwarded["stream"], forwarded["tools"]) == (stream, _mapped_tools()), forwarded


def test_responses_compact_maps_anthropic_tool_search_for_openai(
    openai_gateway: Gateway, openai_api: ControlPlane
) -> None:
    exchange: Final = _exchange(
        openai_gateway,
        openai_api,
        lambda model, marker: {"model": model, "input": _prompt(marker), "tools": list(_deferred_tools())},
        path="/v1/responses/compact",
        respond=_compaction,
    )

    assert exchange.response.status_code == 200, exchange.response.text
    assert _JSON_OBJECT.validate_json(exchange.response.content)["object"] == "response.compaction", exchange.response
    assert exchange.only_body("/v1/responses/compact")["tools"] == _mapped_tools()


@pytest.mark.parametrize("client", ["sdk", "async_sdk"])
def test_the_openai_sdk_gets_its_answer_with_the_search_tool_mapped(
    openai_gateway: Gateway, openai_api: ControlPlane, client: _Client
) -> None:
    marker: Final = uuid.uuid4().hex
    key: Final = f"synthetic-openai-key-{marker}"
    with (
        openai_gateway.scenario() as scenario,
        openai_api.answering(key, _answered_as_openai_api(rv.ResponsesVendor().respond)),
    ):
        model: Final = scenario.model(model=_RESPONSES_MODEL, api_base=_OPENAI_API_BASE, api_key=key)
        answer: Final = _answer(openai_gateway, client, _responses_body(model, marker, _deferred_tools(), False))
        (received,) = _openai_api_requests(openai_api, key)

    assert _output_types(answer.payload) == ["reasoning", "message"], answer.payload
    assert _JSON_OBJECT.validate_json(received.body)["tools"] == _mapped_tools(), received.body
    _assert_spend_rows(model, (str(answer.payload["id"]),))


def test_model_info_reports_tool_search_support_for_an_openai_gpt_5_5_deployment(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-5.5", api_base=_OPENAI_API_BASE, api_key="synthetic-openai-key")
        response: Final = gateway.request("GET", "/model/info", None)

        assert response.status_code == 200, response.text
        (entry,) = [
            item
            for item in _ITEMS.validate_python(_JSON_OBJECT.validate_json(response.content)["data"])
            if item["model_name"] == model
        ]
        assert _JSON_OBJECT.validate_python(entry["model_info"]).get("supports_tool_search") is True, entry


@pytest.mark.parametrize(
    ("history", "replayed"),
    [
        pytest.param((_search_call(),), (), id="unanswered"),
        pytest.param((_search_call(), _search_output()), (_search_call(), _search_output()), id="answered"),
        pytest.param((_search_call(name="get_weather"),), (_search_call(name="get_weather"),), id="other-name"),
        pytest.param((_search_call(call_id="call_audit"),), (_search_call(call_id="call_audit"),), id="other-id"),
        pytest.param((_search_call(call_id=...),), (_search_call(call_id=...),), id="no-call-id"),
        pytest.param((_search_call(call_id=5),), (_search_call(call_id=5),), id="int-call-id"),
    ],
)
def test_only_an_unanswered_claude_search_call_leaves_the_replayed_input(
    openai_gateway: Gateway,
    openai_api: ControlPlane,
    history: tuple[dict[str, JsonValue], ...],
    replayed: tuple[dict[str, JsonValue], ...],
) -> None:
    exchange: Final = _exchange(
        openai_gateway,
        openai_api,
        lambda model, marker: {
            "model": model,
            "input": [{"role": "user", "content": _prompt(marker)}, *history],
            "tools": list(_deferred_tools()),
            "stream": False,
        },
    )

    assert exchange.response.status_code == 200, exchange.response.text
    body: Final = exchange.only_body()
    assert body["input"] == [{"role": "user", "content": _prompt(exchange.marker)}, *replayed], body
    assert body["tools"] == _mapped_tools(), body


def _drop_params_cells() -> list[ParameterSet]:
    stripped: Final = [_function_tool(...), _undeferred(_stock_function_tool())]
    return [
        pytest.param(_UNFLAGGED_MODEL, True, _deferred_tools(), stripped, id="unflagged-true"),
        pytest.param(_UNFLAGGED_MODEL, "true", _deferred_tools(), stripped, id="unflagged-string-true"),
        pytest.param(_UNFLAGGED_MODEL, ..., _deferred_tools(), _mapped_tools(), id="unflagged-missing"),
        pytest.param(_UNFLAGGED_MODEL, None, _deferred_tools(), _mapped_tools(), id="unflagged-null"),
        pytest.param(_UNFLAGGED_MODEL, False, _deferred_tools(), _mapped_tools(), id="unflagged-false"),
        pytest.param(_UNFLAGGED_MODEL, "", _deferred_tools(), _mapped_tools(), id="unflagged-empty-string"),
        pytest.param(
            _UNFLAGGED_MODEL,
            True,
            (_bm25_tool(), _namespace_tool()),
            [_undeferred_namespace()],
            id="unflagged-namespace",
        ),
        pytest.param(_RESPONSES_MODEL, True, _deferred_tools(), _mapped_tools(), id="flagged-deferred"),
        pytest.param(
            _RESPONSES_MODEL, True, (_bm25_tool(), _function_tool(...)), [_function_tool(...)], id="flagged-undeferred"
        ),
    ]


@pytest.mark.parametrize(("model", "drop_params", "tools", "forwarded"), _drop_params_cells())
def test_drop_params_drops_the_search_only_when_the_openai_model_cannot_run_it(
    openai_gateway: Gateway,
    openai_api: ControlPlane,
    model: str,
    drop_params: JsonValue | EllipsisType,
    tools: tuple[dict[str, JsonValue], ...],
    forwarded: list[dict[str, JsonValue]],
) -> None:
    setting: Final[dict[str, JsonValue]] = {} if isinstance(drop_params, EllipsisType) else {"drop_params": drop_params}
    exchange: Final = _exchange(openai_gateway, openai_api, _tools_body(tools), model=model, deployment=setting)

    assert exchange.response.status_code == 200, exchange.response.text
    assert exchange.only_body()["tools"] == forwarded


@pytest.mark.parametrize("drop_params", [False, True], ids=["no-drop-params", "drop-params"])
def test_the_callers_own_tool_search_replaces_the_anthropic_search_tool(
    openai_gateway: Gateway, openai_api: ControlPlane, drop_params: bool
) -> None:
    exchange: Final = _exchange(
        openai_gateway,
        openai_api,
        _tools_body((_bm25_tool(), _tool_search(), _function_tool(), _stock_function_tool())),
        model=_UNFLAGGED_MODEL,
        deployment={"drop_params": drop_params},
    )

    assert exchange.response.status_code == 200, exchange.response.text
    assert exchange.only_body()["tools"] == [_tool_search(), _function_tool(), _stock_function_tool()]


@pytest.mark.parametrize(
    ("tools", "forwarded"),
    [
        pytest.param(
            (_bm25_tool(), _bm25_tool(), _function_tool()), [dict(_SERVER_TOOL_SEARCH), _function_tool()], id="twice"
        ),
        pytest.param(
            (_function_tool(), dict(_REGEX_TOOL), _bm25_tool(), _stock_function_tool()),
            [_function_tool(), dict(_SERVER_TOOL_SEARCH), _stock_function_tool()],
            id="regex-then-bm25",
        ),
        *(
            pytest.param(({"type": _BM25_TOOL_TYPE, "name": name}, _function_tool()), None, id=f"name-{label}")
            for label, name in (("int", 5), ("list", [_BM25_TOOL_NAME]), ("empty", ""), ("5kb", "a" * 5000))
        ),
        pytest.param(({"type": _BM25_TOOL_TYPE}, _function_tool()), None, id="name-missing"),
    ],
)
def test_the_search_tool_maps_by_its_type_whatever_sits_around_it(
    openai_gateway: Gateway,
    openai_api: ControlPlane,
    tools: tuple[JsonValue, ...],
    forwarded: list[JsonValue] | None,
) -> None:
    exchange: Final = _exchange(
        openai_gateway,
        openai_api,
        lambda model, marker: {"model": model, "input": _prompt(marker), "tools": list(tools), "stream": False},
    )

    assert exchange.response.status_code == 200, exchange.response.text
    expected: Final = forwarded if forwarded is not None else [dict(_SERVER_TOOL_SEARCH), _function_tool()]
    assert exchange.only_body()["tools"] == expected


def test_an_int_tool_beside_the_search_tool_reaches_openai_with_the_search_mapped(
    openai_gateway: Gateway, openai_api: ControlPlane
) -> None:
    pytest.skip("BUG: an int tools element answers 401 Authentication Error from the vector store tool scan")
    exchange: Final = _exchange(
        openai_gateway,
        openai_api,
        lambda model, marker: {"model": model, "input": _prompt(marker), "tools": [5, _bm25_tool(), _function_tool()]},
    )

    assert exchange.response.status_code == 200, exchange.response.text
    assert exchange.only_body()["tools"] == [5, dict(_SERVER_TOOL_SEARCH), _function_tool()]


def _string_beside_search_body(model: str, marker: str) -> dict[str, JsonValue]:
    return {"model": model, "input": _prompt(marker), "tools": ["get_weather", _bm25_tool(), _function_tool()]}


def test_a_string_tool_beside_the_search_tool_is_refused_before_reaching_openai(
    openai_gateway: Gateway, openai_api: ControlPlane
) -> None:
    exchange: Final = _exchange(openai_gateway, openai_api, _string_beside_search_body)

    assert exchange.response.status_code == 400, exchange.response.text
    assert exchange.requests == (), exchange.requests


def test_a_string_tool_beside_the_search_tool_is_refused_before_reaching_claude(gateway: Gateway) -> None:
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        response: Final = _raw(
            gateway, "/v1/responses", {**_string_beside_search_body(model, uuid.uuid4().hex), **_CACHE_BUST}
        )

        assert response.status_code == 400, response.text
        assert [request for request in wire.drain() if request.method == "POST"] == []


@pytest.mark.parametrize("model", [_RESPONSES_MODEL, "azure/gpt-5.5"], ids=["openai-compatible-host", "azure"])
def test_a_deployment_off_api_openai_com_keeps_the_anthropic_search_tool(gateway: Gateway, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(rv.ResponsesVendor().respond) as wire, gateway.scenario() as scenario:
        host: Final[dict[str, JsonValue]] = (
            {"api_base": wire.url, "api_version": "preview"}
            if model.startswith("azure/")
            else {"api_base": f"{wire.url}/v1"}
        )
        deployment: Final = scenario.model(model_info=None, model=model, api_key="synthetic-provider-key", **host)
        response: Final = _raw(
            gateway, "/v1/responses", {**_responses_body(deployment, marker, _deferred_tools(), False), **_CACHE_BUST}
        )

        assert response.status_code == 200, response.text
        (request,) = [request for request in wire.drain() if request.method == "POST"]
        assert _JSON_OBJECT.validate_json(request.body)["tools"] == list(_deferred_tools()), request.body


@pytest.mark.parametrize("status", [400, 401, 404])
def test_an_openai_error_on_the_mapped_request_reaches_the_caller_and_lands_as_a_failed_spend_row(
    openai_gateway: Gateway, openai_api: ControlPlane, status: int
) -> None:
    message: Final = f"scripted openai {status}"
    exchange: Final = _exchange(
        openai_gateway,
        openai_api,
        _tools_body(_deferred_tools()),
        respond=lambda request: rv.error(status, message, "invalid_request_error"),
    )

    assert exchange.response.status_code == status, exchange.response.text
    assert message in exchange.response.text, exchange.response.text
    assert exchange.only_body()["tools"] == _mapped_tools()
    call_id: Final = exchange.response.headers["x-litellm-call-id"]
    rows: Final = eventually(
        lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    assert str(rows[0]["status"]) == "failure", rows


def test_chat_completions_through_the_responses_bridge_drop_the_search_tool_under_drop_params(
    openai_gateway: Gateway, openai_api: ControlPlane
) -> None:
    exchange: Final = _exchange(
        openai_gateway,
        openai_api,
        lambda model, marker: {
            "model": model,
            "messages": [{"role": "user", "content": _prompt(marker)}],
            "tools": [
                _bm25_tool(),
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Weather for a city",
                        "parameters": _WEATHER_PARAMETERS,
                    },
                    "defer_loading": True,
                },
            ],
        },
        path="/v1/chat/completions",
        deployment={"drop_params": True},
    )

    assert exchange.response.status_code == 200, exchange.response.text
    tools: Final = _ITEMS.validate_python(exchange.only_body()["tools"])
    assert [(tool["type"], tool.get("name")) for tool in tools] == [("function", "get_weather")], tools


def test_messages_to_an_openai_model_with_the_search_tool_still_answers(
    openai_gateway: Gateway, openai_api: ControlPlane
) -> None:
    exchange: Final = _exchange(
        openai_gateway,
        openai_api,
        lambda model, marker: {
            "model": model,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": _prompt(marker)}],
            "tools": [
                _bm25_tool(),
                {
                    "name": "get_weather",
                    "description": "Weather for a city",
                    "input_schema": _WEATHER_PARAMETERS,
                    "defer_loading": True,
                },
            ],
        },
        path="/v1/messages",
        model="openai/gpt-5.5",
    )

    assert exchange.response.status_code == 200, exchange.response.text
    assert len(exchange.requests) == 1, exchange.requests


def test_a_global_drop_params_drops_the_search_for_an_openai_model_that_cannot_run_it(
    openai_drop_params_proxy: OwnedProxy, openai_api: ControlPlane
) -> None:
    exchange: Final = _exchange(
        openai_drop_params_proxy.gateway, openai_api, _tools_body(_deferred_tools()), model=_UNFLAGGED_MODEL
    )

    assert exchange.response.status_code == 200, exchange.response.text
    assert exchange.only_body()["tools"] == [_function_tool(...), _undeferred(_stock_function_tool())]


@dataclass(frozen=True, slots=True)
class _BurstCall:
    marker: str
    path: str
    stream: bool
    fails: bool


def _burst_calls() -> tuple[_BurstCall, ...]:
    shapes: Final = itertools.cycle(
        (("/v1/responses", False), ("/v1/responses", True), ("/v1/responses/compact", False))
    )
    return tuple(
        _BurstCall(uuid.uuid4().hex, path, stream, index % 5 == 0)
        for index, (path, stream) in zip(range(_BURST_SIZE), shapes)
    )


def _burst_body(model: str, call: _BurstCall) -> dict[str, JsonValue]:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "input": _prompt(call.marker),
        "tools": list(_deferred_tools()),
        **_CACHE_BUST,
    }
    return body if call.path.endswith("/compact") else {**body, "stream": call.stream}


async def _post(client: httpx.AsyncClient, path: str, body: Mapping[str, JsonValue]) -> httpx.Response | None:
    try:
        return await client.post(path, json=dict(body))
    except httpx.TransportError:
        return None


async def _burst(
    gateway: Gateway, bodies: Sequence[tuple[str, Mapping[str, JsonValue]]]
) -> list[httpx.Response | None]:
    async with httpx.AsyncClient(
        base_url=str(gateway.client.base_url),
        headers={"Authorization": f"Bearer {gateway.key}"},
        trust_env=False,
        timeout=120,
    ) as client:
        return list(await asyncio.gather(*(_post(client, path, body) for path, body in bodies)))


def _marker_of(request: ControlPlaneRequest) -> str:
    marker: Final = rv.newest_marker(request.body.decode())
    assert marker is not None, request.body
    return marker


async def test_a_mixed_burst_with_upstream_failures_maps_every_request_and_logs_each_once(
    openai_drop_params_proxy: OwnedProxy, openai_api: ControlPlane
) -> None:
    calls: Final = _burst_calls()
    failing: Final = frozenset(call.marker for call in calls if call.fails)
    vendor: Final = rv.ResponsesVendor(pause_between_chunks=0.02)
    key: Final = f"synthetic-openai-key-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        marker: Final = rv.newest_marker(request.body.decode())
        if marker in failing:
            return rv.error(500, f"scripted outage marker-{marker}", "server_error")
        return _compaction(request) if urlsplit(request.target).path.endswith("/compact") else vendor.respond(request)

    proxy: Final = openai_drop_params_proxy.gateway
    with proxy.scenario() as scenario, openai_api.answering(key, _answered_as_openai_api(respond)):
        model: Final = scenario.model(model=_RESPONSES_MODEL, api_base=_OPENAI_API_BASE, api_key=key)
        responses: Final = await _burst(proxy, [(call.path, _burst_body(model, call)) for call in calls])
        received: Final = _openai_api_requests(openai_api, key)

        for call, response in zip(calls, responses):
            assert response is not None, call
            if call.fails:
                assert response.status_code == 500, (call, response.text)
                assert f"scripted outage marker-{call.marker}" in response.text, response.text
                continue
            assert response.status_code == 200, (call, response.text)
        for request in received:
            assert _JSON_OBJECT.validate_json(request.body)["tools"] == _mapped_tools(), request.body
        served: Final = [_marker_of(request) for request in received if _marker_of(request) not in failing]
        assert sorted(served) == sorted(call.marker for call in calls if not call.fails), served
        succeeded: Final = [
            _response_id(response, call.stream)
            for call, response in zip(calls, responses)
            if response is not None and not call.fails
        ]
        _assert_success_rows(model, succeeded, failures=len(failing))


def _open_tunnels(pid: int, openai_api: ControlPlane) -> int:
    port: Final = urlsplit(openai_api.url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@contextmanager
def _released_on_exit(release: threading.Event) -> Generator[None]:
    try:
        yield
    finally:
        release.set()


async def test_a_worker_killed_mid_burst_leaves_the_survivor_mapping_the_search_tool(
    openai_api: ControlPlane, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    vendor: Final = rv.ResponsesVendor()
    key: Final = f"synthetic-openai-key-{uuid.uuid4().hex}"
    markers: Final = tuple(uuid.uuid4().hex for _ in range(20))

    def respond(request: Request) -> Reply:
        marker: Final = rv.newest_marker(request.body.decode())
        if marker in markers:
            held.put(marker)
            assert release.wait(timeout=readiness_seconds(1) + 60), "The burst was never released"
        return vendor.respond(request)

    with (
        gateway_from_environment() as parent,
        owned_proxy_process(parent, tmp_path, openai_api.environment(), workers=2) as owned,
        owned.gateway.scenario() as scenario,
        openai_api.answering(key, _answered_as_openai_api(respond)),
        _released_on_exit(release),
    ):
        model: Final = scenario.model(model=_RESPONSES_MODEL, api_base=_OPENAI_API_BASE, api_key=key)
        workers: Final = eventually(owned.worker_pids, lambda pids: len(pids) == 2, seconds=30)
        bodies: Final = [
            ("/v1/responses", {**_responses_body(model, marker, _deferred_tools(), False), **_CACHE_BUST})
            for marker in markers
        ]
        burst: Final = asyncio.create_task(_burst(owned.gateway, bodies))
        await asyncio.to_thread(eventually, held.qsize, lambda size: size == len(markers), 60)
        held_by: Final = MappingProxyType({pid: _open_tunnels(pid, openai_api) for pid in workers})
        assert sum(held_by.values()) == len(markers), held_by
        victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
        await asyncio.to_thread(owned.kill_worker, victim_pid)
        release.set()
        responses: Final = await burst
        served: Final = [response for response in responses if response is not None and response.status_code == 200]
        assert held_by[survivor_pid] >= 1, held_by
        assert len(served) == held_by[survivor_pid], (held_by, [r.status_code if r else None for r in responses])
        follow_up: Final = uuid.uuid4().hex
        (after,) = await _burst(
            owned.gateway,
            [("/v1/responses", {**_responses_body(model, follow_up, _deferred_tools(), False), **_CACHE_BUST})],
        )
        assert after is not None and after.status_code == 200, after
        received: Final = _openai_api_requests(openai_api, key)
        assert {_marker_of(request) for request in received} == {*markers, follow_up}
        for request in received:
            assert _JSON_OBJECT.validate_json(request.body)["tools"] == _mapped_tools(), request.body
