import asyncio
import itertools
import json
import re
import socket
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import EllipsisType, MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import unquote, urlsplit

import httpx
import psutil
import pytest
import yaml
from _pytest.mark.structures import ParameterSet
from integration._support import claude_code as cc
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support import tool_search as ts
from integration._support.anthropic_thinking import BEDROCK_MODEL, EVENT_STREAM, Event, aws_chunks, sse_chunks
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, string_value
from integration._support.openai_wire import answering_model_discovery
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_CLAUDE: Final = cc.OPUS
_BEDROCK_CLAUDE: Final = "anthropic.claude-sonnet-5-5"
_MISTRAL: Final = "mistral-large-latest"
_CONFIG_MODEL: Final = "responses-bridge-tool-search-chaos"
_ANTHROPIC_BASE: Final = "http://api.anthropic.com"
_REGEX_TOOL_TYPE: Final = "tool_search_tool_regex_20251119"
_REGEX_TOOL_NAME: Final = "tool_search_tool_regex"
_REGEX_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType({"type": _REGEX_TOOL_TYPE, "name": _REGEX_TOOL_NAME})
_TOOL_SEARCH_BETA: Final = "advanced-tool-use-2025-11-20"
_SEARCH_ARGUMENTS: Final = '{"pattern": "weather"}'
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_CACHE_BUST: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
_OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120
_WEATHER_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

_ORDER_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}},
    "required": ["order_id"],
}

Endpoint = Literal["responses", "chat", "messages"]
StoredShape: TypeAlias = Literal["tool_use", "search_only", "search_and_text"]
OffAnthropicRoute: TypeAlias = Literal["mistral", "bedrock_converse"]


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str
    call_id: str


def _prompt(marker: str) -> str:
    return f"Weather in Paris marker-{marker}"


def _search_result(marker: str) -> Event:
    return {
        "type": "tool_search_tool_result",
        "tool_use_id": f"srvtoolu_{marker}",
        "content": {
            "type": "tool_search_tool_search_result",
            "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
        },
    }


def _message(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{marker}",
        "type": "message",
        "role": "assistant",
        "model": _CLAUDE,
        "content": [
            {"type": "text", "text": f"searching marker-{marker}"},
            {
                "type": "server_tool_use",
                "id": f"srvtoolu_{marker}",
                "name": _REGEX_TOOL_NAME,
                "input": json.loads(_SEARCH_ARGUMENTS),
            },
            _search_result(marker),
            {"type": "tool_use", "id": f"toolu_{marker}", "name": "get_weather", "input": {"city": "Paris"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 7, "server_tool_use": {"tool_search_requests": 1}},
    }


def _events(marker: str) -> tuple[Event, ...]:
    opened: Final = {**_message(marker), "content": [], "stop_reason": None}
    return (
        {"type": "message_start", "message": opened},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": f"searching marker-{marker}"},
        },
        {"type": "content_block_stop", "index": 0},
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
        {
            "type": "content_block_start",
            "index": 3,
            "content_block": {"type": "tool_use", "id": f"toolu_{marker}", "name": "get_weather", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 3,
            "delta": {"type": "input_json_delta", "partial_json": '{"city": "Paris"}'},
        },
        {"type": "content_block_stop", "index": 3},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    )


@dataclass(frozen=True, slots=True)
class _Vendor:
    pause: float = 0

    def respond(self, request: Request) -> Reply:
        if request.method != "POST":
            return Reply()
        marker: Final = rv.newest_marker(request.body.decode()) or "health"
        if request.target.endswith("/invoke-with-response-stream"):
            return Reply(content_type=EVENT_STREAM, chunks=aws_chunks(_events(marker)))
        if rv.JSON_OBJECT.validate_json(request.body).get("stream") is True:
            return Reply(
                content_type="text/event-stream", chunks=sse_chunks(_events(marker)), pause_between_chunks=self.pause
            )
        return Reply(body=json.dumps(_message(marker)).encode())


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "responses":
            return "/v1/responses"
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"


def _body(model: str, call: _Call) -> dict[str, JsonValue]:
    common: Final[dict[str, JsonValue]] = {"model": model, "stream": call.stream, "num_retries": 0, **_CACHE_BUST}
    match call.endpoint:
        case "responses":
            return {
                **common,
                "input": _prompt(call.marker),
                "tools": [
                    {"type": "tool_search"},
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Weather for a city",
                        "parameters": _WEATHER_PARAMETERS,
                        "defer_loading": True,
                    },
                ],
            }
        case "chat":
            return {
                **common,
                "messages": [{"role": "user", "content": _prompt(call.marker)}],
                "tools": [
                    dict(_REGEX_TOOL),
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
            }
        case "messages":
            return {
                **common,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": _prompt(call.marker)}],
                "tools": [
                    dict(_REGEX_TOOL),
                    {
                        "name": "get_weather",
                        "description": "Weather for a city",
                        "input_schema": _WEATHER_PARAMETERS,
                        "defer_loading": True,
                    },
                ],
            }


def _calls(count: int, endpoints: tuple[Endpoint, ...]) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoint=endpoints[index % len(endpoints)], stream=index % 2 == 1, marker=uuid.uuid4().hex)
        for index in range(count)
    )


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_body(model, call),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(call, response.status_code, raw.decode(), response.headers.get("x-litellm-call-id", ""))


async def _burst(
    base_url: str, key: str, model: str, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, model, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _frames(text: str) -> list[dict[str, JsonValue]]:
    return [rv.JSON_OBJECT.validate_json(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


def _responses_payload(served: _Served) -> dict[str, JsonValue]:
    if not served.call.stream:
        return rv.JSON_OBJECT.validate_json(served.text)
    (completed,) = [frame for frame in _frames(served.text) if frame.get("type") == "response.completed"]
    return rv.JSON_OBJECT.validate_python(completed["response"])


def _response_id(served: _Served) -> str:
    if not served.call.stream:
        return str(rv.JSON_OBJECT.validate_json(served.text)["id"])
    frames: Final = _frames(served.text)
    match served.call.endpoint:
        case "responses":
            return str(_responses_payload(served)["id"])
        case "chat":
            return str(frames[0]["id"])
        case "messages":
            (start,) = [frame for frame in frames if frame.get("type") == "message_start"]
            return str(rv.JSON_OBJECT.validate_python(start["message"])["id"])


def _assert_carries_the_search_call(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(rv.MARKER.findall(served.text)) == {served.call.marker}, served.text
    if served.call.endpoint != "responses":
        assert _REGEX_TOOL_NAME in served.text, served.text
        return
    types: Final = [item["type"] for item in rv.ITEMS.validate_python(_responses_payload(served)["output"])]
    assert "tool_search_call" in types and "function_call" in types, types


def _betas(request: Request) -> tuple[str, ...]:
    return tuple(beta.strip() for beta in request.headers.get("anthropic-beta", "").split(",") if beta.strip())


def _assert_forwarded_with_tool_search(request: Request) -> None:
    assert urlsplit(request.target).path == "/v1/messages", request.target
    assert _TOOL_SEARCH_BETA in _betas(request), dict(request.headers)
    assert "no-cache" not in request.body.decode(), request.body
    tools: Final = rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(request.body)["tools"])
    assert tools[0] == dict(_REGEX_TOOL), tools
    assert [(tool["name"], tool.get("defer_loading")) for tool in tools[1:]] == [("get_weather", True)], tools


def _spend_rows(model: str, expected: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=70,
    )


def _assert_each_lands_once(
    rows: list[dict[str, JsonValue]], failed: tuple[_Served, ...], served: tuple[_Served, ...]
) -> None:
    by_status: Final = {str(row["request_id"]): str(row["status"]) for row in rows}
    assert len(by_status) == len(rows) == len(failed) + len(served), rows
    for item in failed:
        assert by_status.get(item.call_id) == "failure", (item.call_id, rows)
    for item in served:
        (match,) = [request_id for request_id in by_status if rv.same_response(request_id, _response_id(item))]
        assert by_status[match] == "success", rows


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _health_counts(gateway: Gateway, model: str) -> tuple[int, int]:
    response: Final = gateway.request("GET", f"/health?model={model}", None)
    assert response.status_code in (200, 503), response.text
    health: Final = rv.JSON_OBJECT.validate_json(response.text)
    return int(str(health["healthy_count"])), int(str(health["unhealthy_count"]))


def _marked(received: tuple[Request, ...]) -> dict[str, Request]:
    marked: Final = {marker: request for request in received if (marker := rv.newest_marker(request.body.decode()))}
    assert len(marked) == sum(1 for request in received if rv.newest_marker(request.body.decode())), received
    return marked


def _register(gateway: Gateway, api_base: str) -> str:
    with gateway.scenario() as scenario:
        return scenario.model(model=f"anthropic/{_CLAUDE}", api_base=api_base, api_key=cc.ANTHROPIC_API_KEY)


async def test_mixed_burst_carries_the_search_call_on_every_endpoint_and_lands_each_once(gateway: Gateway) -> None:
    calls: Final = _calls(24, ("responses", "chat", "messages"))
    with wire_server(_Vendor().respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_CLAUDE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 24
        for item in served:
            _assert_carries_the_search_call(item)
        forwarded: Final = _marked(wire.drain())
        assert set(forwarded) == {call.marker for call in calls}, sorted(forwarded)
        for request in forwarded.values():
            _assert_forwarded_with_tool_search(request)
        _assert_each_lands_once(_spend_rows(model, 24), (), served)


@pytest.mark.timeout(180)
async def test_vendor_outage_fails_each_call_cleanly_and_the_recovered_vendor_gets_the_tool_search(
    gateway: Gateway,
) -> None:
    port: Final = _free_port()
    while_down: Final = _calls(15, ("responses", "chat", "messages"))
    after: Final = _calls(15, ("responses", "chat", "messages"))
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_CLAUDE}", api_base=f"http://127.0.0.1:{port}", api_key=cc.ANTHROPIC_API_KEY
        )
        failed: Final = await _burst(str(gateway.client.base_url), gateway.key, model, while_down)
        assert len(failed) == 15
        for item in failed:
            assert item.status == 500 and "Cannot connect to host" in item.text, (item.status, item.text)
            assert "tool_search_call" not in item.text, item.text
            assert item.call_id, item
        assert _health_counts(gateway, model) == (0, 1)
        with wire_server(_Vendor().respond, port=port) as wire:
            assert _health_counts(gateway, model) == (1, 0)
            wire.drain()
            served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, after)
            assert len(served) == 15
            for item in served:
                _assert_carries_the_search_call(item)
            forwarded: Final = _marked(wire.drain())
            assert set(forwarded) == {call.marker for call in after}, sorted(forwarded)
            for request in forwarded.values():
                _assert_forwarded_with_tool_search(request)
        _assert_each_lands_once(_spend_rows(model, 30), failed, served)


async def test_slow_vendor_streams_each_carry_the_search_call_and_are_forwarded_once(gateway: Gateway) -> None:
    calls: Final = tuple(_Call("responses", True, uuid.uuid4().hex) for _ in range(10))
    with wire_server(_Vendor(pause=0.3).respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_CLAUDE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 10
        for item in served:
            _assert_carries_the_search_call(item)
            assert "response.completed" in item.text, item.text
        received: Final = wire.drain()
        assert len(received) == 10, [request.target for request in received]
        forwarded: Final = _marked(received)
        assert set(forwarded) == {call.marker for call in calls}
        for request in forwarded.values():
            _assert_forwarded_with_tool_search(request)
        _assert_each_lands_once(_spend_rows(model, 10), (), served)


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {
            "model_name": _CONFIG_MODEL,
            "litellm_params": {"model": f"anthropic/{_CLAUDE}", "api_base": wire.url, "api_key": cc.ANTHROPIC_API_KEY},
        }
    ]
    path: Final = tmp_path / "responses-bridge-tool-search-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_returning_the_search_call(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = tuple(_Call("responses", False, uuid.uuid4().hex) for _ in range(20))
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()
    vendor: Final = _Vendor()

    def held(request: Request) -> Reply:
        marker: Final = rv.newest_marker(request.body.decode())
        if request.method != "POST" or marker is None:
            return vendor.respond(request)
        held_markers.put(marker)
        assert release.wait(timeout=60), "The burst was never released"
        return vendor.respond(request)

    with wire_server(held) as wire:
        path: Final = _chaos_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _burst(
                    str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, calls, tolerate_transport_errors=True
                )
            )
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            await asyncio.to_thread(owned.kill_worker, victim_pid)
            release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_carries_the_search_call(item)
            follow_up: Final = _Call("responses", False, uuid.uuid4().hex)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, (follow_up,))
            _assert_carries_the_search_call(answered)
            forwarded: Final = _marked(tuple(request for request in wire.drain() if request.method == "POST"))
            assert set(forwarded) == {call.marker for call in (*calls, follow_up)}, sorted(forwarded)
            for request in forwarded.values():
                _assert_forwarded_with_tool_search(request)


@dataclass(frozen=True, slots=True)
class _OwnedRig:
    wire: Wire
    proxy: OwnedProxy


@pytest.fixture(scope="module")
def owned_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_OwnedRig]:
    directory: Final = tmp_path_factory.mktemp("tool-search-owned")
    with gateway_from_environment() as gateway, wire_server(_Vendor().respond) as wire:
        overrides: Final = {
            "ANTHROPIC_API_BASE": _ANTHROPIC_BASE,
            "ANTHROPIC_API_KEY": cc.ANTHROPIC_API_KEY,
            "AIOHTTP_TRUST_ENV": "True",
            "HTTP_PROXY": wire.url,
            "NO_PROXY": "127.0.0.1,localhost",
            "STORE_PROMPTS_IN_SPEND_LOGS": "true",
        }
        with owned_proxy_process(gateway, directory, overrides, workers=2) as owned:
            yield _OwnedRig(wire, owned)


def _tool_results(messages: Sequence[Mapping[str, JsonValue]]) -> Iterator[dict[str, JsonValue]]:
    for turn in messages:
        if turn["role"] != "user" or not isinstance(turn["content"], list):
            continue
        yield from (block for block in rv.ITEMS.validate_python(turn["content"]) if block["type"] == "tool_result")


def _marked_posts(received: tuple[Request, ...], marker: str) -> tuple[Request, ...]:
    return tuple(request for request in received if request.method == "POST" and marker in request.body.decode())


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_a_previous_response_id_turn_replays_the_function_call_without_the_search_call(owned_rig: _OwnedRig) -> None:
    marker: Final = uuid.uuid4().hex
    owned_rig.wire.drain()
    proxy: Final = owned_rig.proxy.gateway
    with proxy.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_CLAUDE}", api_base=owned_rig.wire.url, api_key=cc.ANTHROPIC_API_KEY
        )
        first_body: Final = _body(model, _Call("responses", False, marker))
        first: Final = proxy.request("POST", "/v1/responses", first_body)
        assert first.status_code == 200, first.text
        payload: Final = rv.JSON_OBJECT.validate_json(first.text)
        types: Final = [item["type"] for item in rv.ITEMS.validate_python(payload["output"])]
        assert "tool_search_call" in types and "function_call" in types, types
        response_id: Final = str(payload["id"])
        eventually(
            lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
            lambda rows: any(rv.same_response(str(row["request_id"]), response_id) for row in rows),
            seconds=70,
        )
        follow_up: Final = proxy.request(
            "POST",
            "/v1/responses",
            {
                **first_body,
                "previous_response_id": response_id,
                "input": [
                    {"type": "function_call_output", "call_id": f"toolu_{marker}", "output": f"22C marker-{marker}"}
                ],
            },
        )
        assert follow_up.status_code == 200, follow_up.text
        assert "tool_search_call" in follow_up.text, follow_up.text
        opening, replay = _marked_posts(owned_rig.wire.drain(), marker)
        _assert_forwarded_with_tool_search(opening)
        _assert_forwarded_with_tool_search(replay)
        messages: Final = rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(replay.body)["messages"])
        assert "srvtoolu_" not in json.dumps(messages) and "tool_search" not in json.dumps(messages), messages
        (assistant,) = [turn for turn in messages if turn["role"] == "assistant"]
        blocks: Final = rv.ITEMS.validate_python(assistant["content"])
        tool_uses: Final = [(block["type"], block.get("id")) for block in blocks if block["type"] == "tool_use"]
        assert tool_uses == [("tool_use", f"toolu_{marker}")], blocks
        assert [block["tool_use_id"] for block in _tool_results(messages)] == [f"toolu_{marker}"], messages


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_anthropic_pass_through_relays_the_search_frames_untouched(owned_rig: _OwnedRig) -> None:
    marker: Final = uuid.uuid4().hex
    owned_rig.wire.drain()
    body: Final[dict[str, JsonValue]] = {
        "model": _CLAUDE,
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": _prompt(marker)}],
        "tools": [
            dict(_REGEX_TOOL),
            {"name": "get_weather", "input_schema": _WEATHER_PARAMETERS, "defer_loading": True},
        ],
    }
    response: Final = owned_rig.proxy.gateway.request(
        "POST", "/anthropic/v1/messages", body, headers={"anthropic-beta": _TOOL_SEARCH_BETA}
    )
    assert response.status_code == 200, response.text
    assert response.content == b"".join(sse_chunks(_events(marker))), response.text
    received: Final = owned_rig.wire.drain()
    assert [request.target for request in received] == [f"{_ANTHROPIC_BASE}/v1/messages"], response.text
    assert _TOOL_SEARCH_BETA in _betas(received[0]), dict(received[0].headers)
    assert rv.JSON_OBJECT.validate_json(received[0].body)["tools"] == body["tools"], received[0].body
    rows: Final = eventually(
        lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (f"msg_{marker}",)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    assert str(rows[0]["status"]) == "success", rows


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_bedrock_invoke_pass_through_relays_the_search_frames_untouched(owned_rig: _OwnedRig) -> None:
    marker: Final = uuid.uuid4().hex
    owned_rig.wire.drain()
    proxy: Final = owned_rig.proxy.gateway
    with proxy.scenario() as scenario:
        deployment: Final = scenario.model(
            model=f"bedrock/{_BEDROCK_CLAUDE}",
            api_base=owned_rig.wire.url,
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=owned_rig.wire.url,
        )
        body: Final[dict[str, JsonValue]] = {
            "anthropic_version": "bedrock-2023-05-31",
            "anthropic_beta": [_TOOL_SEARCH_BETA],
            "max_tokens": 64,
            "messages": [{"role": "user", "content": _prompt(marker)}],
            "tools": [
                dict(_REGEX_TOOL),
                {"name": "get_weather", "input_schema": _WEATHER_PARAMETERS, "defer_loading": True},
            ],
        }
        response: Final = proxy.request("POST", f"/bedrock/model/{deployment}/invoke-with-response-stream", body)
        assert response.status_code == 200, response.text
        assert response.content == b"".join(aws_chunks(_events(marker))), response.text
        received: Final = owned_rig.wire.drain()
        assert [unquote(request.target) for request in received] == [
            f"/model/{_BEDROCK_CLAUDE}/invoke-with-response-stream"
        ], received
        assert rv.JSON_OBJECT.validate_json(received[0].body)["tools"] == body["tools"], received[0].body
        rows: Final = eventually(
            lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (deployment,)),
            lambda found: len(found) == 1,
            seconds=70,
        )
        assert str(rows[0]["status"]) == "success", rows


def _server_tool_use(marker: str) -> Event:
    return {
        "type": "server_tool_use",
        "id": f"srvtoolu_{marker}",
        "name": _REGEX_TOOL_NAME,
        "input": json.loads(_SEARCH_ARGUMENTS),
    }


def _stored_turn(marker: str, shape: StoredShape) -> dict[str, JsonValue]:
    match shape:
        case "tool_use":
            return _message(marker)
        case "search_only":
            return {
                **_message(marker),
                "content": [_server_tool_use(marker), _search_result(marker)],
                "stop_reason": "pause_turn",
            }
        case "search_and_text":
            return {
                **_message(marker),
                "content": [
                    _server_tool_use(marker),
                    _search_result(marker),
                    {"type": "text", "text": f"found marker-{marker}"},
                ],
                "stop_reason": "end_turn",
            }


def _stored_turn_vendor(shape: StoredShape) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply()
        marker: Final = rv.newest_marker(request.body.decode()) or "health"
        return Reply(body=json.dumps(_stored_turn(marker, shape)).encode())

    return respond


def _tool_call_ids(message: Mapping[str, JsonValue]) -> frozenset[str]:
    return frozenset(str(call["id"]) for call in rv.ITEMS.validate_python(message.get("tool_calls") or []))


def _answer_ids_after(messages: Sequence[Mapping[str, JsonValue]], index: int) -> frozenset[str]:
    answers: Final = itertools.takewhile(lambda message: message.get("role") == "tool", messages[index + 1 :])
    return frozenset(str(message.get("tool_call_id")) for message in answers)


def _opening_call_ids(messages: Sequence[Mapping[str, JsonValue]], index: int) -> frozenset[str]:
    opener: Final = next((message for message in reversed(messages[:index]) if message.get("role") != "tool"), None)
    return frozenset() if opener is None else _tool_call_ids(opener)


def _unmatched_tool_ids(messages: Sequence[Mapping[str, JsonValue]]) -> Iterator[str]:
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            yield from sorted(_tool_call_ids(message) - _answer_ids_after(messages, index))
        if message.get("role") == "tool" and str(message.get("tool_call_id")) not in _opening_call_ids(messages, index):
            yield str(message.get("tool_call_id"))


def _chat_answer(marker: str, stream: bool) -> Reply:
    head: Final[Mapping[str, JsonValue]] = {"id": f"cmpl-{marker}", "created": 1, "model": _MISTRAL}
    if not stream:
        message: Final = {"role": "assistant", "content": f"answered marker-{marker}"}
        completion: Final = {
            **head,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": rv.CHAT_USAGE,
        }
        return Reply(body=json.dumps(completion).encode())
    delta: Final[dict[str, JsonValue]] = {"role": "assistant", "content": f"answered marker-{marker}"}
    opened: Final[dict[str, JsonValue]] = {"index": 0, "delta": delta, "finish_reason": None}
    return Reply(
        content_type="text/event-stream",
        chunks=(
            rv.chat_sse({**head, "object": "chat.completion.chunk", "choices": [opened]}),
            rv.chat_sse(
                {
                    **head,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": rv.CHAT_USAGE,
                }
            ),
            b"data: [DONE]\n\n",
        ),
    )


def _strict_mistral(request: Request) -> Reply:
    if request.method != "POST":
        return Reply()
    marker: Final = rv.newest_marker(request.body.decode()) or "health"
    if request.target != "/v1/chat/completions":
        return rv.error(404, f"unexpected target {request.target}", "not_found")
    if "defer_loading" in request.body.decode():
        return rv.error(422, f"tools: defer_loading: Extra inputs are not permitted marker-{marker}", "extra_forbidden")
    body: Final = rv.JSON_OBJECT.validate_json(request.body)
    unmatched: Final = tuple(_unmatched_tool_ids(rv.ITEMS.validate_python(body.get("messages") or [])))
    if unmatched:
        return rv.error(
            400,
            f"Not the same number of function calls and responses: {', '.join(unmatched)} marker-{marker}",
            "invalid_request_message_order",
        )
    return _chat_answer(marker, body.get("stream") is True)


def _converse_ids(message: Mapping[str, JsonValue], block_key: str) -> frozenset[str]:
    return frozenset(
        str(rv.JSON_OBJECT.validate_python(block[block_key])["toolUseId"])
        for block in rv.ITEMS.validate_python(message.get("content") or [])
        if block_key in block
    )


def _unpaired_converse_turns(messages: Sequence[Mapping[str, JsonValue]]) -> tuple[int, ...]:
    return tuple(
        index + 1
        for index, (previous, current) in enumerate(zip(messages, messages[1:]))
        if _converse_ids(previous, "toolUse") != _converse_ids(current, "toolResult")
    )


def _strict_converse(request: Request) -> Reply:
    if request.method != "POST":
        return Reply()
    marker: Final = rv.newest_marker(request.body.decode()) or "health"
    if unquote(request.target) != f"/model/{BEDROCK_MODEL}/converse":
        return rv.error(404, f"unexpected target {request.target}", "not_found")
    body: Final = rv.JSON_OBJECT.validate_json(request.body)
    unpaired: Final = _unpaired_converse_turns(rv.ITEMS.validate_python(body.get("messages") or []))
    if unpaired:
        refusal: Final = (
            f"Expected toolResult blocks at messages.{unpaired[0]}.content for the toolUse ids of the previous turn "
            f"marker-{marker}"
        )
        return Reply(status=400, body=json.dumps({"message": refusal}).encode())
    answer: Final = {
        "output": {"message": {"role": "assistant", "content": [{"text": f"answered marker-{marker}"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15},
        "metrics": {"latencyMs": 1},
    }
    return Reply(body=json.dumps(answer).encode())


def _strict_vendor(route: OffAnthropicRoute) -> Callable[[Request], Reply]:
    match route:
        case "mistral":
            return _strict_mistral
        case "bedrock_converse":
            return _strict_converse


def _off_anthropic_deployment(scenario: Scenario, route: OffAnthropicRoute, wire_url: str) -> str:
    match route:
        case "mistral":
            return scenario.model(model=f"mistral/{_MISTRAL}", api_base=wire_url, api_key="synthetic-mistral-key")
        case "bedrock_converse":
            return scenario.model(
                model=f"bedrock/converse/{BEDROCK_MODEL}",
                aws_access_key_id="AKIASCRIPTEDPROVIDER",
                aws_secret_access_key="scripted-secret",
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=wire_url,
            )


def _stored_first_turn(proxy: Gateway, model: str, marker: str) -> str:
    first: Final = proxy.request("POST", "/v1/responses", _body(model, _Call("responses", False, marker)))
    assert first.status_code == 200, first.text
    payload: Final = rv.JSON_OBJECT.validate_json(first.text)
    types: Final = [item["type"] for item in rv.ITEMS.validate_python(payload["output"])]
    assert "tool_search_call" in types, types
    response_id: Final = str(payload["id"])
    eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda rows: any(rv.same_response(str(row["request_id"]), response_id) for row in rows),
        seconds=70,
    )
    return response_id


def _follow_up_input(marker: str, shape: StoredShape) -> list[JsonValue]:
    if shape == "tool_use":
        return [{"type": "function_call_output", "call_id": f"toolu_{marker}", "output": f"22C marker-{marker}"}]
    return [{"role": "user", "content": f"continue marker-{marker}"}]


def _chat_turn(message: Mapping[str, JsonValue]) -> tuple[str, tuple[str, ...]]:
    if message.get("role") == "tool":
        return "tool", (str(message.get("tool_call_id")),)
    return str(message["role"]), tuple(sorted(_tool_call_ids(message)))


def _expected_chat_turns(marker: str, shape: StoredShape) -> tuple[tuple[str, tuple[str, ...]], ...]:
    match shape:
        case "tool_use":
            return (("user", ()), ("assistant", (f"toolu_{marker}",)), ("tool", (f"toolu_{marker}",)))
        case "search_only":
            return (("user", ()), ("user", ()))
        case "search_and_text":
            return (("user", ()), ("assistant", ()), ("user", ()))


def _assert_converse_follow_up(messages: Sequence[Mapping[str, JsonValue]], marker: str) -> None:
    assert [message["role"] for message in messages] == ["user", "assistant", "user"], messages
    assert _converse_ids(messages[1], "toolUse") == {f"toolu_{marker}"}, messages
    assert _converse_ids(messages[2], "toolResult") == {f"toolu_{marker}"}, messages


def _off_anthropic_cells() -> tuple[ParameterSet, ...]:
    cases: Final[tuple[tuple[OffAnthropicRoute, StoredShape, bool], ...]] = (
        ("mistral", "search_only", False),
        ("mistral", "search_and_text", False),
        ("mistral", "search_and_text", True),
        ("bedrock_converse", "tool_use", False),
    )
    return tuple(
        pytest.param(route, shape, stream, id=f"{route}-{shape}-{'stream' if stream else 'sync'}")
        for route, shape, stream in cases
    )


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
@pytest.mark.parametrize(("route", "shape", "stream"), _off_anthropic_cells())
def test_a_follow_up_off_anthropic_drops_the_stored_search_call_and_is_served(
    owned_rig: _OwnedRig, route: OffAnthropicRoute, shape: StoredShape, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    proxy: Final = owned_rig.proxy.gateway
    with (
        wire_server(_stored_turn_vendor(shape)) as claude_wire,
        wire_server(_strict_vendor(route)) as follow_up_wire,
        proxy.scenario() as scenario,
    ):
        claude: Final = scenario.model(
            model=f"anthropic/{_CLAUDE}", api_base=claude_wire.url, api_key=cc.ANTHROPIC_API_KEY
        )
        target: Final = _off_anthropic_deployment(scenario, route, follow_up_wire.url)
        response_id: Final = _stored_first_turn(proxy, claude, marker)
        follow_up: Final = proxy.request(
            "POST",
            "/v1/responses",
            {
                **_body(target, _Call("responses", stream, marker)),
                "previous_response_id": response_id,
                "input": _follow_up_input(marker, shape),
            },
        )
        assert follow_up.status_code == 200, follow_up.text
        assert f"answered marker-{marker}" in follow_up.text, follow_up.text
        (request,) = _marked_posts(follow_up_wire.drain(), marker)
        sent: Final = request.body.decode()
        assert "srvtoolu_" not in sent and "tool_search" not in sent and "defer_loading" not in sent, sent
        body: Final = rv.JSON_OBJECT.validate_json(request.body)
        messages: Final = rv.ITEMS.validate_python(body["messages"])
        if route == "bedrock_converse":
            _assert_converse_follow_up(messages, marker)
            return
        assert tuple(_chat_turn(message) for message in messages) == _expected_chat_turns(marker, shape), messages
        tools: Final = rv.ITEMS.validate_python(body["tools"])
        assert [rv.JSON_OBJECT.validate_python(tool["function"])["name"] for tool in tools] == ["get_weather"], tools
        if shape == "search_and_text":
            assert f"found marker-{marker}" in json.dumps(messages[1]), messages


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_a_follow_up_on_anthropic_that_answers_the_stored_search_call_replays_it_with_its_answer(
    owned_rig: _OwnedRig,
) -> None:
    marker: Final = uuid.uuid4().hex
    proxy: Final = owned_rig.proxy.gateway
    with wire_server(_stored_turn_vendor("tool_use")) as claude_wire, proxy.scenario() as scenario:
        claude: Final = scenario.model(
            model=f"anthropic/{_CLAUDE}", api_base=claude_wire.url, api_key=cc.ANTHROPIC_API_KEY
        )
        response_id: Final = _stored_first_turn(proxy, claude, marker)
        answers: Final[list[JsonValue]] = [
            {"type": "function_call_output", "call_id": f"srvtoolu_{marker}", "output": f"get_weather marker-{marker}"},
            {"type": "function_call_output", "call_id": f"toolu_{marker}", "output": f"22C marker-{marker}"},
        ]
        follow_up: Final = proxy.request(
            "POST",
            "/v1/responses",
            {**_body(claude, _Call("responses", False, marker)), "previous_response_id": response_id, "input": answers},
        )
        assert follow_up.status_code == 200, follow_up.text
        opening, replay = _marked_posts(claude_wire.drain(), marker)
        _assert_forwarded_with_tool_search(opening)
        _assert_forwarded_with_tool_search(replay)
        messages: Final = rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(replay.body)["messages"])
        stored_turn: Final = next(turn for turn in messages if turn["role"] == "assistant")
        blocks: Final = rv.ITEMS.validate_python(stored_turn["content"])
        assert [(block.get("id"), block.get("name")) for block in blocks if block["type"] != "text"] == [
            (f"srvtoolu_{marker}", _REGEX_TOOL_NAME),
            (f"toolu_{marker}", "get_weather"),
        ], blocks
        assert sorted(str(block["tool_use_id"]) for block in _tool_results(messages)) == [
            f"srvtoolu_{marker}",
            f"toolu_{marker}",
        ], messages


@pytest.fixture(scope="module")
def drop_params_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("tool-search-drop-params")
    config: Final = rv.JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    settings: Final = rv.JSON_OBJECT.validate_python(config["litellm_settings"])
    path: Final = directory / "responses-bridge-tool-search-drop-params.yaml"
    path.write_text(yaml.safe_dump({**config, "litellm_settings": {**settings, "drop_params": True}}))
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(gateway, directory, {}, config=path, workers=2) as owned,
    ):
        yield owned


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
@pytest.mark.parametrize("request_drop_params", [..., False], ids=["absent", "false"])
def test_a_global_drop_params_drops_a_malformed_defer_loading_and_serves_the_search_call(
    drop_params_proxy: OwnedProxy, request_drop_params: bool | EllipsisType
) -> None:
    marker: Final = uuid.uuid4().hex
    proxy: Final = drop_params_proxy.gateway
    with wire_server(_Vendor().respond) as wire, proxy.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_CLAUDE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        opening: Final = _body(model, _Call("responses", False, marker))
        lookup: Final[dict[str, JsonValue]] = {
            "type": "function",
            "name": "lookup_order",
            "description": "Look up an order",
            "parameters": _ORDER_PARAMETERS,
            "defer_loading": "true",
        }
        flag: Final[Mapping[str, JsonValue]] = (
            {} if isinstance(request_drop_params, EllipsisType) else {"drop_params": request_drop_params}
        )
        response: Final = proxy.request(
            "POST",
            "/v1/responses",
            {**opening, "tools": [*rv.ITEMS.validate_python(opening["tools"]), lookup], **flag},
        )
        _assert_carries_the_search_call(
            _Served(_Call("responses", False, marker), response.status_code, response.text, "")
        )
        (request,) = _marked_posts(wire.drain(), marker)
        tools: Final = rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(request.body)["tools"])
        assert tools[0] == dict(_REGEX_TOOL), tools
        assert [(tool["name"], tool.get("defer_loading")) for tool in tools[1:]] == [
            ("get_weather", True),
            ("lookup_order", None),
        ], tools


_BRIDGE_OUTAGE: Final = "scripted: upstream outage"

BridgeEndpoint: TypeAlias = Literal["chat", "messages", "responses"]


@dataclass(frozen=True, slots=True)
class _BridgeCall:
    endpoint: BridgeEndpoint
    marker: str
    stream: bool
    replay: bool


@dataclass(frozen=True, slots=True)
class _BridgeServed:
    call: _BridgeCall
    status: int
    text: str


_BRIDGE_ENDPOINTS: Final[tuple[BridgeEndpoint, ...]] = ("chat", "chat", "messages", "responses")


def _bridge_calls(
    count: int, endpoint: Callable[[int], BridgeEndpoint], stream: Callable[[int], bool]
) -> tuple[_BridgeCall, ...]:
    return tuple(
        _BridgeCall(endpoint(index), uuid.uuid4().hex, stream(index), endpoint(index) != "messages" and index % 3 != 0)
        for index in range(count)
    )


def _bridge_chat_messages(call: _BridgeCall) -> list[JsonValue]:
    user: Final[dict[str, JsonValue]] = {"role": "user", "content": pcb.prompt(call.marker)}
    if not call.replay:
        return [user]
    items: Final[list[JsonValue]] = [
        ts.search_call(call.marker),
        ts.search_output(call.marker, [ts.loaded(ts.WEATHER)]),
    ]
    tool_call: Final[dict[str, JsonValue]] = {
        "id": ts.call_id(call.marker),
        "type": "function",
        "function": {"name": ts.WEATHER, "arguments": '{"city": "Paris"}'},
        "provider_specific_fields": {"namespace": ts.WEATHER},
    }
    return [
        user,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call],
            "provider_specific_fields": {"tool_search_items": items},
        },
        ts.tool_result(call.marker),
    ]


def _bridge_responses_input(call: _BridgeCall) -> list[JsonValue]:
    user: Final[dict[str, JsonValue]] = {"type": "message", "role": "user", "content": pcb.prompt(call.marker)}
    if not call.replay:
        return [user]
    return [
        user,
        *ts.searched(call.marker, "a", ts.WEATHER),
        ts.replayed_call(call.marker),
        {"type": "function_call_output", "call_id": ts.call_id(call.marker), "output": "sunny"},
    ]


def _bridge_expected_replay(call: _BridgeCall) -> list[dict[str, JsonValue]]:
    if not call.replay:
        return []
    match call.endpoint:
        case "chat":
            return [
                ts.search_call(call.marker),
                ts.search_output(call.marker, [ts.loaded(ts.WEATHER)]),
                ts.replayed_call(call.marker),
            ]
        case "responses":
            return [*ts.searched(call.marker, "a", ts.WEATHER), ts.replayed_call(call.marker)]
        case "messages":
            return []


def _bridge_request(call: _BridgeCall, model: str) -> tuple[str, dict[str, JsonValue]]:
    common: Final[dict[str, JsonValue]] = {"model": model, "stream": call.stream, "num_retries": 0, **pcb.NO_CACHE}
    match call.endpoint:
        case "chat":
            return "/v1/chat/completions", {
                **common,
                "messages": _bridge_chat_messages(call),
                "tools": ts.chat_tools(ts.WEATHER),
            }
        case "messages":
            return "/v1/messages", {
                **common,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": pcb.prompt(call.marker)}],
                "tools": [{"name": ts.WEATHER, "description": f"Look up {ts.WEATHER}", "input_schema": ts.PARAMETERS}],
            }
        case "responses":
            return "/v1/responses", {
                **common,
                "input": _bridge_responses_input(call),
                "tools": [ts.TOOL_SEARCH, ts.loaded(ts.WEATHER)],
            }


async def _bridge_send(client: httpx.AsyncClient, key: str, model: str, call: _BridgeCall) -> _BridgeServed:
    route, body = _bridge_request(call, model)
    headers: Final = {"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"}
    async with client.stream("POST", route, json=body, headers=headers) as response:
        raw: Final = await response.aread()
    return _BridgeServed(call, response.status_code, raw.decode())


async def _bridge_burst(
    gateway: Gateway, model: str, calls: tuple[_BridgeCall, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_BridgeServed, ...]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=120, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_bridge_send(client, gateway.key, model, call) for call in calls),
            return_exceptions=tolerate_transport_errors,
        )
    unexpected: Final = [
        result
        for result in results
        if isinstance(result, BaseException) and not isinstance(result, httpx.TransportError)
    ]
    assert unexpected == [], unexpected
    return tuple(result for result in results if isinstance(result, _BridgeServed))


def _bridge_assert_chat_answered(served: _BridgeServed) -> None:
    assistant: Final = (
        ts.assistant_from_stream(ts.chunks_of(served.text)) if served.call.stream else ts.first_message(served.text)
    )
    if served.call.replay:
        assert assistant["content"] == rv.answer(served.call.marker), served.text
        return
    assert ts.provider_fields(assistant).get("tool_search_items") == [
        *ts.searched(served.call.marker, "a", ts.WEATHER)
    ], served.text
    assert ts.namespaces(assistant) == [ts.WEATHER], served.text


def _bridge_assert_answered(served: _BridgeServed) -> None:
    assert served.status == 200, served.text
    match served.call.endpoint:
        case "chat":
            _bridge_assert_chat_answered(served)
        case "messages":
            assert '"tool_use"' in served.text and ts.call_id(served.call.marker) in served.text, served.text
            assert "tool_search" not in served.text, served.text
        case "responses":
            expected: Final = rv.answer(served.call.marker) if served.call.replay else f"tso_a_{served.call.marker}"
            assert expected in served.text, served.text


def _bridge_assert_upstream_saw_each_call_once(received: tuple[Request, ...], calls: tuple[_BridgeCall, ...]) -> None:
    posts: Final = tuple(request for request in received if request.method == "POST")
    assert sorted(rv.newest_marker(request.body.decode()) or "" for request in posts) == sorted(
        call.marker for call in calls
    ), [request.body for request in posts]
    expected: Final = {call.marker: _bridge_expected_replay(call) for call in calls}
    assert [ts.replayed(request) for request in posts] == [
        expected[rv.newest_marker(request.body.decode()) or ""] for request in posts
    ], [request.body for request in posts]


def _bridge_caller_id(served: _BridgeServed) -> str:
    if not served.call.stream:
        return string_value(rv.JSON_OBJECT.validate_json(served.text)["id"])
    first: Final = ts.chunks_of(served.text)[0]
    match served.call.endpoint:
        case "chat":
            return string_value(first["id"])
        case "messages":
            return string_value(rv.JSON_OBJECT.validate_python(first["message"])["id"])
        case "responses":
            return string_value(rv.JSON_OBJECT.validate_python(first["response"])["id"])


def _bridge_assert_one_success_row(model: str, served: _BridgeServed) -> None:
    caller_id: Final = _bridge_caller_id(served)
    rows: Final = eventually(lambda: ts.spend_rows_of(model, caller_id), lambda found: len(found) >= 1, seconds=60)
    assert [row["status"] for row in rows] == ["success"], rows


def _bridge_outage_for(down: frozenset[str]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if rv.newest_marker(request.body.decode()) in down:
            return rv.error(503, _BRIDGE_OUTAGE, "server_error")
        return ts.scripted(request)

    return answering_model_discovery(respond)


async def test_upstream_outage_mid_burst_fails_only_its_calls_and_every_replay_lands_once(gateway: Gateway) -> None:
    calls: Final = _bridge_calls(24, lambda index: _BRIDGE_ENDPOINTS[index % 4], lambda index: (index // 4) % 2 == 0)
    down: Final = frozenset(call.marker for index, call in enumerate(calls) if index % 5 == 1)
    with wire_server(_bridge_outage_for(down)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        served: Final = await _bridge_burst(gateway, model, calls)
        assert len(served) == len(calls)
        for item in served:
            if item.call.marker in down:
                assert item.status == 503 and _BRIDGE_OUTAGE in item.text, item.text
                assert rv.answer(item.call.marker) not in item.text, item.text
            else:
                _bridge_assert_answered(item)
        recovery: Final = _BridgeCall("chat", uuid.uuid4().hex, stream=True, replay=True)
        (recovered,) = await _bridge_burst(gateway, model, (recovery,))
        _bridge_assert_answered(recovered)
        _bridge_assert_upstream_saw_each_call_once(wire.drain(), (*calls, recovery))
        for item in (*served, recovered):
            if item.call.marker not in down:
                _bridge_assert_one_success_row(model, item)


async def test_slow_upstream_streams_keep_items_intact_without_a_duplicate_upstream_call(gateway: Gateway) -> None:
    calls: Final = _bridge_calls(12, lambda index: _BRIDGE_ENDPOINTS[index % 4], lambda index: (index // 4) % 2 == 0)
    with wire_server(ts.scripted_with_pause(0.3)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        served: Final = await _bridge_burst(gateway, model, calls)
        assert len(served) == len(calls)
        for item in served:
            _bridge_assert_answered(item)
        _bridge_assert_upstream_saw_each_call_once(wire.drain(), calls)
        for item in served:
            _bridge_assert_one_success_row(model, item)


def _bridge_chaos_config(wire: Wire, directory: Path) -> Path:
    template: Final = Path("tests/integration/proxy_config.yaml").read_text()
    base: Final = rv.JSON_OBJECT.validate_python(yaml.safe_load(template))
    config: Final = {
        **base,
        "model_list": [
            {
                "model_name": _CONFIG_MODEL,
                "litellm_params": {"model": pcb.MODEL, "api_base": f"{wire.url}/v1", "api_key": "sk-upstream"},
            }
        ],
    }
    path: Final = directory / "responses-bridge-tool-search-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _bridge_open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
async def test_worker_kill_mid_replay_burst_leaves_the_sibling_replaying_search_items(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _bridge_calls(20, lambda index: _BRIDGE_ENDPOINTS[index % 4], lambda _: False)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        held_markers.put(rv.newest_marker(request.body.decode()) or "")
        assert release.wait(timeout=90), "The burst was never released"
        return ts.scripted(request)

    with wire_server(answering_model_discovery(held)) as wire:
        with owned_proxy_process(
            gateway, tmp_path, {}, config=_bridge_chaos_config(wire, tmp_path), workers=2
        ) as owned:
            workers: Final = eventually(
                lambda: pcb.started_worker_pids(owned.log),
                lambda pids: len(pids) == 2,
                seconds=graceful_stop_seconds() + 60,
            )
            burst: Final = asyncio.create_task(
                _bridge_burst(owned.gateway, _CONFIG_MODEL, calls, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == len(calls), 60)
            held_by: Final = MappingProxyType(
                {pid: _bridge_open_upstream_connections(pid, wire.url) for pid in workers}
            )
            assert sum(held_by.values()) == len(calls), held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            with owned.replacing_worker(victim_pid) as kill:
                kill.freeze()
                await asyncio.to_thread(kill)
                release.set()
                served: Final = await burst
                assert len(served) == held_by[survivor_pid], (held_by, len(served))
                for item in served:
                    _bridge_assert_answered(item)
            follow_up: Final = _BridgeCall("chat", uuid.uuid4().hex, stream=True, replay=True)
            (answered,) = await _bridge_burst(owned.gateway, _CONFIG_MODEL, (follow_up,))
            _bridge_assert_answered(answered)
            _bridge_assert_upstream_saw_each_call_once(wire.drain(), (*calls, follow_up))
            for item in (*served, answered):
                _bridge_assert_one_success_row(_CONFIG_MODEL, item)
