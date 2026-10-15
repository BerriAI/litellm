import asyncio
import re
import socket
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support import claude_code as cc
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.openai_wire import chat_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.test_chat_completions_tool_search_wire import (
    DEFERRED_NAMES,
    UNLISTED_CLAUDE,
    Answer,
    assert_lands_once_as,
    assert_no_search_and_no_deferral,
    assert_search_call_answered,
    chat_body,
    chat_frames,
    claude_vendor,
    deferred_tools,
    response_id,
    tool_search,
)
from pydantic import JsonValue

_CLAUDE: Final = cc.OPUS
_REGEX_TOOL_TYPE: Final = "tool_search_tool_regex_20251119"
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_CACHE_BUST: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
_OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120
_SEARCH_ARGUMENTS: Final = '{"pattern": "weather"}'
_CONFIG_MODEL: Final = "chat-tool-search-chaos"

Kind: TypeAlias = Literal["search", "history"]


@dataclass(frozen=True, slots=True)
class _Call:
    kind: Kind
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    answer: Answer
    call_id: str


def _history(marker: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": f"Weather in Paris marker-{marker}"},
        {
            "role": "assistant",
            "content": f"searching marker-{marker}",
            "tool_calls": [
                {
                    "id": f"srvtoolu_{marker}",
                    "type": "function",
                    "function": {"name": "tool_search_tool_regex", "arguments": _SEARCH_ARGUMENTS},
                },
                {
                    "id": f"toolu_{marker}",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": f"toolu_{marker}", "content": f"22C marker-{marker}"},
        {"role": "user", "content": f"And tomorrow? marker-{marker}"},
    ]


def _strict_openai(request: Request) -> Reply:
    if request.method != "POST":
        return Reply()
    body: Final = rv.JSON_OBJECT.validate_json(request.body)
    marker: Final = rv.newest_marker(request.body.decode()) or "health"
    if "srvtoolu_" in request.body.decode():
        return rv.error(400, f"No tool output found for function call srvtoolu_{marker}.", "invalid_request_error")
    return chat_reply(f"chatcmpl-{marker}", "gpt-5.6", f"answer marker-{marker}", stream=body.get("stream") is True)


def _vendor(request: Request) -> Reply:
    if request.target.endswith("/v1/chat/completions"):
        return _strict_openai(request)
    return claude_vendor(request)


def _body(models: Mapping[Kind, str], call: _Call) -> dict[str, JsonValue]:
    match call.kind:
        case "search":
            return chat_body(models["search"], call.marker, (tool_search(), *deferred_tools()), call.stream)
        case "history":
            return {
                "model": models["history"],
                "messages": _history(call.marker),
                "stream": call.stream,
                "num_retries": 0,
            }


def _calls(count: int, kinds: tuple[Kind, ...]) -> tuple[_Call, ...]:
    return tuple(
        _Call(kind=kinds[index % len(kinds)], stream=index % 2 == 1, marker=uuid.uuid4().hex) for index in range(count)
    )


async def _send(client: httpx.AsyncClient, key: str, models: Mapping[Kind, str], call: _Call) -> _Served:
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={**_body(models, call), **_CACHE_BUST},
        headers={"Authorization": f"Bearer {key}"},
    ) as response:
        raw: Final = (await response.aread()).decode()
    payload: Final = rv.JSON_OBJECT.validate_json(raw) if raw.startswith("{") else {}
    answer: Final = Answer(response.status_code, payload, chat_frames(raw) if call.stream else (), raw)
    return _Served(call, answer, response.headers.get("x-litellm-call-id", ""))


async def _burst(
    gateway: Gateway, models: Mapping[Kind, str], calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, gateway.key, models, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    unexpected: Final = [
        result
        for result in results
        if isinstance(result, BaseException) and not isinstance(result, httpx.TransportError)
    ]
    assert unexpected == [], unexpected
    return tuple(result for result in results if isinstance(result, _Served))


def _assert_served(item: _Served) -> None:
    match item.call.kind:
        case "search":
            assert_search_call_answered(item.answer, item.call.marker)
        case "history":
            assert item.answer.status == 200, item.answer.text
            assert f"answer marker-{item.call.marker}" in item.answer.text, item.answer.text


def _marked(received: tuple[Request, ...]) -> dict[str, Request]:
    posted: Final = tuple(request for request in received if request.method == "POST")
    marked: Final = {marker: request for request in posted if (marker := rv.newest_marker(request.body.decode()))}
    assert len(marked) == len(posted), [request.target for request in posted]
    return marked


def _assert_forwarded(request: Request, kind: Kind) -> None:
    sent: Final = request.body.decode()
    body: Final = rv.JSON_OBJECT.validate_json(request.body)
    match kind:
        case "search":
            assert request.target == "/v1/messages", request.target
            tools: Final = rv.ITEMS.validate_python(body["tools"])
            assert tools[0]["type"] == _REGEX_TOOL_TYPE, tools
            assert [(tool["name"], tool.get("defer_loading")) for tool in tools[1:]] == [
                (name, True) for name in DEFERRED_NAMES
            ], tools
        case "history":
            assert request.target == "/v1/chat/completions", request.target
            assert "srvtoolu_" not in sent and "tool_search_tool_regex" not in sent, body


def _assert_every_call_forwarded(wire: Wire, calls: tuple[_Call, ...]) -> None:
    forwarded: Final = _marked(wire.drain())
    assert set(forwarded) == {call.marker for call in calls}, sorted(forwarded)
    kinds: Final = {call.marker: call.kind for call in calls}
    for marker, request in forwarded.items():
        _assert_forwarded(request, kinds[marker])


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _identities(served: tuple[_Served, ...], kind: Kind) -> tuple[str, ...]:
    return tuple(response_id(item.answer) for item in served if item.call.kind == kind)


async def test_a_mixed_burst_carries_the_regex_tool_and_drops_the_history_call_and_lands_each_once(
    gateway: Gateway,
) -> None:
    calls: Final = _calls(24, ("search", "search", "history"))
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        models: Final = MappingProxyType(
            {
                "search": scenario.model(model=f"anthropic/{_CLAUDE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY),
                "history": scenario.model(model="openai/gpt-5.6", api_base=f"{wire.url}/v1", api_key="scripted"),
            }
        )
        served: Final = await _burst(gateway, models, calls)
        assert len(served) == 24
        for item in served:
            _assert_served(item)
        _assert_every_call_forwarded(wire, calls)
        assert_lands_once_as(models["search"], _identities(served, "search"), "success")
        assert_lands_once_as(models["history"], _identities(served, "history"), "success")


def _spend_status(model: str, expected: int) -> dict[str, str]:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=70,
    )
    by_id: Final = {str(row["request_id"]): str(row["status"]) for row in rows}
    assert len(by_id) == len(rows) == expected, rows
    return by_id


@pytest.mark.timeout(180)
async def test_a_vendor_outage_fails_each_call_cleanly_and_the_recovered_vendor_gets_the_regex_tool(
    gateway: Gateway,
) -> None:
    port: Final = _free_port()
    while_down: Final = _calls(15, ("search",))
    after: Final = _calls(15, ("search",))
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_CLAUDE}", api_base=f"http://127.0.0.1:{port}", api_key=cc.ANTHROPIC_API_KEY
        )
        models: Final = MappingProxyType({"search": model, "history": model})
        failed: Final = await _burst(gateway, models, while_down)
        assert len(failed) == 15
        for item in failed:
            assert item.answer.status == 500 and "Cannot connect to host" in item.answer.text, item.answer.text
            assert "srvtoolu_" not in item.answer.text and item.call_id, item
        with wire_server(claude_vendor, port=port) as wire:
            wire.drain()
            served: Final = await _burst(gateway, models, after)
            assert len(served) == 15
            for item in served:
                _assert_served(item)
            _assert_every_call_forwarded(wire, after)
        by_id: Final = _spend_status(model, 30)
        assert {by_id.get(item.call_id) for item in failed} == {"failure"}, by_id
        matched: Final = [
            [logged for logged in by_id if rv.same_response(logged, response_id(item.answer))] for item in served
        ]
        assert all(len(ids) == 1 for ids in matched), by_id
        assert {by_id[ids[0]] for ids in matched} == {"success"}, by_id


def _config(tmp_path: Path, name: str, model: Mapping[str, JsonValue], settings: Mapping[str, JsonValue]) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [{"model_name": name, "litellm_params": dict(model)}],
                "litellm_settings": {**config["litellm_settings"], **settings},
            }
        )
    )
    return path


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _holding(
    vendor: Callable[[Request], Reply], release: threading.Event, held: SimpleQueue[str]
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        marker: Final = rv.newest_marker(request.body.decode())
        if request.method != "POST" or marker is None:
            return vendor(request)
        held.put(marker)
        assert release.wait(timeout=60), "The burst was never released"
        return vendor(request)

    return respond


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
async def test_a_worker_sigkill_mid_burst_leaves_the_sibling_returning_the_regex_tool_search(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20, ("search",))
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    models: Final = MappingProxyType({"search": _CONFIG_MODEL, "history": _CONFIG_MODEL})
    with wire_server(_holding(claude_vendor, release, held)) as wire:
        model: Final[Mapping[str, JsonValue]] = {
            "model": f"anthropic/{_CLAUDE}",
            "api_base": wire.url,
            "api_key": cc.ANTHROPIC_API_KEY,
        }
        path: Final = _config(tmp_path, _CONFIG_MODEL, model, {})
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(_burst(candidate, models, calls, tolerate_transport_errors=True))
            await asyncio.to_thread(eventually, held.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            await asyncio.to_thread(owned.kill_worker, victim_pid)
            release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_served(item)
            follow_up: Final = _Call("search", False, uuid.uuid4().hex)
            (answered,) = await _burst(candidate, models, (follow_up,))
            _assert_served(answered)
            _assert_every_call_forwarded(wire, (*calls, follow_up))


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_global_drop_params_drops_the_search_and_every_deferral_for_a_claude_without_tool_search(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(claude_vendor) as wire:
        model: Final[Mapping[str, JsonValue]] = {
            "model": f"anthropic/{UNLISTED_CLAUDE}",
            "api_base": wire.url,
            "api_key": cc.ANTHROPIC_API_KEY,
        }
        path: Final = _config(tmp_path, _CONFIG_MODEL, model, {"drop_params": True})
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            body: Final = {**chat_body(_CONFIG_MODEL, marker, (tool_search(), *deferred_tools()), False), **_CACHE_BUST}
            response: Final = owned.gateway.request("POST", "/v1/chat/completions", body)
            assert response.status_code == 200, response.text
            assert f"toolu_{marker}" in response.text and f"srvtoolu_{marker}" not in response.text, response.text
            posted: Final = tuple(request for request in wire.drain() if request.method == "POST")
            assert len(posted) == 1, [request.target for request in posted]
            assert posted[0].target == "/v1/messages", posted[0].target
            assert_no_search_and_no_deferral(rv.JSON_OBJECT.validate_json(posted[0].body), DEFERRED_NAMES)
