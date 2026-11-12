import asyncio
import re
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue
from integration.providers._tool_search_support import (
    MODEL,
    SEARCH_RESULT,
    SEARCH_TURN,
    WEATHER_ID,
    Event,
    answer,
    answered_ids,
    assistant_from,
    chat_body,
    deployment,
    first_messages,
    hand_built_assistant,
    KEPT_FIELDS,
    SEARCH_CALL,
    WEATHER_CALL,
    kept_results,
    marker_of,
    question,
    replay_messages,
    request_messages,
    response_id,
    scripted_peer,
    wire_assistant,
)

Kind = Literal["replay", "first", "messages", "responses"]

_OWNED_MODEL: Final = "tool-search-chaos"
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120
_NO_CACHE: Final[Event] = {"cache": {"no-cache": True}}


@dataclass(frozen=True, slots=True)
class _Call:
    kind: Kind
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


def _path(kind: Kind) -> str:
    match kind:
        case "replay" | "first":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _body(model: str, call: _Call) -> Event:
    match call.kind:
        case "replay":
            assistant: Final = hand_built_assistant((SEARCH_CALL, WEATHER_CALL), KEPT_FIELDS)
            return chat_body(model, replay_messages(call.marker, assistant), call.stream)
        case "first":
            return chat_body(model, first_messages(call.marker), call.stream)
        case "messages":
            return {
                "model": model,
                "max_tokens": 64,
                "stream": call.stream,
                "messages": [{"role": "user", "content": question(call.marker)}],
            }
        case "responses":
            return {"model": model, "input": question(call.marker), "max_output_tokens": 64, **_NO_CACHE}


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    try:
        async with client.stream(
            "POST", _path(call.kind), json=_body(model, call), headers={"Authorization": f"Bearer {key}"}
        ) as response:
            raw: Final = await response.aread()
            return _Served(call=call, status=response.status_code, text=raw.decode())
    except httpx.TransportError as error:
        return _Served(call=call, status=0, text=repr(error))


async def _burst(base_url: str, key: str, model: str, calls: Sequence[_Call]) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=90, trust_env=False) as client:
        return tuple(await asyncio.gather(*(_send(client, key, model, call) for call in calls)))


def _calls(count: int, kinds: Sequence[Kind]) -> tuple[_Call, ...]:
    return tuple(
        _Call(kind=kinds[index % len(kinds)], stream=index % 2 == 0, marker=uuid.uuid4().hex) for index in range(count)
    )


def _chat_response(item: _Served) -> httpx.Response:
    return httpx.Response(item.status, content=item.text.encode())


def _assert_served(item: _Served) -> None:
    assert item.status == 200, (item.call, item.text)
    match item.call.kind:
        case "first":
            assistant: Final = assistant_from(_chat_response(item), item.call.stream)
            assert kept_results(assistant) == [SEARCH_RESULT], item.text
        case "replay":
            assert answer(item.call.marker) in item.text, item.text
        case "messages":
            assert "tool_search_tool_result" in item.text, item.text
            assert "get_weather" in item.text, item.text
        case "responses":
            assert "get_weather" in item.text, item.text


def _assert_upstream_saw_each_once(received: Sequence[Request], served: Sequence[_Served]) -> None:
    markers: Final = tuple(marker_of(request) for request in received)
    for item in served:
        assert markers.count(item.call.marker) == 1, (item.call, markers)
    for request in received:
        if len(request_messages(request)) > 1:
            assert wire_assistant(request) == list(SEARCH_TURN), request.body
            assert answered_ids(request) == [WEATHER_ID], request.body


def _chat_ids(served: Sequence[_Served]) -> tuple[str, ...]:
    return tuple(
        response_id(_chat_response(item), item.call.stream)
        for item in served
        if item.status == 200 and item.call.kind in ("replay", "first")
    )


def _success_rows(model: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE model_group=%s AND status=%s', (model, "success")
    )


async def test_upstream_outage_mid_burst_recovers_with_every_search_result_kept_and_logged_once(
    gateway: Gateway,
) -> None:
    kinds: Final[tuple[Kind, ...]] = ("replay", "first", "messages", "responses", "replay", "first")
    before: Final = _calls(12, kinds)
    during: Final = _calls(6, ("replay", "first"))
    after: Final = _calls(12, kinds)
    with gateway.scenario() as scenario:
        with wire_server(scripted_peer()) as wire:
            port: Final = int(wire.url.rsplit(":", 1)[1])
            model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
            served_before: Final = await _burst(str(gateway.client.base_url), gateway.key, model, before)
            received_before: Final = wire.drain()
        served_during: Final = await _burst(str(gateway.client.base_url), gateway.key, model, during)
        assert all(item.status == 0 or item.status >= 500 for item in served_during), served_during
        assert gateway.client.get("/health/liveliness").status_code == 200
        with wire_server(scripted_peer(), port=port) as revived:
            served_after: Final = await _burst(str(gateway.client.base_url), gateway.key, model, after)
            received_after: Final = revived.drain()
    for item in (*served_before, *served_after):
        _assert_served(item)
    _assert_upstream_saw_each_once(received_before, served_before)
    _assert_upstream_saw_each_once(received_after, served_after)
    ids: Final = _chat_ids((*served_before, *served_after))
    assert len(ids) == 16, ids
    rows: Final = await asyncio.to_thread(
        eventually, lambda: _success_rows(model), lambda found: len(found) == len(before) + len(after), 70
    )
    logged: Final = tuple(str(row["request_id"]) for row in rows)
    assert len(logged) == len(before) + len(after), logged
    for request_id in ids:
        assert logged.count(request_id) == 1, (request_id, logged)


async def test_slow_parallel_streams_each_keep_their_search_result(gateway: Gateway) -> None:
    def slow(request: Request) -> Reply:
        return replace(scripted_peer()(request), pause_between_chunks=0.2)

    calls: Final = tuple(_Call(kind="first", stream=True, marker=uuid.uuid4().hex) for _ in range(10))
    with wire_server(slow) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        burst: Final = asyncio.create_task(_burst(str(gateway.client.base_url), gateway.key, model, calls))
        await asyncio.to_thread(eventually, wire.received.qsize, lambda count: count >= 1, 30)
        assert gateway.client.get("/health/liveliness").status_code == 200
        served: Final = await burst
        received: Final = wire.drain()
    for item in served:
        _assert_served(item)
    _assert_upstream_saw_each_once(received, served)


def _chaos_config(wire: Wire, directory: Path) -> Path:
    config: Final = yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    path: Final = directory / "tool-search-chaos.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    {
                        "model_name": _OWNED_MODEL,
                        "litellm_params": {
                            "model": f"anthropic/{MODEL}",
                            "api_base": wire.url,
                            "api_key": "scripted-anthropic-key",
                        },
                    }
                ],
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


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
async def test_worker_sigkill_mid_burst_leaves_the_survivor_replaying_the_search_result(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20, ("replay",))
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        held_markers.put(marker_of(request))
        assert release.wait(timeout=graceful_stop_seconds() + 60), "The burst was never released"
        return scripted_peer()(request)

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
                _burst(str(candidate.client.base_url), candidate.key, _OWNED_MODEL, calls)
            )
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            with owned.replacing_worker(victim_pid) as kill:
                kill.freeze()
                await asyncio.to_thread(kill)
                release.set()
                served: Final = await burst
                completed: Final = tuple(item for item in served if item.status == 200)
                assert len(completed) == held_by[survivor_pid], (held_by, [item.status for item in served])
                for item in completed:
                    _assert_served(item)
            follow_up: Final = _Call(kind="replay", stream=True, marker=uuid.uuid4().hex)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, _OWNED_MODEL, (follow_up,))
            _assert_served(answered)
            received: Final = wire.drain()
            assert len(received) == 21, len(received)
            for request in received:
                assert wire_assistant(request) == list(SEARCH_TURN), request.body
                assert answered_ids(request) == [WEATHER_ID], request.body
