import asyncio
import threading
import uuid
from collections.abc import Callable
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
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support import tool_search as ts
from integration._support.client import Gateway, eventually, string_value
from integration._support.openai_wire import answering_model_discovery
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_OUTAGE: Final = "scripted: upstream outage"
_CONFIG_MODEL: Final = "responses-bridge-tool-search-chaos"


Endpoint: TypeAlias = Literal["chat", "messages", "responses"]


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    marker: str
    stream: bool
    replay: bool


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "chat", "messages", "responses")


def _calls(count: int, endpoint: Callable[[int], Endpoint], stream: Callable[[int], bool]) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoint(index), uuid.uuid4().hex, stream(index), endpoint(index) != "messages" and index % 3 != 0)
        for index in range(count)
    )


def _chat_messages(call: _Call) -> list[JsonValue]:
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


def _responses_input(call: _Call) -> list[JsonValue]:
    user: Final[dict[str, JsonValue]] = {"type": "message", "role": "user", "content": pcb.prompt(call.marker)}
    if not call.replay:
        return [user]
    return [
        user,
        *ts.searched(call.marker, "a", ts.WEATHER),
        ts.replayed_call(call.marker),
        {"type": "function_call_output", "call_id": ts.call_id(call.marker), "output": "sunny"},
    ]


def _expected_replay(call: _Call) -> list[dict[str, JsonValue]]:
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


def _request(call: _Call, model: str) -> tuple[str, dict[str, JsonValue]]:
    common: Final[dict[str, JsonValue]] = {"model": model, "stream": call.stream, "num_retries": 0, **pcb.NO_CACHE}
    match call.endpoint:
        case "chat":
            return "/v1/chat/completions", {
                **common,
                "messages": _chat_messages(call),
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
                "input": _responses_input(call),
                "tools": [ts.TOOL_SEARCH, ts.loaded(ts.WEATHER)],
            }


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    route, body = _request(call, model)
    headers: Final = {"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"}
    async with client.stream("POST", route, json=body, headers=headers) as response:
        raw: Final = await response.aread()
    return _Served(call, response.status_code, raw.decode())


async def _burst(
    gateway: Gateway, model: str, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=120, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, gateway.key, model, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    unexpected: Final = [
        result
        for result in results
        if isinstance(result, BaseException) and not isinstance(result, httpx.TransportError)
    ]
    assert unexpected == [], unexpected
    return tuple(result for result in results if isinstance(result, _Served))


def _assert_chat_answered(served: _Served) -> None:
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


def _assert_answered(served: _Served) -> None:
    assert served.status == 200, served.text
    match served.call.endpoint:
        case "chat":
            _assert_chat_answered(served)
        case "messages":
            assert '"tool_use"' in served.text and ts.call_id(served.call.marker) in served.text, served.text
            assert "tool_search" not in served.text, served.text
        case "responses":
            expected: Final = rv.answer(served.call.marker) if served.call.replay else f"tso_a_{served.call.marker}"
            assert expected in served.text, served.text


def _assert_upstream_saw_each_call_once(received: tuple[Request, ...], calls: tuple[_Call, ...]) -> None:
    posts: Final = tuple(request for request in received if request.method == "POST")
    assert sorted(rv.newest_marker(request.body.decode()) or "" for request in posts) == sorted(
        call.marker for call in calls
    ), [request.body for request in posts]
    expected: Final = {call.marker: _expected_replay(call) for call in calls}
    assert [ts.replayed(request) for request in posts] == [
        expected[rv.newest_marker(request.body.decode()) or ""] for request in posts
    ], [request.body for request in posts]


def _caller_id(served: _Served) -> str:
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


def _assert_one_success_row(model: str, served: _Served) -> None:
    caller_id: Final = _caller_id(served)
    rows: Final = eventually(
        lambda: ts.spend_rows_of(model, caller_id), lambda found: len(found) >= 1, seconds=60
    )
    assert [row["status"] for row in rows] == ["success"], rows


def _outage_for(down: frozenset[str]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if rv.newest_marker(request.body.decode()) in down:
            return rv.error(503, _OUTAGE, "server_error")
        return ts.scripted(request)

    return answering_model_discovery(respond)


async def test_upstream_outage_mid_burst_fails_only_its_calls_and_every_replay_lands_once(gateway: Gateway) -> None:
    calls: Final = _calls(24, lambda index: _ENDPOINTS[index % 4], lambda index: (index // 4) % 2 == 0)
    down: Final = frozenset(call.marker for index, call in enumerate(calls) if index % 5 == 1)
    with wire_server(_outage_for(down)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        served: Final = await _burst(gateway, model, calls)
        assert len(served) == len(calls)
        for item in served:
            if item.call.marker in down:
                assert item.status == 503 and _OUTAGE in item.text, item.text
                assert rv.answer(item.call.marker) not in item.text, item.text
            else:
                _assert_answered(item)
        recovery: Final = _Call("chat", uuid.uuid4().hex, stream=True, replay=True)
        (recovered,) = await _burst(gateway, model, (recovery,))
        _assert_answered(recovered)
        _assert_upstream_saw_each_call_once(wire.drain(), (*calls, recovery))
        for item in (*served, recovered):
            if item.call.marker not in down:
                _assert_one_success_row(model, item)


async def test_slow_upstream_streams_keep_items_intact_without_a_duplicate_upstream_call(gateway: Gateway) -> None:
    calls: Final = _calls(12, lambda index: _ENDPOINTS[index % 4], lambda index: (index // 4) % 2 == 0)
    with wire_server(ts.scripted_with_pause(0.3)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        served: Final = await _burst(gateway, model, calls)
        assert len(served) == len(calls)
        for item in served:
            _assert_answered(item)
        _assert_upstream_saw_each_call_once(wire.drain(), calls)
        for item in served:
            _assert_one_success_row(model, item)


def _chaos_config(wire: Wire, directory: Path) -> Path:
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


def _open_upstream_connections(pid: int, upstream: str) -> int:
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
    calls: Final = _calls(20, lambda index: _ENDPOINTS[index % 4], lambda _: False)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        held_markers.put(rv.newest_marker(request.body.decode()) or "")
        assert release.wait(timeout=90), "The burst was never released"
        return ts.scripted(request)

    with wire_server(answering_model_discovery(held)) as wire:
        with owned_proxy_process(gateway, tmp_path, {}, config=_chaos_config(wire, tmp_path), workers=2) as owned:
            workers: Final = eventually(
                lambda: pcb.started_worker_pids(owned.log),
                lambda pids: len(pids) == 2,
                seconds=graceful_stop_seconds() + 60,
            )
            burst: Final = asyncio.create_task(
                _burst(owned.gateway, _CONFIG_MODEL, calls, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == len(calls), 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == len(calls), held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            with owned.replacing_worker(victim_pid) as kill:
                kill.freeze()
                await asyncio.to_thread(kill)
                release.set()
                served: Final = await burst
                assert len(served) == held_by[survivor_pid], (held_by, len(served))
                for item in served:
                    _assert_answered(item)
            follow_up: Final = _Call("chat", uuid.uuid4().hex, stream=True, replay=True)
            (answered,) = await _burst(owned.gateway, _CONFIG_MODEL, (follow_up,))
            _assert_answered(answered)
            _assert_upstream_saw_each_call_once(wire.drain(), (*calls, follow_up))
            for item in (*served, answered):
                _assert_one_success_row(_CONFIG_MODEL, item)
