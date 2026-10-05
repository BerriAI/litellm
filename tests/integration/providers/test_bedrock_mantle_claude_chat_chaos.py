import asyncio
import json
import re
import signal
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "anthropic.claude-haiku-4-5"
_API_KEY: Final = "synthetic-mantle-bearer"
_CONFIG_MODEL: Final = "bedrock-mantle-claude-chat-chaos"
_MESSAGES_PATH: Final = "/anthropic/v1/messages"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_ROWS_BY_CALL: Final = (
    "SELECT litellm_call_id, status FROM \"LiteLLM_SpendLogs\" WHERE litellm_call_id = ANY(string_to_array(%s, ','))"
)

Endpoint = Literal["chat", "messages", "responses"]


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


def _answer(marker: str) -> str:
    return f"answer marker-{marker}"


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _body(model: str, call: _Call) -> dict[str, JsonValue]:
    question: Final = f"Question marker-{call.marker}"
    common: Final[dict[str, JsonValue]] = {"model": model, "stream": call.stream}
    match call.endpoint:
        case "chat":
            return {**common, "messages": [{"role": "user", "content": question}]}
        case "messages":
            return {**common, "max_tokens": 64, "messages": [{"role": "user", "content": question}]}
        case "responses":
            return {**common, "input": question}


def _mantle_reply(marker: str, stream: bool, *, abort: bool = False, pause: float = 0) -> Reply:
    message: Final = {
        "id": f"msg_bdrk_{marker}",
        "type": "message",
        "role": "assistant",
        "model": _BACKEND,
        "stop_sequence": None,
    }
    if not stream:
        payload: Final = json.dumps(
            {
                **message,
                "content": [{"type": "text", "text": _answer(marker)}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 23, "output_tokens": 7},
            }
        ).encode()
        return Reply(chunks=(payload,), abort_after=0) if abort else Reply(body=payload)
    opening: Final = {**message, "content": [], "stop_reason": None, "usage": {"input_tokens": 23, "output_tokens": 1}}
    events: Final = (
        ("message_start", {"message": opening}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "answer "}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": f"marker-{marker}"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 7}}),
        ("message_stop", {}),
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(
            f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n".encode() for kind, payload in events
        ),
        abort_after=0 if abort else None,
        pause_between_chunks=pause,
    )


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body
    return found.group(1)


def _native_peer(aborted: frozenset[str] = frozenset(), pause: float = 0) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _MESSAGES_PATH), request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", sorted(request.headers)
        marker: Final = _marker_of(request)
        stream: Final = _JSON_OBJECT.validate_json(request.body).get("stream") is True
        return _mantle_reply(marker, stream, abort=marker in aborted, pause=pause)

    return respond


def _assert_native_requests_for(received: tuple[Request, ...], calls: tuple[_Call, ...]) -> None:
    assert {(request.method, request.target) for request in received} == {("POST", _MESSAGES_PATH)}
    assert sorted(_marker_of(request) for request in received) == sorted(call.marker for call in calls)


def _spend_status_by_call(served: tuple[_Served, ...]) -> dict[str, JsonValue]:
    wanted: Final = sorted(item.call_id for item in served)
    rows: Final = eventually(
        lambda: read_rows(_ROWS_BY_CALL, (",".join(wanted),)), lambda found: len(found) >= len(wanted), seconds=70
    )
    assert sorted(string_value(row["litellm_call_id"]) for row in rows) == wanted, rows
    return {string_value(row["litellm_call_id"]): row["status"] for row in rows}


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_body(model, call),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(
        call=call,
        status=response.status_code,
        text=raw.decode(),
        call_id=response.headers.get("x-litellm-call-id", ""),
    )


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


def _calls(count: int, endpoints: tuple[Endpoint, ...], stream: Callable[[int], bool]) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoint=endpoints[index % len(endpoints)], stream=stream(index), marker=uuid.uuid4().hex)
        for index in range(count)
    )


def _assert_answered_with_its_own_marker(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(_MARKER.findall(served.text)) == {served.call.marker}, served.text


async def test_concurrent_burst_across_endpoints_is_answered_from_the_native_route_and_logged_once_per_call(
    gateway: Gateway,
) -> None:
    calls: Final = _calls(30, ("chat", "messages", "responses"), lambda index: index % 2 == 0)
    with wire_server(_native_peer()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 30
        for item in served:
            _assert_answered_with_its_own_marker(item)
        _assert_native_requests_for(wire.drain(), calls)
        assert _spend_status_by_call(served) == {item.call_id: "success" for item in served}


def _unhealthy_count(gateway: Gateway, model: str) -> JsonValue:
    response: Final = gateway.request("GET", "/health", params={"model": model})
    return _JSON_OBJECT.validate_json(response.content).get("unhealthy_count")


async def test_upstream_aborts_then_an_outage_fail_each_call_once_and_the_native_route_recovers(
    gateway: Gateway,
) -> None:
    calls: Final = _calls(21, ("chat", "messages", "responses"), lambda index: index % 2 == 0)
    aborted: Final = frozenset(call.marker for call in calls if call.endpoint == "chat")
    during_outage: Final = (
        _Call(endpoint="chat", stream=True, marker=uuid.uuid4().hex),
        _Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex),
        _Call(endpoint="messages", stream=False, marker=uuid.uuid4().hex),
        _Call(endpoint="responses", stream=False, marker=uuid.uuid4().hex),
    )
    after_restart: Final = _calls(3, ("chat", "messages", "responses"), lambda index: index == 0)
    proxy: Final = str(gateway.client.base_url)
    with gateway.scenario() as scenario:
        with wire_server(_native_peer(aborted)) as wire:
            upstream: Final = wire.url
            model: Final = scenario.model(model=f"bedrock_mantle/{_BACKEND}", api_base=upstream, api_key=_API_KEY)
            burst: Final = await _burst(proxy, gateway.key, model, calls)
            assert len(burst) == 21
            for item in burst:
                if item.call.marker in aborted:
                    assert item.status == (500 if item.call.stream else 503), item.text
                    assert "Response payload is not completed" in item.text and "answer marker-" not in item.text
                else:
                    _assert_answered_with_its_own_marker(item)
            _assert_native_requests_for(wire.drain(), calls)
        refused: Final = await _burst(proxy, gateway.key, model, during_outage)
        assert len(refused) == 4
        for item in refused:
            assert item.status == 503, item.text
            assert "Cannot connect to host" in item.text, item.text
        assert await asyncio.to_thread(
            eventually, lambda: _unhealthy_count(gateway, model), lambda count: count == 1, 30
        )
        with wire_server(_native_peer(), port=urlsplit(upstream).port or 0) as restarted:
            recovered: Final = await _burst(proxy, gateway.key, model, after_restart)
            assert len(recovered) == 3
            for item in recovered:
                _assert_answered_with_its_own_marker(item)
            _assert_native_requests_for(restarted.drain(), after_restart)
        failed: Final = frozenset(item.call_id for item in (*burst, *refused) if item.status != 200)
        assert len(failed) == len(aborted) + 4
        assert _spend_status_by_call((*burst, *refused, *recovered)) == {
            item.call_id: "failure" if item.call_id in failed else "success" for item in (*burst, *refused, *recovered)
        }


async def test_slow_native_streams_reach_every_caller_whole_and_are_logged_once(gateway: Gateway) -> None:
    calls: Final = _calls(10, ("chat", "responses"), lambda _: True)
    with wire_server(_native_peer(pause=0.3)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 10
        for item in served:
            _assert_answered_with_its_own_marker(item)
            assert item.text.rstrip().endswith("data: [DONE]"), item.text
        _assert_native_requests_for(wire.drain(), calls)
        assert _spend_status_by_call(served) == {item.call_id: "success" for item in served}


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    config: Final = {
        **yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()),
        "model_list": [
            {
                "model_name": _CONFIG_MODEL,
                "litellm_params": {"model": f"bedrock_mantle/{_BACKEND}", "api_base": wire.url, "api_key": _API_KEY},
            }
        ],
    }
    target: Final = tmp_path / "bedrock-mantle-claude-chat-chaos.yaml"
    target.write_text(yaml.safe_dump(config))
    return target


def _live_workers(log: Path) -> tuple[int, ...]:
    started: Final = (int(pid) for pid in _STARTED_WORKER.findall(log.read_text()))
    return tuple(pid for pid in started if psutil.pid_exists(pid))


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(300)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_serving_the_native_route(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20, ("chat",), lambda _: False)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()
    answer: Final = _native_peer()

    def held(request: Request) -> Reply:
        held_markers.put(_marker_of(request))
        assert release.wait(timeout=60), "The burst was never released"
        return answer(request)

    with wire_server(held) as wire:
        config: Final = _chaos_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(lambda: _live_workers(owned.log), lambda pids: len(pids) == 2, seconds=30)
            burst: Final = asyncio.create_task(
                _burst(
                    str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, calls, tolerate_transport_errors=True
                )
            )
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_answered_with_its_own_marker(item)
            follow_up: Final = _Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, (follow_up,))
            _assert_answered_with_its_own_marker(answered)
            _assert_native_requests_for(wire.drain(), (*calls, follow_up))
            assert _spend_status_by_call((*served, answered)) == {
                item.call_id: "success" for item in (*served, answered)
            }
