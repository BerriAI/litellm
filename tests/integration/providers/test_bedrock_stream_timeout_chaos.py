import asyncio
import json
import re
import signal
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from urllib.parse import unquote, urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_CONVERSE_MODEL: Final = "bedrock/converse/global.moonshotai.kimi-k3"
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_ANSWER: Final = "bedrock stream timeout chaos control"
_PROMPT: Final = "How long does the gateway wait for this stream?"
_USER_TURN: Final[dict[str, JsonValue]] = {"role": "user", "content": _PROMPT}
_TIMEOUT_SECONDS: Final = 1
_KILL_TIMEOUT_SECONDS: Final = 10
_LONG_TIMEOUT_SECONDS: Final = 30
_HOLD_SECONDS: Final = 120.0
_CLIENT_WINDOW: Final = 10.0
_SHORT_WINDOW: Final = 3.0
_BURST_WINDOW: Final = 40.0
_BARE_MODEL: Final = "bedrock-stream-timeout-bare"
_SLOW_MODEL: Final = "bedrock-stream-timeout-thirty"
_KILL_MODEL: Final = "bedrock-stream-timeout-ten"
_DEPLOYMENTS: Final = ((_BARE_MODEL, None), (_SLOW_MODEL, _LONG_TIMEOUT_SECONDS), (_KILL_MODEL, _KILL_TIMEOUT_SECONDS))
_AWS: Final[dict[str, JsonValue]] = {
    "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
    "aws_secret_access_key": "scripted-secret",
    "aws_region_name": "us-east-1",
}
_EXTRA: Final[dict[str, JsonValue]] = {"num_retries": 0, "cache": {"no-cache": True}}
_TIMEOUT_PASSED: Final = re.compile(r"Timeout passed=(?:Timeout\(timeout=)?(-?\d+\.\d+)")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_USAGE: Final[dict[str, JsonValue]] = {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}
_CONVERSE_RESPONSE: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": _ANSWER}]}},
        "stopReason": "end_turn",
        **_USAGE,
        "metrics": {"latencyMs": 1},
    }
).encode()

Endpoint = Literal["chat", "messages", "responses"]
_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")


def _frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return _aws_event_frame(event_type, payload, "sc", "u")


_CONVERSE_FRAMES: Final = b"".join(
    (
        _frame("messageStart", {"role": "assistant"}),
        _frame("contentBlockDelta", {"delta": {"text": _ANSWER}, "contentBlockIndex": 0}),
        _frame("contentBlockStop", {"contentBlockIndex": 0}),
        _frame("messageStop", {"stopReason": "end_turn"}),
        _frame("metadata", _USAGE),
    )
)


@dataclass(frozen=True, slots=True)
class _Peer:
    wire: Wire
    hold: threading.Event
    drop: threading.Event
    released: threading.Event

    def respond(self, request: Request) -> Reply:
        if self.hold.is_set():
            self.released.wait(timeout=_HOLD_SECONDS)
        if self.drop.is_set():
            return Reply(drop_connection=True)
        if unquote(request.target).endswith("converse-stream"):
            return Reply(body=_CONVERSE_FRAMES, content_type=_EVENT_STREAM)
        return Reply(body=_CONVERSE_RESPONSE)


@contextmanager
def _peer() -> Iterator[_Peer]:
    hold: Final = threading.Event()
    drop: Final = threading.Event()
    released: Final = threading.Event()

    def respond(request: Request) -> Reply:
        return _Peer(wire, hold, drop, released).respond(request)

    with wire_server(respond) as wire:
        try:
            yield _Peer(wire, hold, drop, released)
        finally:
            released.set()


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    stream: bool = True


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


def _calls(count: int) -> tuple[_Call, ...]:
    return tuple(_Call(_ENDPOINTS[index % len(_ENDPOINTS)]) for index in range(count))


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _body(model: str, call: _Call) -> dict[str, JsonValue]:
    match call.endpoint:
        case "chat":
            return {"model": model, "messages": [_USER_TURN], "stream": call.stream, **_EXTRA}
        case "messages":
            return {"model": model, "messages": [_USER_TURN], "max_tokens": 16, "stream": call.stream, **_EXTRA}
        case "responses":
            return {"model": model, "input": _PROMPT, "stream": call.stream, **_EXTRA}


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    response: Final = await client.post(
        _path(call.endpoint), json=_body(model, call), headers={"Authorization": f"Bearer {key}"}
    )
    return _Served(call, response.status_code, response.text)


async def _burst(
    gateway: Gateway, model: str, calls: tuple[_Call, ...], *, window: float, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=_proxy_url(gateway), timeout=window, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, gateway.key, model, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


async def _one(gateway: Gateway, model: str, call: _Call, *, window: float = _CLIENT_WINDOW) -> _Served:
    (served,) = await _burst(gateway, model, (call,), window=window)
    return served


def _timeout_passed(text: str) -> str:
    found: Final = _TIMEOUT_PASSED.search(text)
    assert found is not None, text
    return found.group(1)


def _assert_timed_out(served: _Served, seconds: float = _TIMEOUT_SECONDS) -> None:
    assert served.status == 408, served.text
    assert _timeout_passed(served.text) == f"{seconds:.1f}", served.text
    assert _ANSWER not in served.text, served.text


def _assert_answered(served: _Served) -> None:
    assert served.status == 200, served.text
    assert _ANSWER in served.text, served.text
    assert "Timeout" not in served.text, served.text


def _assert_cut_off(served: _Served) -> None:
    assert served.status >= 500, served.text
    assert "error" in served.text, served.text
    assert _ANSWER not in served.text, served.text


async def _wait_for_received(peer: _Peer, expected: int) -> None:
    await asyncio.to_thread(eventually, peer.wire.received.qsize, lambda size: size == expected, 60)


def _failure_rows(model: str, expected: int) -> None:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=60,
    )
    assert [row["status"] for row in rows] == ["failure"] * expected, rows
    assert len({row["request_id"] for row in rows}) == expected, rows


def _converse(scenario: Scenario, wire: Wire, **extra: JsonValue) -> str:
    return scenario.model(model=_CONVERSE_MODEL, api_base=wire.url, api_key=None, **_AWS, **extra)


def _deployment(name: str, wire: Wire, timeout: int | None) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": _CONVERSE_MODEL,
            "api_base": wire.url,
            **_AWS,
            **({} if timeout is None else {"timeout": timeout}),
        },
    }


def _config(wire: Wire, directory: Path, *, router_timeout: int | None = None) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    router_settings: Final = {
        **base["router_settings"],
        **({} if router_timeout is None else {"timeout": router_timeout}),
    }
    config: Final = {
        **base,
        "model_list": [_deployment(name, wire, timeout) for name, timeout in _DEPLOYMENTS],
        "router_settings": router_settings,
    }
    path: Final = directory / f"bedrock-stream-timeout-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _worker_pids(log: Path) -> tuple[int, ...]:
    return eventually(
        lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(log.read_text())),
        lambda pids: len(pids) == 2,
        seconds=30,
    )


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(300)
async def test_p7_p8_the_global_request_timeout_bounds_a_stream_only_when_the_deployment_sets_none(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _peer() as peer:
        peer.hold.set()
        path: Final = _config(peer.wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {"REQUEST_TIMEOUT": "1"}, config=path, workers=2) as owned:
            _assert_timed_out(await _one(owned.gateway, _BARE_MODEL, _Call("chat")))
            with pytest.raises(httpx.ReadTimeout):
                await _one(owned.gateway, _SLOW_MODEL, _Call("messages"), window=_SHORT_WINDOW)
            await _wait_for_received(peer, 2)
            peer.released.set()


@pytest.mark.timeout(300)
async def test_p9_under_router_settings_timeout_the_global_request_timeout_precedes_the_deployment_for_streams(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _peer() as peer:
        peer.hold.set()
        path: Final = _config(peer.wire, tmp_path, router_timeout=_LONG_TIMEOUT_SECONDS)
        with owned_proxy_process(gateway, tmp_path, {"REQUEST_TIMEOUT": "1"}, config=path, workers=2) as owned:
            _assert_timed_out(await _one(owned.gateway, _SLOW_MODEL, _Call("chat")))
            with pytest.raises(httpx.ReadTimeout):
                await _one(owned.gateway, _SLOW_MODEL, _Call("chat", stream=False), window=_SHORT_WINDOW)
            await _wait_for_received(peer, 2)
            peer.released.set()


@pytest.mark.timeout(120)
async def test_x1_thirty_stalled_streams_across_every_endpoint_each_time_out_while_the_proxy_stays_live(
    gateway: Gateway,
) -> None:
    calls: Final = _calls(30)
    with _peer() as peer, gateway.scenario() as scenario:
        peer.hold.set()
        model: Final = _converse(scenario, peer.wire, timeout=_TIMEOUT_SECONDS)
        burst: Final = asyncio.create_task(_burst(gateway, model, calls, window=_CLIENT_WINDOW))
        await _wait_for_received(peer, 30)
        async with httpx.AsyncClient(base_url=_proxy_url(gateway), timeout=2, trust_env=False) as client:
            liveliness: Final = await client.get("/health/liveliness")
        assert liveliness.status_code == 200, liveliness.text
        served: Final = await burst
        assert len(served) == 30
        for item in served:
            _assert_timed_out(item)
        assert len(peer.wire.drain()) == 30
        _failure_rows(model, 30)


@pytest.mark.timeout(120)
async def test_x2_an_upstream_dropping_every_held_connection_fails_each_stream_once_and_then_recovers(
    gateway: Gateway,
) -> None:
    calls: Final = _calls(20)
    with _peer() as peer, gateway.scenario() as scenario:
        peer.hold.set()
        peer.drop.set()
        model: Final = _converse(scenario, peer.wire, timeout=_LONG_TIMEOUT_SECONDS)
        burst: Final = asyncio.create_task(
            _burst(gateway, model, calls, window=_CLIENT_WINDOW, tolerate_transport_errors=True)
        )
        await _wait_for_received(peer, 20)
        peer.released.set()
        await asyncio.to_thread(eventually, peer.wire.disconnected.qsize, lambda size: size == 20, 30)
        served: Final = await burst
        assert len(served) == 20
        for item in served:
            _assert_cut_off(item)
        peer.drop.clear()
        peer.hold.clear()
        _assert_answered(await _one(gateway, model, _Call("chat")))
        assert len(peer.wire.drain()) == 21


@pytest.mark.timeout(360)
async def test_x3_killing_one_worker_mid_burst_leaves_the_sibling_timing_its_streams_out(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20)
    with _peer() as peer:
        peer.hold.set()
        path: Final = _config(peer.wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            workers: Final = _worker_pids(owned.log)
            burst: Final = asyncio.create_task(
                _burst(owned.gateway, _KILL_MODEL, calls, window=_BURST_WINDOW, tolerate_transport_errors=True)
            )
            await _wait_for_received(peer, 20)
            held_by: Final = {pid: _open_upstream_connections(pid, peer.wire.url) for pid in workers}
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_timed_out(item, _KILL_TIMEOUT_SECONDS)
            peer.hold.clear()
            _assert_answered(await _one(owned.gateway, _KILL_MODEL, _Call("chat")))
            assert len(peer.wire.drain()) == 21


@pytest.mark.timeout(480)
async def test_x4_a_proxy_stopped_mid_burst_still_times_its_held_streams_out_and_replays_none(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20)
    with _peer() as peer:
        peer.hold.set()
        path: Final = _config(peer.wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as first:
            burst: Final = asyncio.create_task(_burst(first.gateway, _KILL_MODEL, calls, window=_BURST_WINDOW))
            await _wait_for_received(peer, 20)
        served: Final = await burst
        assert len(served) == 20
        for item in served:
            _assert_timed_out(item, _KILL_TIMEOUT_SECONDS)
        assert len(peer.wire.drain()) == 20
        peer.hold.clear()
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as second:
            _assert_answered(await _one(second.gateway, _KILL_MODEL, _Call("chat")))
        assert len(peer.wire.drain()) == 1
