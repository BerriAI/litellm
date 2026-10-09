import asyncio
import re
import signal
import socket
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
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
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.responses_stream import (
    AZURE_TARGET,
    chat_content,
    function_tools,
    healthy_stream,
    rate_limited_stream,
)
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_MODEL: Final = "bridged-stream-chaos"
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_SENTINEL_PREFIX: Final = "litellm.MidStreamFallbackError: "
_RATE_LIMIT_PREFIX: Final = "litellm.RateLimitError: "
_HEADERS: Final[Mapping[str, str]] = MappingProxyType({"anthropic-version": "2023-06-01"})

Kind: TypeAlias = Literal["chat_limited", "chat_healthy", "messages_limited", "responses_limited"]
_KINDS: Final[tuple[Kind, ...]] = ("chat_limited", "chat_healthy", "messages_limited", "responses_limited")
_CHAT_KINDS: Final[tuple[Kind, ...]] = ("chat_limited", "chat_healthy")
_LOGGED_KINDS: Final[frozenset[Kind]] = frozenset({"chat_limited", "chat_healthy", "responses_limited"})


@dataclass(frozen=True, slots=True)
class _Call:
    kind: Kind
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str
    call_id: str


@dataclass(frozen=True, slots=True)
class _Rig:
    port: int
    proxy: OwnedProxy

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return reserve.getsockname()[1]


def _config(port: int, directory: Path) -> Path:
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    router_settings: Final = object_value(stock.get("router_settings") or {})
    path: Final = directory / "bridged-stream-chaos.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **stock,
                "model_list": [
                    {
                        "model_name": _MODEL,
                        "litellm_params": {
                            "model": "azure/gpt-6",
                            "api_base": f"http://127.0.0.1:{port}",
                            "api_key": "synthetic-azure-key",
                        },
                    }
                ],
                "router_settings": {**router_settings, "num_retries": 0},
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("bridged-stream-chaos")
    port: Final = _free_port()
    with (
        gateway_from_environment() as shared,
        owned_proxy_process(shared, directory, {}, config=_config(port, directory), workers=2) as owned,
    ):
        yield _Rig(port, owned)


def _newest_marker(text: str) -> str | None:
    found: Final = _MARKER.findall(text)
    return found[-1] if found else None


def _respond(request: Request) -> Reply:
    assert request.method == "POST" and request.target.startswith(AZURE_TARGET), request.target
    marker: Final = _newest_marker(request.body.decode())
    assert marker is not None, request.body
    identity: Final = f"resp_{uuid.uuid4().hex}"
    if b"chat_healthy" in request.body:
        return Reply(content_type="text/event-stream", chunks=healthy_stream(identity, f"answer marker-{marker}"))
    return Reply(content_type="text/event-stream", chunks=rate_limited_stream(identity))


def _path(kind: Kind) -> str:
    match kind:
        case "chat_limited" | "chat_healthy":
            return "/v1/chat/completions"
        case "messages_limited":
            return "/v1/messages"
        case "responses_limited":
            return "/v1/responses"


def _body(call: _Call) -> Mapping[str, JsonValue]:
    prompt: Final = f"{call.kind} marker-{call.marker}"
    common: Final[Mapping[str, JsonValue]] = {
        "model": _MODEL,
        "stream": True,
        "num_retries": 0,
        "cache": {"no-cache": True},
    }
    match call.kind:
        case "chat_limited" | "chat_healthy":
            return {**common, "messages": [{"role": "user", "content": prompt}], "tools": function_tools()}
        case "messages_limited":
            return {
                **common,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Weather for a city",
                        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                    }
                ],
            }
        case "responses_limited":
            return {**common, "input": prompt}


def _calls(count: int, kinds: tuple[Kind, ...]) -> tuple[_Call, ...]:
    return tuple(_Call(kinds[index % len(kinds)], uuid.uuid4().hex) for index in range(count))


async def _send(client: httpx.AsyncClient, key: str, call: _Call) -> _Served:
    async with client.stream(
        "POST", _path(call.kind), json=_body(call), headers={"Authorization": f"Bearer {key}", **_HEADERS}
    ) as response:
        raw: Final = await response.aread()
    return _Served(call, response.status_code, raw.decode(), response.headers["x-litellm-call-id"])


async def _burst(
    gateway: Gateway, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, gateway.key, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _data_frames(text: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: {")
    )


def _sse_events(text: str) -> tuple[tuple[str, Mapping[str, JsonValue]], ...]:
    def parse(block: str) -> tuple[str, Mapping[str, JsonValue]]:
        lines: Final = block.splitlines()
        event: Final = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data: Final = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        return event, JSON_OBJECT.validate_json(data)

    return tuple(parse(block) for block in text.strip().split("\n\n") if "event: " in block)


def _assert_answered_in_its_own_shape(served: _Served) -> None:
    match served.call.kind:
        case "chat_healthy":
            assert served.status == 200, served.text
            assert chat_content(_data_frames(served.text)) == f"answer marker-{served.call.marker}", served.text
        case "chat_limited":
            assert served.status == 429, served.text
            error: Final = object_value(JSON_OBJECT.validate_json(served.text)["error"])
            assert error["type"] == "throttling_error" and str(error["code"]) == "429", error
            message: Final = string_value(error["message"])
            assert message.startswith(_RATE_LIMIT_PREFIX) and _SENTINEL_PREFIX not in message, message
        case "messages_limited":
            assert served.status == 200, served.text
            events: Final = _sse_events(served.text)
            assert events[0][0] == "message_start" and events[-1][0] == "error", events
            frame_error: Final = object_value(events[-1][1]["error"])
            assert frame_error["type"] == "rate_limit_error", frame_error
            frame_message: Final = string_value(frame_error["message"])
            assert frame_message.startswith(_RATE_LIMIT_PREFIX) and _SENTINEL_PREFIX not in frame_message, frame_message
        case "responses_limited":
            assert served.status == 200, served.text
            kinds: Final = [frame["type"] for frame in _data_frames(served.text)]
            assert kinds == ["response.created", "response.failed"], served.text


def _assert_forwarded(received: tuple[Request, ...], calls: tuple[_Call, ...]) -> None:
    posts: Final = tuple(request for request in received if request.method == "POST")
    forwarded: Final = sorted(_newest_marker(request.body.decode()) or "" for request in posts)
    assert forwarded == sorted(call.marker for call in calls), forwarded


def _rows(call_ids: Sequence[str]) -> Sequence[Mapping[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT litellm_call_id, status FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = ANY(string_to_array(%s, %s))',
            (",".join(call_ids), ","),
        ),
        lambda found: len(found) >= len(call_ids),
        seconds=70,
    )


def _assert_each_lands_once(served: tuple[_Served, ...]) -> None:
    logged: Final = tuple(item for item in served if item.call.kind in _LOGGED_KINDS)
    rows: Final = _rows(tuple(item.call_id for item in logged))
    by_call: Final = {string_value(row["litellm_call_id"]): row for row in rows}
    assert len(by_call) == len(rows) == len(logged), rows
    for item in logged:
        expected: Final = "success" if item.call.kind == "chat_healthy" else "failure"
        assert by_call[item.call_id]["status"] == expected, (item.call_id, rows)


def _worker_pids(log: Path, count: int) -> tuple[int, ...]:
    return eventually(
        lambda: tuple(int(found.group(1)) for found in _STARTED_WORKER.finditer(log.read_text())),
        lambda pids: len(pids) == count,
        seconds=30,
    )


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@dataclass(frozen=True, slots=True)
class _Held:
    release: threading.Event
    markers: SimpleQueue[str]

    def respond(self, request: Request) -> Reply:
        marker: Final = _newest_marker(request.body.decode())
        assert marker is not None, request.body
        self.markers.put(marker)
        assert self.release.wait(timeout=60), "The burst was never released"
        return _respond(request)


async def _held_burst(gateway: Gateway, calls: tuple[_Call, ...], held: _Held) -> asyncio.Task[tuple[_Served, ...]]:
    burst: Final = asyncio.create_task(_burst(gateway, calls, tolerate_transport_errors=True))
    await asyncio.to_thread(eventually, held.markers.qsize, lambda size: size == len(calls), 60)
    return burst


async def test_mixed_burst_of_bridged_streams_answers_each_call_in_its_own_shape_and_logs_each_once(
    rig: _Rig,
) -> None:
    calls: Final = _calls(24, _KINDS)
    with wire_server(_respond, port=rig.port) as wire:
        served: Final = await _burst(rig.gateway, calls)
        assert len(served) == 24
        for item in served:
            _assert_answered_in_its_own_shape(item)
        _assert_forwarded(wire.drain(), calls)
    _assert_each_lands_once(served)


async def test_worker_sigkill_mid_burst_leaves_the_sibling_answering_the_bridged_streams(rig: _Rig) -> None:
    calls: Final = _calls(20, _CHAT_KINDS)
    held: Final = _Held(threading.Event(), SimpleQueue())
    with wire_server(held.respond, port=rig.port) as wire:
        workers: Final = _worker_pids(rig.proxy.log, 2)
        burst: Final = await _held_burst(rig.gateway, calls, held)
        held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
        assert sum(held_by.values()) == 20, held_by
        victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
        victim: Final = psutil.Process(victim_pid)
        victim.suspend()
        victim.send_signal(signal.SIGKILL)
        held.release.set()
        served: Final = await burst
        assert held_by[survivor_pid] >= 10, held_by
        assert len(served) == held_by[survivor_pid], (held_by, len(served))
        for item in served:
            _assert_answered_in_its_own_shape(item)
        follow_up: Final = _Call("chat_healthy", uuid.uuid4().hex)
        (answered,) = await _burst(rig.gateway, (follow_up,))
        _assert_answered_in_its_own_shape(answered)
        _assert_forwarded(wire.drain(), (*calls, follow_up))
    _assert_each_lands_once((*served, answered))


@pytest.mark.timeout(4 * graceful_stop_seconds() + 240)
async def test_proxy_sigterm_mid_burst_drains_the_spend_log_queue_and_the_restarted_proxy_serves(
    gateway: Gateway, tmp_path: Path
) -> None:
    port: Final = _free_port()
    config: Final = _config(port, tmp_path)
    calls: Final = _calls(20, _CHAT_KINDS)
    held: Final = _Held(threading.Event(), SimpleQueue())
    with wire_server(held.respond, port=port) as wire:
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            burst: Final = await _held_burst(owned.gateway, calls, held)
            owned.process.terminate()
            held.release.set()
            served: Final = await burst
            await asyncio.to_thread(
                eventually, owned.process.poll, lambda code: code is not None, graceful_stop_seconds()
            )
        assert len(served) == 20, len(served)
        for item in served:
            _assert_answered_in_its_own_shape(item)
        _assert_forwarded(wire.drain(), calls)
        _assert_each_lands_once(served)
        follow_up: Final = _Call("chat_healthy", uuid.uuid4().hex)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as restarted:
            (answered,) = await _burst(restarted.gateway, (follow_up,))
        _assert_answered_in_its_own_shape(answered)
        _assert_forwarded(wire.drain(), (follow_up,))
    _assert_each_lands_once((answered,))
