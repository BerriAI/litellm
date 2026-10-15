import asyncio
import json
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
    list_value,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from integration._support.ports import allocate_port
from integration.streaming.faked_responses_case import (
    ANTHROPIC_TOOLS,
    API_KEY,
    CHAT_TOOLS,
    FAKED_STREAM,
    NO_CACHE,
    RESPONSES_TOOLS,
    assert_each_value_arrives_once,
    function_call_item,
    message_item,
    response_body,
)
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_RESPONSES_MODEL: Final = "faked-stream-chaos"
_BRIDGE_MODEL: Final = "faked-stream-chaos-bridge"
_MARKER_PREFIX: Final = "marker-"
_DROP: Final = "drop-now"
_HEADERS: Final[Mapping[str, str]] = MappingProxyType({"anthropic-version": "2023-06-01"})

Kind: TypeAlias = Literal["responses", "chat", "messages"]
_KINDS: Final[tuple[Kind, ...]] = ("responses", "chat", "messages")


@dataclass(frozen=True, slots=True)
class _Call:
    kind: Kind
    marker: str
    dropped: bool = False


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


def _deployment(name: str, model: str, port: int) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_base": f"http://127.0.0.1:{port}/v1", "api_key": API_KEY},
        "model_info": FAKED_STREAM,
    }


def _config(port: int, directory: Path) -> Path:
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    router_settings: Final = object_value(stock.get("router_settings") or {})
    path: Final = directory / "faked-stream-chaos.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **stock,
                "model_list": [
                    _deployment(_RESPONSES_MODEL, "openai/faked-stream-backend", port),
                    _deployment(_BRIDGE_MODEL, "openai/responses/faked-stream-backend", port),
                ],
                "router_settings": {**router_settings, "num_retries": 0},
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("faked-stream-chaos")
    port: Final = allocate_port()
    with (
        gateway_from_environment() as shared,
        owned_proxy_process(shared, directory, {}, config=_config(port, directory), workers=2) as owned,
    ):
        yield _Rig(port, owned)


def _answer(marker: str) -> str:
    return f"answer marker-{marker}"


def _arguments(marker: str) -> str:
    return json.dumps({"city": "Paris", "marker": marker})


def _output(marker: str) -> tuple[dict[str, JsonValue], ...]:
    return (message_item(marker, _answer(marker)), function_call_item(marker, _arguments(marker)))


def _markers(text: str) -> tuple[str, ...]:
    return tuple(part[:32] for part in text.split(_MARKER_PREFIX)[1:])


def _newest_marker(request: Request) -> str:
    found: Final = _markers(request.body.decode())
    assert request.method == "POST" and found, (request.target, request.body)
    return found[-1]


def _respond(request: Request) -> Reply:
    marker: Final = _newest_marker(request)
    if _DROP.encode() in request.body:
        return Reply(drop_connection=True)
    return Reply(body=json.dumps(response_body(marker, _output(marker))).encode())


def _body(call: _Call) -> Mapping[str, JsonValue]:
    prompt: Final = f"{call.kind} marker-{call.marker}" + (f" {_DROP}" if call.dropped else "")
    common: Final[Mapping[str, JsonValue]] = {"stream": True, "num_retries": 0, "cache": NO_CACHE}
    match call.kind:
        case "responses":
            return {**common, "model": _RESPONSES_MODEL, "input": prompt, "tools": RESPONSES_TOOLS}
        case "chat":
            return {
                **common,
                "model": _BRIDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "tools": CHAT_TOOLS,
            }
        case "messages":
            return {
                **common,
                "model": _RESPONSES_MODEL,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
                "tools": ANTHROPIC_TOOLS,
            }


def _path(kind: Kind) -> str:
    match kind:
        case "responses":
            return "/v1/responses"
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"


def _calls(count: int, drop_every: int = 0) -> tuple[_Call, ...]:
    return tuple(
        _Call(_KINDS[index % len(_KINDS)], uuid.uuid4().hex, drop_every > 0 and index % drop_every == drop_every - 1)
        for index in range(count)
    )


async def _send(client: httpx.AsyncClient, key: str, call: _Call) -> _Served:
    async with client.stream(
        "POST", _path(call.kind), json=_body(call), headers={"Authorization": f"Bearer {key}", **_HEADERS}
    ) as response:
        raw: Final = await response.aread()
    return _Served(call, response.status_code, raw.decode(), response.headers["x-litellm-call-id"])


async def _burst(
    gateway: Gateway, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=90, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, gateway.key, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _frames(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: {")
    )


def _chat_deltas(frames: Sequence[Mapping[str, JsonValue]]) -> Iterator[dict[str, JsonValue]]:
    for frame in frames:
        for choice in list_value(frame.get("choices") or []):
            yield object_value(object_value(choice)["delta"])


def _chat_tool_calls(deltas: Sequence[Mapping[str, JsonValue]]) -> Iterator[JsonValue]:
    for delta in deltas:
        yield from list_value(delta.get("tool_calls") or [])


def _function_arguments(call: JsonValue) -> str:
    function: Final = object_value(call).get("function")
    return string_value(function.get("arguments") or "") if isinstance(function, dict) else ""


def _streamed_values(served: _Served) -> tuple[str, str]:
    frames: Final = _frames(served.text)
    match served.call.kind:
        case "responses":
            assert_each_value_arrives_once(frames, _output(served.call.marker))
            return _answer(served.call.marker), _arguments(served.call.marker)
        case "chat":
            deltas: Final = tuple(_chat_deltas(frames))
            return "".join(string_value(delta.get("content") or "") for delta in deltas), "".join(
                map(_function_arguments, _chat_tool_calls(deltas))
            )
        case "messages":
            blocks: Final = tuple(
                object_value(frame["delta"]) for frame in frames if frame["type"] == "content_block_delta"
            )
            return "".join(string_value(block.get("text", "")) for block in blocks), "".join(
                string_value(block.get("partial_json", "")) for block in blocks
            )


def _assert_each_value_once(served: _Served) -> None:
    assert served.status == 200, (served.call, served.status, served.text[:500])
    assert _streamed_values(served) == (_answer(served.call.marker), _arguments(served.call.marker)), (
        served.call,
        served.text[:2000],
    )


def _assert_failed_without_a_value(served: _Served) -> None:
    assert served.status >= 400 or '"error"' in served.text, (served.call, served.status, served.text[:500])
    assert _arguments(served.call.marker) not in served.text, (served.call, served.text[:500])


def _assert_forwarded(received: tuple[Request, ...], calls: Sequence[_Call]) -> None:
    posts: Final = tuple(request for request in received if request.method == "POST")
    assert all(JSON_OBJECT.validate_json(request.body).get("stream") is not True for request in posts), posts
    reached: Final = frozenset(_markers(b"\n".join(request.body for request in posts).decode()))
    assert reached >= frozenset(call.marker for call in calls), sorted(reached)


def _rows(call_ids: Sequence[str]) -> Sequence[Mapping[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT litellm_call_id, status FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = ANY(string_to_array(%s, %s))',
            (",".join(call_ids), ","),
        ),
        lambda found: len(found) >= len(call_ids),
        seconds=70,
    )


def _assert_each_logged_once(served: Sequence[_Served]) -> None:
    rows: Final = _rows(tuple(item.call_id for item in served))
    assert sorted(string_value(row["litellm_call_id"]) for row in rows) == sorted(item.call_id for item in served), rows
    assert all(row["status"] == "success" for row in rows), rows


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
        self.markers.put(_newest_marker(request))
        assert self.release.wait(timeout=60), "The burst was never released"
        return _respond(request)


async def _held_burst(gateway: Gateway, calls: tuple[_Call, ...], held: _Held) -> asyncio.Task[tuple[_Served, ...]]:
    burst: Final = asyncio.create_task(_burst(gateway, calls, tolerate_transport_errors=True))
    await asyncio.to_thread(eventually, held.markers.qsize, lambda size: size == len(calls), 60)
    return burst


def _held_by(rig: _Rig, upstream: str) -> Mapping[int, int]:
    return MappingProxyType({pid: _open_upstream_connections(pid, upstream) for pid in rig.proxy.worker_pids()})


async def test_held_mixed_burst_spreads_over_both_workers_and_each_value_arrives_once(rig: _Rig) -> None:
    calls: Final = _calls(30)
    held: Final = _Held(threading.Event(), SimpleQueue())
    with wire_server(held.respond, port=rig.port) as wire:
        burst: Final = await _held_burst(rig.gateway, calls, held)
        held_by: Final = _held_by(rig, wire.url)
        held.release.set()
        served: Final = await burst
        assert len(held_by) == 2 and all(held_by.values()) and sum(held_by.values()) == 30, held_by
        assert len(served) == 30
        for item in served:
            _assert_each_value_once(item)
        _assert_forwarded(wire.drain(), calls)
    _assert_each_logged_once(served)


async def test_upstream_dropping_every_third_connection_fails_only_those_calls(rig: _Rig) -> None:
    calls: Final = _calls(24, drop_every=3)
    with wire_server(_respond, port=rig.port) as wire:
        served: Final = await _burst(rig.gateway, calls)
        assert len(served) == 24
        healthy: Final = tuple(item for item in served if not item.call.dropped)
        for item in healthy:
            _assert_each_value_once(item)
        dropped: Final = tuple(item for item in served if item.call.dropped)
        assert len(dropped) == 8, dropped
        for item in dropped:
            _assert_failed_without_a_value(item)
        follow_up: Final = await _burst(rig.gateway, _calls(3))
        for item in follow_up:
            _assert_each_value_once(item)
        _assert_forwarded(wire.drain(), calls)
    _assert_each_logged_once((*healthy, *follow_up))


async def test_worker_sigkill_mid_burst_leaves_the_sibling_streaming_each_value_once(rig: _Rig) -> None:
    calls: Final = _calls(21)
    held: Final = _Held(threading.Event(), SimpleQueue())
    with wire_server(held.respond, port=rig.port) as wire:
        burst: Final = await _held_burst(rig.gateway, calls, held)
        held_by: Final = _held_by(rig, wire.url)
        assert len(held_by) == 2 and sum(held_by.values()) == 21, held_by
        victim_pid, survivor_pid = sorted(held_by, key=held_by.__getitem__)
        with rig.proxy.replacing_worker(victim_pid) as kill:
            kill.freeze()
            await asyncio.to_thread(kill)
            held.release.set()
            served: Final = await burst
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_each_value_once(item)
            follow_up: Final = await _burst(rig.gateway, _calls(3))
            for item in follow_up:
                _assert_each_value_once(item)
        _assert_forwarded(wire.drain(), (*calls, *(item.call for item in follow_up)))
    _assert_each_logged_once((*served, *follow_up))
