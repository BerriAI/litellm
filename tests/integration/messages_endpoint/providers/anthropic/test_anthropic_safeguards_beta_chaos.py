import asyncio
import json
import re
import signal
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.messages_endpoint.providers.anthropic.test_anthropic_safeguards_beta_wire import (
    ANTHROPIC_KEY,
    ANTHROPIC_MODEL,
    BETA,
    anthropic_peer,
    marker_of,
    messages_body,
    stream_frames,
)
from pydantic import JsonValue, TypeAdapter

_CONFIG_MODEL: Final = "anthropic-safeguards-chaos"
_YAML_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_MODEL_LIST: Final = json.dumps(
    {"object": "list", "data": [{"id": ANTHROPIC_MODEL, "object": "model", "owned_by": "anthropic"}]}
).encode()


@dataclass(frozen=True, slots=True)
class _Call:
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


def _calls(count: int, stream: bool) -> tuple[_Call, ...]:
    return tuple(_Call(stream=stream, marker=uuid.uuid4().hex) for _ in range(count))


def _streamed_reply(marker: str, abort_after: int | None = None, pause: float = 0) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=stream_frames(marker),
        abort_after=abort_after,
        pause_between_chunks=pause,
    )


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        "/v1/messages",
        json=messages_body(model, call.marker, stream=call.stream),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(call=call, status=response.status_code, text=raw.decode())


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


def _assert_answered_with_its_own_marker(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(_MARKER.findall(served.text)) == {served.call.marker}, served.text
    assert f"msg_{served.call.marker}" in served.text, served.text
    if served.call.stream:
        assert "event: message_stop" in served.text, served.text


def _assert_every_request_carried_the_beta(received: tuple[Request, ...], markers: frozenset[str]) -> None:
    assert sorted(marker_of(request) for request in received) == sorted(markers)
    missing: Final = [
        frozenset(request.headers) for request in received if request.headers.get("anthropic-beta") != BETA
    ]
    assert missing == [], missing


def _spend_success_ids(model: str, expected: frozenset[str]) -> None:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE model_group=%s AND status=%s', (model, "success")
        ),
        lambda found: expected <= {str(row["request_id"]) for row in found},
        seconds=70,
    )
    identities: Final = [row["request_id"] for row in rows]
    assert len(set(identities)) == len(identities), identities


async def test_burst_with_aborted_streams_keeps_the_beta_on_every_forwarded_request(gateway: Gateway) -> None:
    calls: Final = tuple(_Call(stream=index % 2 == 0, marker=uuid.uuid4().hex) for index in range(30))
    aborted: Final = frozenset(call.marker for index, call in enumerate(calls) if call.stream and index % 3 == 0)

    def respond(request: Request) -> Reply:
        marker: Final = marker_of(request)
        if marker in aborted:
            return _streamed_reply(marker, abort_after=0)
        return anthropic_peer(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=ANTHROPIC_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 30
        for item in served:
            if item.call.marker in aborted:
                assert item.status == 500, item.text
                assert '"type":"api_error"' in item.text and "marker-" not in item.text, item.text
            else:
                _assert_answered_with_its_own_marker(item)
        recovery: Final = _Call(stream=True, marker=uuid.uuid4().hex)
        (recovered,) = await _burst(str(gateway.client.base_url), gateway.key, model, (recovery,))
        _assert_answered_with_its_own_marker(recovered)
        _assert_every_request_carried_the_beta(wire.drain(), frozenset(call.marker for call in (*calls, recovery)))
        answered: Final = frozenset(f"msg_{call.marker}" for call in (*calls, recovery) if call.marker not in aborted)
        _spend_success_ids(model, answered)


async def test_slow_streams_complete_with_the_beta_on_every_request(gateway: Gateway) -> None:
    calls: Final = _calls(10, stream=True)
    with (
        wire_server(lambda request: _streamed_reply(marker_of(request), pause=0.3)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=ANTHROPIC_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 10
        for item in served:
            _assert_answered_with_its_own_marker(item)
        _assert_every_request_carried_the_beta(wire.drain(), frozenset(call.marker for call in calls))
        _spend_success_ids(model, frozenset(f"msg_{call.marker}" for call in calls))


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    base: Final = _YAML_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    deployment: Final[dict[str, JsonValue]] = {
        "model_name": _CONFIG_MODEL,
        "litellm_params": {"model": f"anthropic/{ANTHROPIC_MODEL}", "api_base": wire.url, "api_key": ANTHROPIC_KEY},
    }
    config: Final = {**base, "model_list": [deployment]}
    path: Final = tmp_path / "anthropic-safeguards-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(180)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_sending_the_beta(gateway: Gateway, tmp_path: Path) -> None:
    calls: Final = _calls(20, stream=False)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        if (request.method, request.target) == ("GET", "/v1/models"):
            return Reply(body=_MODEL_LIST)
        held_markers.put(marker_of(request))
        assert release.wait(timeout=60), "The burst was never released"
        return anthropic_peer(request)

    with wire_server(held) as wire:
        path: Final = _chaos_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(found.group(1)) for found in _STARTED_WORKER.finditer(owned.log.read_text())),
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
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_answered_with_its_own_marker(item)
            follow_up: Final = _Call(stream=False, marker=uuid.uuid4().hex)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, (follow_up,))
            _assert_answered_with_its_own_marker(answered)
            received: Final = wire.drain()
            posts: Final = tuple(request for request in received if request.method == "POST")
            probes: Final[list[tuple[str, str]]] = [
                (request.method, request.target) for request in received if request.method != "POST"
            ]
            assert set(probes) <= {("GET", "/v1/models")}, probes
            _assert_every_request_carried_the_beta(posts, frozenset(call.marker for call in (*calls, follow_up)))
