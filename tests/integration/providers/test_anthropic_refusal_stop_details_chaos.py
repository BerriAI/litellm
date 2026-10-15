import asyncio
import json
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.anthropic_refusal import (
    REFUSAL_FIELDS,
    Ledger,
    Location,
    refusal_prompt,
    relayed,
    respond,
    sink_config,
)
from integration._support.anthropic_thinking import JSON_OBJECT, MODEL, Event, chunks_of
from integration._support.client import Gateway, eventually, held
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.timing import assert_not_delayed_by
from integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = [
    pytest.mark.timeout(2 * graceful_stop_seconds() + 120),
    pytest.mark.xdist_group("anthropic_refusal_stop_details_chaos"),
]

_MODEL_NAME: Final = "anthropic-refusal-chaos"
_SLOW_SINK_SECONDS: Final = 4
_SINK: Final = "refusal_chaos_sink"
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_CHAT_FIELDS: Final[Location] = ("choices", 0, "message", "provider_specific_fields")
_DELTA_FIELDS: Final[Location] = ("choices", 0, "delta", "provider_specific_fields")
_PATHS: Final[Mapping[str, str]] = MappingProxyType(
    {"chat": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}
)

Surface: TypeAlias = Literal["chat", "messages", "responses"]
_RecordProperty: TypeAlias = Callable[[str, object], None]


@dataclass(frozen=True, slots=True)
class _Call:
    surface: Surface
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


@dataclass(frozen=True, slots=True)
class _Held:
    release: threading.Event
    markers: SimpleQueue[str]

    def respond(self, request: Request) -> Reply:
        self.markers.put(_markers_in(request.body.decode())[0])
        assert self.release.wait(timeout=90), "The burst was never released"
        return respond(request)


def _markers_in(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_MARKER.findall(text)))


def _calls(count: int) -> tuple[_Call, ...]:
    surfaces: Final[tuple[Surface, ...]] = ("chat", "messages", "responses")
    return tuple(
        _Call(surfaces[index % 3], stream=(index // 3) % 2 == 0, marker=uuid.uuid4().hex) for index in range(count)
    )


def _body(call: _Call) -> Event:
    text: Final = refusal_prompt("refusal", call.marker)
    shared: Final[Event] = {"model": _MODEL_NAME, "stream": call.stream, "cache": {"no-cache": True}}
    match call.surface:
        case "chat":
            return {**shared, "messages": [{"role": "user", "content": text}], "max_tokens": 64}
        case "messages":
            return {**shared, "messages": [{"role": "user", "content": text}], "max_tokens": 64}
        case "responses":
            return {**shared, "input": text, "max_output_tokens": 64}


async def _send(client: httpx.AsyncClient, key: str, call: _Call) -> _Served:
    try:
        response: Final = await client.post(
            _PATHS[call.surface], json=_body(call), headers={"Authorization": f"Bearer {key}"}
        )
    except httpx.TransportError as error:
        return _Served(call, 0, repr(error))
    return _Served(call, response.status_code, response.text)


async def _burst(gateway: Gateway, calls: Sequence[_Call]) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=120, trust_env=False) as client:
        return tuple(await asyncio.gather(*(_send(client, gateway.key, call) for call in calls)))


def _chaos_config(directory: Path, upstream: Wire, sink: str, *, retries: int) -> Path:
    base: Final = sink_config(directory, {_SINK: sink}, retries=retries)
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(base.read_text()))
    model_list: Final = [
        {
            "model_name": _MODEL_NAME,
            "litellm_params": {"model": f"anthropic/{MODEL}", "api_base": upstream.url, "api_key": "scripted-key"},
        }
    ]
    path: Final = directory / "anthropic-refusal-chaos.yaml"
    path.write_text(yaml.safe_dump({**config, "model_list": model_list}))
    return path


def _ledger_events(ledgers: Sequence[Ledger]) -> tuple[Event, ...]:
    return tuple(chain.from_iterable(ledger.events() for ledger in ledgers))


def _marker_counts(ledgers: Sequence[Ledger], markers: frozenset[str]) -> Mapping[str, int]:
    found: Final = chain.from_iterable(_markers_in(json.dumps(event)) for event in _ledger_events(ledgers))
    return MappingProxyType(Counter(marker for marker in found if marker in markers))


def _events_for(ledgers: Sequence[Ledger], marker: str) -> tuple[Event, ...]:
    return tuple(event for event in _ledger_events(ledgers) if marker in json.dumps(event))


def _assert_answered(item: _Served) -> None:
    assert item.status == 200, (item.call, item.text)
    if item.call.surface != "chat":
        return
    if item.call.stream:
        found: Final = tuple(chain.from_iterable(relayed(chunk) for chunk in chunks_of(item.text)))
        assert found == ((_DELTA_FIELDS, REFUSAL_FIELDS),), item.text
        return
    assert relayed(JSON_OBJECT.validate_json(item.text)) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), item.text


def _assert_logged_chat_relays(ledgers: Sequence[Ledger], calls: Sequence[_Call]) -> None:
    for call in calls:
        if call.surface != "chat":
            continue
        (event,) = _events_for(ledgers, call.marker)
        assert relayed(event["response"]) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), event


def _worker_pids(owned: OwnedProxy) -> tuple[int, ...]:
    return eventually(
        lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
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


def _port(wire: Wire) -> int:
    port: Final = urlsplit(wire.url).port
    assert port is not None, wire.url
    return port


def _accept(_request: Request) -> Reply:
    return Reply()


async def _timed_chat(gateway: Gateway) -> float:
    started: Final = time.monotonic()
    (served,) = await _burst(gateway, (_Call("chat", stream=False, marker=uuid.uuid4().hex),))
    _assert_answered(served)
    return time.monotonic() - started


async def test_sink_outage_mid_burst_delivers_every_refusal_event_once(gateway: Gateway, tmp_path: Path) -> None:
    calls: Final = _calls(30)
    markers: Final = frozenset(call.marker for call in calls)
    with wire_server(respond) as upstream, ExitStack() as first_sink:
        sink: Final = first_sink.enter_context(wire_server(_accept))
        first: Final = Ledger(sink)
        config: Final = _chaos_config(tmp_path, upstream, sink.url, retries=6)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            _worker_pids(owned)
            served_first: Final = await _burst(owned.gateway, calls[:10])
            await asyncio.to_thread(
                eventually, lambda: sum(_marker_counts((first,), markers).values()), lambda total: total >= 1, 30
            )
            first.events()
            first_sink.close()
            served_during: Final = await _burst(owned.gateway, calls[10:20])
            with wire_server(_accept, port=_port(sink)) as restarted:
                second: Final = Ledger(restarted)
                served_after: Final = await _burst(owned.gateway, calls[20:])
                for item in (*served_first, *served_during, *served_after):
                    _assert_answered(item)
                counts: Final = await asyncio.to_thread(
                    held,
                    lambda: _marker_counts((first, second), markers),
                    lambda found: set(found) == markers and set(found.values()) == {1},
                    holding=3,
                    seconds=60,
                )
                assert sum(counts.values()) == 30, counts
                _assert_logged_chat_relays((first, second), calls)
    assert len(upstream.drain()) == 30


async def test_slow_sink_neither_deadlocks_nor_uploads_a_refusal_event_twice(gateway: Gateway, tmp_path: Path) -> None:
    calls: Final = _calls(30)
    markers: Final = frozenset(call.marker for call in calls)

    def slow(_request: Request) -> Reply:
        time.sleep(_SLOW_SINK_SECONDS)
        return Reply()

    with wire_server(respond) as upstream, wire_server(slow) as sink:
        ledger: Final = Ledger(sink)
        config: Final = _chaos_config(tmp_path, upstream, sink.url, retries=0)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            _worker_pids(owned)
            control: Final = await _timed_chat(owned.gateway)
            served: Final = await _burst(owned.gateway, calls)
            for item in served:
                _assert_answered(item)
            follow_up: Final = await _timed_chat(owned.gateway)
            assert_not_delayed_by((follow_up,), control=(control,), injected_delay=_SLOW_SINK_SECONDS)
            counts: Final = await asyncio.to_thread(
                held,
                lambda: _marker_counts((ledger,), markers),
                lambda found: set(found) == markers and set(found.values()) == {1},
                holding=5,
                seconds=90,
            )
            assert sum(counts.values()) == 30, counts
            _assert_logged_chat_relays((ledger,), calls)


async def test_worker_sigkill_mid_burst_leaves_the_sibling_relaying_and_logging_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20)
    hold: Final = _Held(threading.Event(), SimpleQueue())
    with wire_server(hold.respond) as upstream, wire_server(_accept) as sink:
        ledger: Final = Ledger(sink)
        config: Final = _chaos_config(tmp_path, upstream, sink.url, retries=0)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            workers: Final = _worker_pids(owned)
            burst: Final = asyncio.create_task(_burst(owned.gateway, calls))
            await asyncio.to_thread(eventually, hold.markers.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, upstream.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            with owned.replacing_worker(victim_pid) as kill:
                kill.freeze()
                await asyncio.to_thread(kill)
                hold.release.set()
                served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            completed: Final = tuple(item for item in served if item.status == 200)
            assert len(completed) == held_by[survivor_pid], (held_by, [item.status for item in served])
            for item in completed:
                _assert_answered(item)
            follow_up: Final = _Call("chat", stream=True, marker=uuid.uuid4().hex)
            (answered,) = await _burst(owned.gateway, (follow_up,))
            _assert_answered(answered)
            survivors: Final = frozenset((*(item.call.marker for item in completed), follow_up.marker))
            counts: Final = await asyncio.to_thread(
                held,
                lambda: _marker_counts((ledger,), frozenset(call.marker for call in (*calls, follow_up))),
                lambda found: set(found) == survivors and set(found.values()) == {1},
                holding=3,
                seconds=60,
            )
            assert sum(counts.values()) == len(survivors), counts
            _assert_logged_chat_relays((ledger,), tuple(item.call for item in completed))


async def test_proxy_restart_mid_burst_never_logs_a_refusal_twice(
    gateway: Gateway, tmp_path: Path, record_property: _RecordProperty
) -> None:
    calls: Final = _calls(20)
    markers: Final = frozenset(call.marker for call in calls)
    hold: Final = _Held(threading.Event(), SimpleQueue())
    with wire_server(hold.respond) as upstream, wire_server(_accept) as sink:
        ledger: Final = Ledger(sink)
        config: Final = _chaos_config(tmp_path, upstream, sink.url, retries=0)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as first:
            _worker_pids(first)
            burst: Final = asyncio.create_task(_burst(first.gateway, calls))
            await asyncio.to_thread(eventually, hold.markers.qsize, lambda size: size == 20, 60)
            first.begin_shutdown()
            hold.release.set()
            served: Final = await burst
        for item in served:
            _assert_answered(item)
        second_directory: Final = tmp_path / "second"
        second_directory.mkdir()
        with owned_proxy_process(gateway, second_directory, {}, config=config, workers=2) as second:
            follow_up: Final = _Call("chat", stream=False, marker=uuid.uuid4().hex)
            (answered,) = await _burst(second.gateway, (follow_up,))
            _assert_answered(answered)
            await asyncio.to_thread(
                eventually,
                lambda: _marker_counts((ledger,), frozenset({follow_up.marker})),
                lambda found: found[follow_up.marker] == 1,
                30,
            )
        counts: Final = _marker_counts((ledger,), markers)
        assert set(counts.values()) <= {1}, counts
        _assert_logged_chat_relays((ledger,), tuple(call for call in calls if call.marker in counts))
        record_property("served", len(served))
        record_property("landed", sum(counts.values()))
