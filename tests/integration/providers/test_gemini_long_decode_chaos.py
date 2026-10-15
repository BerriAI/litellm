from __future__ import annotations

import asyncio
import dataclasses
import re
import threading
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
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
from integration._support.client import JSON_OBJECT, Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._gemini_long_decode_support import (
    API_KEY,
    BACKEND,
    CHECKPOINT,
    ENDPOINTS,
    INPUT_COST,
    OUTPUT_COST,
    PLAIN,
    PROVIDERS,
    SPEND_COLUMNS,
    Endpoint,
    Hop,
    Seen,
    call_type,
    carries_token,
    complete,
    cost_of,
    cut_off,
    hop_reply,
    number,
    path_of,
    payloads,
    post,
    register,
    request_body,
    seen,
    single,
    streams,
    summed,
)
from pydantic import JsonValue

Continuation = Callable[[Request, str], Reply]

_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_HOP_TOKEN: Final = "hop-token-"
_PROBE_ID: Final = "probe-"
_MODES: Final = len(ENDPOINTS) * 2


@dataclass(frozen=True, slots=True)
class Call:
    endpoint: Endpoint
    stream: bool
    marker: str
    model: str


@dataclass(frozen=True, slots=True)
class Served:
    call: Call
    status: int
    text: str

    @property
    def observed(self) -> Seen:
        assert self.status == 200, self.text
        return seen(self.call.endpoint, self.call.stream, payloads(self.text, self.call.stream), self.text)


def _opening(marker: str) -> Hop:
    return Hop((f"opening marker-{marker} ", "and "), "CONTINUATION", f"{_HOP_TOKEN}{marker}", 20, 7)


def _closing(marker: str) -> Hop:
    return Hop((f"closing marker-{marker}", "."), "STOP", None, 27, 5)


def _joined(marker: str) -> str:
    return _opening(marker).text + _closing(marker).text


_OPENING_USAGE: Final = _opening("").usage
_CHAIN_USAGE: Final = summed(_opening(""), _closing(""))


def _marker_asked(request: Request) -> str | None:
    found: Final = _MARKER.search(request.body.decode())
    return found.group(1) if found is not None else None


def _token_sent(request: Request) -> str | None:
    token: Final = JSON_OBJECT.validate_json(request.body).get("continuationToken")
    return token if isinstance(token, str) else None


def _continues(request: Request, token: str) -> Reply:
    return hop_reply(f"second-{uuid.uuid4().hex}", _closing(token.removeprefix(_HOP_TOKEN)), streams(request))


def _continues_slowly(request: Request, token: str) -> Reply:
    return dataclasses.replace(_continues(request, token), pause_between_chunks=0.3)


def _held(release: threading.Event, held: SimpleQueue[str], then: Continuation = _continues) -> Continuation:
    def hold(request: Request, token: str) -> Reply:
        held.put(token)
        assert release.wait(timeout=90), "The continuation requests were never released"
        return then(request, token)

    return hold


def _down_for(doomed: frozenset[str]) -> Continuation:
    def answer(request: Request, token: str) -> Reply:
        if token.removeprefix(_HOP_TOKEN) in doomed:
            return Reply(drop_connection=True)
        return _continues(request, token)

    return answer


def _google(continuation: Continuation = _continues) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        marker: Final = _marker_asked(request)
        token: Final = _token_sent(request)
        if marker is None:
            return hop_reply(f"{_PROBE_ID}{uuid.uuid4().hex}", PLAIN, streams(request))
        if token is None:
            return hop_reply(f"first-{marker}", _opening(marker), streams(request))
        return continuation(request, token)

    return respond


@contextmanager
def _released_on_exit(release: threading.Event) -> Generator[threading.Event]:
    try:
        yield release
    finally:
        release.set()


def _calls(count: int, models: Sequence[str], stream: Callable[[int], bool]) -> tuple[Call, ...]:
    return tuple(
        Call(
            endpoint=ENDPOINTS[index % len(ENDPOINTS)],
            stream=stream(index),
            marker=uuid.uuid4().hex,
            model=models[(index // _MODES) % len(models)],
        )
        for index in range(count)
    )


def _alternating(index: int) -> bool:
    return index % 2 == 0


async def _send(client: httpx.AsyncClient, key: str, call: Call) -> Served:
    question: Final = f"Write the long answer for marker-{call.marker}"
    async with client.stream(
        "POST",
        path_of(call.endpoint),
        json=request_body(call.endpoint, call.model, call.stream, prompt=question),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return Served(call=call, status=response.status_code, text=raw.decode())


async def _burst(
    gateway: Gateway, calls: Sequence[Call], *, tolerate_transport_errors: bool = False
) -> tuple[Served, ...]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=90, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, gateway.key, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, Served))


async def _all_mid_chain(held: SimpleQueue[str], burst: asyncio.Task[tuple[Served, ...]], count: int) -> None:
    await asyncio.to_thread(eventually, lambda: held.qsize() == count or burst.done(), bool, 60)
    assert held.qsize() == count, f"{held.qsize()} of {count} answers reached their continuation request"


def _assert_joined(served: Served) -> None:
    observed: Final = served.observed
    whole: Final = _joined(served.call.marker)
    assert (observed.answer, observed.streamed) == (whole, whole), served.text
    assert observed.finishes == (complete(served.call.endpoint),), served.text
    assert (observed.usage, observed.tokens, observed.errors) == (_CHAIN_USAGE, (), ()), served.text
    assert CHECKPOINT not in served.text and _HOP_TOKEN not in served.text, served.text
    assert set(_MARKER.findall(served.text)) == {served.call.marker}, served.text


def _assert_cut_off_with_its_own_token(served: Served) -> None:
    observed: Final = served.observed
    call: Final = served.call
    so_far: Final = _opening(call.marker).text
    own: Final = (f"{_HOP_TOKEN}{call.marker}",) if carries_token(call.endpoint, call.stream) else ()
    assert (observed.answer, observed.streamed) == (so_far, so_far), served.text
    assert observed.finishes == (cut_off(call.endpoint),), served.text
    assert (observed.usage, observed.tokens, observed.errors) == (_OPENING_USAGE, own, ()), served.text
    assert CHECKPOINT not in served.text, served.text
    assert set(_MARKER.findall(served.text)) == {call.marker}, served.text


def _assert_each_answer_sent_its_own_token_once(received: Sequence[Request], calls: Sequence[Call]) -> None:
    sent: Final = sorted(
        (marker, _token_sent(request) or "") for request in received if (marker := _marker_asked(request)) is not None
    )
    assert sent == sorted(
        (*((call.marker, "") for call in calls), *((call.marker, f"{_HOP_TOKEN}{call.marker}") for call in calls))
    ), sent


def _spend_rows(model: str) -> Sequence[Mapping[str, JsonValue]]:
    return read_rows(
        f"{SPEND_COLUMNS} WHERE model_group=%s AND request_id NOT LIKE '{_PROBE_ID}%%' ORDER BY request_id", (model,)
    )


def _billed_as(row: Mapping[str, JsonValue], served: Served, usage: tuple[int, int]) -> bool:
    identities: Final = (single(served.observed.ids), f"first-{served.call.marker}")
    return (
        any(same_response(string_value(row["request_id"]), identity) for identity in identities)
        and (row["call_type"], row["status"], row["prompt_tokens"], row["completion_tokens"])
        == (call_type(served.call.endpoint), "success", *usage)
        and number(row["spend"]) == pytest.approx(cost_of(usage))
    )


def _billings(rows: Sequence[Mapping[str, JsonValue]], served: Served, usage: tuple[int, int]) -> int:
    return sum(1 for row in rows if _billed_as(row, served, usage))


def _assert_landed_once(model: str, joined: Sequence[Served], cut_off_early: Sequence[Served] = ()) -> None:
    mine: Final = tuple(
        (served, usage)
        for served, usage in (
            *((served, _CHAIN_USAGE) for served in joined),
            *((served, _OPENING_USAGE) for served in cut_off_early),
        )
        if served.call.model == model
    )
    rows: Final = eventually(lambda: _spend_rows(model), lambda found: len(found) >= len(mine), seconds=90)
    assert len(rows) == len(mine), (len(rows), len(mine), rows)
    unbilled: Final = tuple(served.call for served, usage in mine if _billings(rows, served, usage) != 1)
    assert unbilled == (), (unbilled, rows)


@pytest.mark.timeout(400)
async def test_mixed_burst_of_continued_answers_keeps_every_answer_on_its_own_token_and_bills_it_once(
    gateway: Gateway,
) -> None:
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    with (
        wire_server(_google(_held(release, held))) as wire,
        _released_on_exit(release),
        gateway.scenario() as scenario,
    ):
        models: Final = tuple(register(gateway, scenario, provider, wire) for provider in PROVIDERS)
        calls: Final = _calls(30, models, _alternating)
        burst: Final = asyncio.create_task(_burst(gateway, calls))
        await _all_mid_chain(held, burst, 30)
        release.set()
        served: Final = await burst
        assert len(served) == 30
        for item in served:
            _assert_joined(item)
        _assert_each_answer_sent_its_own_token_once(wire.drain(), calls)
        for model in models:
            _assert_landed_once(model, served)


@pytest.mark.timeout(400)
async def test_slow_continuation_streams_are_joined_once_each_under_concurrency(gateway: Gateway) -> None:
    with wire_server(_google(_continues_slowly)) as wire, gateway.scenario() as scenario:
        model: Final = register(gateway, scenario, "gemini", wire)
        calls: Final = _calls(20, (model,), lambda _: True)
        served: Final = await _burst(gateway, calls)
        assert len(served) == 20
        for item in served:
            _assert_joined(item)
        _assert_each_answer_sent_its_own_token_once(wire.drain(), calls)
        _assert_landed_once(model, served)


@pytest.mark.timeout(400)
async def test_google_outage_on_half_the_continuation_requests_mid_burst_cuts_those_off_and_recovers(
    gateway: Gateway,
) -> None:
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    with gateway.scenario() as scenario:
        calls: Final = _calls(30, ("unregistered",), _alternating)
        doomed: Final = frozenset(call.marker for call in calls[::2])
        with wire_server(_google(_held(release, held, _down_for(doomed)))) as wire, _released_on_exit(release):
            model: Final = register(gateway, scenario, "gemini", wire)
            routed: Final = tuple(dataclasses.replace(call, model=model) for call in calls)
            burst: Final = asyncio.create_task(_burst(gateway, routed))
            await _all_mid_chain(held, burst, 30)
            release.set()
            served: Final = await burst
            assert len(served) == 30
            lost: Final = tuple(item for item in served if item.call.marker in doomed)
            kept: Final = tuple(item for item in served if item.call.marker not in doomed)
            for item in lost:
                _assert_cut_off_with_its_own_token(item)
            for item in kept:
                _assert_joined(item)
            after: Final = _calls(_MODES, (model,), _alternating)
            recovered: Final = await _burst(gateway, after)
            assert len(recovered) == _MODES
            for item in recovered:
                _assert_joined(item)
            _assert_each_answer_sent_its_own_token_once(wire.drain(), (*routed, *after))
            _assert_landed_once(model, (*kept, *recovered), lost)


def _chaos_config(wire: Wire, directory: Path, model_name: str) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "model_list": [
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": f"gemini/{BACKEND}",
                    "api_base": wire.url,
                    "api_key": API_KEY,
                    "input_cost_per_token": INPUT_COST,
                    "output_cost_per_token": OUTPUT_COST,
                },
            }
        ],
    }
    path: Final = directory / "gemini-long-decode-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _answers_probes(gateway: Gateway, model: str) -> bool:
    try:
        return post(gateway, "chat", request_body("chat", model, False)).status_code == 200
    except httpx.TransportError:
        return False


def _ready(gateway: Gateway, model: str) -> None:
    eventually(
        lambda: tuple(_answers_probes(gateway, model) for _ in range(2)), all, seconds=graceful_stop_seconds() + 60
    )


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _worker_pids(log: Path) -> tuple[int, ...]:
    return eventually(
        lambda: tuple(int(started.group(1)) for started in _STARTED_WORKER.finditer(log.read_text())),
        lambda pids: len(pids) == 2,
        seconds=graceful_stop_seconds() + 60,
    )


@pytest.mark.timeout(2 * graceful_stop_seconds() + 240)
async def test_worker_sigkill_mid_answer_leaves_the_sibling_joining_its_own_answers(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    alias: Final = f"gemini-long-decode-chaos-{uuid.uuid4().hex[:12]}"
    calls: Final = _calls(20, (alias,), _alternating)
    with wire_server(_google(_held(release, held))) as wire, _released_on_exit(release):
        config: Final = _chaos_config(wire, tmp_path, alias)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = _worker_pids(owned.log)
            _ready(candidate, alias)
            wire.drain()
            burst: Final = asyncio.create_task(_burst(candidate, calls, tolerate_transport_errors=True))
            await _all_mid_chain(held, burst, 20)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            with owned.replacing_worker(victim_pid) as kill:
                kill.freeze()
                await asyncio.to_thread(kill)
                release.set()
                served: Final = await burst
                assert held_by[survivor_pid] >= 10, held_by
                assert len(served) == held_by[survivor_pid], (held_by, len(served))
                for item in served:
                    _assert_joined(item)
            follow_up: Final = _calls(1, (alias,), _alternating)
            (answered,) = await _burst(candidate, follow_up)
            _assert_joined(answered)
            _assert_each_answer_sent_its_own_token_once(wire.drain(), (*calls, *follow_up))
            _assert_landed_once(alias, (*served, answered))
            eventually(
                lambda: owned.log.read_text().count("Application startup complete."),
                lambda started: started == 3,
                seconds=graceful_stop_seconds() + 60,
            )


@pytest.mark.timeout(4 * graceful_stop_seconds() + 240)
async def test_proxy_sigterm_mid_answer_finishes_every_answer_and_the_rebooted_proxy_bills_each_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    alias: Final = f"gemini-long-decode-chaos-{uuid.uuid4().hex[:12]}"
    calls: Final = _calls(20, (alias,), _alternating)
    with wire_server(_google(_held(release, held))) as wire, _released_on_exit(release):
        config: Final = _chaos_config(wire, tmp_path, alias)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            _worker_pids(owned.log)
            _ready(owned.gateway, alias)
            wire.drain()
            burst: Final = asyncio.create_task(_burst(owned.gateway, calls))
            await _all_mid_chain(held, burst, 20)
            owned.begin_shutdown()
            release.set()
            served: Final = await burst
            assert len(served) == 20
            for item in served:
                _assert_joined(item)
            await asyncio.to_thread(
                eventually, owned.process.poll, lambda code: code is not None, graceful_stop_seconds() + 60
            )
            _assert_each_answer_sent_its_own_token_once(wire.drain(), calls)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as rebooted:
            _worker_pids(rebooted.log)
            _ready(rebooted.gateway, alias)
            wire.drain()
            follow_up: Final = _calls(_MODES, (alias,), _alternating)
            recovered: Final = await _burst(rebooted.gateway, follow_up)
            assert len(recovered) == _MODES
            for item in recovered:
                _assert_joined(item)
            _assert_each_answer_sent_its_own_token_once(wire.drain(), follow_up)
            _assert_landed_once(alias, (*served, *recovered))
