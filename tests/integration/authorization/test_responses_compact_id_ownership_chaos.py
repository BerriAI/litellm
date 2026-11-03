from __future__ import annotations

import threading
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
from integration._support.client import Gateway, eventually
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Wire, wire_server
from integration._support.ports import allocate_port
from integration.authorization._responses_compact import (
    ALIASES,
    OTHER_TEAM_DETAIL,
    OTHER_USER_DETAIL,
    UNMANAGED_DETAIL,
    CompactUpstream,
    Tenants,
    Upstreamed,
    assert_compact_rows,
    assert_compacted_reply,
    assert_refused,
    compact_body,
    compact_hits,
    compact_post,
    marker,
    mint,
    minted_id,
    successful_markers,
    tenants,
    upstream_compact_body,
    upstreamed,
    vendor_id,
)

pytestmark: Final = pytest.mark.shares_worker_state

_Kind = Literal["owner", "other_user", "other_team", "unmanaged"]
_KINDS: Final[tuple[_Kind, ...]] = ("owner", "other_user", "other_team", "unmanaged")
_REFUSING: Final[tuple[_Kind, ...]] = ("other_user", "other_team", "unmanaged")
_OWNER: Final[tuple[_Kind, ...]] = ("owner",)
_REFUSALS: Final[Mapping[str, str]] = {
    "other_user": OTHER_USER_DETAIL,
    "other_team": OTHER_TEAM_DETAIL,
    "unmanaged": UNMANAGED_DETAIL,
}


@dataclass(frozen=True, slots=True)
class _Call:
    kind: _Kind
    prefix: str
    text: str


@dataclass(frozen=True, slots=True)
class _Outcome:
    call: _Call
    response: httpx.Response | None


@dataclass(frozen=True, slots=True)
class _Target:
    gateway: Gateway
    actors: Tenants
    owned: str
    foreign: str


def _calls(label: str, count: int, kinds: tuple[_Kind, ...] = _KINDS) -> tuple[_Call, ...]:
    return tuple(
        _Call(kinds[index % len(kinds)], ALIASES[index % len(ALIASES)], marker(label)) for index in range(count)
    )


def _send(target: _Target, call: _Call) -> _Outcome:
    plans: Final[Mapping[str, tuple[str, str]]] = {
        "owner": (target.actors.owner, target.owned),
        "other_user": (target.actors.other_user, target.owned),
        "other_team": (target.actors.other_team, target.owned),
        "unmanaged": (target.actors.owner, target.foreign),
    }
    key, previous = plans[call.kind]
    body: Final = compact_body(target.actors.model, call.text, {"previous_response_id": previous})
    try:
        return _Outcome(call, compact_post(target.gateway, key, call.prefix, body, call.text))
    except httpx.TransportError:
        return _Outcome(call, None)


def _submit(pool: ThreadPoolExecutor, target: _Target, calls: tuple[_Call, ...]) -> tuple[Future[_Outcome], ...]:
    return tuple(pool.submit(_send, target, call) for call in calls)


def _results(futures: tuple[Future[_Outcome], ...]) -> tuple[_Outcome, ...]:
    return tuple(future.result(timeout=180) for future in futures)


def _send_all(target: _Target, calls: tuple[_Call, ...]) -> tuple[_Outcome, ...]:
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return _results(_submit(pool, target, calls))


def _done(futures: tuple[Future[_Outcome], ...]) -> int:
    return sum(future.done() for future in futures)


def _assert_verdict(outcome: _Outcome, model: str) -> None:
    assert outcome.response is not None, outcome.call
    if outcome.call.kind in _REFUSALS:
        assert_refused(outcome.response, _REFUSALS[outcome.call.kind])
        return
    assert_compacted_reply(outcome.response, model, outcome.call.text)


def _served(outcome: _Outcome, model: str) -> bool:
    if outcome.response is None or outcome.response.status_code >= 500:
        return False
    _assert_verdict(outcome, model)
    return True


def _texts(outcomes: tuple[_Outcome, ...], kinds: tuple[_Kind, ...]) -> frozenset[str]:
    return frozenset(outcome.call.text for outcome in outcomes if outcome.call.kind in kinds)


def _assert_upstream(
    received: tuple[Upstreamed, ...], owner_texts: frozenset[str], refused_texts: frozenset[str], created: str
) -> None:
    assert {text: compact_hits(received, text) for text in owner_texts} == {
        text: (upstream_compact_body(text, {"previous_response_id": minted_id(created)}),) for text in owner_texts
    }
    assert {text: compact_hits(received, text) for text in refused_texts} == dict.fromkeys(refused_texts, ())


def _held_connections(pid: int, wire: Wire) -> int:
    port: Final = urlsplit(wire.url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _settled(target: _Target) -> bool:
    text: Final = marker("settle")
    body: Final = compact_body(target.actors.model, text, {})
    try:
        return compact_post(target.gateway, target.actors.owner, "/v1", body, text).status_code == 200
    except httpx.TransportError:
        return False


def test_upstream_outage_mid_burst_keeps_refusals_and_lands_each_owner_compact_once(gateway: Gateway) -> None:
    upstream: Final = CompactUpstream()
    created: Final = marker("mint")
    port: Final = allocate_port()
    with gateway.scenario() as scenario:
        with wire_server(upstream.respond, port=port) as first:
            actors: Final = tenants(scenario, first)
            target: Final = _Target(gateway, actors, mint(gateway, actors.owner, actors.model, created), vendor_id())
            upstreamed(first)
            before: Final = _send_all(target, _calls("before", 16))
            before_received: Final = upstreamed(first)
        during: Final = _send_all(target, _calls("during", 16))
        alive: Final = gateway.request("GET", "/health/liveliness")
        with wire_server(upstream.respond, port=port) as second:
            eventually(lambda: _settled(target), lambda settled: settled, seconds=60)
            after: Final = _send_all(target, _calls("after", 16))
            after_received: Final = upstreamed(second)
    assert alive.status_code == 200, alive.text
    for outcome in (*before, *after):
        _assert_verdict(outcome, actors.model)
    for outcome in during:
        assert outcome.response is not None, outcome.call
        if outcome.call.kind in _REFUSALS:
            assert_refused(outcome.response, _REFUSALS[outcome.call.kind])
        else:
            assert outcome.response.status_code >= 500, outcome.response.text
    _assert_upstream(before_received, _texts(before, _OWNER), _texts(before, _REFUSING), created)
    _assert_upstream(after_received, _texts(after, _OWNER), _texts((*during, *after), _REFUSING), created)
    assert {text: compact_hits(after_received, text) for text in _texts(during, _OWNER)} == dict.fromkeys(
        _texts(during, _OWNER), ()
    )
    assert_compact_rows(actors.owner, _texts((*before, *after), _OWNER))
    refused: Final = _texts((*before, *during, *after), _REFUSING)
    assert successful_markers(actors.other_user, refused) == frozenset()
    assert successful_markers(actors.other_team, refused) == frozenset()
    assert successful_markers(actors.owner, refused | _texts(during, _OWNER)) == frozenset()


def test_held_owner_compacts_do_not_hold_up_refusals(gateway: Gateway) -> None:
    release: Final = threading.Event()
    upstream: Final = CompactUpstream(MappingProxyType({"held": release}))
    created: Final = marker("mint")
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        target: Final = _Target(gateway, actors, mint(gateway, actors.owner, actors.model, created), vendor_id())
        upstreamed(wire)
        held: Final = _calls("held", 8, _OWNER)
        with ThreadPoolExecutor(max_workers=len(held)) as pool:
            try:
                pending: Final = _submit(pool, target, held)
                eventually(lambda: wire.received.qsize(), lambda size: size >= len(held), seconds=90)
                refused: Final = _send_all(target, _calls("refusal", 12, _REFUSING))
                done_while_held: Final = _done(pending)
            finally:
                release.set()
            served: Final = _results(pending)
        received: Final = upstreamed(wire)
    assert done_while_held == 0
    for outcome in (*refused, *served):
        _assert_verdict(outcome, actors.model)
    _assert_upstream(received, _texts(served, _OWNER), _texts(refused, _REFUSING), created)
    assert_compact_rows(actors.owner, _texts(served, _OWNER))


@pytest.mark.timeout(4 * graceful_stop_seconds() + 300)
def test_worker_kill_and_proxy_restart_keep_compact_ownership(gateway: Gateway, tmp_path: Path) -> None:
    kill_gate: Final = threading.Event()
    stop_gate: Final = threading.Event()
    upstream: Final = CompactUpstream(MappingProxyType({"heldkill": kill_gate, "heldstop": stop_gate}))
    created: Final = marker("mint")
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        with owned_proxy_process(gateway, tmp_path, {}, workers=2) as first:
            workers: Final = eventually(
                first.worker_pids, lambda found: len(found) == 2, seconds=graceful_stop_seconds() + 60
            )
            target: Final = _Target(
                first.gateway, actors, mint(first.gateway, actors.owner, actors.model, created), vendor_id()
            )
            upstreamed(wire)
            held_kill: Final = _calls("heldkill", 12, _OWNER)
            with ThreadPoolExecutor(max_workers=len(held_kill)) as pool:
                try:
                    in_flight: Final = _submit(pool, target, held_kill)
                    eventually(lambda: wire.received.qsize(), lambda size: size >= len(held_kill), seconds=90)
                    held_by: Final = MappingProxyType({pid: _held_connections(pid, wire) for pid in workers})
                    victim: Final = max(workers, key=held_by.__getitem__)
                    with first.replacing_worker(victim) as kill:
                        kill()
                        kill_gate.set()
                        killed: Final = _results(in_flight)
                finally:
                    kill_gate.set()
            after_kill: Final = _send_all(target, _calls("after_kill", 12))
            kill_received: Final = upstreamed(wire)
            held_stop: Final = _calls("heldstop", 8, _OWNER)
            with ThreadPoolExecutor(max_workers=len(held_stop)) as pool:
                try:
                    stopping: Final = _submit(pool, target, held_stop)
                    eventually(lambda: wire.received.qsize(), lambda size: size >= len(held_stop), seconds=90)
                    first.begin_shutdown()
                    stop_gate.set()
                    stopped: Final = _results(stopping)
                finally:
                    stop_gate.set()
        stop_received: Final = upstreamed(wire)
        with owned_proxy_process(gateway, tmp_path, {}, workers=2) as second:
            rebooted: Final = _send_all(
                _Target(second.gateway, actors, target.owned, target.foreign), _calls("rebooted", 12)
            )
            reboot_received: Final = upstreamed(wire)
    assert sum(held_by.values()) == len(held_kill) and held_by[victim] > 0, held_by
    served_through_kill: Final = tuple(outcome for outcome in killed if _served(outcome, actors.model))
    assert len(served_through_kill) == len(held_kill) - held_by[victim], (held_by, killed)
    for outcome in (*after_kill, *stopped, *rebooted):
        _assert_verdict(outcome, actors.model)
    _assert_upstream(kill_received, _texts((*killed, *after_kill), _OWNER), _texts(after_kill, _REFUSING), created)
    _assert_upstream(stop_received, _texts(stopped, _OWNER), frozenset(), created)
    _assert_upstream(reboot_received, _texts(rebooted, _OWNER), _texts(rebooted, _REFUSING), created)
    assert_compact_rows(actors.owner, _texts((*served_through_kill, *after_kill, *rebooted), _OWNER))
