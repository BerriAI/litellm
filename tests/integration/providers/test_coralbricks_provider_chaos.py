from __future__ import annotations

import asyncio
import re
import signal
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
from integration._support.client import JSON_OBJECT, Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._coralbricks import (
    API_KEY,
    MARKER,
    MODEL,
    NO_CACHE,
    PROVIDER,
    SPEND_COLUMNS,
    deployment,
    marker_of,
    marker_provider,
    ready,
)
from pydantic import JsonValue

Endpoint = Literal["chat", "messages", "responses"]
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")


@dataclass(frozen=True, slots=True)
class Call:
    endpoint: Endpoint
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class Served:
    call: Call
    status: int
    text: str
    call_id: str


def path_of(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def body_of(model: str, call: Call) -> dict[str, JsonValue]:
    question: Final = f"Say hello marker-{call.marker}"
    common: Final[dict[str, JsonValue]] = {"model": model, "stream": call.stream, "cache": NO_CACHE}
    match call.endpoint:
        case "chat":
            return {**common, "messages": [{"role": "user", "content": question}]}
        case "messages":
            return {**common, "max_tokens": 64, "messages": [{"role": "user", "content": question}]}
        case "responses":
            return {**common, "input": question}


def calls_of(count: int, endpoints: tuple[Endpoint, ...], stream: Callable[[int], bool]) -> tuple[Call, ...]:
    return tuple(
        Call(endpoint=endpoints[index % len(endpoints)], stream=stream(index), marker=uuid.uuid4().hex)
        for index in range(count)
    )


async def send(client: httpx.AsyncClient, key: str, model: str, call: Call) -> Served:
    async with client.stream(
        "POST",
        path_of(call.endpoint),
        json=body_of(model, call),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return Served(
        call=call, status=response.status_code, text=raw.decode(), call_id=response.headers.get("x-litellm-call-id", "")
    )


async def burst(
    base_url: str, key: str, model: str, calls: tuple[Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=90, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(send(client, key, model, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, Served))


def assert_answered_with_its_own_marker(served: Served) -> None:
    assert served.status == 200, served.text
    assert set(MARKER.findall(served.text)) == {served.call.marker}, served.text


def expected_row_id(call: Call) -> str:
    match call.endpoint:
        case "chat":
            return f"req_{call.marker}"
        case "messages":
            return f"msg_{call.marker}"
        case "responses":
            return f"resp_{call.marker}"


def landed_once(model_group: str, served: tuple[Served, ...]) -> None:
    expected: Final = tuple(expected_row_id(item.call) for item in served)
    rows: Final = eventually(
        lambda: read_rows(
            f"{SPEND_COLUMNS} WHERE model_group=%s AND status='success' AND request_id NOT LIKE 'probe_%%'",
            (model_group,),
        ),
        lambda found: len(found) >= len(expected),
        seconds=90,
    )
    identities: Final = [string_value(row["request_id"]) for row in rows]
    for identity in expected:
        matches: Final = [found for found in identities if same_response(found, identity)]
        assert len(matches) == 1, (identity, matches)
    assert len(rows) == len(expected), (len(rows), len(expected))
    assert {row["model"] for row in rows} == {f"{PROVIDER}/{MODEL}"}, rows
    assert {row["custom_llm_provider"] for row in rows} == {PROVIDER}, rows


def failed_once(model_group: str, served: tuple[Served, ...]) -> None:
    call_ids: Final = tuple(item.call_id for item in served)
    assert all(call_ids), [item.call_id for item in served]
    rows: Final = eventually(
        lambda: read_rows(f"{SPEND_COLUMNS} WHERE model_group=%s AND status='failure'", (model_group,)),
        lambda found: {string_value(row["request_id"]) for row in found} >= set(call_ids),
        seconds=90,
    )
    identities: Final = [string_value(row["request_id"]) for row in rows]
    assert sorted(identities) == sorted(call_ids), (sorted(identities), sorted(call_ids))


def assert_no_bleed(received: tuple[Request, ...], markers: frozenset[str]) -> None:
    forwarded: Final = [marker_of(request) for request in received if marker_of(request) is not None]
    assert sorted(marker for marker in forwarded if marker is not None) == sorted(markers), forwarded


@contextmanager
def released_on_exit(release: threading.Event) -> Iterator[threading.Event]:
    try:
        yield release
    finally:
        release.set()


def held_provider(release: threading.Event, held: SimpleQueue[str]) -> Callable[[Request], Reply]:
    respond: Final = marker_provider()

    def hold(request: Request) -> Reply:
        marker: Final = marker_of(request)
        if marker is None:
            return respond(request)
        held.put(marker)
        assert release.wait(timeout=90), "The burst was never released"
        return respond(request)

    return hold


def health_of(gateway: Gateway, model: str) -> tuple[int, int, int, str]:
    response: Final = gateway.request("GET", "/health", params={"model": model})
    health: Final = JSON_OBJECT.validate_json(response.content)
    healthy_count: Final = health["healthy_count"]
    unhealthy_count: Final = health["unhealthy_count"]
    assert isinstance(healthy_count, int) and isinstance(unhealthy_count, int), response.text
    return response.status_code, healthy_count, unhealthy_count, response.text


def assert_unhealthy(gateway: Gateway, model: str) -> None:
    observed: Final = eventually(
        lambda: health_of(gateway, model), lambda found: found[:3] == (503, 0, 1), seconds=30, return_last_on_timeout=True
    )
    assert observed[:3] == (503, 0, 1), observed[3]


def assert_healthy(gateway: Gateway, model: str) -> None:
    observed: Final = eventually(
        lambda: health_of(gateway, model), lambda found: found[:3] == (200, 1, 0), seconds=30, return_last_on_timeout=True
    )
    assert observed[:3] == (200, 1, 0), observed[3]


@pytest.mark.timeout(400)
async def test_provider_outage_mid_burst_fails_fast_reports_unhealthy_and_recovers(gateway: Gateway) -> None:
    before: Final = calls_of(15, ("chat", "messages", "responses"), lambda index: index % 2 == 0)
    during: Final = calls_of(15, ("chat", "messages", "responses"), lambda index: index % 2 == 1)
    after: Final = calls_of(15, ("chat", "messages", "responses"), lambda index: index % 2 == 0)
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    with gateway.scenario() as scenario:
        with wire_server(held_provider(release, held)) as wire, released_on_exit(release):
            port: Final = int(urlsplit(wire.url).port or 0)
            alias: Final = deployment(scenario, wire)
            wire.drain()
            first: Final = asyncio.create_task(burst(str(gateway.client.base_url), gateway.key, alias, before))
            await asyncio.to_thread(eventually, held.qsize, lambda size: size == 15, 60)
            release.set()
            served_before: Final = await first
            assert len(served_before) == 15
            for item in served_before:
                assert_answered_with_its_own_marker(item)
            assert_no_bleed(wire.drain(), frozenset(call.marker for call in before))
        served_during: Final = await burst(str(gateway.client.base_url), gateway.key, alias, during)
        assert len(served_during) == 15
        for item in served_during:
            assert item.status == 500, item.text
            assert "CoralbricksException" in item.text, item.text
            assert "Connection error" in item.text or "Cannot connect to host" in item.text, item.text
            assert "marker-" not in item.text, item.text
        assert_unhealthy(gateway, alias)
        with wire_server(marker_provider(), port=port) as restarted:
            served_after: Final = await burst(str(gateway.client.base_url), gateway.key, alias, after)
            assert len(served_after) == 15
            for item in served_after:
                assert_answered_with_its_own_marker(item)
            assert_no_bleed(restarted.drain(), frozenset(call.marker for call in after))
            assert_healthy(gateway, alias)
        landed_once(alias, (*served_before, *served_after))
        failed_once(alias, served_during)


@pytest.mark.timeout(300)
async def test_slow_provider_streams_are_forwarded_once_under_concurrency(gateway: Gateway) -> None:
    calls: Final = calls_of(20, ("chat", "messages", "responses"), lambda _: True)
    with wire_server(marker_provider(stream_pause=0.3)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        wire.drain()
        served: Final = await burst(str(gateway.client.base_url), gateway.key, alias, calls)
        assert len(served) == 20
        for item in served:
            assert_answered_with_its_own_marker(item)
            if item.call.endpoint == "chat":
                assert item.text.rstrip().endswith("data: [DONE]"), item.text
        assert_no_bleed(wire.drain(), frozenset(call.marker for call in calls))
        landed_once(alias, served)


def chaos_model() -> str:
    return f"coralbricks-chaos-{uuid.uuid4().hex[:12]}"


def chaos_config(wire: Wire, directory: Path, model_name: str) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = {
        **base,
        "model_list": [
            {
                "model_name": model_name,
                "litellm_params": {"model": f"{PROVIDER}/{MODEL}", "api_base": f"{wire.url}/v1", "api_key": API_KEY},
            }
        ],
    }
    path: Final = directory / "coralbricks-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def worker_pids(log: Path) -> tuple[int, ...]:
    return eventually(
        lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(log.read_text())),
        lambda pids: len(pids) == 2,
        seconds=60,
    )


@pytest.mark.timeout(2 * graceful_stop_seconds() + 240)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_serving_coralbricks(gateway: Gateway, tmp_path: Path) -> None:
    calls: Final = calls_of(20, ("chat", "messages", "responses"), lambda _: False)
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    alias: Final = chaos_model()
    with wire_server(held_provider(release, held)) as wire, released_on_exit(release):
        config: Final = chaos_config(wire, tmp_path, alias)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = worker_pids(owned.log)
            ready(candidate, alias, seconds=120)
            wire.drain()
            task: Final = asyncio.create_task(
                burst(str(candidate.client.base_url), candidate.key, alias, calls, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, held.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            served: Final = await task
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                assert_answered_with_its_own_marker(item)
            follow_up: Final = Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex)
            (answered,) = await burst(str(candidate.client.base_url), candidate.key, alias, (follow_up,))
            assert_answered_with_its_own_marker(answered)
            assert_no_bleed(wire.drain(), frozenset(call.marker for call in (*calls, follow_up)))
            landed_once(alias, (*served, answered))
            eventually(
                lambda: owned.log.read_text().count("Application startup complete."),
                lambda started: started == 3,
                seconds=graceful_stop_seconds(),
            )


@pytest.mark.timeout(480)
async def test_proxy_sigterm_mid_burst_then_reboot_serves_coralbricks_again(gateway: Gateway, tmp_path: Path) -> None:
    calls: Final = calls_of(20, ("chat", "messages", "responses"), lambda index: index % 2 == 0)
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    alias: Final = chaos_model()
    with wire_server(held_provider(release, held)) as wire, released_on_exit(release):
        config: Final = chaos_config(wire, tmp_path, alias)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            worker_pids(owned.log)
            ready(owned.gateway, alias, seconds=120)
            wire.drain()
            task: Final = asyncio.create_task(
                burst(
                    str(owned.gateway.client.base_url),
                    owned.gateway.key,
                    alias,
                    calls,
                    tolerate_transport_errors=True,
                )
            )
            await asyncio.to_thread(eventually, held.qsize, lambda size: size == 20, 60)
            owned.process.send_signal(signal.SIGTERM)
            release.set()
            served: Final = await task
            answered: Final = tuple(item for item in served if item.status == 200)
            for item in answered:
                assert_answered_with_its_own_marker(item)
            await asyncio.to_thread(eventually, owned.process.poll, lambda code: code is not None, 90)
            assert_no_bleed(wire.drain(), frozenset(call.marker for call in calls))
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as rebooted:
            worker_pids(rebooted.log)
            ready(rebooted.gateway, alias, seconds=120)
            wire.drain()
            follow_up: Final = Call(endpoint="responses", stream=True, marker=uuid.uuid4().hex)
            (recovered,) = await burst(str(rebooted.gateway.client.base_url), rebooted.gateway.key, alias, (follow_up,))
            assert_answered_with_its_own_marker(recovered)
            assert_no_bleed(wire.drain(), frozenset({follow_up.marker}))
            rows: Final = eventually(
                lambda: read_rows(
                    f"{SPEND_COLUMNS} WHERE model_group=%s AND status='success' AND request_id NOT LIKE 'probe_%%'",
                    (alias,),
                ),
                lambda found: any(same_response(string_value(row["request_id"]), expected_row_id(follow_up)) for row in found),
                seconds=90,
            )
            identities: Final = [string_value(row["request_id"]) for row in rows]
            assert len(identities) == len(set(identities)), identities
            allowed: Final = (*(expected_row_id(call) for call in calls), expected_row_id(follow_up))
            for identity in identities:
                assert any(same_response(identity, candidate) for candidate in allowed), identity
            billed: Final = {
                item.call.marker: [found for found in identities if same_response(found, expected_row_id(item.call))]
                for item in answered
            }
            assert all(len(found) == 1 for found in billed.values()), billed
