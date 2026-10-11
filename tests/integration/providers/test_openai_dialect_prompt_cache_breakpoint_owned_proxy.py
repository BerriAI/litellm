import asyncio
import re
import signal
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.responses_vendor import newest_marker
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import owned_config
from integration.providers._mantle_gpt_prompt_cache_support import (
    EXPLICIT,
    GPT,
    IMPLICIT,
    SYSTEM,
    SYSTEM_POINT,
    TOKEN,
    Outcome,
    assert_answered,
    assert_burst_landed,
    assert_wire,
    body_of,
    breakpoint_count,
    expected_wire,
    fresh_marker,
    mantle_deployment,
    mantle_peer,
    observe,
    openai_shaped_peer,
    parse_outcome,
    plan_burst,
    prompt_text,
    request_body,
    row_key,
    send,
    settled,
    spend_rows,
    success_row,
)
from pydantic import JsonValue

_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_FOUNDRY_BASE: Final = "http://prompt-cache-breakpoint-audit.services.ai.azure.com"
_FOUNDRY_HOST: Final = urlsplit(_FOUNDRY_BASE).netloc
_FOUNDRY_MODEL: Final = "gpt-6-astra"
_FOUNDRY_KEY: Final = "synthetic-foundry-key"
_FOUNDRY_CHAT_PATH: Final = "/models/chat/completions"
_CELL_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)
_RESTART_TIMEOUT: Final = int(4 * graceful_stop_seconds() + 240)
_HELD_BURST: Final = 20
_SPREAD_BURST: Final = 12
_SPREAD_ATTEMPTS: Final = 20
_MANTLE_NAME: Final = "mantle-gpt-cache-owned"


def _peer(release: threading.Event, held: SimpleQueue[str]) -> Callable[[Request], Reply]:
    mantle: Final = mantle_peer()
    foundry: Final = openai_shaped_peer()

    def respond(request: Request) -> Reply:
        if urlsplit(request.target).netloc == _FOUNDRY_HOST:
            return foundry(request)
        marker: Final = newest_marker(request.body.decode())
        assert marker is not None, request.body
        held.put(marker)
        assert release.wait(timeout=120), "The held burst was never released"
        return mantle(request)

    return respond


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    wire: Wire
    owned: OwnedProxy
    release: threading.Event
    held: SimpleQueue[str]


def _overrides(wire: Wire) -> MappingProxyType[str, str]:
    return MappingProxyType({"HTTP_PROXY": wire.url, "NO_PROXY": "127.0.0.1,localhost", "AIOHTTP_TRUST_ENV": "True"})


def _started_workers(log: Path) -> tuple[int, ...]:
    return tuple(int(match.group(1)) for match in _STARTED_WORKER.finditer(log.read_text()))


def _live_workers(log: Path) -> tuple[int, ...]:
    return tuple(pid for pid in _started_workers(log) if psutil.pid_exists(pid))


def _open_peer_connections(pid: int, peer_url: str) -> int:
    port: Final = urlsplit(peer_url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _drain(queue: SimpleQueue[str]) -> None:
    while not queue.empty():
        queue.get_nowait()


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("openai-dialect-prompt-cache-breakpoint")
    release: Final = threading.Event()
    release.set()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    with gateway_from_environment() as environment, wire_server(_peer(release, held)) as wire:
        with owned_proxy_process(environment, directory, _overrides(wire), workers=2) as owned:
            eventually(lambda: len(_started_workers(owned.log)), lambda count: count == 2, seconds=60)
            wire.drain()
            yield _Rig(owned.gateway, wire, owned, release, held)


def _foundry_deployment(rig: _Rig, scenario: Scenario, *, bare: bool) -> str:
    name: Final = (
        scenario.model(
            model=_FOUNDRY_MODEL,
            custom_llm_provider="azure_ai",
            api_base=_FOUNDRY_BASE,
            api_key=_FOUNDRY_KEY,
            cache_control_injection_points=SYSTEM_POINT,
        )
        if bare
        else scenario.model(
            model=f"azure_ai/{_FOUNDRY_MODEL}",
            api_base=_FOUNDRY_BASE,
            api_key=_FOUNDRY_KEY,
            cache_control_injection_points=SYSTEM_POINT,
        )
    )
    settled(rig.gateway, name, rig.wire)
    return name


def _assert_foundry_chat_wire(received: Request, prompt: str, *, options: JsonValue | None) -> None:
    body: Final = body_of(received)
    assert urlsplit(received.target).netloc == _FOUNDRY_HOST, received.target
    assert urlsplit(received.target).path == _FOUNDRY_CHAT_PATH, received.target
    assert "prompt_cache_breakpoint" not in received.body.decode(), received.body
    assert body["model"] == _FOUNDRY_MODEL, received.body
    assert body["messages"] == [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], (
        received.body
    )
    assert body.get("prompt_cache_options") == options, received.body


@pytest.mark.timeout(_CELL_TIMEOUT)
@pytest.mark.parametrize("bare", [False, True], ids=["prefixed", "bare-provider-field"])
def test_b9_b12_foundry_gpt6_responses_reach_the_openai_dialect_on_the_foundry_host(rig: _Rig, bare: bool) -> None:
    marker: Final = fresh_marker()
    with rig.gateway.scenario() as scenario:
        name: Final = _foundry_deployment(rig, scenario, bare=bare)
        outcome, received = observe(
            rig.gateway, rig.wire, "responses", request_body("responses", name, prompt_text(marker))
        )
        assert_answered(outcome, marker)
        assert urlsplit(received.target).netloc == _FOUNDRY_HOST, received.target
        assert_wire(
            received,
            expected_wire(_FOUNDRY_MODEL, prompt_text(marker), endpoint="responses", marked=True),
            streaming=False,
        )
        success_row(name, marker)


@pytest.mark.timeout(_CELL_TIMEOUT)
def test_b10_foundry_gpt6_messages_bridge_to_chat_without_a_breakpoint(rig: _Rig) -> None:
    marker: Final = fresh_marker()
    with rig.gateway.scenario() as scenario:
        name: Final = _foundry_deployment(rig, scenario, bare=False)
        outcome, received = observe(
            rig.gateway, rig.wire, "messages", request_body("messages", name, prompt_text(marker))
        )
        assert_answered(outcome, marker)
        _assert_foundry_chat_wire(received, prompt_text(marker), options=IMPLICIT)
        success_row(name, marker)


@pytest.mark.timeout(_CELL_TIMEOUT)
def test_b11_foundry_gpt6_chat_stays_on_the_plain_chat_wire(rig: _Rig) -> None:
    marker: Final = fresh_marker()
    with rig.gateway.scenario() as scenario:
        name: Final = _foundry_deployment(rig, scenario, bare=False)
        outcome, received = observe(rig.gateway, rig.wire, "chat", request_body("chat", name, prompt_text(marker)))
        assert_answered(outcome, marker)
        _assert_foundry_chat_wire(received, prompt_text(marker), options=IMPLICIT)
        success_row(name, marker)


def _spread_burst(rig: _Rig, name: str, workers: tuple[int, ...]) -> dict[int, int]:
    plan: Final = plan_burst(_SPREAD_BURST)
    _drain(rig.held)
    rig.wire.drain()
    rig.release.clear()
    with ThreadPoolExecutor(max_workers=_SPREAD_BURST) as pool:
        futures: Final = tuple(
            pool.submit(send, rig.gateway, endpoint, request_body(endpoint, name, prompt_text(marker), stream=stream))
            for endpoint, stream, marker in plan
        )
        eventually(rig.held.qsize, lambda size: size == _SPREAD_BURST, seconds=60)
        held_by: Final = {pid: _open_peer_connections(pid, rig.wire.url) for pid in workers}
        rig.release.set()
        outcomes: Final = tuple(future.result() for future in futures)
    assert sum(held_by.values()) == _SPREAD_BURST, held_by
    assert_burst_landed(
        rig.wire, name, tuple((marker, outcome) for (_, _, marker), outcome in zip(plan, outcomes)), marked=True
    )
    return held_by


@pytest.mark.timeout(_CELL_TIMEOUT)
def test_e6_every_worker_of_a_two_worker_proxy_marks_the_mantle_request(rig: _Rig) -> None:
    with rig.gateway.scenario() as scenario:
        name: Final = mantle_deployment(rig.gateway, scenario, rig.wire)
        workers: Final = _live_workers(rig.owned.log)
        assert len(workers) == 2, workers
        for _ in range(_SPREAD_ATTEMPTS):
            if all(count > 0 for count in _spread_burst(rig, name, workers).values()):
                break
        else:
            raise AssertionError("One worker never took a marked request")


def _mantle_config(wire: Wire, directory: Path, **litellm_params: JsonValue) -> Path:
    return owned_config(
        directory,
        [
            {
                "model_name": _MANTLE_NAME,
                "litellm_params": {
                    "model": GPT,
                    "api_base": wire.url,
                    "api_key": TOKEN,
                    "aws_region_name": "us-east-1",
                    "cache_control_injection_points": SYSTEM_POINT,
                    **litellm_params,
                },
            }
        ],
    )


async def _one(client: httpx.AsyncClient, key: str, body: dict[str, JsonValue]) -> Outcome:
    response: Final = await client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})
    lines: Final = tuple(line for line in response.text.splitlines() if line)
    return parse_outcome("chat", stream=False, status=response.status_code, headers=response.headers, lines=lines)


async def _held_burst(url: str, key: str, markers: tuple[str, ...]) -> tuple[tuple[str, Outcome], ...]:
    async with httpx.AsyncClient(base_url=url, timeout=180, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_one(client, key, request_body("chat", _MANTLE_NAME, prompt_text(marker))) for marker in markers),
            return_exceptions=True,
        )
    return tuple((marker, result) for marker, result in zip(markers, results) if isinstance(result, Outcome))


def _assert_served_and_landed(wire: Wire, served: tuple[tuple[str, Outcome], ...], *, expected_received: int) -> None:
    for marker, outcome in served:
        assert_answered(outcome, marker)
    received: Final = wire.drain()
    assert len(received) == expected_received, (len(received), expected_received)
    for request in received:
        assert breakpoint_count(body_of(request)) == 1, request.body
    identities: Final = frozenset(outcome.response_id for _, outcome in served)
    assert len(identities) == len(served), identities
    rows: Final = spend_rows(_MANTLE_NAME, frozenset(marker for marker, _ in served), expected=len(served), seconds=120)
    assert sorted(row_key(str(row["request_id"])) for row in rows) == sorted(identities), rows


@pytest.mark.timeout(_CELL_TIMEOUT)
async def test_f2_a_worker_killed_mid_burst_leaves_the_survivor_marking_requests(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    markers: Final = tuple(fresh_marker() for _ in range(_HELD_BURST))
    with wire_server(_peer(release, held)) as wire:
        config: Final = _mantle_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            url: Final = str(candidate.client.base_url).rstrip("/")
            workers: Final = eventually(lambda: _live_workers(owned.log), lambda pids: len(pids) == 2, seconds=60)
            burst: Final = asyncio.create_task(_held_burst(url, candidate.key, markers))
            await asyncio.to_thread(eventually, held.qsize, lambda size: size == _HELD_BURST, 60)
            held_by: Final = MappingProxyType({pid: _open_peer_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == _HELD_BURST, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            follow_up: Final = fresh_marker()
            (answered,) = await _held_burst(url, candidate.key, (follow_up,))
            _assert_served_and_landed(wire, (*served, answered), expected_received=_HELD_BURST + 1)
            eventually(lambda: len(_started_workers(owned.log)), lambda count: count == 3, seconds=90)


def _model_id(name: str) -> str:
    (row,) = read_rows('SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_name=%s', (name,))
    return str(row["model_id"])


@pytest.mark.timeout(_RESTART_TIMEOUT)
async def test_f3_a_graceful_restart_mid_burst_drains_the_held_requests_and_keeps_the_stored_explicit_mode(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    markers: Final = tuple(fresh_marker() for _ in range(_HELD_BURST))
    stored: Final = f"mantle-gpt-explicit-stored-{fresh_marker()}"
    with wire_server(_peer(release, held)) as wire:
        config: Final = _mantle_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            url: Final = str(candidate.client.base_url).rstrip("/")
            eventually(lambda: len(_live_workers(owned.log)), lambda count: count == 2, seconds=60)
            candidate.post(
                "/model/new",
                {
                    "model_name": stored,
                    "litellm_params": {
                        "model": GPT,
                        "api_base": wire.url,
                        "api_key": TOKEN,
                        "aws_region_name": "us-east-1",
                        "cache_control_injection_points": SYSTEM_POINT,
                        "prompt_cache_options": EXPLICIT,
                    },
                },
            )
            release.set()
            settled(candidate, stored, wire)
            before, before_received = observe(
                candidate, wire, "responses", request_body("responses", stored, prompt_text(markers[0]))
            )
            assert_answered(before, markers[0])
            assert_wire(
                before_received,
                expected_wire(GPT, prompt_text(markers[0]), endpoint="responses", marked=True, options=EXPLICIT),
                streaming=False,
            )
            success_row(stored, markers[0])
            _drain(held)
            release.clear()
            burst: Final = asyncio.create_task(_held_burst(url, candidate.key, markers))
            await asyncio.to_thread(eventually, held.qsize, lambda size: size == _HELD_BURST, 60)
            owned.process.terminate()
            release.set()
            served: Final = await burst
        assert len(served) == _HELD_BURST, len(served)
        _assert_served_and_landed(wire, served, expected_received=_HELD_BURST)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as restarted:
            settled(restarted.gateway, stored, wire)
            after, after_received = observe(
                restarted.gateway, wire, "responses", request_body("responses", stored, prompt_text(markers[1]))
            )
            assert_answered(after, markers[1])
            assert_wire(
                after_received,
                expected_wire(GPT, prompt_text(markers[1]), endpoint="responses", marked=True, options=EXPLICIT),
                streaming=False,
            )
            success_row(stored, markers[1])
            restarted.gateway.post("/model/delete", {"id": _model_id(stored)})
