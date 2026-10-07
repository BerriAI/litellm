import asyncio
import re
import signal
import socket
import threading
import uuid
from collections.abc import Iterator
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
from integration._support.client import Gateway, eventually, gateway_from_environment, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy, owned_proxy_process
from integration._support.sagemaker_embedding import (
    COMPONENT,
    COMPONENT_HEADER,
    JSON_OBJECT,
    SERVED_MODEL,
    json_reply,
    openai_deployment,
    openai_payload,
    respond,
)
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(300)

_MODEL: Final = "sm-openai-chaos"
_ALIASES: Final = (
    "/v1/embeddings",
    "/embeddings",
    f"/openai/deployments/{_MODEL}/embeddings",
    f"/engines/{_MODEL}/embeddings",
)
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_ROWS_BY_CALL: Final = (
    "SELECT litellm_call_id, status FROM \"LiteLLM_SpendLogs\" WHERE litellm_call_id = ANY(string_to_array(%s, ','))"
)


@dataclass(frozen=True, slots=True)
class _Served:
    text: str
    status: int
    body: str
    call_id: str


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _config(directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [openai_deployment(_MODEL, "openai", model_info={"mode": "embedding"})]
    path: Final = directory / "sagemaker-embedding-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _runtime(peer_url: str) -> dict[str, str]:
    return {"AWS_ENDPOINT_URL_SAGEMAKER_RUNTIME": peer_url, "AWS_MAX_ATTEMPTS": "1"}


@pytest.fixture(scope="module")
def rig() -> Iterator[Gateway]:
    with gateway_from_environment() as gateway:
        yield gateway


@pytest.fixture(scope="module")
def peer_port() -> int:
    return _free_port()


@pytest.fixture(scope="module")
def proxy(rig: Gateway, peer_port: int, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("sagemaker-embedding-chaos")
    runtime: Final = _runtime(f"http://127.0.0.1:{peer_port}")
    with owned_proxy(rig, directory, runtime, config=_config(directory), workers=2) as owned:
        yield owned


def _texts(count: int) -> tuple[str, ...]:
    return tuple(f"integration embedding {uuid.uuid4().hex}" for _ in range(count))


async def _send(client: httpx.AsyncClient, key: str, index: int, text: str) -> _Served:
    path: Final = _ALIASES[index % len(_ALIASES)]
    body: Final[dict[str, JsonValue]] = {
        "input": text,
        "cache": {"no-cache": True},
        **({"model": _MODEL} if path.endswith("/v1/embeddings") or path == "/embeddings" else {}),
    }
    response: Final = await client.post(path, json=body, headers={"Authorization": f"Bearer {key}"})
    return _Served(text, response.status_code, response.text, response.headers.get("x-litellm-call-id", ""))


async def _burst(
    base_url: str, key: str, texts: tuple[str, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=90, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, index, text) for index, text in enumerate(texts)),
            return_exceptions=tolerate_transport_errors,
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _assert_embedded(served: _Served) -> None:
    assert served.status == 200, served.body
    payload: Final = JSON_OBJECT.validate_json(served.body)
    assert payload["data"] == [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 1.0]}], served.body


def _assert_peer_saw(received: tuple[Request, ...], texts: tuple[str, ...]) -> None:
    assert {request.headers.get(COMPONENT_HEADER) for request in received} == {COMPONENT}, received
    bodies: Final = [JSON_OBJECT.validate_json(request.body) for request in received]
    assert sorted(string_value(body["input"]) for body in bodies) == sorted(texts), bodies
    assert {body["model"] for body in bodies} == {SERVED_MODEL}, bodies


def _spend_status_by_call(served: tuple[_Served, ...]) -> dict[str, JsonValue]:
    wanted: Final = sorted(item.call_id for item in served)
    rows: Final = eventually(
        lambda: read_rows(_ROWS_BY_CALL, (",".join(wanted),)), lambda found: len(found) >= len(wanted), seconds=70
    )
    assert sorted(string_value(row["litellm_call_id"]) for row in rows) == wanted, rows
    return {string_value(row["litellm_call_id"]): row["status"] for row in rows}


def _unhealthy_count(gateway: Gateway) -> JsonValue:
    response: Final = gateway.request("GET", "/health", params={"model": _MODEL})
    return JSON_OBJECT.validate_json(response.content).get("unhealthy_count")


async def test_concurrent_burst_across_every_alias_reaches_the_component_and_is_logged_once_per_call(
    proxy: Gateway, peer_port: int
) -> None:
    texts: Final = _texts(24)
    with wire_server(respond, port=peer_port) as wire:
        served: Final = await _burst(str(proxy.client.base_url), proxy.key, texts)
        assert len(served) == 24
        for item in served:
            _assert_embedded(item)
        _assert_peer_saw(wire.drain(), texts)
    assert _spend_status_by_call(served) == {item.call_id: "success" for item in served}


async def test_peer_outage_fails_each_call_once_reports_unhealthy_and_recovers_on_restart(
    proxy: Gateway, peer_port: int
) -> None:
    before: Final = _texts(12)
    during: Final = _texts(4)
    after: Final = _texts(4)
    base_url: Final = str(proxy.client.base_url)
    with wire_server(respond, port=peer_port) as wire:
        served: Final = await _burst(base_url, proxy.key, before)
        assert len(served) == 12
        for item in served:
            _assert_embedded(item)
        _assert_peer_saw(wire.drain(), before)
    refused: Final = await _burst(base_url, proxy.key, during)
    assert len(refused) == 4
    for item in refused:
        assert item.status == 503, item.body
        assert "Could not connect to the endpoint URL" in item.body, item.body
    assert await asyncio.to_thread(eventually, lambda: _unhealthy_count(proxy), lambda count: count == 1, 60)
    with wire_server(respond, port=peer_port) as restarted:
        recovered: Final = await _burst(base_url, proxy.key, after)
        assert len(recovered) == 4
        for item in recovered:
            _assert_embedded(item)
        healthy: Final = [request for request in restarted.drain() if "test from litellm" not in request.body.decode()]
        _assert_peer_saw(tuple(healthy), after)
    assert _spend_status_by_call((*served, *refused, *recovered)) == {
        **{item.call_id: "success" for item in served},
        **{item.call_id: "failure" for item in refused},
        **{item.call_id: "success" for item in recovered},
    }


def _slow(request: Request) -> Reply:
    body: Final = json_reply(openai_payload(JSON_OBJECT.validate_json(request.body))).body
    return Reply(chunks=(body[:8], body[8:]), pause_between_chunks=0.5)


async def test_slow_peer_answers_every_caller_whole_without_duplicate_rows(proxy: Gateway, peer_port: int) -> None:
    texts: Final = _texts(10)
    with wire_server(_slow, port=peer_port) as wire:
        served: Final = await _burst(str(proxy.client.base_url), proxy.key, texts)
        assert len(served) == 10
        for item in served:
            _assert_embedded(item)
        _assert_peer_saw(wire.drain(), texts)
    assert _spend_status_by_call(served) == {item.call_id: "success" for item in served}


def _live_workers(log: Path) -> tuple[int, ...]:
    started: Final = (int(pid) for pid in _STARTED_WORKER.findall(log.read_text()))
    return tuple(pid for pid in started if psutil.pid_exists(pid))


def _open_peer_connections(pid: int, peer_url: str) -> int:
    port: Final = urlsplit(peer_url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@dataclass(frozen=True, slots=True)
class _HeldPeer:
    release: threading.Event
    held: SimpleQueue[str]

    def respond(self, request: Request) -> Reply:
        self.held.put(string_value(JSON_OBJECT.validate_json(request.body)["input"]))
        assert self.release.wait(timeout=120), "The burst was never released"
        return respond(request)


@pytest.mark.timeout(600)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_serving_the_component(rig: Gateway, tmp_path: Path) -> None:
    texts: Final = _texts(8)
    peer: Final = _HeldPeer(threading.Event(), SimpleQueue())
    with wire_server(peer.respond) as wire:
        with owned_proxy_process(rig, tmp_path, _runtime(wire.url), config=_config(tmp_path), workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(lambda: _live_workers(owned.log), lambda pids: len(pids) == 2, seconds=30)
            burst: Final = asyncio.create_task(
                _burst(str(candidate.client.base_url), candidate.key, texts, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, peer.held.qsize, lambda size: size == len(texts), 120)
            held_by: Final = MappingProxyType({pid: _open_peer_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == len(texts), held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            peer.release.set()
            served: Final = await burst
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_embedded(item)
            (follow_up,) = _texts(1)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, (follow_up,))
            _assert_embedded(answered)
            _assert_peer_saw(wire.drain(), (*texts, follow_up))
            assert _spend_status_by_call((*served, answered)) == {
                item.call_id: "success" for item in (*served, answered)
            }


@pytest.mark.timeout(600)
async def test_proxy_sigterm_mid_burst_finishes_in_flight_calls_and_a_relaunch_serves_again(
    rig: Gateway, tmp_path: Path
) -> None:
    texts: Final = _texts(8)
    peer: Final = _HeldPeer(threading.Event(), SimpleQueue())
    with wire_server(peer.respond) as wire:
        first: Final = tmp_path / "first"
        first.mkdir()
        with owned_proxy_process(rig, first, _runtime(wire.url), config=_config(first), workers=2) as owned:
            burst: Final = asyncio.create_task(
                _burst(str(owned.gateway.client.base_url), owned.gateway.key, texts, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, peer.held.qsize, lambda size: size == len(texts), 120)
            owned.process.send_signal(signal.SIGTERM)
            peer.release.set()
            served: Final = await burst
            exit_code: Final = await asyncio.to_thread(
                eventually, owned.process.poll, lambda code: code is not None, 120
            )
            assert exit_code is not None
        for item in served:
            _assert_embedded(item)
        _assert_peer_saw(wire.drain(), texts)
        assert len(served) == len(texts), (len(served), len(texts))
        second: Final = tmp_path / "second"
        second.mkdir()
        with owned_proxy_process(rig, second, _runtime(wire.url), config=_config(second), workers=2) as relaunched:
            (follow_up,) = _texts(1)
            (answered,) = await _burst(str(relaunched.gateway.client.base_url), relaunched.gateway.key, (follow_up,))
            _assert_embedded(answered)
            _assert_peer_saw(wire.drain(), (follow_up,))
            assert _spend_status_by_call((*served, answered)) == {
                item.call_id: "success" for item in (*served, answered)
            }
