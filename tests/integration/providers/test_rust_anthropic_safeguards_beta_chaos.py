from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpcore
import httpx
import psutil
import pytest
from integration._support.client import Gateway, eventually
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import wire_server
from integration.providers._anthropic_safeguards_beta_support import (
    ANTHROPIC_DEPLOYMENT,
    DANGEROUS_TOOL_USE,
    SAFEGUARDS,
    deliveries,
    holding_peer,
    messages_body,
    new_marker,
    owned_config,
    rust_routed,
    safeguarded,
    started_workers,
)
from pydantic import TypeAdapter

_CELL_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)
_RESTART_TIMEOUT: Final = int(4 * graceful_stop_seconds() + 240)
_BURST: Final = 12
_SECOND_WAVE: Final = 6
_CLIENT_ADDRESS: Final = TypeAdapter(tuple[str, int])


def _clients(stack: ExitStack, base_url: str, count: int) -> tuple[httpx.Client, ...]:
    return tuple(
        stack.enter_context(httpx.Client(base_url=base_url, timeout=120, trust_env=False)) for _ in range(count)
    )


def _post(client: httpx.Client, key: str, marker: str) -> int:
    body: Final = safeguarded(messages_body(ANTHROPIC_DEPLOYMENT, marker), list(SAFEGUARDS))
    response: Final = client.post("/v1/messages", json=body, headers={"Authorization": f"Bearer {key}"})
    assert response.status_code != 200 or rust_routed(response.headers), dict(response.headers)
    return response.status_code


def _post_or_dropped(client: httpx.Client, key: str, marker: str) -> int | None:
    try:
        return _post(client, key, marker)
    except httpx.TransportError:
        return None


def _local_port(client: httpx.Client) -> int:
    with client.stream("GET", "/health/liveliness") as response:
        stream: Final[object] = response.extensions["network_stream"]  # pyright: ignore[reportAny]  # httpx types extensions as Any
        assert isinstance(stream, httpcore.NetworkStream), stream
        port: Final = _CLIENT_ADDRESS.validate_python(stream.get_extra_info("client_addr"))[1]
        response.read()
    assert response.status_code == 200, response.text
    return port


def _accepted_client_ports(pid: int, proxy_port: int) -> frozenset[int]:
    return frozenset(
        connection.raddr.port
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.raddr and connection.laddr.port == proxy_port
    )


@pytest.mark.timeout(_CELL_TIMEOUT)
def test_worker_sigkill_mid_safeguards_burst_leaves_the_replacement_adding_the_beta(
    gateway: Gateway, tmp_path: Path
) -> None:
    held: Final[SimpleQueue[str]] = SimpleQueue()
    release: Final = threading.Event()
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(holding_peer(held, release)))
        config: Final = owned_config(tmp_path, gateway, wire.url)
        owned: Final = stack.enter_context(
            owned_proxy_process(gateway, tmp_path, {"LITELLM_RUST": "true"}, config=config, workers=2)
        )
        proxy_url: Final = owned.gateway.client.base_url
        workers: Final = eventually(lambda: started_workers(owned.log), lambda pids: len(pids) == 2, seconds=30)
        clients: Final = _clients(stack, str(proxy_url), _BURST)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        stack.callback(release.set)
        ports: Final = tuple(_local_port(client) for client in clients)
        markers: Final = tuple(new_marker() for _ in clients)
        futures: Final = tuple(
            pool.submit(_post_or_dropped, client, gateway.key, marker)
            for client, marker in zip(clients, markers, strict=True)
        )
        eventually(held.qsize, lambda size: size == len(clients), seconds=30)
        shares: Final = {pid: _accepted_client_ports(pid, proxy_url.port or 0) & frozenset(ports) for pid in workers}
        assert sum(map(len, shares.values())) == len(clients), shares
        victim: Final = min((pid for pid in workers if shares[pid]), key=lambda pid: len(shares[pid]))
        with owned.replacing_worker(victim) as kill:
            kill()
            release.set()
            results: Final = tuple(future.result(timeout=60) for future in futures)
            for port, result in zip(ports, results, strict=True):
                assert result == (None if port in shares[victim] else 200), (
                    port,
                    result,
                    shares,
                )
        second_wave: Final = tuple(new_marker() for _ in range(_SECOND_WAVE))
        for marker in second_wave:
            assert _post(owned.gateway.client, gateway.key, marker) == 200
        delivered: Final = deliveries(wire)
        assert tuple(sent.beta for sent in delivered) == (DANGEROUS_TOOL_USE,) * (len(clients) + len(second_wave)), (
            delivered
        )
        seen: Final = tuple(sum(sent.marker == marker for sent in delivered) for marker in markers + second_wave)
        assert seen == (1,) * len(seen), seen
        assert owned.process.poll() is None


@pytest.mark.timeout(_RESTART_TIMEOUT)
def test_proxy_shutdown_mid_safeguards_burst_then_a_fresh_boot_keeps_adding_the_beta(
    gateway: Gateway, tmp_path: Path
) -> None:
    held: Final[SimpleQueue[str]] = SimpleQueue()
    release: Final = threading.Event()
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(holding_peer(held, release)))
        config: Final = owned_config(tmp_path, gateway, wire.url)
        markers: Final = tuple(new_marker() for _ in range(_BURST))
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(markers)))
        stack.callback(release.set)
        with owned_proxy_process(gateway, tmp_path, {"LITELLM_RUST": "true"}, config=config, workers=2) as first:
            clients: Final = _clients(stack, str(first.gateway.client.base_url), len(markers))
            futures: Final = tuple(
                pool.submit(_post_or_dropped, client, gateway.key, marker)
                for client, marker in zip(clients, markers, strict=True)
            )
            eventually(held.qsize, lambda size: size == len(markers), seconds=30)
            first.begin_shutdown()
            release.set()
            results: Final = tuple(future.result(timeout=120) for future in futures)
        during: Final = deliveries(wire)
        assert all(result in (None, 200) for result in results), results
        assert tuple(sent.beta for sent in during) == (DANGEROUS_TOOL_USE,) * len(markers), during
        assert sorted(sent.marker for sent in during) == sorted(markers), during
        with owned_proxy_process(gateway, tmp_path, {"LITELLM_RUST": "true"}, config=config, workers=2) as second:
            after: Final = tuple(new_marker() for _ in range(_SECOND_WAVE))
            statuses: Final = tuple(_post(second.gateway.client, gateway.key, marker) for marker in after)
        assert statuses == (200,) * len(after), statuses
        delivered: Final = deliveries(wire)
        assert tuple(sent.beta for sent in delivered) == (DANGEROUS_TOOL_USE,) * len(after), delivered
        assert sorted(sent.marker for sent in delivered) == sorted(after), delivered
