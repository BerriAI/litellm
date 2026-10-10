from __future__ import annotations

import asyncio
import re
import signal
import threading
import uuid
from contextlib import ExitStack
from pathlib import Path
from queue import SimpleQueue
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._responses_bridge_prompt_cache_breakpoint import (
    _Call,
    _JSON_OBJECT,
    _MODEL,
    _assert_spend_for_result,
    _burst,
    _calls,
    _peer_marker_matches_response,
    _request_marker,
    _responses_reply,
    _send_call,
)

_CONFIG_MODEL: Final = "responses-bridge-cache-breakpoint-chaos"

_API_KEY: Final = "synthetic-responses-bridge-key"

_STARTED_WORKER: Final[re.Pattern[str]] = re.compile(r"Started server process \[(\d+)\]")

def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    base_config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    config: Final = {
        **base_config,
        "model_list": [
            {
                "model_name": _CONFIG_MODEL,
                "litellm_params": {
                    "model": _MODEL,
                    "api_base": wire.url + "/v1",
                    "api_key": _API_KEY,
                },
            },
        ],
    }
    path: Final = tmp_path / "responses-bridge-cache-breakpoint-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path

def _open_upstream_connections(pid: int, port: int) -> int:
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )

@pytest.mark.timeout(180)
async def test_worker_and_peer_outages_preserve_markers_and_recover(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    calls: Final = _calls(30)
    release: Final = threading.Event()
    early_release: Final = threading.Event()
    outage_release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()
    early_calls: Final = calls[:10]
    early_markers: Final = frozenset(call.marker for call in early_calls)

    def held(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return _responses_reply(request)
        marker: Final = _request_marker(request)
        held_markers.put(marker)
        gate: Final = early_release if marker in early_markers else release
        assert gate.wait(timeout=60), "The worker-kill burst was never released"
        return _responses_reply(request)

    with ExitStack() as peer_stack:
        wire: Final = peer_stack.enter_context(wire_server(held))
        config: Final = _chaos_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            try:
                candidate: Final = owned.gateway
                workers: Final[tuple[int, ...]] = eventually(
                    lambda: tuple(int(match.group(1)) for match in _STARTED_WORKER.finditer(owned.log.read_text())),
                    lambda pids: len(pids) == 2,
                    seconds=30,
                )
                async with httpx.AsyncClient(
                    base_url=str(candidate.client.base_url),
                    headers={"Authorization": f"Bearer {candidate.key}"},
                    timeout=20,
                    trust_env=False,
                    limits=httpx.Limits(max_connections=100),
                ) as client:
                    burst_tasks: Final = tuple(
                        asyncio.create_task(_send_call(client, _CONFIG_MODEL, call)) for call in calls
                    )
                    await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == len(calls), 60)
                    early_release.set()
                    early_served: Final = await asyncio.gather(*burst_tasks[: len(early_calls)])
                    early_successful: Final = tuple(item for item in early_served if item.status == 200)
                    for item in early_successful:
                        _assert_spend_for_result(item, _CONFIG_MODEL)
                    upstream_port_value: Final = urlsplit(wire.url).port
                    assert upstream_port_value is not None
                    upstream_port: Final = upstream_port_value
                    active_by_worker: Final = eventually(
                        lambda: {pid: _open_upstream_connections(pid, upstream_port) for pid in workers},
                        lambda counts: sum(counts.values()) == len(calls) - len(early_calls),
                        seconds=30,
                    )
                    victim_pid: Final = max(workers, key=active_by_worker.__getitem__)
                    survivor_pids: Final = tuple(pid for pid in workers if pid != victim_pid)
                    assert active_by_worker[victim_pid] > 0 and len(survivor_pids) == 1, active_by_worker
                    (survivor_pid,) = survivor_pids
                    victim: Final = psutil.Process(victim_pid)
                    victim.suspend()
                    victim.send_signal(signal.SIGKILL)
                    release.set()
                    remaining_served: Final = await asyncio.gather(*burst_tasks[len(early_calls) :])
                    served: Final = (*early_served, *remaining_served)
                successful: Final = tuple(item for item in served if item.status == 200)
                connection_errors: Final = tuple(item for item in served if item.status == 0)
                print(f"worker-kill burst: {len(successful)} HTTP 200, {len(connection_errors)} connection errors")
                assert len(successful) + len(connection_errors) == len(calls), {
                    "successes": len(successful),
                    "connection_errors": len(connection_errors),
                    "responses": served,
                }
                assert successful and connection_errors, {
                    "successes": len(successful),
                    "connection_errors": len(connection_errors),
                }
                follow_ups: Final = (
                    _Call("chat", False, uuid.uuid4().hex),
                    _Call("responses", False, uuid.uuid4().hex),
                )
                recovered: Final = await _burst(
                    str(candidate.client.base_url),
                    candidate.key,
                    _CONFIG_MODEL,
                    follow_ups,
                )
                assert all(item.status == 200 for item in recovered), recovered
                assert psutil.pid_exists(survivor_pid), survivor_pid
                received_after_worker_kill: Final = wire.drain()
                worker_marker_failures: Final = tuple(
                    item.call.marker
                    for item in (*successful, *recovered)
                    if not _peer_marker_matches_response(item, received_after_worker_kill)
                )
                for item in (*successful, *recovered):
                    assert item.response_id is not None, item
                    _assert_spend_for_result(item, _CONFIG_MODEL)

                peer_stack.close()
                outage_seen: Final[SimpleQueue[str]] = SimpleQueue()

                def outage(request: Request) -> Reply:
                    if request.method == "GET" and request.target == "/v1/models":
                        return _responses_reply(request)
                    outage_seen.put(_request_marker(request))
                    assert outage_release.wait(timeout=60), "The peer-outage burst was never stopped"
                    return Reply(
                        status=503,
                        body=b'{"error":{"message":"synthetic peer outage","type":"server_error"}}',
                    )

                peer_stack.enter_context(wire_server(outage, port=upstream_port))
                outage_calls: Final = _calls(12)
                outage_burst: Final = asyncio.create_task(
                    _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, outage_calls)
                )
                await asyncio.to_thread(eventually, outage_seen.qsize, lambda size: size == len(outage_calls), 30)
                outage_release.set()
                peer_stack.close()
                outage_served: Final = await outage_burst
                assert len(outage_served) == len(outage_calls), outage_served
                assert all(item.status >= 400 and "error" in item.text.lower() for item in outage_served), outage_served
                down_call: Final = _Call("chat", False, uuid.uuid4().hex)
                (down_response,) = await _burst(
                    str(candidate.client.base_url),
                    candidate.key,
                    _CONFIG_MODEL,
                    (down_call,),
                )
                assert down_response.status >= 400 and "error" in down_response.text.lower(), down_response

                restarted_wire: Final = peer_stack.enter_context(wire_server(_responses_reply, port=upstream_port))
                recovery_calls: Final = (
                    _Call("chat", False, uuid.uuid4().hex),
                    _Call("responses", False, uuid.uuid4().hex),
                )
                recovered_after_peer_restart: Final = await _burst(
                    str(candidate.client.base_url),
                    candidate.key,
                    _CONFIG_MODEL,
                    recovery_calls,
                )
                assert all(item.status == 200 for item in recovered_after_peer_restart), recovered_after_peer_restart
                restarted_requests: Final = restarted_wire.drain()
                recovery_marker_failures: Final = tuple(
                    item.call.marker
                    for item in recovered_after_peer_restart
                    if not _peer_marker_matches_response(item, restarted_requests)
                )
                for item in recovered_after_peer_restart:
                    assert item.response_id is not None, item
                    _assert_spend_for_result(item, _CONFIG_MODEL)
                assert not (*worker_marker_failures, *recovery_marker_failures), {
                    "worker_marker_failures": worker_marker_failures,
                    "recovery_marker_failures": recovery_marker_failures,
                }
            finally:
                release.set()
                outage_release.set()
