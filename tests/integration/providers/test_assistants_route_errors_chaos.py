import asyncio
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.client import Gateway, eventually
from integration._support.process import WorkerHealthcheck, graceful_stop_seconds, owned_proxy_process
from integration.providers._assistants_route_errors_support import (
    MAPPED_BY_STATUS,
    MARKED_ROUTES,
    ROUTE_BY_NAME,
    Call,
    Held,
    answer_all,
    assert_answered_with_its_own_marker,
    assistants_wire,
    held_error_peer,
    held_one_by_one,
    held_upstream_connections,
    marker_of,
    new_marker,
    openai_assistants_config,
    provider_requests,
    worker_pids,
)

_STATUSES: Final = (400, 404, 429, 500, 503)
_PER_WORKER: Final = 10
_STARTUP_COMPLETE: Final = "Application startup complete."


def _startups(log: Path) -> tuple[int, int]:
    return len(worker_pids(log)), log.read_text().count(_STARTUP_COMPLETE)


@pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120))
async def test_worker_sigkill_mid_burst_leaves_the_sibling_answering_each_mapped_status(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = tuple(
        Call(MARKED_ROUTES[index % len(MARKED_ROUTES)], new_marker()) for index in range(2 * _PER_WORKER)
    )
    follow_up: Final = Call(ROUTE_BY_NAME["get_thread"], new_marker())
    status_by_marker: Final = MappingProxyType(
        {
            **{call.marker: _STATUSES[index % len(_STATUSES)] for index, call in enumerate(calls)},
            follow_up.marker: 404,
        }
    )
    release: Final = threading.Event()
    arrived: Final[SimpleQueue[str]] = SimpleQueue()
    with assistants_wire(held_error_peer(status_by_marker, arrived, release)) as wire:
        port: Final = urlsplit(wire.url).port
        assert port is not None
        config: Final = openai_assistants_config(tmp_path, port, 60)
        with owned_proxy_process(
            gateway, tmp_path, {}, config=config, workers=2, healthcheck=WorkerHealthcheck.LENIENT
        ) as owned:
            eventually(lambda: _startups(owned.log), lambda found: found == (2, 2), seconds=120)
            victim_pid, survivor_pid = sorted(eventually(owned.worker_pids, lambda pids: len(pids) == 2))
            try:
                async with AsyncExitStack() as bursts:
                    with owned.suspended(survivor_pid):
                        on_victim: Final = await bursts.enter_async_context(
                            held_one_by_one(owned.gateway, calls[:_PER_WORKER], arrived, lambda: False)
                        )
                    with owned.suspended(victim_pid):
                        on_survivor: Final = await bursts.enter_async_context(
                            held_one_by_one(owned.gateway, calls[_PER_WORKER:], arrived, lambda: False)
                        )
                    held: Final = Held(on_victim.calls + on_survivor.calls, on_victim.pending + on_survivor.pending)
                    held_by: Final = await asyncio.to_thread(
                        held_upstream_connections, (victim_pid, survivor_pid), port, len(held.calls)
                    )
                    assert held_by == {victim_pid: _PER_WORKER, survivor_pid: _PER_WORKER}, held_by
                    with owned.replacing_worker(victim_pid) as kill:
                        await asyncio.to_thread(kill)
                        release.set()
                        served: Final = await held.answers()
            finally:
                release.set()
            (answered,) = await answer_all(owned.gateway, (follow_up,))
            await asyncio.to_thread(eventually, lambda: _startups(owned.log), lambda found: found == (3, 3), 180)
        received: Final = provider_requests(wire.drain())
    assert len(served) == held_by[survivor_pid], (held_by, len(served))
    for answer in (*served, answered):
        assert_answered_with_its_own_marker(answer, MAPPED_BY_STATUS[status_by_marker[answer.call.marker]])
    assert sorted(marker_of(request) or "" for request in received) == sorted(
        (*(call.marker for call in held.calls), follow_up.marker)
    ), received
