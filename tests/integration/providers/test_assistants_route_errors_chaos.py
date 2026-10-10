import asyncio
import signal
import threading
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import psutil
import pytest
from integration._support.client import Gateway, eventually
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration.providers._assistants_route_errors_support import (
    MAPPED_BY_STATUS,
    MARKED_ROUTES,
    ROUTE_BY_NAME,
    Call,
    answer_all,
    assert_answered_with_its_own_marker,
    assistants_wire,
    held_error_peer,
    held_one_by_one,
    held_upstream_connections,
    live_worker_pids,
    marker_of,
    new_marker,
    open_upstream_connections,
    openai_assistants_config,
    provider_requests,
    worker_pids,
)

_STATUSES: Final = (400, 404, 429, 500, 503)
_MIN_CALLS: Final = 20
_MAX_CALLS: Final = 60
_STARTUP_COMPLETE: Final = "Application startup complete."


def _startups(log: Path) -> tuple[int, int]:
    return len(worker_pids(log)), log.read_text().count(_STARTUP_COMPLETE)


@pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120))
async def test_worker_sigkill_mid_burst_leaves_the_sibling_answering_each_mapped_status(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = tuple(Call(MARKED_ROUTES[index % len(MARKED_ROUTES)], new_marker()) for index in range(_MAX_CALLS))
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
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            eventually(lambda: _startups(owned.log), lambda found: found == (2, 2), seconds=120)
            workers: Final = live_worker_pids(owned.log)
            assert len(workers) == 2, workers

            def both_workers_hold_enough() -> bool:
                held: Final = tuple(open_upstream_connections(pid, port) for pid in workers)
                return sum(held) >= _MIN_CALLS and min(held) > 0

            try:
                async with held_one_by_one(owned.gateway, calls, arrived, both_workers_hold_enough) as held:
                    held_by: Final = await asyncio.to_thread(held_upstream_connections, workers, port, len(held.calls))
                    assert min(held_by.values()) > 0, held_by
                    victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
                    victim: Final = psutil.Process(victim_pid)
                    victim.suspend()
                    victim.send_signal(signal.SIGKILL)
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
