import threading
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
from integration._support.client import Gateway, eventually
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.prometheus_series import Sample, scrape
from integration._support.wire import Wire, wire_server
from test_prometheus_provider_auth_failure import (
    FAILURES,
    LLM_FAILED,
    STATE,
    TOTALS,
    Provider,
    config,
    identity,
    name,
    send,
    values,
)

BURST: Final = 12
ENDPOINTS: Final = ("chat", "chat-stream", "messages", "responses")


@dataclass(frozen=True, slots=True)
class ChaosRig:
    owned: OwnedProxy
    upstream: Wire
    release: threading.Event


@contextmanager
def _rig(gateway: Gateway, tmp_path: Path) -> Iterator[ChaosRig]:
    release: Final = threading.Event()
    prom_dir: Final = tmp_path / "prom"
    prom_dir.mkdir()
    with (
        wire_server(Provider(release)) as upstream,
        owned_proxy_process(
            gateway,
            tmp_path,
            {"PROMETHEUS_MULTIPROC_DIR": str(prom_dir)},
            config=config(tmp_path, upstream),
            workers=2,
        ) as owned,
    ):
        try:
            yield ChaosRig(owned, upstream, release)
        finally:
            release.set()


def _status(gateway: Gateway, cell: str, model: str, key: str | None = None) -> int:
    try:
        return send(gateway, cell, model=model, key=key).status_code
    except httpx.HTTPError:
        return -1


def _holds_a_connection_to(pid: int, port: int) -> bool:
    return any(connection.raddr[1:] == (port,) for connection in psutil.Process(pid).net_connections(kind="tcp"))


def _workers_calling_upstream(rig: ChaosRig) -> frozenset[int]:
    port: Final = int(rig.upstream.url.rsplit(":", 1)[1])
    return frozenset(pid for pid in rig.owned.worker_pids() if _holds_a_connection_to(pid, port))


def _counted(samples: Sequence[Sample], cell: str) -> tuple[float, float, float]:
    return (
        sum(values(samples, FAILURES, model_id=identity(cell))),
        sum(values(samples, TOTALS, model_id=identity(cell))),
        sum(values(samples, LLM_FAILED, model_id=identity(cell))),
    )


def test_a_burst_of_provider_401s_and_proxy_rejects_across_two_workers_counts_only_the_provider_401s(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path) as rig:
        rig.release.set()
        workers: Final = eventually(rig.owned.worker_pids, lambda pids: len(pids) == 2)
        invalid: Final = f"sk-not-a-virtual-key-{uuid.uuid4().hex}"
        calls: Final[tuple[tuple[str, str, str | None], ...]] = tuple(
            (ENDPOINTS[index % len(ENDPOINTS)], name("burst"), None) for index in range(BURST)
        ) + tuple(("chat", name("invalid-key"), invalid) for _ in range(BURST))

        def call(cell: str, model: str, key: str | None) -> int:
            return _status(rig.owned.gateway, cell, model, key)

        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses: Final = tuple(pool.map(call, *zip(*calls, strict=True)))
        assert statuses == (401,) * (2 * BURST), statuses
        assert len(rig.upstream.drain()) == BURST
        samples: Final = eventually(
            lambda: scrape(rig.owned.gateway),
            lambda observed: _counted(observed, "burst") == (BURST, BURST, BURST),
            seconds=70,
            return_last_on_timeout=True,
        )
        assert _counted(samples, "burst") == (BURST, BURST, BURST), (workers, _counted(samples, "burst"))
        assert set(values(samples, STATE, model_id=identity("burst"))) == {1.0}
        assert values(samples, FAILURES, requested_model=name("invalid-key")) == []
        assert values(samples, LLM_FAILED, model=name("invalid-key")) == []


def test_a_worker_killed_while_provider_401s_are_in_flight_loses_no_answered_failure(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path) as rig:
        eventually(rig.owned.worker_pids, lambda pids: len(pids) == 2)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures: Final = tuple(pool.submit(_status, rig.owned.gateway, "chat", name("held")) for _ in range(BURST))
            eventually(lambda: rig.upstream.received.qsize() >= 8, bool, seconds=30)
            victim: Final = min(eventually(lambda: _workers_calling_upstream(rig), bool))
            with rig.owned.replacing_worker(victim) as kill:
                kill()
                rig.release.set()
            statuses: Final = tuple(future.result() for future in futures)
        reached: Final = len(rig.upstream.drain())
        answered: Final = statuses.count(401)
        lost: Final = statuses.count(-1)
        assert answered + lost == BURST and lost > 0, statuses
        samples: Final = eventually(
            lambda: scrape(rig.owned.gateway),
            lambda observed: answered <= _counted(observed, "held")[0] <= reached,
            seconds=70,
            return_last_on_timeout=True,
        )
        counted: Final = _counted(samples, "held")
        assert answered <= counted[0] <= reached and counted[0] == counted[1] == counted[2], (statuses, counted)
        follow_up: Final = _status(rig.owned.gateway, "chat", name("held"))
        assert follow_up == 401
        recounted: Final = eventually(
            lambda: _counted(scrape(rig.owned.gateway), "held"),
            lambda observed: observed[0] == counted[0] + 1,
            seconds=70,
        )
        assert recounted == (counted[0] + 1,) * 3
