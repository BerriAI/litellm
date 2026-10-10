from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import psutil
import pytest
from integration._support.aws_control_plane import ControlPlane, ControlPlaneRequest, control_plane
from integration._support.bedrock_discovery import (
    CLOSE,
    CONTROL_MODEL,
    HOSTED_REGIONS,
    LISTING_TIMEOUT_SECONDS,
    WORKERS,
    catalog_for,
    control_plane_host,
    credential,
    discovered,
    discovery_proxy,
    from_stem,
    listed_ids,
    live_worker_pids,
    mine,
    open_connections_to,
    sigv4_deployment,
    stem,
    wait_for_replacement_worker,
)
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import graceful_stop_seconds
from integration._support.wire import Reply

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    log: Path
    plane: ControlPlane


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    hosts: Final = tuple(control_plane_host(region) for region in HOSTED_REGIONS)
    directory: Final = tmp_path_factory.mktemp("bedrock_discovery_chaos")
    with control_plane(directory, hosts) as plane, gateway_from_environment() as parent:
        with discovery_proxy(parent, directory, plane, check_provider_endpoint=True, workers=WORKERS) as owned:
            yield Rig(owned.gateway, owned.log, plane)


def _victim(pids: tuple[int, ...], plane_url: str) -> int | None:
    busy: Final = tuple(pid for pid in pids if psutil.pid_exists(pid) and open_connections_to(pid, plane_url) > 0)
    return busy[0] if busy else None


def test_worker_killed_mid_listing_leaves_the_sibling_serving_and_a_replacement_listing_again(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    release: Final = threading.Event()
    attempts: Final[list[str]] = []  # mutable-ok: appended by the control plane's handler threads

    def hang(request: ControlPlaneRequest) -> Reply:
        attempts.append(request.path)
        if not release.is_set():
            release.wait(timeout=LISTING_TIMEOUT_SECONDS * 3)
        return catalog.respond(request)

    original: Final = live_worker_pids(rig.log)
    assert len(original) == WORKERS, original
    try:
        with rig.gateway.scenario() as scenario, rig.plane.answering(key, hang):
            sigv4_deployment(scenario, key, secret, "us-east-1")
            with ThreadPoolExecutor(max_workers=1) as pool:
                in_flight: Final = pool.submit(lambda: rig.gateway.request("GET", "/v1/models", headers=CLOSE))
                victim: Final = eventually(
                    lambda: _victim(original, rig.plane.url),
                    lambda pid: pid is not None,
                    seconds=LISTING_TIMEOUT_SECONDS,
                )
                assert victim is not None
                psutil.Process(victim).send_signal(signal.SIGKILL)
                sibling: Final = next(pid for pid in original if pid != victim)
                for _ in range(5):
                    assert rig.gateway.chat(CONTROL_MODEL)["choices"], "the sibling worker stopped answering"
                assert psutil.pid_exists(sibling)
                severed: Final = in_flight.exception(timeout=LISTING_TIMEOUT_SECONDS * 2)
                assert severed is not None or in_flight.result().status_code >= 500
            wait_for_replacement_worker(rig.log, original)
            release.set()
            assert discovered(rig.gateway, catalog, marker) == catalog.invocable_ids()
            for _ in range(2 * WORKERS):
                assert from_stem(marker, listed_ids(rig.gateway)) == catalog.invocable_ids()
            assert attempts, "the victim never reached the control plane"
            assert mine(rig.plane, key) != ()
            survivors: Final = frozenset(live_worker_pids(rig.log))
            replacements: Final = survivors - frozenset(original)
            assert replacements, (survivors, original, victim)
            assert survivors == (frozenset(original) - {victim}) | replacements, (survivors, original, victim)
    finally:
        release.set()
