import os
import re
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.process import graceful_stop_seconds, owned_proxy_process

WORKERS: Final = 2
DEPLOYMENT_HEALTHCHECK_SECONDS: Final = 5
WORKER_IMPORT_DELAY_SECONDS: Final = 3 * DEPLOYMENT_HEALTHCHECK_SECONDS
STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
DIED_WORKER: Final = re.compile(r"Child process \[(\d+)\] died")
SLOW_WORKER_HOOK: Final = f"""\
import sys
import time

if "--multiprocessing-fork" in sys.argv:
    time.sleep({WORKER_IMPORT_DELAY_SECONDS})
"""


def _slow_worker_environment(directory: Path) -> dict[str, str]:
    hook: Final = directory / "slow_worker"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(SLOW_WORKER_HOOK)
    return {
        "PYTHONPATH": os.pathsep.join((str(hook), os.environ.get("PYTHONPATH", ""))),
        "TIMEOUT_WORKER_HEALTHCHECK": str(DEPLOYMENT_HEALTHCHECK_SECONDS),
    }


@pytest.mark.timeout(2 * graceful_stop_seconds() + 60)
def test_owned_proxy_workers_outlast_the_deployments_healthcheck_default(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(
        gateway,
        tmp_path,
        _slow_worker_environment(tmp_path),
        workers=WORKERS,
        extra_arguments=("--timeout_worker_healthcheck", str(int(graceful_stop_seconds()))),
    ) as owned:
        started: Final = eventually(
            lambda: STARTED_WORKER.findall(owned.log.read_text()),
            lambda pids: len(pids) >= WORKERS,
            seconds=graceful_stop_seconds(),
        )
        response: Final = owned.gateway.request("GET", "/health/readiness")
        assert response.status_code == 200, response.text
        log: Final = owned.log.read_text()
    assert len(started) == WORKERS, log
    assert DIED_WORKER.findall(log) == [], log
