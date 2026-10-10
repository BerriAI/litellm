from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import psutil
import psycopg
import pytest
from psycopg import sql
from pydantic import BaseModel, ConfigDict

from tests.integration._support.database import read_rows

FAULT_CASE: Final = """import json, os, pathlib, psutil
from urllib.parse import urlsplit

def test_worker_hard_exit():
    worker = psutil.Process()
    owner = psutil.Process(worker.ppid())
    pathlib.Path(os.environ["WORKER_CRASH_RECORD"]).write_text(json.dumps({
        "database": urlsplit(os.environ["DATABASE_URL"]).path.removeprefix("/"),
        "processes": [[p.pid, p.create_time()] for p in owner.children(recursive=True) if p.pid != worker.pid],
    }))
    os._exit(23)
"""


class CrashResources(BaseModel):
    model_config = ConfigDict(frozen=True)
    database: str
    processes: tuple[tuple[int, float], ...]


def _survivor(identity: tuple[int, float]) -> psutil.Process | None:
    pid, created = identity
    try:
        process: Final = psutil.Process(pid)
        return process if process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE else None
    except psutil.NoSuchProcess:
        return None


def _clean_fault_resources(resources: CrashResources) -> None:
    processes: Final = tuple(process for identity in resources.processes if (process := _survivor(identity)))
    for process in processes:
        process.terminate()
    _stopped, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        process.kill()
    psutil.wait_procs(alive, timeout=3)
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(resources.database)))


def test_worker_crash_fails_the_run_and_releases_its_database_and_processes(tmp_path: Path) -> None:
    root: Final = Path(__file__).resolve().parents[3]
    case: Final = tmp_path / "test_worker_fault.py"
    record: Final = tmp_path / "resources.json"
    case.write_text(FAULT_CASE)
    environment: Final = {
        **os.environ,
        "PYTHONPATH": f"{root}:{root / 'tests'}:{root / 'tests' / 'e2e'}",
        "PYTEST_ADDOPTS": "",
        "INTEGRATION_WORKER_ISOLATION": "1",
        "INTEGRATION_PEER_URL": "",
        "INTEGRATION_PROXY_READ_REPLICA_URL": "",
        "INTEGRATION_RESULTS_DIR": str(tmp_path / "results"),
        "WORKER_CRASH_RECORD": str(record),
    }
    try:
        with (tmp_path / "fault-run.log").open("w") as log, pytest.raises(subprocess.CalledProcessError) as refused:
            subprocess.run(
                (
                    sys.executable,
                    "-P",
                    "-m",
                    "pytest",
                    "-p",
                    "tests.integration.conftest",
                    "-p",
                    "no:cacheprovider",
                    "-n",
                    "1",
                    "--max-worker-restart=0",
                    str(case),
                ),
                cwd=tmp_path,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=True,
            )
        resources: Final = CrashResources.model_validate_json(record.read_text())
        assert resources.database.startswith("integration_")
        assert refused.value.returncode == 1
        assert resources.processes
        assert not read_rows("SELECT datname FROM pg_database WHERE datname=%s", (resources.database,))
        assert all(_survivor(identity) is None for identity in resources.processes)
    finally:
        if record.exists():
            recorded: Final = CrashResources.model_validate_json(record.read_text())
            assert recorded.database.startswith("integration_")
            _clean_fault_resources(recorded)
