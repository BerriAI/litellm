import multiprocessing
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest

from litellm.proxy.dev_reload import multi_worker_reload_supported, stop_pool_when_reloader_exits
from tests.integration._support.client import eventually
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import group_members, kill_stray_process, signal_group, stop_processes

POOL_WORKERS: Final = 2
POOL_TIMEOUT_SECONDS: Final = 120
WORKER_HEALTHCHECK_SECONDS: Final = 30
WORKER_STARTED: Final = re.compile(r"Started server process \[(\d+)\]")
POOL_STOPPED: Final = "failed to start, stopping the parent process"
APP_LOAD_ERROR: Final = "Error loading ASGI app"
PID_APP: Final = """import os


async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": str(os.getpid()).encode()})
"""
REQUIRES_UVICORN_STARTUP_FAILURE: Final = pytest.mark.skipif(
    not multi_worker_reload_supported(), reason="uvicorn<0.50.0 has no startup failure exit code"
)
CORE_DUMP_SECONDS: Final = 3
SLOW_CORE_DUMP_ON_QUIT: Final = f"""
import os
import signal
import subprocess


def freeze_until_killed(signum, frame):
    subprocess.Popen(("sh", "-c", f"sleep {CORE_DUMP_SECONDS}; kill -9 {{os.getpid()}}"))
    os.kill(os.getpid(), signal.SIGSTOP)


signal.signal(signal.SIGQUIT, freeze_until_killed)
"""
POOL_RUNNER: Final = """import sys

sys.path.insert(0, sys.argv[1])

from litellm.proxy.dev_reload import run_multi_worker_reload

run_multi_worker_reload(
    {
        "app": "reload_pool_app:app",
        "host": "127.0.0.1",
        "port": int(sys.argv[2]),
        "timeout_worker_healthcheck": int(sys.argv[3]),
        "reload": True,
        "reload_dirs": (sys.argv[1],),
        "reload_includes": ("*.py",),
    },
    workers=int(sys.argv[4]),
)
"""


def _started_workers(log_text: str) -> frozenset[int]:
    return frozenset(int(match.group(1)) for match in WORKER_STARTED.finditer(log_text))


def _until(done: Callable[[], bool], log: Path, seconds: float = POOL_TIMEOUT_SECONDS) -> None:
    eventually(lambda: (done(), log.read_text()), lambda observed: observed[0], seconds=seconds)


def _wait_for_log(log: Path, process: subprocess.Popen[bytes], seen: Callable[[str], bool]) -> str:
    log_text, _ = eventually(
        lambda: (log.read_text(), process.poll()),
        lambda observed: seen(observed[0]) or observed[1] is not None,
        seconds=POOL_TIMEOUT_SECONDS,
    )
    assert seen(log_text), log_text
    return log_text


def _start_pool(app_dir: Path, port: int, log: Path, runner: str = POOL_RUNNER) -> subprocess.Popen[bytes]:
    with log.open("wb") as output:
        return subprocess.Popen(
            (
                sys.executable,
                "-I",
                "-c",
                runner,
                str(app_dir),
                str(port),
                str(WORKER_HEALTHCHECK_SECONDS),
                str(POOL_WORKERS),
            ),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _port_refuses_connections(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _next_pool(log: Path, process: subprocess.Popen[bytes], port: int, previous: frozenset[int]) -> frozenset[int]:
    log_text: Final = _wait_for_log(log, process, lambda text: len(_started_workers(text) - previous) == POOL_WORKERS)
    pool: Final = _started_workers(log_text) - previous
    assert not any(psutil.pid_exists(pid) for pid in previous), log_text
    served: Final = httpx.get(f"http://127.0.0.1:{port}/", timeout=POOL_TIMEOUT_SECONDS)
    assert int(served.text) in pool, log_text
    return pool


def test_the_pool_is_told_to_stop_once_the_reloader_process_exits() -> None:
    reloader: Final = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(POOL_TIMEOUT_SECONDS,))
    reloader.start()
    pool_should_exit: Final = threading.Event()
    threading.Thread(target=stop_pool_when_reloader_exits, args=(reloader, pool_should_exit), daemon=True).start()
    try:
        assert not pool_should_exit.wait(0.5)
        kill_stray_process(reloader.pid)
        assert pool_should_exit.wait(POOL_TIMEOUT_SECONDS)
    finally:
        if reloader.is_alive():
            kill_stray_process(reloader.pid)
        reloader.join()


@REQUIRES_UVICORN_STARTUP_FAILURE
@pytest.mark.timeout(5 * POOL_TIMEOUT_SECONDS)
def test_a_save_replaces_every_worker_and_a_broken_save_stops_the_pool_until_the_next_good_one(
    tmp_path: Path,
) -> None:
    app: Final = tmp_path / "reload_pool_app.py"
    app.write_text(PID_APP)
    log: Final = tmp_path / "reload.log"
    port: Final = allocate_port()
    process: Final = _start_pool(tmp_path, port, log)
    try:
        first_pool: Final = _next_pool(log, process, port, frozenset())
        app.write_text("async def app(\n")
        _wait_for_log(log, process, lambda text: POOL_STOPPED in text)
        assert not any(psutil.pid_exists(pid) for pid in first_pool), log.read_text()
        app.write_text(PID_APP)
        last_pool: Final = _next_pool(log, process, port, first_pool)
        signal_group(process.pid, signal.SIGINT)
        assert process.wait(timeout=POOL_TIMEOUT_SECONDS) == 0, log.read_text()
        assert not any(psutil.pid_exists(pid) for pid in last_pool), log.read_text()
        assert 1 <= log.read_text().count(APP_LOAD_ERROR) <= POOL_WORKERS, log.read_text()
    finally:
        stop_processes(group_members(process.pid))
        process.wait()


@REQUIRES_UVICORN_STARTUP_FAILURE
@pytest.mark.timeout(3 * POOL_TIMEOUT_SECONDS)
@pytest.mark.parametrize(
    "kill_reloader",
    (
        pytest.param(kill_stray_process, id="sigkill-reloader"),
        pytest.param(lambda reloader: signal_group(reloader, signal.SIGHUP), id="terminal-closed"),
        pytest.param(lambda reloader: signal_group(reloader, signal.SIGQUIT), id="ctrl-backslash"),
    ),
)
def test_the_pool_stops_and_frees_the_port_when_the_reloader_dies(
    tmp_path: Path, kill_reloader: Callable[[int], None]
) -> None:
    (tmp_path / "reload_pool_app.py").write_text(PID_APP)
    log: Final = tmp_path / "reload.log"
    port: Final = allocate_port()
    process: Final = _start_pool(tmp_path, port, log)
    try:
        workers: Final = _next_pool(log, process, port, frozenset())
        pool_parent: Final = psutil.Process(min(workers)).ppid()
        assert psutil.Process(pool_parent).ppid() == process.pid, log.read_text()
        kill_reloader(process.pid)
        process.wait(timeout=POOL_TIMEOUT_SECONDS)
        _until(lambda: not psutil.pid_exists(pool_parent) and _port_refuses_connections(port), log)
    finally:
        stop_processes(group_members(process.pid))
        process.wait()


@REQUIRES_UVICORN_STARTUP_FAILURE
@pytest.mark.timeout(3 * POOL_TIMEOUT_SECONDS)
def test_a_killed_pool_parent_takes_its_workers_with_it_so_the_next_save_serves_only_new_workers(
    tmp_path: Path,
) -> None:
    app: Final = tmp_path / "reload_pool_app.py"
    app.write_text(PID_APP)
    log: Final = tmp_path / "reload.log"
    port: Final = allocate_port()
    process: Final = _start_pool(tmp_path, port, log)
    try:
        orphans: Final = _next_pool(log, process, port, frozenset())
        kill_stray_process(psutil.Process(min(orphans)).ppid())
        _until(lambda: not any(psutil.pid_exists(pid) for pid in orphans), log)
        app.write_text(f"{PID_APP}\n")
        _next_pool(log, process, port, orphans)
    finally:
        stop_processes(group_members(process.pid))
        process.wait()


@REQUIRES_UVICORN_STARTUP_FAILURE
@pytest.mark.timeout(3 * POOL_TIMEOUT_SECONDS)
def test_ctrl_backslash_frees_the_port_without_waiting_out_the_worker_healthcheck_when_core_dumps_are_slow(
    tmp_path: Path,
) -> None:
    (tmp_path / "reload_pool_app.py").write_text(PID_APP + SLOW_CORE_DUMP_ON_QUIT)
    log: Final = tmp_path / "reload.log"
    port: Final = allocate_port()
    process: Final = _start_pool(tmp_path, port, log, runner=SLOW_CORE_DUMP_ON_QUIT + POOL_RUNNER)
    try:
        workers: Final = _next_pool(log, process, port, frozenset())
        pool_parent: Final = psutil.Process(min(workers)).ppid()
        signal_group(process.pid, signal.SIGQUIT)
        _until(
            lambda: (
                process.poll() is not None and not psutil.pid_exists(pool_parent) and _port_refuses_connections(port)
            ),
            log,
            seconds=WORKER_HEALTHCHECK_SECONDS / 2,
        )
    finally:
        stop_processes(group_members(process.pid))
        process.wait()
