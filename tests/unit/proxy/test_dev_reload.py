import contextlib
import importlib
import logging
import logging.handlers
import multiprocessing
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
from uvicorn import config as uvicorn_config

from litellm.proxy.dev_reload import (
    RELOAD_WORKER_HEALTHCHECK_SECONDS,
    ReloadOptions,
    load_app_or_exit,
    multi_worker_reload_supported,
    pool_config,
    stop_pool_when_reloader_exits,
)

UNCHANGED_LOGGING: Final = {"version": 1, "incremental": True}
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


def _options(app: str) -> ReloadOptions:
    return {
        "app": app,
        "log_config": UNCHANGED_LOGGING,
        "reload": True,
        "reload_dirs": (),
        "reload_includes": ("*.py",),
    }


@pytest.mark.parametrize(
    ("module", "source"),
    (
        ("reload_syntax_error_app", "def app(\n"),
        ("reload_name_error_app", "app = undefined_name\n"),
        ("reload_runtime_error_app", "raise RuntimeError('broken edit')\n"),
        ("reload_missing_attribute_app", "value = 1\n"),
    ),
)
def test_a_broken_app_exits_the_worker_as_a_startup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: str, source: str
) -> None:
    (tmp_path / f"{module}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SystemExit) as exit_info:
        pool_config(_options(f"{module}:app"), workers=2).load()
    assert exit_info.value.code == uvicorn_config.STARTUP_FAILURE


@pytest.mark.parametrize(
    ("module", "source"),
    (
        ("reload_import_error_app", "import os\nfrom os import missing_name\n"),
        ("reload_value_error_app", "import os\nraise ValueError('broken edit')\n"),
    ),
)
def test_a_broken_import_logs_the_line_that_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: str, source: str
) -> None:
    app_file: Final = tmp_path / f"{module}.py"
    app_file.write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    uvicorn_logger: Final = logging.getLogger("uvicorn.error")
    captured: Final = logging.handlers.BufferingHandler(capacity=POOL_WORKERS)
    uvicorn_logger.addHandler(captured)
    try:
        with pytest.raises(SystemExit):
            load_app_or_exit(f"{module}:app")
    finally:
        uvicorn_logger.removeHandler(captured)
    (record,) = captured.buffer
    assert record.exc_info is not None
    raised_at: Final = frozenset((frame.filename, frame.lineno) for frame in traceback.extract_tb(record.exc_info[2]))
    assert (str(app_file), 2) in raised_at, record.exc_text


def test_the_reload_pool_defaults_the_worker_healthcheck_and_keeps_an_explicit_one() -> None:
    explicit: Final = RELOAD_WORKER_HEALTHCHECK_SECONDS + 7
    assert pool_config(_options("app:app"), workers=2).timeout_worker_healthcheck == RELOAD_WORKER_HEALTHCHECK_SECONDS
    assert (
        pool_config({**_options("app:app"), "timeout_worker_healthcheck": explicit}, workers=2).timeout_worker_healthcheck
        == explicit
    )


def test_the_pool_is_told_to_stop_once_the_reloader_process_exits() -> None:
    reloader: Final = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(POOL_TIMEOUT_SECONDS,))
    reloader.start()
    pool_should_exit: Final = threading.Event()
    threading.Thread(target=stop_pool_when_reloader_exits, args=(reloader, pool_should_exit), daemon=True).start()
    try:
        assert not pool_should_exit.wait(0.5)
        reloader.kill()
        assert pool_should_exit.wait(POOL_TIMEOUT_SECONDS)
    finally:
        reloader.kill()
        reloader.join()


def test_a_working_app_loads_as_the_module_defines_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "reload_working_app.py").write_text("async def app(scope, receive, send):\n    return None\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert load_app_or_exit("reload_working_app:app") is importlib.import_module("reload_working_app").app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _started_workers(log_text: str) -> frozenset[int]:
    return frozenset(int(match.group(1)) for match in WORKER_STARTED.finditer(log_text))


def _wait_for_log(log: Path, process: subprocess.Popen[bytes], seen: Callable[[str], bool]) -> str:
    deadline: Final = time.monotonic() + POOL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        log_text: Final = log.read_text()
        if seen(log_text):
            return log_text
        assert process.poll() is None, log_text
        time.sleep(0.2)
    raise AssertionError(log.read_text())


def _start_pool(app_dir: Path, port: int, log: Path) -> subprocess.Popen[bytes]:
    with log.open("wb") as output:
        return subprocess.Popen(
            (
                sys.executable,
                "-I",
                "-c",
                POOL_RUNNER,
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


@pytest.mark.skipif(
    not multi_worker_reload_supported(), reason="uvicorn<0.50.0 cannot stop a worker pool on a startup failure"
)
@pytest.mark.timeout(5 * POOL_TIMEOUT_SECONDS)
def test_a_save_replaces_every_worker_and_a_broken_save_stops_the_pool_until_the_next_good_one(
    tmp_path: Path,
) -> None:
    app: Final = tmp_path / "reload_pool_app.py"
    app.write_text(PID_APP)
    log: Final = tmp_path / "reload.log"
    port: Final = _free_port()
    process: Final = _start_pool(tmp_path, port, log)
    try:
        first_pool: Final = _next_pool(log, process, port, frozenset())
        app.write_text("async def app(\n")
        _wait_for_log(log, process, lambda text: POOL_STOPPED in text)
        assert not any(psutil.pid_exists(pid) for pid in first_pool), log.read_text()
        app.write_text(PID_APP)
        last_pool: Final = _next_pool(log, process, port, first_pool)
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=POOL_TIMEOUT_SECONDS) == 0, log.read_text()
        assert not any(psutil.pid_exists(pid) for pid in last_pool), log.read_text()
        assert 1 <= log.read_text().count(APP_LOAD_ERROR) <= POOL_WORKERS, log.read_text()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@pytest.mark.skipif(
    not multi_worker_reload_supported(), reason="uvicorn<0.50.0 cannot stop a worker pool on a startup failure"
)
@pytest.mark.timeout(3 * POOL_TIMEOUT_SECONDS)
@pytest.mark.parametrize(
    "kill_reloader",
    (
        pytest.param(lambda reloader: os.kill(reloader, signal.SIGKILL), id="sigkill-reloader"),
        pytest.param(lambda reloader: os.killpg(reloader, signal.SIGHUP), id="terminal-closed"),
        pytest.param(lambda reloader: os.killpg(reloader, signal.SIGQUIT), id="ctrl-backslash"),
    ),
)
def test_the_pool_stops_and_frees_the_port_when_the_reloader_dies(
    tmp_path: Path, kill_reloader: Callable[[int], None]
) -> None:
    (tmp_path / "reload_pool_app.py").write_text(PID_APP)
    log: Final = tmp_path / "reload.log"
    port: Final = _free_port()
    process: Final = _start_pool(tmp_path, port, log)
    try:
        workers: Final = _next_pool(log, process, port, frozenset())
        pool_parent: Final = psutil.Process(min(workers)).ppid()
        assert psutil.Process(pool_parent).ppid() == process.pid, log.read_text()
        kill_reloader(process.pid)
        process.wait(timeout=POOL_TIMEOUT_SECONDS)
        deadline: Final = time.monotonic() + POOL_TIMEOUT_SECONDS
        while psutil.pid_exists(pool_parent) or not _port_refuses_connections(port):
            assert time.monotonic() < deadline, log.read_text()
            time.sleep(0.2)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
