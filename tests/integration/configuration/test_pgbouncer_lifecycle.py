import http.client
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Final, cast

import pytest

from tests.integration._support.ports import allocate_port
from tests.integration._support.timing import assert_enforced_within

ROOT: Final = Path(os.getenv("LITELLM_TEST_ROOT", str(Path(__file__).resolve().parents[3])))
LAUNCHER_SOURCE: Final = """\
import ctypes
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PR_SET_CHILD_SUBREAPER = 36
if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER) failed")

root, launcher_pid_file, pooler_pid_file, config_file, port_file, result_file, *command = sys.argv[1:]
started = time.monotonic()
launcher = subprocess.Popen(command, cwd=root, env=os.environ.copy())
Path(launcher_pid_file).write_text(str(launcher.pid))
status = launcher.wait()
try:
    pooler_pid = int(Path(pooler_pid_file).read_text())
except (OSError, ValueError):
    pooler_pid = None
try:
    config_path = Path(Path(config_file).read_text())
except OSError:
    config_path = None
try:
    port = int(Path(port_file).read_text())
except (OSError, ValueError):
    port = None
pooler_present = pooler_pid is not None and Path(f"/proc/{pooler_pid}").exists()
listener_open = False
if port is not None:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            listener_open = True
    except OSError:
        pass
result = {
    "launcher_exit_code": status,
    "elapsed": time.monotonic() - started,
    "pooler_pid": pooler_pid,
    "pooler_present": pooler_present,
    "listener_open": listener_open,
    "runtime_dir_present": config_path is not None and config_path.parent.exists(),
}
Path(result_file).write_text(json.dumps(result))
if pooler_pid is not None and Path(f"/proc/{pooler_pid}").exists():
    try:
        os.kill(pooler_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pooler_pid, 0)
    except ChildProcessError:
        pass
if config_path is not None:
    shutil.rmtree(config_path.parent, ignore_errors=True)
raise SystemExit(status)
"""

PGBOUNCER_SOURCE: Final = """\
import configparser
import os
import select
import signal
import socket
import sys
import time
from pathlib import Path

signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGHUP, signal.SIG_IGN)

if sys.argv[1:] == ["--version"]:
    print("PgBouncer 1.26.0")
    raise SystemExit(0)

config_path = Path(sys.argv[1])
parser = configparser.ConfigParser()
parser.read(config_path)
port = parser.getint("pgbouncer", "listen_port")
socket_path = Path(parser.get("pgbouncer", "unix_socket_dir")) / f".s.PGSQL.{port}"
Path(os.environ["LITELLM_TEST_POOLER_PID_FILE"]).write_text(str(os.getpid()))
Path(os.environ["LITELLM_TEST_POOLER_CONFIG_FILE"]).write_text(str(config_path))
Path(os.environ["LITELLM_TEST_POOLER_PORT_FILE"]).write_text(str(port))
if os.environ["LITELLM_TEST_POOLER_MODE"] == "slow":
    while True:
        signal.pause()
if os.environ["LITELLM_TEST_POOLER_MODE"] == "exit":
    time.sleep(0.25)
    raise SystemExit(3)

tcp = socket.socket()
tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
tcp.bind(("127.0.0.1", port))
tcp.listen()
unix = socket.socket(socket.AF_UNIX)
unix.bind(str(socket_path))
unix.listen()
while True:
    for listener in select.select((tcp, unix), (), ())[0]:
        listener.accept()[0].close()
"""

SERVE_SOURCE: Final = """\
import sys
import uvicorn
from gateway.launch import main

async def app(scope, receive, send):
    if scope["type"] == "http":
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ready"})

def serve(argv):
    port = int(argv[argv.index("--port") + 1])
    uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off", workers=1)
    ).run()

main(sys.argv[1:], serve=serve)
"""


def _write_script(path: Path, source: str) -> Path:
    path.write_text(f"#!/usr/bin/env python3\n{textwrap.dedent(source)}")
    path.chmod(0o700)
    return path


def _wait_for_file(path: Path, timeout_seconds: float = 15.0) -> str:
    deadline: Final = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            return path.read_text()
        except OSError:
            time.sleep(0.05)
    pytest.fail(f"timed out waiting for {path}")


def _wait_for_new_pid(path: Path, previous_pid: int, timeout_seconds: float = 5.0) -> int:
    deadline: Final = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            pid: Final = int(path.read_text())
            if pid != previous_pid:
                return pid
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    pytest.fail(f"timed out waiting for a replacement process in {path}")


def _http_health(port: int) -> bool:
    try:
        connection: Final = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
        connection.request("GET", "/health/liveliness")
        response: Final = connection.getresponse()
        response.read()
        connection.close()
        return response.status == 200
    except (OSError, http.client.HTTPException):
        return False


def _wait_for_health(port: int, timeout_seconds: float = 15.0) -> None:
    deadline: Final = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _http_health(port):
            return
        time.sleep(0.05)
    pytest.fail(f"gateway did not become healthy on port {port}")


def _run_black_box(
    tmp_path: Path, pooler_mode: str, signal_number: signal.Signals, require_health: bool
) -> dict[str, object]:
    launcher_pid_file: Final = tmp_path / "launcher.pid"
    pooler_pid_file: Final = tmp_path / "pooler.pid"
    config_file: Final = tmp_path / "pooler-config"
    port_file: Final = tmp_path / "pooler-port"
    result_file: Final = tmp_path / "result.json"
    pooler_port: Final = allocate_port()
    gateway_port: Final = allocate_port()
    pooler_binary: Final = _write_script(tmp_path / "fake-pgbouncer", PGBOUNCER_SOURCE)
    wrapper: Final = _write_script(tmp_path / "subreaper-wrapper.py", LAUNCHER_SOURCE)
    environment: Final = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH", "")))),
        "DATABASE_URL": "postgresql://app:password@127.0.0.1:5432/litellm",
        "LITELLM_PGBOUNCER_ENABLED": "true",
        "LITELLM_PGBOUNCER_BINARY": str(pooler_binary),
        "LITELLM_PGBOUNCER_PORT": str(pooler_port),
        "LITELLM_TEST_POOLER_MODE": pooler_mode,
        "LITELLM_TEST_POOLER_PID_FILE": str(pooler_pid_file),
        "LITELLM_TEST_POOLER_CONFIG_FILE": str(config_file),
        "LITELLM_TEST_POOLER_PORT_FILE": str(port_file),
    }
    serve_script: Final = _write_script(tmp_path / "serve-gateway.py", SERVE_SOURCE)
    command: Final = (
        (
            sys.executable,
            str(serve_script),
            "--host",
            "127.0.0.1",
            "--port",
            str(gateway_port),
            "--workers",
            "1",
        )
        if require_health
        else (
            sys.executable,
            "-m",
            "gateway.launch",
            "--host",
            "127.0.0.1",
            "--port",
            str(gateway_port),
            "--workers",
            "1",
        )
    )
    launch_log: Final = (tmp_path / "gateway.log").open("w")
    wrapper_process: Final = subprocess.Popen(
        (
            sys.executable,
            "-P",
            str(wrapper),
            str(ROOT),
            str(launcher_pid_file),
            str(pooler_pid_file),
            str(config_file),
            str(port_file),
            str(result_file),
            *command,
        ),
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=launch_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        launcher_pid: Final = int(_wait_for_file(launcher_pid_file))
        _wait_for_file(pooler_pid_file)
        if require_health:
            try:
                _wait_for_health(gateway_port)
            except BaseException as error:
                raise AssertionError((tmp_path / "gateway.log").read_text()) from error
        os.kill(launcher_pid, signal_number)  # test-quality-ok: signal starts the lifecycle under test
        wrapper_process.wait(timeout=35)
        return cast(dict[str, object], json.loads(result_file.read_text()))
    finally:
        if wrapper_process.poll() is None:
            try:
                launcher_pid = int(launcher_pid_file.read_text())
                os.kill(launcher_pid, signal.SIGKILL)  # test-quality-ok: bounded fixture cleanup
            except (OSError, ValueError):
                pass
            try:
                wrapper_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.kill(wrapper_process.pid, signal.SIGKILL)  # test-quality-ok: bounded fixture cleanup
                wrapper_process.wait()
        launch_log.close()


@pytest.mark.parametrize(
    ("signal_number", "expected_exit"),
    ((signal.SIGTERM, 143), (signal.SIGINT, 130)),
)
def test_startup_signal_removes_slow_pooler_and_runtime(
    tmp_path: Path, signal_number: signal.Signals, expected_exit: int
) -> None:
    result: Final = _run_black_box(tmp_path, "slow", signal_number, require_health=False)
    assert result["pooler_present"] is False, result
    assert result["listener_open"] is False, result
    assert result["runtime_dir_present"] is False, result
    assert result["launcher_exit_code"] == expected_exit, result
    assert cast(float, result["elapsed"]) <= 30


@pytest.mark.parametrize(
    ("signal_number", "expected_exit"),
    ((signal.SIGTERM, 143), (signal.SIGINT, 130)),
)
def test_running_gateway_signal_removes_pooler_and_runtime(
    tmp_path: Path, signal_number: signal.Signals, expected_exit: int
) -> None:
    result: Final = _run_black_box(tmp_path, "serve", signal_number, require_health=True)
    assert result["pooler_present"] is False, result
    assert result["listener_open"] is False, result
    assert result["runtime_dir_present"] is False, result
    assert result["launcher_exit_code"] == expected_exit, result
    assert cast(float, result["elapsed"]) <= 30


def test_managed_pooler_stop_reaps_owned_pid_and_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.db.pgbouncer import ManagedPgBouncerRuntime, PgBouncerSettings, start_managed_pgbouncer

    pid_file: Final = tmp_path / "pooler.pid"
    config_file: Final = tmp_path / "pooler-config"
    port_file: Final = tmp_path / "pooler-port"
    pooler_binary: Final = _write_script(tmp_path / "fake-pgbouncer", PGBOUNCER_SOURCE)
    settings: Final = PgBouncerSettings(enabled=True, binary=str(pooler_binary), port=allocate_port())
    environment: Final = {
        "LITELLM_TEST_POOLER_MODE": "serve",
        "LITELLM_TEST_POOLER_PID_FILE": str(pid_file),
        "LITELLM_TEST_POOLER_CONFIG_FILE": str(config_file),
        "LITELLM_TEST_POOLER_PORT_FILE": str(port_file),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    runtime_result: Final = start_managed_pgbouncer(settings, "postgresql://app:password@db/litellm")
    assert isinstance(runtime_result, ManagedPgBouncerRuntime), runtime_result
    runtime: Final = runtime_result
    pid: Final = runtime.pooler.pid
    try:
        assert pid is not None
        with runtime.owned_pid_guard() as guarded_pid:
            assert guarded_pid == pid
    finally:
        runtime.stop(time.monotonic() + 15)
    assert pid is not None
    assert not Path(f"/proc/{pid}").exists()
    assert not runtime.runtime_dir.exists()


def test_pooler_start_failure_stops_injected_fast_refresher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Callable

    from litellm.proxy.db.pgbouncer import (
        PgBouncerError,
        PgBouncerSettings,
        PgBouncerTokenRefresher,
        PgBouncerTokenSource,
        start_managed_pgbouncer,
    )
    from litellm.proxy.db.token_auth import AzureEntraTokenAuth

    pid_file: Final = tmp_path / "pooler.pid"
    config_file: Final = tmp_path / "pooler-config"
    port_file: Final = tmp_path / "pooler-port"
    pooler_binary: Final = _write_script(tmp_path / "fake-pgbouncer", PGBOUNCER_SOURCE)
    settings: Final = PgBouncerSettings(enabled=True, binary=str(pooler_binary), port=allocate_port())
    monkeypatch.setenv("LITELLM_TEST_POOLER_MODE", "exit")
    monkeypatch.setenv("LITELLM_TEST_POOLER_PID_FILE", str(pid_file))
    monkeypatch.setenv("LITELLM_TEST_POOLER_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("LITELLM_TEST_POOLER_PORT_FILE", str(port_file))
    first_mint: Final = threading.Event()
    refreshed_during_start: Final = threading.Event()
    start_failed: Final = threading.Event()
    refreshed_after_failure: Final = threading.Event()
    initial_install_done: Final = threading.Event()

    def provide_token() -> str:
        if first_mint.is_set():
            refreshed_during_start.set()
            if start_failed.is_set():
                refreshed_after_failure.set()
        else:
            first_mint.set()
        return "opaque-token"

    def fast_refresher_factory(source: PgBouncerTokenSource, install: Callable[[str], None]) -> PgBouncerTokenRefresher:
        def install_once(token: str) -> None:
            if initial_install_done.is_set():
                return
            install(token)
            initial_install_done.set()

        return PgBouncerTokenRefresher(source, install_once, fallback_seconds=0.02, retry_seconds=0.02)

    outcome: Final = start_managed_pgbouncer(
        settings,
        "postgresql://app@db/litellm",
        token_auth=AzureEntraTokenAuth(token_provider=provide_token),
        refresher_factory=fast_refresher_factory,
    )
    assert isinstance(outcome, PgBouncerError), outcome
    assert pid_file.exists(), outcome
    assert refreshed_during_start.wait(2)
    start_failed.set()
    time.sleep(0.1)
    assert not refreshed_after_failure.is_set()
    runtime_dir: Final = Path(config_file.read_text()).parent
    assert not runtime_dir.exists()


def test_managed_stop_reaps_replacement_before_refresher_deadline_and_skips_late_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Callable

    from litellm.proxy.db.pgbouncer import (
        ManagedPgBouncerRuntime,
        PgBouncerSettings,
        PgBouncerTokenRefresher,
        PgBouncerTokenSource,
        start_managed_pgbouncer,
    )
    from litellm.proxy.db.token_auth import AzureEntraTokenAuth

    pid_file: Final = tmp_path / "pooler.pid"
    config_file: Final = tmp_path / "pooler-config"
    port_file: Final = tmp_path / "pooler-port"
    monkeypatch.setenv("LITELLM_TEST_POOLER_MODE", "serve")
    monkeypatch.setenv("LITELLM_TEST_POOLER_PID_FILE", str(pid_file))
    monkeypatch.setenv("LITELLM_TEST_POOLER_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("LITELLM_TEST_POOLER_PORT_FILE", str(port_file))
    first_mint: Final = threading.Event()
    mint_blocked: Final = threading.Event()
    release_mint: Final = threading.Event()
    first_install: Final = threading.Event()
    late_install: Final = threading.Event()

    def provide_token() -> str:
        if not first_mint.is_set():
            first_mint.set()
            return "initial-token"
        mint_blocked.set()
        release_mint.wait(10)
        return "late-token"

    def refresher_factory(source: PgBouncerTokenSource, install: Callable[[str], None]) -> PgBouncerTokenRefresher:
        def observe_install(token: str) -> None:
            if first_install.is_set():
                late_install.set()
                return
            first_install.set()
            install(token)

        return PgBouncerTokenRefresher(source, observe_install, fallback_seconds=0.01, retry_seconds=0.01)

    pooler_binary: Final = _write_script(tmp_path / "fake-pgbouncer", PGBOUNCER_SOURCE)
    settings: Final = PgBouncerSettings(enabled=True, binary=str(pooler_binary), port=allocate_port())
    outcome: Final = start_managed_pgbouncer(
        settings,
        "postgresql://app@db/litellm",
        token_auth=AzureEntraTokenAuth(token_provider=provide_token),
        refresher_factory=refresher_factory,
    )
    assert isinstance(outcome, ManagedPgBouncerRuntime), outcome
    runtime: Final = outcome
    refresher: Final = runtime.refresher
    assert refresher is not None
    try:
        assert first_install.is_set()
        assert mint_blocked.wait(3)
        initial_pid: Final = runtime.pooler.pid
        assert initial_pid is not None
        os.kill(initial_pid, signal.SIGKILL)  # test-quality-ok: crash the owned child to exercise watchdog replacement
        replacement_pid: Final = _wait_for_new_pid(pid_file, initial_pid)
        assert runtime.pooler.pid == replacement_pid
        started: Final = time.monotonic()
        stop_budget_seconds: Final = 0.23
        runtime.stop(started + stop_budget_seconds)
        assert_enforced_within(
            time.monotonic() - started,
            configured=stop_budget_seconds,
            slack=0.27,
        )
        assert not Path(f"/proc/{replacement_pid}").exists()
        assert not runtime.runtime_dir.exists()
        release_mint.set()
        refresher.stop(time.monotonic() + 2)
        assert not late_install.is_set()
    finally:
        release_mint.set()
        runtime.stop(time.monotonic() + 5)
