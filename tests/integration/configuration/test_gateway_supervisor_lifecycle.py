import http.client
import os
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from gateway.launch import (
    _pooled_child_environments,  # pyright: ignore[reportPrivateUsage]  # exercise the child env policy
)
from gateway.supervisor import ChildProcess, SupervisorConfig, SupervisorFailure, supervise
from litellm.proxy.db.pgbouncer import (
    ManagedPgBouncerRuntime,
    PgBouncerSettings,
    start_managed_pgbouncer,
)
from tests.integration._support.ports import allocate_port

ROOT: Final = Path(__file__).resolve().parents[3]
SERVER_SOURCE: Final = """\
import http.server
import signal
import subprocess
import sys
import threading
from pathlib import Path

role, port, events, mode, child_pid_file = sys.argv[1:]
if role == "python" and mode == "startup-failure":
    raise SystemExit(17)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        if self.path == "/events":
            self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: connected\\n\\n" if self.path == "/events" else b"ok")
        self.wfile.flush()
        if self.path == "/events":
            threading.Event().wait()
        if role == "rust" and mode == "exit":
            threading.Timer(0.1, server.shutdown).start()

    def log_message(self, *_args):
        pass

server = http.server.ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
server.daemon_threads = True
if mode == "orphan":
    child = subprocess.Popen(
        (
            sys.executable,
            "-P",
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
        ),
        start_new_session=True,
    )
    Path(child_pid_file).write_text(str(child.pid))
elif mode == "orphan-exit":
    middle_command = "import subprocess,sys,time; child=subprocess.Popen((sys.executable,'-P','-c','import time; time.sleep(0.35)'), start_new_session=True); open(sys.argv[1],'w').write(str(child.pid)); time.sleep(0.2)"
    subprocess.Popen((sys.executable, "-P", "-c", middle_command, child_pid_file))

def stop(_signum, _frame):
    with Path(events).open("a") as output:
        output.write(role + "\\n")
    if mode not in ("ignore", "orphan"):
        threading.Thread(target=server.shutdown).start()

signal.signal(signal.SIGTERM, stop)
if role == "python" and mode == "exit":
    threading.Timer(0.5, server.shutdown).start()
server.serve_forever()
"""
LAUNCH_MAIN_SOURCE: Final = """\
import os
from pathlib import Path
from gateway.launch import main

def serve(_argv):
    Path(os.environ["TEST_SERVE_MARKER"]).write_text(
        os.environ["LITELLM_PGBOUNCER_ENABLED"] + "|" + os.environ.get("DATABASE_URL", "")
    )

main((), serve)
"""
SUPERVISOR_SOURCE: Final = """\
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from types import MappingProxyType
from pathlib import Path
from gateway.supervisor import SupervisorConfig, supervise

server, private_port, public_port, events, python_mode, rust_mode, child_pid_file, rust_timeout = sys.argv[1:]
environment = MappingProxyType(dict(os.environ))
python_command = (sys.executable, server, "python", private_port, events, python_mode, child_pid_file)
rust_command = (
    ("/missing/rust-gateway",)
    if rust_mode == "missing"
    else (sys.executable, server, "rust", public_port, events, rust_mode, child_pid_file)
)
config = SupervisorConfig(
    python_command=python_command,
    rust_command=rust_command,
    python_environment=environment,
    rust_environment=environment,
    public_probe_host="127.0.0.1",
    public_port=int(public_port),
    private_port=int(private_port),
    startup_timeout=5,
    rust_drain_timeout=int(rust_timeout),
    python_grace_timeout=2,
)
if "TEST_MANAGED_PGBOUNCER_BINARY" in os.environ:
    from litellm.proxy.db.pgbouncer import PgBouncerError, PgBouncerSettings, start_managed_pgbouncer

    managed_pooler = start_managed_pgbouncer(
        PgBouncerSettings(
            enabled=True,
            port=int(os.environ["TEST_MANAGED_PGBOUNCER_PORT"]),
            binary=os.environ["TEST_MANAGED_PGBOUNCER_BINARY"],
        ),
        "postgresql://litellm:password@127.0.0.1/litellm",
    )
    if isinstance(managed_pooler, PgBouncerError):
        print(managed_pooler.reason, file=sys.stderr)
        sys.exit(1)
    pooler_pid_file = Path(os.environ["TEST_MANAGED_PGBOUNCER_PID_FILE"])

    def write_pooler_pid():
        temporary = pooler_pid_file.with_suffix(".tmp")
        temporary.write_text(str(managed_pooler.pooler.pid))
        temporary.replace(pooler_pid_file)

    def record_pooler_pid():
        while True:
            write_pooler_pid()
            time.sleep(0.05)

    write_pooler_pid()
    threading.Thread(target=record_pooler_pid, daemon=True).start()
    config = replace(config, managed_pooler=managed_pooler)
failure = supervise(config)
if failure is not None:
    print(failure.reason, file=sys.stderr)
    sys.exit(1)
"""


@dataclass(frozen=True, slots=True)
class _FailAfterPythonStartRunner:
    python_command: tuple[str, ...]
    python_pid_file: Path
    cleanup: ExitStack

    def start(self, command: tuple[str, ...], environment: Mapping[str, str]) -> ChildProcess:
        if command != self.python_command:
            raise OSError("injected Rust gateway start failure")
        process: Final = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.cleanup.callback(_terminate_if_running, process)
        self.python_pid_file.write_text(str(process.pid))
        return process


def _start_supervisor(
    tmp_path: Path,
    private_port: int,
    public_port: int,
    events: Path,
    python_mode: str = "normal",
    rust_mode: str = "normal",
    rust_timeout: int = 2,
    managed_pooler: tuple[Path, int, Path] | None = None,
) -> subprocess.Popen[bytes]:
    server: Final = tmp_path / "server.py"
    launcher: Final = tmp_path / "supervisor.py"
    child_pid_file: Final = tmp_path / "orphan.pid"
    server.write_text(SERVER_SOURCE)
    launcher.write_text(SUPERVISOR_SOURCE)
    environment: Final = {
        **os.environ,
        **(
            {}
            if managed_pooler is None
            else {
                "TEST_MANAGED_PGBOUNCER_BINARY": str(managed_pooler[0]),
                "TEST_MANAGED_PGBOUNCER_PORT": str(managed_pooler[1]),
                "TEST_MANAGED_PGBOUNCER_PID_FILE": str(managed_pooler[2]),
            }
        ),
    }
    return subprocess.Popen(
        (
            sys.executable,
            "-P",
            str(launcher),
            str(server),
            str(private_port),
            str(public_port),
            str(events),
            python_mode,
            rust_mode,
            str(child_pid_file),
            str(rust_timeout),
        ),
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _fake_pgbouncer(tmp_path: Path) -> Path:
    binary: Final = tmp_path / "fake-pgbouncer"
    binary.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import configparser, select, socket, sys
            if sys.argv[1:] == ["--version"]:
                print("PgBouncer 1.25.2")
                sys.exit(0)
            configuration = configparser.ConfigParser()
            configuration.read(sys.argv[1])
            port = configuration.getint("pgbouncer", "listen_port")
            tcp = socket.socket()
            tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            tcp.bind(("127.0.0.1", port))
            tcp.listen()
            unix = socket.socket(socket.AF_UNIX)
            unix.bind(configuration.get("pgbouncer", "unix_socket_dir") + f"/.s.PGSQL.{{port}}")
            unix.listen()
            while True:
                for listener in select.select((tcp, unix), [], [])[0]:
                    listener.accept()[0].close()
            """
        )
    )
    binary.chmod(0o700)
    return binary


def _slow_starting_pgbouncer(tmp_path: Path, pid_file: Path, config_file: Path) -> Path:
    binary: Final = tmp_path / "slow-pgbouncer"
    binary.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os, signal, sys, time
            from pathlib import Path
            if sys.argv[1:] == ["--version"]:
                print("PgBouncer 1.26.0")
                sys.exit(0)
            Path({str(config_file)!r}).write_text(sys.argv[1])
            Path({str(pid_file)!r}).write_text(str(os.getpid()))
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(60)
            """
        )
    )
    binary.chmod(0o700)
    return binary


def test_managed_pgbouncer_stop_is_bounded_and_guards_its_owned_pid(tmp_path: Path) -> None:
    port: Final = allocate_port()
    runtime: Final = start_managed_pgbouncer(
        PgBouncerSettings(enabled=True, port=port, binary=str(_fake_pgbouncer(tmp_path))),
        "postgresql://litellm:password@127.0.0.1/litellm",
    )
    assert isinstance(runtime, ManagedPgBouncerRuntime), runtime
    pid: Final = runtime.pooler.pid
    deadline: Final = time.monotonic() + 2
    try:
        with runtime.owned_pid_guard() as guarded_pid:
            assert guarded_pid == pid
    finally:
        runtime.stop(deadline)
    assert pid is not None
    assert not Path(f"/proc/{pid}").exists()
    assert not runtime.runtime_dir.exists()


def test_launch_cleans_up_managed_pooler_when_startup_is_interrupted(tmp_path: Path) -> None:
    pid_file: Final = tmp_path / "pooler.pid"
    config_file: Final = tmp_path / "pooler-config"
    process: Final = subprocess.Popen(  # test-quality-ok: child must import the working-tree gateway.launch module
        (
            sys.executable,
            "-m",
            "gateway.launch",
            "--host",
            "127.0.0.1",
            "--port",
            str(allocate_port()),
        ),
        cwd=ROOT,
        env={
            **os.environ,
            "DATABASE_URL": "postgresql://litellm:password@127.0.0.1/litellm",
            "LITELLM_PGBOUNCER_BINARY": str(_slow_starting_pgbouncer(tmp_path, pid_file, config_file)),
            "LITELLM_PGBOUNCER_ENABLED": "false",
            "LITELLM_PGBOUNCER_PORT": str(allocate_port()),
            "LITELLM_RUST_GATEWAY": "true",
            "LITELLM_RUST_GATEWAY_BINARY": "/bin/true",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        pooler_pid: Final = _wait_for_pid(pid_file)
        os.kill(process.pid, signal.SIGTERM)  # test-quality-ok: Interrupt startup to exercise managed cleanup
        assert process.wait(timeout=20) == 143
        assert not Path(f"/proc/{pooler_pid}").exists()
        runtime_dir: Final = Path(config_file.read_text()).parent
        assert not runtime_dir.exists()
    finally:
        _terminate_if_running(process)


def _wait_for_pid(pid_file: Path) -> int:
    deadline: Final = time.monotonic() + 5
    pooler_pid: Final = next(
        (pid for pid in _recorded_pooler_pids(pid_file, deadline) if Path(f"/proc/{pid}").exists()),
        None,
    )
    if pooler_pid is None:
        pytest.fail("managed PgBouncer did not record a live process")
    return pooler_pid


def _recorded_pooler_pids(pid_file: Path, deadline: float) -> Iterator[int]:
    while time.monotonic() < deadline:
        try:
            yield int(pid_file.read_text())
        except (OSError, ValueError):
            pass
        time.sleep(0.05)


def _wait_for_replacement_pooler(pid_file: Path, original_pid: int) -> int | None:
    deadline: Final = time.monotonic() + 5
    return next(
        (
            pid
            for pid in _recorded_pooler_pids(pid_file, deadline)
            if pid != original_pid and Path(f"/proc/{pid}").exists()
        ),
        None,
    )


def test_database_free_python_child_runs_launch_with_pooling_disabled(tmp_path: Path) -> None:
    serve_marker: Final = tmp_path / "serve"
    inherited_environment: Final = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("DATABASE_") and name not in ("DIRECT_URL", "LITELLM_PGBOUNCER_POOLED_DATABASE_URL")
    }
    rust_environment: Final = {
        **inherited_environment,
        "LITELLM_RUST_GATEWAY": "false",
        "LITELLM_PGBOUNCER_ENABLED": "true",
        "TEST_SERVE_MARKER": str(serve_marker),
    }
    python_environment, _ = _pooled_child_environments(rust_environment, {}, None)
    child: Final = subprocess.run(
        (sys.executable, "-P", "-c", LAUNCH_MAIN_SOURCE),
        cwd=ROOT,
        env=python_environment,
        capture_output=True,
        timeout=10,
    )
    assert child.returncode == 0, child.stderr.decode(errors="replace")
    assert serve_marker.read_text() == "false|"


def _wait_for_gateway(process: subprocess.Popen[bytes], port: int) -> None:
    deadline: Final = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"supervisor exited early with status {process.returncode}")
        if _gateway_readiness_status(port) == 200:
            return
        time.sleep(0.05)
    pytest.fail("supervisor did not start both child listeners")


def _gateway_readiness_status(port: int) -> int | None:
    connection: Final = http.client.HTTPConnection("127.0.0.1", port, timeout=0.2)
    try:
        connection.request("GET", "/health/readiness")
        response: Final = connection.getresponse()
        response.read()
        return response.status
    except OSError:
        return None
    finally:
        connection.close()


def _terminate_if_running(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGKILL)  # test-quality-ok: Cleanup terminates only this test-owned supervisor group
    process.wait(timeout=5)


def test_supervisor_drains_rust_before_python(tmp_path: Path) -> None:
    private_port: Final = allocate_port()
    public_port: Final = allocate_port()
    events: Final = tmp_path / "signals"
    process: Final = _start_supervisor(tmp_path, private_port, public_port, events)
    try:
        _wait_for_gateway(process, public_port)
        os.kill(process.pid, signal.SIGTERM)  # test-quality-ok: The signal is the ordered-shutdown event under test
        assert process.wait(timeout=10) == 0
        assert events.read_text().splitlines() == ["rust", "python"]
    finally:
        _terminate_if_running(process)


def test_supervisor_fails_when_python_cannot_start(tmp_path: Path) -> None:
    process: Final = _start_supervisor(
        tmp_path,
        allocate_port(),
        allocate_port(),
        tmp_path / "signals",
        python_mode="startup-failure",
    )
    try:
        assert process.wait(timeout=10) == 1
    finally:
        _terminate_if_running(process)


def test_supervisor_cleans_up_python_when_rust_cannot_start(tmp_path: Path) -> None:
    events: Final = tmp_path / "signals"
    process: Final = _start_supervisor(tmp_path, allocate_port(), allocate_port(), events, rust_mode="missing")
    try:
        assert process.wait(timeout=10) == 1
        assert events.read_text().splitlines() == ["python"]
    finally:
        _terminate_if_running(process)


def test_supervisor_cleans_children_and_managed_pooler_when_rust_start_raises(tmp_path: Path) -> None:
    server: Final = tmp_path / "server.py"
    server.write_text(SERVER_SOURCE)
    python_pid_file: Final = tmp_path / "python.pid"
    python_command: Final = (
        sys.executable,
        "-P",
        str(server),
        "python",
        str(allocate_port()),
        str(tmp_path / "events"),
        "normal",
        str(tmp_path / "orphan.pid"),
    )
    pooler: Final = start_managed_pgbouncer(
        PgBouncerSettings(enabled=True, port=allocate_port(), binary=str(_fake_pgbouncer(tmp_path))),
        "postgresql://litellm:password@127.0.0.1/litellm",
    )
    assert isinstance(pooler, ManagedPgBouncerRuntime), pooler
    pooler_pid: Final = pooler.pooler.pid
    config: Final = SupervisorConfig(
        python_command=python_command,
        rust_command=("/injected/rust-gateway",),
        python_environment={},
        rust_environment={},
        public_probe_host="127.0.0.1",
        public_port=allocate_port(),
        private_port=int(python_command[4]),
        startup_timeout=5,
        rust_drain_timeout=1,
        python_grace_timeout=1,
        termination_grace_timeout=5,
        managed_pooler=pooler,
    )
    try:
        with ExitStack() as child_cleanup:
            failure: Final = supervise(
                config,
                runner=_FailAfterPythonStartRunner(python_command, python_pid_file, child_cleanup),
            )
            assert isinstance(failure, SupervisorFailure)
            assert "injected Rust gateway start failure" in failure.reason
            python_pid: Final = int(python_pid_file.read_text())
            assert not Path(f"/proc/{python_pid}").exists()
            assert pooler_pid is not None
            assert not Path(f"/proc/{pooler_pid}").exists()
            assert not pooler.runtime_dir.exists()
    finally:
        pooler.stop(time.monotonic() + 2)


@pytest.mark.parametrize("exiting_child", ("python", "rust"))
def test_supervisor_fails_when_either_child_exits(tmp_path: Path, exiting_child: str) -> None:
    events: Final = tmp_path / "signals"
    process: Final = _start_supervisor(
        tmp_path,
        allocate_port(),
        allocate_port(),
        events,
        python_mode="exit" if exiting_child == "python" else "normal",
        rust_mode="exit" if exiting_child == "rust" else "normal",
    )
    try:
        assert process.wait(timeout=10) == 1
    finally:
        _terminate_if_running(process)


def test_supervisor_kills_unresponsive_rust_group_and_reaps_orphan(tmp_path: Path) -> None:
    private_port: Final = allocate_port()
    public_port: Final = allocate_port()
    events: Final = tmp_path / "signals"
    child_pid_file: Final = tmp_path / "orphan.pid"
    process: Final = _start_supervisor(
        tmp_path,
        private_port,
        public_port,
        events,
        rust_mode="orphan",
        rust_timeout=1,
    )
    try:
        _wait_for_gateway(process, public_port)
        deadline: Final = time.monotonic() + 5
        while not child_pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid_file.is_file()
        child_pid: Final = int(child_pid_file.read_text())
        os.kill(process.pid, signal.SIGTERM)  # test-quality-ok: The signal is the orphan-cleanup event under test
        assert process.wait(timeout=10) == 0
        assert events.read_text().splitlines() == ["rust", "python"]
        assert not Path(f"/proc/{child_pid}").exists()
    finally:
        _terminate_if_running(process)


def test_supervisor_reaps_adopted_child_while_gateway_is_ready(tmp_path: Path) -> None:
    public_port: Final = allocate_port()
    child_pid_file: Final = tmp_path / "orphan.pid"
    pooler_pid_file: Final = tmp_path / "pooler.pid"
    process: Final = _start_supervisor(
        tmp_path,
        allocate_port(),
        public_port,
        tmp_path / "signals",
        rust_mode="orphan-exit",
        managed_pooler=(_fake_pgbouncer(tmp_path), allocate_port(), pooler_pid_file),
    )
    try:
        _wait_for_gateway(process, public_port)
        pooler_pid: Final = _wait_for_pid(pooler_pid_file)
        deadline: Final = time.monotonic() + 5
        while not child_pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid_file.is_file()
        child_pid: Final = int(child_pid_file.read_text())
        child_deadline: Final = time.monotonic() + 5
        while Path(f"/proc/{child_pid}").exists() and time.monotonic() < child_deadline:
            time.sleep(0.05)
        assert not Path(f"/proc/{child_pid}").exists()
        assert process.poll() is None
        assert Path(f"/proc/{pooler_pid}").exists()
        os.kill(pooler_pid, signal.SIGKILL)  # test-quality-ok: The signal exercises supervised PgBouncer restart
        replacement_pid: Final = _wait_for_replacement_pooler(pooler_pid_file, pooler_pid)
        assert replacement_pid is not None
        assert replacement_pid != pooler_pid
        _wait_for_gateway(process, public_port)
        os.kill(process.pid, signal.SIGTERM)  # test-quality-ok: Signal shutdown after observing the adopted child reap
        assert process.wait(timeout=10) == 0
        assert not Path(f"/proc/{replacement_pid}").exists()
    finally:
        _terminate_if_running(process)


def test_supervisor_stops_managed_pooler_when_python_startup_fails(tmp_path: Path) -> None:
    pooler_pid_file: Final = tmp_path / "pooler.pid"
    process: Final = _start_supervisor(
        tmp_path,
        allocate_port(),
        allocate_port(),
        tmp_path / "signals",
        python_mode="startup-failure",
        managed_pooler=(_fake_pgbouncer(tmp_path), allocate_port(), pooler_pid_file),
    )
    try:
        assert process.wait(timeout=10) == 1
        pooler_pid: Final = int(pooler_pid_file.read_text())
        assert not Path(f"/proc/{pooler_pid}").exists()
    finally:
        _terminate_if_running(process)


def test_supervisor_bounds_shutdown_with_an_open_sse_response(tmp_path: Path) -> None:
    public_port: Final = allocate_port()
    events: Final = tmp_path / "signals"
    process: Final = _start_supervisor(
        tmp_path,
        allocate_port(),
        public_port,
        events,
        rust_mode="ignore",
        rust_timeout=1,
    )
    connection: Final = http.client.HTTPConnection("127.0.0.1", public_port, timeout=2)
    try:
        _wait_for_gateway(process, public_port)
        connection.request("GET", "/events")
        response: Final = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream"
        assert response.readline() == b"data: connected\n"
        os.kill(process.pid, signal.SIGTERM)  # test-quality-ok: The signal starts the bounded SSE drain under test
        assert process.wait(timeout=10) == 0
        assert events.read_text().splitlines() == ["rust", "python"]
    finally:
        connection.close()
        _terminate_if_running(process)
