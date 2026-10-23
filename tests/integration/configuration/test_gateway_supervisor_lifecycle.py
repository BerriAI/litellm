import http.client
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

import pytest

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
SUPERVISOR_SOURCE: Final = """\
import os
import subprocess
import sys
from types import MappingProxyType
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
failure = supervise(config)
if failure is not None:
    print(failure.reason, file=sys.stderr)
    sys.exit(1)
"""


def _start_supervisor(
    tmp_path: Path,
    private_port: int,
    public_port: int,
    events: Path,
    python_mode: str = "normal",
    rust_mode: str = "normal",
    rust_timeout: int = 2,
) -> subprocess.Popen[bytes]:
    server: Final = tmp_path / "server.py"
    launcher: Final = tmp_path / "supervisor.py"
    child_pid_file: Final = tmp_path / "orphan.pid"
    server.write_text(SERVER_SOURCE)
    launcher.write_text(SUPERVISOR_SOURCE)
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
        env=os.environ,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_for_gateway(process: subprocess.Popen[bytes], port: int) -> None:
    deadline: Final = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"supervisor exited early with status {process.returncode}")
        connection: Final = http.client.HTTPConnection("127.0.0.1", port, timeout=0.2)
        try:
            connection.request("GET", "/health/readiness")
            response: Final = connection.getresponse()
            response.read()
            if response.status == 200:
                return
        except OSError:
            pass
        finally:
            connection.close()
    pytest.fail("supervisor did not start both child listeners")


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
