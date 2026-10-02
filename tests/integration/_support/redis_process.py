import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError


@dataclass
class OwnedRedis:
    host: str
    port: int
    command: tuple[str, ...]
    log: TextIO
    pid_file: str
    process: subprocess.Popen | None = None
    server_pid: int | None = None

    def start(self) -> None:
        assert self.process is None
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline: Final = time.monotonic() + 8
        with Redis(host=self.host, port=self.port, socket_connect_timeout=0.2, socket_timeout=0.2) as client:
            while True:
                assert self.process.poll() is None, "Owned Redis exited before readiness"
                try:
                    if client.ping():
                        actual: Final = int(client.info("server")["process_id"])
                        expected: Final = self.process.pid if self.command[0] != "docker" else int(subprocess.check_output(["docker", "exec", "redis-cache", "cat", self.pid_file], timeout=2))
                        assert actual == expected, "Redis readiness reached a different process"
                        self.server_pid = actual
                        return
                except RedisConnectionError:
                    pass
                assert time.monotonic() < deadline, "Owned Redis readiness deadline exceeded"
                time.sleep(0.05)

    def stop(self) -> None:
        assert self.process is not None
        failure = None
        forced = False
        try:
            if self.process.poll() is None:
                with Redis(host=self.host, port=self.port, socket_connect_timeout=1, socket_timeout=1) as client:
                    assert int(client.info("server")["process_id"]) == self.server_pid, "Redis ownership changed before shutdown"
                    client.shutdown(nosave=True)
        except Exception as error:
            failure = error
        finally:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                forced = True
                self.signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.signal(signal.SIGKILL)
                    self.process.wait(timeout=3)
        self.process = None
        self.server_pid = None
        with Redis(host=self.host, port=self.port, socket_connect_timeout=0.2, socket_timeout=0.2) as client:
            try:
                client.ping()
            except RedisConnectionError:
                stopped = True
            else:
                stopped = False
        assert stopped, "Owned Redis still serves after shutdown"
        assert failure is None and not forced, f"Owned Redis required shutdown recovery: {failure!r}"

    def signal(self, action: signal.Signals) -> None:
        assert self.process is not None
        if self.command[0] != "docker":
            self.process.send_signal(action)
            return
        pid: Final = int(subprocess.check_output(["docker", "exec", "redis-cache", "cat", self.pid_file], timeout=2))
        command: Final = subprocess.check_output(["docker", "exec", "redis-cache", "cat", f"/proc/{pid}/cmdline"], timeout=2)
        assert self.pid_file.encode() in command, "Redis process ownership changed"
        subprocess.run(["docker", "exec", "redis-cache", "kill", f"-{int(action)}", str(pid)], check=True, timeout=2)


@contextmanager
def owned_redis(directory: Path) -> Iterator[OwnedRedis]:
    binary: Final = shutil.which("redis-server")
    if binary:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        host = "127.0.0.1"
        prefix = (binary,)
    else:
        host = subprocess.check_output(["docker", "inspect", "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", "redis-cache"], text=True).strip()
        assert host, "CircleCI owned Redis container has no address"
        port = 16379
        prefix = ("docker", "exec", "redis-cache", "redis-server")
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(directory)))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "owned-redis-recovery.log").open("w") as log:
        pid_file: Final = str(directory / "owned-redis.pid") if binary else f"/tmp/integration-redis-{uuid.uuid4().hex}.pid"
        server: Final = OwnedRedis(host, port, (*prefix, "--port", str(port), "--set-proc-title", "no", "--pidfile", pid_file, "--bind", "0.0.0.0" if not binary else "127.0.0.1", "--protected-mode", "no", "--save", "", "--appendonly", "no"), log, pid_file)
        try:
            server.start()
            yield server
        finally:
            if server.process is not None:
                server.stop()
