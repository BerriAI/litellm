import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

PORT_ATTEMPTS: Final = 5


@dataclass(frozen=True, slots=True)
class PortUnavailable:
    reason: str


def free_loopback_port() -> int:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


@dataclass
class OwnedRedis:
    host: str
    port: int
    command: tuple[str, ...]
    log: TextIO
    pid_file: str
    process: subprocess.Popen | None = None
    server_pid: int | None = None

    def serves(self) -> bool:
        with Redis(host=self.host, port=self.port, socket_connect_timeout=0.2, socket_timeout=0.2) as client:
            try:
                return bool(client.ping())
            except RedisConnectionError:
                return False

    def owned_pid(self) -> int | None:
        assert self.process is not None
        if self.command[0] != "docker":
            return self.process.pid
        result: Final = subprocess.run(
            ["docker", "exec", "redis-cache", "cat", self.pid_file],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            return None
        return int(result.stdout)

    def start(self) -> None:
        unavailable: Final = self.try_start()
        assert unavailable is None, unavailable.reason

    def try_start(self) -> PortUnavailable | None:
        assert self.process is None
        if self.serves():
            return PortUnavailable(f"Another Redis already serves {self.host}:{self.port} before the owned one starts")
        process: Final = subprocess.Popen(
            self.command, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True
        )
        self.process = process
        deadline: Final = time.monotonic() + 8
        with Redis(host=self.host, port=self.port, socket_connect_timeout=0.2, socket_timeout=0.2) as client:
            while True:
                code: Final = process.poll()
                if code is not None:
                    self.process = None
                    return PortUnavailable(
                        f"Owned Redis exited with code {code} before readiness on {self.host}:{self.port}"
                    )
                try:
                    if client.ping():
                        actual: Final = int(client.info("server")["process_id"])
                        expected: Final = self.owned_pid()
                        if actual == expected:
                            self.server_pid = actual
                            return None
                        try:
                            process.wait(timeout=max(deadline - time.monotonic(), 0))
                        except subprocess.TimeoutExpired as error:
                            raise AssertionError("Redis readiness reached a different process") from error
                        self.process = None
                        return PortUnavailable(f"A different Redis answered on {self.host}:{self.port} during startup")
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
                    assert int(client.info("server")["process_id"]) == self.server_pid, (
                        "Redis ownership changed before shutdown"
                    )
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
        assert not self.serves(), "Owned Redis still serves after shutdown"
        assert failure is None and not forced, f"Owned Redis required shutdown recovery: {failure!r}"

    def signal(self, action: signal.Signals) -> None:
        assert self.process is not None
        if self.command[0] != "docker":
            self.process.send_signal(action)
            return
        pid: Final = int(subprocess.check_output(["docker", "exec", "redis-cache", "cat", self.pid_file], timeout=2))
        command: Final = subprocess.check_output(
            ["docker", "exec", "redis-cache", "cat", f"/proc/{pid}/cmdline"], timeout=2
        )
        assert self.pid_file.encode() in command, "Redis process ownership changed"
        subprocess.run(["docker", "exec", "redis-cache", "kill", f"-{int(action)}", str(pid)], check=True, timeout=2)

    @contextmanager
    def paused(self) -> Iterator[None]:
        self.signal(signal.SIGSTOP)
        try:
            yield
        finally:
            self.signal(signal.SIGCONT)


@contextmanager
def _started(
    build: Callable[[int], OwnedRedis],
    choose_port: Callable[[], int],
    attempts: int,
    log_name: str,
) -> Iterator[OwnedRedis]:
    server: Final = build(choose_port())
    try:
        unavailable: Final = server.try_start()
        if unavailable is None:
            yield server
            return
        assert attempts > 1, (
            f"Owned Redis found no usable port after {PORT_ATTEMPTS} attempts: {unavailable.reason} (log {log_name})"
        )
        with _started(build, choose_port, attempts - 1, log_name) as started:
            yield started
    finally:
        if server.process is not None:
            server.stop()


@contextmanager
def owned_redis(directory: Path, choose_port: Callable[[], int] = free_loopback_port) -> Iterator[OwnedRedis]:
    binary: Final = shutil.which("redis-server")
    if binary:
        host: Final = "127.0.0.1"
        prefix: Final = (binary,)
    else:
        host: Final = subprocess.check_output(
            [
                "docker",
                "inspect",
                "--format",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                "redis-cache",
            ],
            text=True,
        ).strip()
        assert host, "CircleCI owned Redis container has no address"
        prefix: Final = ("docker", "exec", "redis-cache", "redis-server")
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(directory)))
    output.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    with (output / f"owned-redis-{uuid.uuid4().hex}.log").open("w") as log:

        def build(port: int) -> OwnedRedis:
            pid_file: Final = (
                str(directory / f"owned-redis-{port}.pid")
                if binary
                else f"/tmp/integration-redis-{uuid.uuid4().hex}.pid"
            )
            command: Final = (
                *prefix,
                "--port",
                str(port),
                "--set-proc-title",
                "no",
                "--pidfile",
                pid_file,
                "--bind",
                "0.0.0.0" if not binary else "127.0.0.1",
                "--protected-mode",
                "no",
                "--save",
                "",
                "--appendonly",
                "no",
            )
            return OwnedRedis(host, port, command, log, pid_file)

        with _started(build, choose_port, PORT_ATTEMPTS, log.name) as server:
            yield server
