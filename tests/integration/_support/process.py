import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import psutil
from integration._support.client import GATEWAY_LIMITS, Gateway

DB_PUSH: Final = ("--use_prisma_db_push",)
MIGRATE_DEPLOY: Final = ()
LEGACY_MIGRATE_DEPLOY: Final = ("--use_legacy_migration_resolver",)


def proxy_database_environment() -> Mapping[str, str]:
    writer: Final = os.environ.get("INTEGRATION_PROXY_DATABASE_URL", "")
    reader: Final = os.environ.get("INTEGRATION_PROXY_READ_REPLICA_URL", "")
    return MappingProxyType(
        {
            **({"DATABASE_URL": writer} if writer else {}),
            **({"DATABASE_URL_READ_REPLICA": reader} if reader else {}),
        }
    )


def in_group(process: psutil.Process, group: int) -> bool:
    try:
        return os.getpgid(process.pid) == group
    except ProcessLookupError:
        return False


def group_members(group: int) -> tuple[psutil.Process, ...]:
    return tuple(process for process in psutil.process_iter() if in_group(process, group))


def signal_group(group: int, action: int) -> None:
    try:
        os.killpg(group, action)
    except ProcessLookupError:
        pass


def graceful_stop_seconds() -> float:
    return max(30.0, float(os.environ.get("INTEGRATION_PROXY_READY_SECONDS", "70")))


def stop_root_process(process: subprocess.Popen[bytes]) -> bool:
    if process.poll() is not None:
        return True
    process.terminate()
    try:
        process.wait(timeout=graceful_stop_seconds())
    except subprocess.TimeoutExpired:
        return False
    return True


@dataclass(frozen=True, slots=True)
class OwnedProxy:
    gateway: Gateway
    process: subprocess.Popen[bytes]
    log: Path


@contextmanager
def owned_proxy(
    gateway: Gateway,
    directory: Path,
    overrides: Mapping[str, str],
    *,
    config: Path | None = None,
    remove_environment: tuple[str, ...] = (),
    workers: int = 1,
    database_setup: tuple[str, ...] = DB_PUSH,
) -> Iterator[Gateway]:
    with owned_proxy_process(
        gateway,
        directory,
        overrides,
        config=config,
        remove_environment=remove_environment,
        workers=workers,
        database_setup=database_setup,
    ) as owned:
        yield owned.gateway


def _stop(process: subprocess.Popen[bytes]) -> None:
    root_stopped: Final = stop_root_process(process)
    residual: Final = group_members(process.pid)
    if residual:
        signal_group(process.pid, signal.SIGTERM)
        psutil.wait_procs(residual, timeout=5)
    remaining: Final = group_members(process.pid)
    if remaining:
        signal_group(process.pid, signal.SIGKILL)
        psutil.wait_procs(remaining, timeout=3)
    process.wait(timeout=3)
    survivors: Final = group_members(process.pid)
    assert not survivors, "Owned proxy child survived cleanup"
    assert root_stopped and not remaining, "Owned proxy required forced cleanup"


_PORT_ATTEMPTS: Final = 3


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return reserve.getsockname()[1]


@dataclass(frozen=True, slots=True)
class _Launch:
    process: subprocess.Popen[bytes]
    port: int
    log: Path


def _launch(command: tuple[str, ...], root: Path, environment: Mapping[str, str], output: Path) -> _Launch:
    port: Final = _free_port()
    log_path: Final = output / f"owned-proxy-{uuid.uuid4().hex}.log"
    with log_path.open("w") as log:
        process: Final = subprocess.Popen(
            [*command, "--port", str(port)],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return _Launch(process, port, log_path)


def _lost_port_race(launch: _Launch) -> bool:
    return launch.process.poll() is not None and "address already in use" in launch.log.read_text()


def _wait_until_ready(launch: _Launch) -> None:
    with httpx.Client(base_url=f"http://127.0.0.1:{launch.port}", timeout=15, trust_env=False) as client:
        deadline: Final = time.monotonic() + float(os.environ.get("INTEGRATION_PROXY_READY_SECONDS", "70"))
        while launch.process.poll() is None:
            try:
                if client.get("/health/readiness", timeout=2).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline, "Owned proxy readiness deadline exceeded"
            time.sleep(0.1)


def _launch_until_bound(
    command: tuple[str, ...], root: Path, environment: Mapping[str, str], output: Path, attempts: int
) -> _Launch:
    launch: Final = _launch(command, root, environment, output)
    try:
        _wait_until_ready(launch)
        assert launch.process.poll() is None or (attempts > 1 and _lost_port_race(launch)), (
            "Owned proxy exited before readiness"
        )
    except BaseException:
        _stop(launch.process)
        raise
    if launch.process.poll() is None:
        return launch
    _stop(launch.process)
    return _launch_until_bound(command, root, environment, output, attempts - 1)


@contextmanager
def owned_proxy_process(
    gateway: Gateway,
    directory: Path,
    overrides: Mapping[str, str],
    *,
    config: Path | None = None,
    remove_environment: tuple[str, ...] = (),
    workers: int = 1,
    database_setup: tuple[str, ...] = DB_PUSH,
    extra_arguments: tuple[str, ...] = (),
) -> Iterator[OwnedProxy]:
    root: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
    environment: Final = {
        **{
            name: value
            for name, value in {**os.environ, **proxy_database_environment()}.items()
            if name not in remove_environment
        },
        "LITELLM_MASTER_KEY": gateway.key,
        "LITELLM_SALT_KEY": os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt"),
        "STORE_MODEL_IN_DB": "True",
        **overrides,
    }
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(directory)))
    output.mkdir(parents=True, exist_ok=True)
    command: Final = (
        sys.executable,
        "-m",
        "integration._support.proxy",
        "--config",
        str(config or "tests/integration/proxy_config.yaml"),
        "--host",
        "127.0.0.1",
        "--num_workers",
        str(workers),
        *database_setup,
        *extra_arguments,
    )
    launch: Final = _launch_until_bound(command, root, environment, output, _PORT_ATTEMPTS)
    process: Final = launch.process
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{launch.port}", timeout=15, trust_env=False, limits=GATEWAY_LIMITS
        ) as client:
            yield OwnedProxy(Gateway(client, gateway.key, gateway.upstream_url), process, launch.log)
    finally:
        _stop(process)


_UPSTREAM_READY_SECONDS: Final = 60


class UpstreamSlot:
    """A scripted upstream a test module owns on a fixed port, so a cell can take it down and bring it back."""

    __slots__ = ("directory", "port", "process", "root")

    def __init__(self, directory: Path, port: int, root: Path) -> None:
        self.directory = directory
        self.port = port
        self.root = root
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        assert self.process is None, "Owned upstream is already running"
        output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR") or self.directory)
        log_path: Final = output / f"owned-upstream-{self.port}-{uuid.uuid4().hex}.log"
        with log_path.open("w") as log:
            process: Final = subprocess.Popen(
                [sys.executable, "-m", "integration._support.upstream", "--port", str(self.port)],
                cwd=self.root,
                env=dict(os.environ),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.process = process
        deadline: Final = time.monotonic() + _UPSTREAM_READY_SECONDS
        while process.poll() is None:
            try:
                if httpx.get(f"{self.url}/health", timeout=2, trust_env=False).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline, f"Owned upstream readiness deadline exceeded: {log_path}"
            time.sleep(0.1)
        raise AssertionError(f"Owned upstream exited before readiness: {log_path}")

    def stop(self) -> None:
        process: Final = self.process
        assert process is not None, "Owned upstream is not running"
        self.process = None
        _stop(process)


@contextmanager
def owned_upstream(directory: Path) -> Generator[UpstreamSlot]:
    root: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
    slot: Final = UpstreamSlot(directory, _free_port(), root)
    slot.start()
    try:
        yield slot
    finally:
        if slot.process is not None:
            slot.stop()
