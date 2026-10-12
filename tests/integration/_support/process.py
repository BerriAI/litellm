import errno
import json
import threading
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from datetime import UTC, datetime, timedelta
from ipaddress import IPv4Address
from itertools import accumulate
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from pydantic import TypeAdapter

from tests.integration._support.budget import case_timeout_seconds, import_seconds, readiness_for, readiness_seconds
from tests.integration._support.client import GATEWAY_LIMITS, Gateway

DB_PUSH: Final = ("--use_prisma_db_push",)
MIGRATE_DEPLOY: Final = ()
LEGACY_MIGRATE_DEPLOY: Final = ("--use_legacy_migration_resolver",)
CONFIGURATION: Final = TypeAdapter(dict[str, object])
MCP_COVERAGE_CONFIG: Final = Path(__file__).resolve().parents[1] / "mcp_coverage.toml"


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
    return readiness_for(import_seconds(), 1)


def stop_root_process(process: subprocess.Popen[bytes]) -> bool:
    if process.poll() is not None:
        return True
    process.terminate()
    try:
        process.wait(timeout=graceful_stop_seconds())
    except subprocess.TimeoutExpired:
        return False
    return True


class WorkerHealthcheck(Enum):
    STRICT = "strict"
    LENIENT = "lenient"


STRICT_HEALTHCHECK_SECONDS: Final = 5
WORKER_ENTRYPOINT: Final = "spawn_main"


def healthcheck_seconds(healthcheck: WorkerHealthcheck) -> int:
    match healthcheck:
        case WorkerHealthcheck.STRICT:
            return STRICT_HEALTHCHECK_SECONDS
        case WorkerHealthcheck.LENIENT:
            return int(case_timeout_seconds())


def _is_serving_worker(process: psutil.Process) -> bool:
    try:
        return _ready_worker_pid(process) is not None and WORKER_ENTRYPOINT in " ".join(process.cmdline())
    except psutil.Error:
        return False


def _is_gone(process: psutil.Process) -> bool:
    try:
        return process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _wait_until_gone(process: psutil.Process, seconds: float) -> None:
    deadline: Final = time.monotonic() + seconds
    while not _is_gone(process):
        assert time.monotonic() < deadline, f"Process {process.pid} is still running after {seconds:.0f}s"
        time.sleep(0.05)


@dataclass(frozen=True, slots=True)
class WorkerKill:
    victim: psutil.Process
    fired: threading.Event

    def freeze(self) -> psutil.Process:
        self.victim.suspend()
        return self.victim

    def __call__(self) -> None:
        assert not self.fired.is_set(), f"Worker {self.victim.pid} was already killed"
        with suppress(psutil.NoSuchProcess):
            self.victim.kill()
        _wait_until_gone(self.victim, STRICT_HEALTHCHECK_SECONDS)
        self.fired.set()


def replacement_pid(before: frozenset[int], answered: int | None, serving: frozenset[int]) -> int | None:
    return answered if answered is not None and answered not in before and answered in serving else None


@dataclass(frozen=True, slots=True)
class OwnedProxy:
    gateway: Gateway
    process: subprocess.Popen[bytes]
    log: Path
    healthcheck: WorkerHealthcheck = WorkerHealthcheck.STRICT

    def worker_pids(self) -> frozenset[int]:
        try:
            children: Final = psutil.Process(self.process.pid).children()
        except psutil.NoSuchProcess:
            return frozenset()
        return frozenset(child.pid for child in children if _is_serving_worker(child))

    def _answering_pid(self) -> int | None:
        try:
            with httpx.Client(
                base_url=str(self.gateway.client.base_url),
                timeout=2,
                trust_env=False,
                limits=httpx.Limits(max_keepalive_connections=0),
            ) as client:
                response: Final = client.get("/health/integration-readiness")
        except httpx.TransportError:
            return None
        identity: Final = response.headers.get("x-integration-worker-pid", "")
        return int(identity) if response.status_code == 200 and identity.isdecimal() else None

    def _await_replacement(self, before: frozenset[int]) -> int:
        budget: Final = readiness_seconds(1)
        deadline: Final = time.monotonic() + budget
        while (replacement := replacement_pid(before, self._answering_pid(), self.worker_pids())) is None:
            assert self.process.poll() is None, "Owned proxy exited while a worker was being replaced"
            assert time.monotonic() < deadline, f"No replacement worker served within {budget:.0f}s"
            time.sleep(0.05)
        return replacement

    def _victim(self, pid: int) -> tuple[psutil.Process, frozenset[int]]:
        before: Final = self.worker_pids()
        assert pid in before, f"{pid} is not a serving worker of this proxy: {sorted(before)}"
        return psutil.Process(pid), before

    @contextmanager
    def replacing_worker(self, pid: int) -> Iterator[WorkerKill]:
        victim, before = self._victim(pid)
        kill: Final = WorkerKill(victim, threading.Event())
        yield kill
        assert kill.fired.is_set(), f"replacing_worker({pid}) exited without killing the worker"
        self._await_replacement(before)

    def kill_worker(self, pid: int) -> int:
        victim, before = self._victim(pid)
        WorkerKill(victim, threading.Event())()
        return self._await_replacement(before)

    @contextmanager
    def suspended(self, pid: int) -> Iterator[psutil.Process]:
        assert self.healthcheck is WorkerHealthcheck.LENIENT, (
            "uvicorn kills a suspended worker after the strict worker healthcheck; "
            "launch this proxy with healthcheck=WorkerHealthcheck.LENIENT"
        )
        assert pid == self.process.pid or pid in self.worker_pids(), f"{pid} is not part of this proxy"
        target: Final = psutil.Process(pid)
        target.suspend()
        try:
            yield target
        finally:
            with suppress(psutil.NoSuchProcess):
                target.resume()
        assert not _is_gone(target), (
            f"Process {pid} died while suspended under a {healthcheck_seconds(self.healthcheck)}s worker healthcheck"
        )

    def stall_worker(self, pid: int) -> int:
        assert self.healthcheck is WorkerHealthcheck.STRICT, "Only the strict worker healthcheck replaces a stall"
        victim, before = self._victim(pid)
        victim.suspend()
        try:
            replacement: Final = self._await_replacement(before)
            _wait_until_gone(victim, readiness_seconds(1))
            return replacement
        finally:
            with suppress(psutil.NoSuchProcess):
                victim.kill()

    def begin_shutdown(self) -> None:
        self.process.terminate()

    def kill_supervisor(self) -> int:
        self.process.kill()
        return self.process.wait(timeout=STRICT_HEALTHCHECK_SECONDS)


def stop_child(process: subprocess.Popen[bytes] | subprocess.Popen[str], grace_seconds: float = 10) -> int:
    if process.poll() is None:
        process.terminate()
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=grace_seconds)


def stop_processes(processes: Iterable[psutil.Process], grace_seconds: float = 5) -> None:
    members: Final = tuple(processes)
    for member in members:
        with suppress(psutil.NoSuchProcess):
            member.terminate()
    _stopped, alive = psutil.wait_procs(members, timeout=grace_seconds)
    for member in alive:
        with suppress(psutil.NoSuchProcess):
            member.kill()
    psutil.wait_procs(alive, timeout=grace_seconds)


def stop_process_group(process: subprocess.Popen[bytes] | subprocess.Popen[str], grace_seconds: float) -> None:
    if process.poll() is None:
        signal_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        signal_group(process.pid, signal.SIGKILL)
        process.wait(timeout=grace_seconds)


def kill_stray_process(pid: int) -> None:
    with suppress(psutil.NoSuchProcess):
        stray: Final = psutil.Process(pid)
        stray.kill()
        _wait_until_gone(stray, STRICT_HEALTHCHECK_SECONDS)


@contextmanager
def paused(process: subprocess.Popen[bytes] | subprocess.Popen[str]) -> Iterator[None]:
    target: Final = psutil.Process(process.pid)
    target.suspend()
    try:
        yield
    finally:
        with suppress(psutil.NoSuchProcess):
            target.resume()


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
    coverage_file: Path | None = None,
) -> Iterator[Gateway]:
    with owned_proxy_process(
        gateway,
        directory,
        overrides,
        config=config,
        remove_environment=remove_environment,
        workers=workers,
        database_setup=database_setup,
        coverage_file=coverage_file,
    ) as owned:
        yield owned.gateway


def _stop(process: subprocess.Popen[bytes]) -> None:
    try:
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
    except BaseException:
        signal_group(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
        psutil.wait_procs(group_members(process.pid), timeout=3)
        assert not group_members(process.pid), "Owned proxy child survived emergency cleanup"
        raise


_PORT_ATTEMPTS: Final = 3
_BIND_COLLISION: Final = os.strerror(errno.EADDRINUSE).lower()


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return reserve.getsockname()[1]


@dataclass(frozen=True, slots=True)
class _Launch:
    process: subprocess.Popen[bytes]
    port: int
    log: Path
    started: float


def _stop_launch(launch: _Launch) -> None:
    started: Final = time.monotonic()
    exit_before: Final = launch.process.poll()
    exception_before: Final = sys.exc_info()[0]
    try:
        _stop(launch.process)
    finally:
        exception_at_record: Final = sys.exc_info()[0]
        launch.log.with_suffix(".exit.json").write_text(
            json.dumps(
                {
                    "root_pid": launch.process.pid,
                    "exit_before_stop": exit_before,
                    "exit_after_stop": launch.process.poll(),
                    "exception_before_stop": exception_before.__name__ if exception_before else None,
                    "exception_at_record": exception_at_record.__name__ if exception_at_record else None,
                    "cleanup_seconds": time.monotonic() - started,
                    "process_lifetime_seconds": time.monotonic() - launch.started,
                },
                indent=2,
            )
            + "\n"
        )


def _launch(command: tuple[str, ...], root: Path, environment: Mapping[str, str], output: Path) -> _Launch:
    port: Final = _free_port()
    log_path: Final = output / f"owned-proxy-{uuid.uuid4().hex}.log"
    started: Final = time.monotonic()
    with log_path.open("w") as log:
        process: Final = subprocess.Popen(
            [*command, "--port", str(port)],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return _Launch(process, port, log_path, started)


def _lost_port_race(exit_code: int | None, log: Path) -> bool:
    return exit_code is not None and _BIND_COLLISION in log.read_text().lower()


def _owns_listener(launch: _Launch) -> bool:
    for member in group_members(launch.process.pid):
        try:
            if any(
                connection.status == psutil.CONN_LISTEN and connection.laddr.port == launch.port
                for connection in member.net_connections(kind="inet")
            ):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def _readiness_pids(launch: _Launch, workers: int) -> Iterator[int]:
    with httpx.Client(
        base_url=f"http://127.0.0.1:{launch.port}",
        timeout=15,
        trust_env=False,
        limits=httpx.Limits(max_keepalive_connections=0),
    ) as client:
        deadline: Final = time.monotonic() + readiness_seconds(workers)
        while launch.process.poll() is None:
            try:
                response: Final = client.get("/health/integration-readiness", timeout=2)
                identity: Final = response.headers.get("x-integration-worker-pid", "")
                if response.status_code == 200 and identity.isdecimal() and _owns_listener(launch):
                    yield int(identity)
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline, "Owned proxy readiness deadline exceeded"
            time.sleep(0.1)


def _ready_workers(pids: frozenset[int], pid: int) -> frozenset[int]:
    return pids | frozenset((pid,))


def _ready_worker_pid(member: psutil.Process) -> int | None:
    try:
        status: Final = member.status()
        return (
            None
            if status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED, psutil.STATUS_TRACING_STOP)
            else member.pid
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _wait_until_ready(launch: _Launch, workers: int = 1) -> None:
    for ready in accumulate(_readiness_pids(launch, workers), _ready_workers, initial=frozenset[int]()):
        members: Final = frozenset(
            pid for member in group_members(launch.process.pid) if (pid := _ready_worker_pid(member)) is not None
        )
        if len(ready & members) >= max(1, workers):
            return


def _launch_until_bound(
    command: tuple[str, ...], root: Path, environment: Mapping[str, str], output: Path, attempts: int, workers: int
) -> _Launch:
    launch: Final = _launch(command, root, environment, output)
    try:
        _wait_until_ready(launch, workers)
        exit_code: Final = launch.process.poll()
        assert exit_code is None or (attempts > 1 and _lost_port_race(exit_code, launch.log)), (
            "Owned proxy exited before readiness"
        )
    except BaseException:
        _stop_launch(launch)
        raise
    if exit_code is None:
        launch.log.with_suffix(".startup.json").write_text(
            json.dumps(
                {
                    "root_pid": launch.process.pid,
                    "readiness_seconds": time.monotonic() - launch.started,
                    "logical_cpus": os.cpu_count(),
                }
            )
            + "\n"
        )
        return launch
    _stop_launch(launch)
    return _launch_until_bound(command, root, environment, output, attempts - 1, workers)


def _proxy_root() -> Path:
    return Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])


def worker_configuration(config: Path, directory: Path) -> Path:
    if os.environ.get("INTEGRATION_WORKER_ISOLATION") != "1":
        return config

    def endpoint(match: re.Match[str]) -> str:
        if match.group(1) == "8190":
            return os.environ["INTEGRATION_UPSTREAM_URL"]
        return f"http://127.0.0.1:{os.environ['INTEGRATION_PROVIDER_PORT']}"

    contents: Final = re.sub(r"http://127[.]0[.]0[.]1:(8190|8191)(?![0-9])", endpoint, config.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    destination: Final = directory / f"worker-config-{uuid.uuid4().hex}.yaml"
    loaded: Final[object] = yaml.safe_load(contents)
    if not isinstance(loaded, dict):
        destination.write_text(contents)
        return destination
    settings: Final = CONFIGURATION.validate_python(loaded)
    general: Final = settings.get("general_settings", {})
    if not isinstance(general, dict):
        destination.write_text(contents)
        return destination
    configured: Final = {
        **settings,
        "general_settings": {"database_connection_pool_limit": 2, **CONFIGURATION.validate_python(general)},
    }
    destination.write_text(yaml.safe_dump(configured, sort_keys=False))
    return destination


def _proxy_environment(
    gateway: Gateway, overrides: Mapping[str, str], remove_environment: tuple[str, ...]
) -> Mapping[str, str]:
    inherited: Final = {**os.environ, **proxy_database_environment()}
    writer: Final = overrides.get("DATABASE_URL")
    reader: Final = inherited.get("DATABASE_URL_READ_REPLICA")
    unpaired_reader: Final = (
        writer is not None
        and reader is not None
        and "DATABASE_URL_READ_REPLICA" not in overrides
        and urlsplit(writer).path != urlsplit(reader).path
    )
    removed: Final = frozenset((*remove_environment, *(("DATABASE_URL_READ_REPLICA",) if unpaired_reader else ())))
    configured_hooks: Final = overrides.get(
        "LITELLM_WORKER_STARTUP_HOOKS",
        inherited.get("LITELLM_WORKER_STARTUP_HOOKS", "") if "LITELLM_WORKER_STARTUP_HOOKS" not in removed else "",
    )
    return MappingProxyType(
        {
            **{name: value for name, value in inherited.items() if name not in removed},
            "LITELLM_MASTER_KEY": gateway.key,
            "LITELLM_SALT_KEY": os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt"),
            "STORE_MODEL_IN_DB": "True",
            "PYTHON_DOTENV_DISABLED": "1",
            **overrides,
            "LITELLM_WORKER_STARTUP_HOOKS": ",".join(
                hook
                for hook in (
                    "integration._support.runtime:configure_executor",
                    "integration._support.runtime:configure_readiness",
                    configured_hooks,
                )
                if hook
            ),
        }
    )


def setup_only_proxy_run(
    gateway: Gateway,
    overrides: Mapping[str, str],
    *,
    config: Path,
    workers: int,
    database_setup: tuple[str, ...] = DB_PUSH,
) -> subprocess.CompletedProcess[str]:
    """The proxy CLI's `--skip_server_startup` pass (the image's setup step), run to completion with the
    environment an owned proxy gets."""
    return subprocess.run(  # test-quality-ok: the checkout at the working directory is the proxy under test
        (
            sys.executable,
            "-m",
            "integration._support.proxy",
            "--config",
            str(config),
            "--num_workers",
            str(workers),
            *database_setup,
            "--skip_server_startup",
        ),
        cwd=_proxy_root(),
        env=dict(_proxy_environment(gateway, overrides, ())),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


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
    coverage_file: Path | None = None,
    healthcheck: WorkerHealthcheck = WorkerHealthcheck.STRICT,
) -> Iterator[OwnedProxy]:
    assert "--timeout_worker_healthcheck" not in extra_arguments, "Choose the worker healthcheck with `healthcheck`"
    root: Final = _proxy_root()
    environment: Final = _proxy_environment(
        gateway, {**overrides, **({"COVERAGE_FILE": str(coverage_file)} if coverage_file else {})}, remove_environment
    )
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(directory)))
    output.mkdir(parents=True, exist_ok=True)
    command: Final = (
        sys.executable,
        "-m",
        *(("coverage", "run", f"--rcfile={MCP_COVERAGE_CONFIG}", "-m") if coverage_file else ()),
        "integration._support.proxy",
        "--config",
        str(worker_configuration(config or root / "tests/integration/proxy_config.yaml", directory)),
        "--host",
        "127.0.0.1",
        "--num_workers",
        str(workers),
        "--timeout_worker_healthcheck",
        str(healthcheck_seconds(healthcheck)),
        *database_setup,
        *extra_arguments,
    )
    launch: Final = _launch_until_bound(command, root, environment, output, _PORT_ATTEMPTS, workers)
    process: Final = launch.process
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{launch.port}", timeout=15, trust_env=False, limits=GATEWAY_LIMITS
        ) as client:
            yield OwnedProxy(Gateway(client, gateway.key, gateway.upstream_url), process, launch.log, healthcheck)
    finally:
        _stop_launch(launch)


@contextmanager
def owned_gateway_image(
    gateway: Gateway, directory: Path, overrides: Mapping[str, str], *, config: Path, workers: int
) -> Iterator[OwnedProxy]:
    """The componentized gateway started the way its image starts it: `docker/component_entrypoint.sh` running
    `python -m gateway.launch`, with the config handed over as `CONFIG_FILE_PATH`. It serves the data plane only,
    so keys come from a proxy that shares its database."""
    root: Final = _proxy_root()
    environment: Final = _proxy_environment(gateway, {**overrides, "CONFIG_FILE_PATH": str(config)}, ())
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(directory)))
    output.mkdir(parents=True, exist_ok=True)
    command: Final = (
        str(root / "docker" / "component_entrypoint.sh"),
        sys.executable,
        "-m",
        "gateway.launch",
        "--workers",
        str(workers),
        "--host",
        "127.0.0.1",
        "--timeout-worker-healthcheck",
        str(int(graceful_stop_seconds())),
    )
    launch: Final = _launch_until_bound(command, root, environment, output, _PORT_ATTEMPTS, workers)
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{launch.port}", timeout=15, trust_env=False, limits=GATEWAY_LIMITS
        ) as client:
            yield OwnedProxy(Gateway(client, gateway.key, gateway.upstream_url), launch.process, launch.log)
    finally:
        _stop_launch(launch)


def _is_ready(client: httpx.Client) -> bool:
    try:
        return client.get("/health/readiness", timeout=2).status_code == 200
    except httpx.TransportError:
        return False


def refused_boot_log(
    gateway: Gateway,
    directory: Path,
    overrides: Mapping[str, str],
    *,
    config: Path | None = None,
) -> str:
    """Start the proxy and return its log once it exits non-zero instead of becoming ready."""
    root: Final = _proxy_root()
    environment: Final = _proxy_environment(gateway, overrides, ())
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
        "1",
        *DB_PUSH,
    )
    launch: Final = _launch(command, root, environment, output)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{launch.port}", timeout=15, trust_env=False) as client:
            deadline: Final = time.monotonic() + readiness_seconds(1)
            while launch.process.poll() is None:
                assert not _is_ready(client), (
                    f"Proxy became ready instead of refusing to boot:\n{launch.log.read_text()}"
                )
                assert time.monotonic() < deadline, "Proxy neither exited nor became ready within the deadline"
                time.sleep(0.1)
        assert launch.process.returncode != 0, f"Proxy exited 0 instead of refusing to boot:\n{launch.log.read_text()}"
        return launch.log.read_text()
    finally:
        _stop_launch(launch)


_UPSTREAM_READY_SECONDS: Final = 60
_LOOPBACK: Final = "127.0.0.1"


@dataclass(frozen=True, slots=True)
class UpstreamCertificate:
    certificate: Path
    key: Path


def self_signed_certificate(directory: Path) -> UpstreamCertificate:
    """A one-day self-signed certificate for 127.0.0.1, for an owned upstream a provider only reaches over TLS."""
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _LOOPBACK)])
    issued: Final = datetime.now(UTC)
    certificate: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(issued - timedelta(minutes=5))
        .not_valid_after(issued + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(IPv4Address(_LOOPBACK))]), critical=False)
        .sign(private_key, hashes.SHA256())
    )
    certificate_path: Final = directory / f"owned-upstream-{uuid.uuid4().hex}.crt"
    key_path: Final = directory / f"owned-upstream-{uuid.uuid4().hex}.key"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    return UpstreamCertificate(certificate_path, key_path)


class UpstreamSlot:
    """A scripted upstream a test module owns on a fixed port, so a cell can take it down and bring it back.

    It runs from the harness's own checkout, never ``INTEGRATION_PROXY_ROOT``: the double belongs to the tests,
    the proxy root only names the litellm under test."""

    __slots__ = ("certificate", "directory", "port", "process", "root")

    def __init__(self, directory: Path, port: int, root: Path, certificate: UpstreamCertificate | None = None) -> None:
        self.directory = directory
        self.port = port
        self.root = root
        self.certificate = certificate
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def url(self) -> str:
        scheme: Final = "http" if self.certificate is None else "https"
        return f"{scheme}://{_LOOPBACK}:{self.port}"

    def _tls_arguments(self) -> tuple[str, ...]:
        if self.certificate is None:
            return ()
        return ("--ssl-certfile", str(self.certificate.certificate), "--ssl-keyfile", str(self.certificate.key))

    def start(self) -> None:
        assert self.process is None, "Owned upstream is already running"
        output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR") or self.directory)
        log_path: Final = output / f"owned-upstream-{self.port}-{uuid.uuid4().hex}.log"
        with log_path.open("w") as log:
            process: Final = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "integration._support.upstream",
                    "--port",
                    str(self.port),
                    *self._tls_arguments(),
                ],
                cwd=self.root,
                env=dict(os.environ),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.process = process
        try:
            deadline: Final = time.monotonic() + _UPSTREAM_READY_SECONDS
            while process.poll() is None:
                try:
                    probe: Final = httpx.get(
                        f"{self.url}/health", timeout=2, trust_env=False, verify=self.certificate is None
                    )
                    if probe.status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                assert time.monotonic() < deadline, f"Owned upstream readiness deadline exceeded: {log_path}"
                time.sleep(0.1)
            raise AssertionError(f"Owned upstream exited before readiness: {log_path}")
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        process: Final = self.process
        assert process is not None, "Owned upstream is not running"
        self.process = None
        _stop(process)


def _harness_root() -> Path:
    return Path(__file__).resolve().parents[3]


@contextmanager
def owned_upstream(directory: Path, *, tls: bool = False, port: int | None = None) -> Generator[UpstreamSlot]:
    certificate: Final = self_signed_certificate(directory) if tls else None
    slot: Final = UpstreamSlot(directory, port or _free_port(), _harness_root(), certificate)
    try:
        slot.start()
        yield slot
    finally:
        if slot.process is not None:
            slot.stop()
