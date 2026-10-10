import errno
import json
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
from datetime import UTC, datetime, timedelta
from ipaddress import IPv4Address
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
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
    return _launch_until_bound(command, root, environment, output, attempts - 1)


def _proxy_root() -> Path:
    return Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])


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
                hook for hook in ("integration._support.runtime:configure_executor", configured_hooks) if hook
            ),
        }
    )


def setup_only_proxy_run(
    gateway: Gateway, overrides: Mapping[str, str], *, config: Path, workers: int
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
            *DB_PUSH,
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
) -> Iterator[OwnedProxy]:
    root: Final = _proxy_root()
    environment: Final = _proxy_environment(gateway, overrides, remove_environment)
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
        "--timeout_worker_healthcheck",
        "5",
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
    launch: Final = _launch_until_bound(command, root, environment, output, _PORT_ATTEMPTS)
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
            deadline: Final = time.monotonic() + 70
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
def owned_upstream(directory: Path, *, tls: bool = False) -> Generator[UpstreamSlot]:
    certificate: Final = self_signed_certificate(directory) if tls else None
    slot: Final = UpstreamSlot(directory, _free_port(), _harness_root(), certificate)
    try:
        slot.start()
        yield slot
    finally:
        if slot.process is not None:
            slot.stop()
