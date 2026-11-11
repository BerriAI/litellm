import json
import threading
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from datetime import UTC, datetime, timedelta
from ipaddress import IPv4Address
from itertools import accumulate
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias
from urllib.parse import urlsplit

import httpx
import psutil
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from pydantic import TypeAdapter

from tests.integration._support.budget import import_seconds, readiness_for, readiness_seconds
from tests.integration._support.client import GATEWAY_LIMITS, Gateway, gateway_from_environment
from tests.integration._support.database import scratch_database
from tests.integration._support.ports import allocate_port
from tests.integration._support.redis_process import owned_redis
from tests.integration._support.workers import gateway_profile, replica_urls

if TYPE_CHECKING:
    import pytest

DB_PUSH: Final = ("--use_prisma_db_push",)
MIGRATE_DEPLOY: Final = ()
LEGACY_MIGRATE_DEPLOY: Final = ("--use_legacy_migration_resolver",)
CONFIGURATION: Final = TypeAdapter(dict[str, object])
_DEPLOYMENT_IDS: Final = TypeAdapter(dict[str, list[str]])
MCP_COVERAGE_CONFIG: Final = Path(__file__).resolve().parents[1] / "mcp_coverage.toml"
OWNED_STATE_VARIABLE: Final = "INTEGRATION_OWNED_STATE"


class OwnedState(Enum):
    PRIVATE = "private"
    SHARED_WITH_WORKER = "shared_with_worker"


@dataclass(frozen=True, slots=True)
class PrivateState:
    database_url: str
    environment: Mapping[str, str]


PrivateStateFactory: TypeAlias = Callable[[Path, Mapping[str, str]], AbstractContextManager[PrivateState]]


def boots_private_state(
    state: OwnedState | None,
    overrides: Mapping[str, str],
    remove_environment: tuple[str, ...],
    inherited: Mapping[str, str],
) -> bool:
    chosen: Final = state or OwnedState(inherited.get(OWNED_STATE_VARIABLE, OwnedState.PRIVATE.value))
    return chosen is OwnedState.PRIVATE and "DATABASE_URL" not in overrides and "DATABASE_URL" not in remove_environment


@contextmanager
def private_state(directory: Path, inherited: Mapping[str, str]) -> Iterator[PrivateState]:
    with scratch_database("owned") as database, owned_redis(directory, inherited=inherited) as cache:
        replica: Final = replica_urls(database, inherited) if gateway_profile(inherited).replica else None
        yield PrivateState(
            database,
            MappingProxyType(
                {
                    "DATABASE_URL": database if replica is None else replica["INTEGRATION_PROXY_DATABASE_URL"],
                    **(
                        {}
                        if replica is None
                        else {"DATABASE_URL_READ_REPLICA": replica["INTEGRATION_PROXY_READ_REPLICA_URL"]}
                    ),
                    "REDIS_HOST": cache.host,
                    "REDIS_PORT": str(cache.port),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class OwnedEnvironment:
    overrides: Mapping[str, str]
    database_url: str | None


@contextmanager
def owned_environment(
    directory: Path,
    overrides: Mapping[str, str],
    remove_environment: tuple[str, ...],
    state: OwnedState | None,
    inherited: Mapping[str, str],
    private: PrivateStateFactory = private_state,
) -> Iterator[OwnedEnvironment]:
    if not boots_private_state(state, overrides, remove_environment, inherited):
        yield OwnedEnvironment(overrides, None)
        return
    with private(directory, inherited) as booted:
        kept: Final = {name: value for name, value in booted.environment.items() if name not in remove_environment}
        yield OwnedEnvironment(MappingProxyType({**kept, **overrides}), booted.database_url)


def proxy_database_environment(inherited: Mapping[str, str] = os.environ) -> Mapping[str, str]:
    writer: Final = inherited.get("INTEGRATION_PROXY_DATABASE_URL", "")
    reader: Final = inherited.get("INTEGRATION_PROXY_READ_REPLICA_URL", "")
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
            return int(graceful_stop_seconds())


def _is_serving_worker(process: psutil.Process) -> bool:
    try:
        return _ready_worker_pid(process) is not None and WORKER_ENTRYPOINT in " ".join(process.cmdline())
    except psutil.Error:
        return False


def _is_gone(process: psutil.Process) -> bool:
    try:
        return process.status() == psutil.STATUS_ZOMBIE and process.num_threads() <= 1
    except psutil.NoSuchProcess:
        return True


def _wait_until_gone(process: psutil.Process, seconds: float) -> None:
    deadline: Final = time.monotonic() + seconds
    while not _is_gone(process):
        assert time.monotonic() < deadline, f"Process {process.pid} is still running after {seconds:.0f}s"
        time.sleep(0.05)


def _is_stopped(process: psutil.Process) -> bool:
    try:
        return process.status() == psutil.STATUS_STOPPED
    except psutil.NoSuchProcess:
        return False


def _suspend(process: psutil.Process) -> None:
    process.suspend()
    deadline: Final = time.monotonic() + STRICT_HEALTHCHECK_SECONDS
    while not _is_stopped(process) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _is_stopped(process):
        return
    with suppress(psutil.NoSuchProcess):
        process.resume()
    raise AssertionError(f"Process {process.pid} did not stop within {STRICT_HEALTHCHECK_SECONDS}s of SIGSTOP")


def _is_reaped(process: psutil.Process) -> bool:
    try:
        process.status()
    except psutil.NoSuchProcess:
        return True
    return False


def _wait_until_reaped(process: psutil.Process, seconds: float) -> None:
    deadline: Final = time.monotonic() + seconds
    while not _is_reaped(process):
        assert time.monotonic() < deadline, f"The supervisor did not reap worker {process.pid} within {seconds:.0f}s"
        time.sleep(0.05)


@dataclass(frozen=True, slots=True)
class WorkerKill:
    victim: psutil.Process
    fired: threading.Event

    def freeze(self) -> psutil.Process:
        _suspend(self.victim)
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

    def _deployment_ids_on_a_worker(self, model_name: str) -> tuple[int, frozenset[str]] | None:
        try:
            with httpx.Client(
                base_url=str(self.gateway.client.base_url),
                timeout=2,
                trust_env=False,
                limits=httpx.Limits(max_keepalive_connections=0),
            ) as client:
                response: Final = client.get("/health/integration-deployment-ids", params={"model_name": model_name})
        except httpx.TransportError:
            return None
        identity: Final = response.headers.get("x-integration-worker-pid", "")
        if response.status_code != 200 or not identity.isdecimal():
            return None
        return int(identity), frozenset(_DEPLOYMENT_IDS.validate_python(response.json())["model_ids"])

    def serves_only_on_every_worker(self, model_name: str, model_id: str) -> bool:
        with ThreadPoolExecutor(max_workers=8) as pool:
            answers: Final = tuple(pool.map(self._deployment_ids_on_a_worker, (model_name,) * 16))
        samples: Final = tuple(answer for answer in answers if answer is not None)
        workers: Final = self.worker_pids()
        return (
            bool(workers)
            and workers <= frozenset(pid for pid, _ in samples)
            and all(model_ids == {model_id} for _, model_ids in samples)
        )

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

    def _replacement_of(self, victim: psutil.Process, before: frozenset[int]) -> int:
        _wait_until_reaped(victim, healthcheck_seconds(self.healthcheck) + STRICT_HEALTHCHECK_SECONDS)
        return self._await_replacement(before)

    @contextmanager
    def replacing_worker(self, pid: int) -> Iterator[WorkerKill]:
        victim, before = self._victim(pid)
        kill: Final = WorkerKill(victim, threading.Event())
        yield kill
        assert kill.fired.is_set(), f"replacing_worker({pid}) exited without killing the worker"
        self._replacement_of(victim, before)

    def kill_worker(self, pid: int) -> int:
        victim, before = self._victim(pid)
        WorkerKill(victim, threading.Event())()
        return self._replacement_of(victim, before)

    @contextmanager
    def suspended(self, pid: int) -> Iterator[psutil.Process]:
        assert self.healthcheck is WorkerHealthcheck.LENIENT, (
            "uvicorn kills a suspended worker after the strict worker healthcheck; "
            "launch this proxy with healthcheck=WorkerHealthcheck.LENIENT"
        )
        assert pid == self.process.pid or pid in self.worker_pids(), f"{pid} is not part of this proxy"
        target: Final = psutil.Process(pid)
        _suspend(target)
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
    _suspend(target)
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
    inherited: Mapping[str, str] = os.environ,
    state: OwnedState | None = None,
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
        inherited=inherited,
        state=state,
    ) as owned:
        yield owned.gateway


@contextmanager
def isolated_gateway(directory: Path, environment: "pytest.MonkeyPatch") -> Iterator[Gateway]:
    with gateway_from_environment() as shared, owned_proxy(shared, directory, {}, state=OwnedState.PRIVATE) as owned:
        assert owned.database_url is not None
        environment.setenv("DATABASE_URL", owned.database_url)
        environment.setenv("INTEGRATION_PROXY_DATABASE_URL", owned.database_url)
        environment.delenv("INTEGRATION_PROXY_READ_REPLICA_URL", raising=False)
        environment.delenv("DATABASE_URL_READ_REPLICA", raising=False)
        owned.post("/config/update", {"router_settings": {"num_retries": 0}})
        yield owned


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


@dataclass(frozen=True, slots=True)
class _Launch:
    process: subprocess.Popen[bytes]
    port: int
    log: Path
    started: float


def _wall_clock(monotonic: float) -> float:
    return time.time() - (time.monotonic() - monotonic)


def _stop_launch(launch: _Launch) -> None:
    started: Final = time.monotonic()
    stop_started_at: Final = time.time()
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
                    "spawned_at": _wall_clock(launch.started),
                    "stop_started_at": stop_started_at,
                    "stopped_at": time.time(),
                },
                indent=2,
            )
            + "\n"
        )


def _launch(command: tuple[str, ...], root: Path, environment: Mapping[str, str], output: Path) -> _Launch:
    port: Final = allocate_port()
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


@dataclass(frozen=True, slots=True)
class BootPhases:
    prisma_started: float | None = None
    prisma_finished: float | None = None
    listening: float | None = None
    first_ready: float | None = None
    ready_pids: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class _Probe:
    elapsed: float
    listening: bool
    prisma: bool
    ready_pid: int | None


def _runs_prisma(process: psutil.Process) -> bool:
    try:
        return any(Path(part).name == "prisma" for part in process.cmdline())
    except psutil.Error:
        return False


def _prisma_running(launch: _Launch) -> bool:
    try:
        children: Final = psutil.Process(launch.process.pid).children()
    except psutil.Error:
        return False
    return any(_runs_prisma(child) for child in children)


def _answer(client: httpx.Client, launch: _Launch) -> tuple[bool, int | None]:
    if not _owns_listener(launch):
        return False, None
    try:
        response: Final = client.get("/health/integration-readiness", timeout=2)
    except httpx.ConnectError:
        return False, None
    except httpx.TransportError:
        return True, None
    identity: Final = response.headers.get("x-integration-worker-pid", "")
    ready: Final = response.status_code == 200 and identity.isdecimal()
    return True, int(identity) if ready else None


def _probes(launch: _Launch, workers: int) -> Iterator[_Probe]:
    with httpx.Client(
        base_url=f"http://127.0.0.1:{launch.port}",
        timeout=15,
        trust_env=False,
        limits=httpx.Limits(max_keepalive_connections=0),
    ) as client:
        deadline: Final = time.monotonic() + readiness_seconds(workers)
        while launch.process.poll() is None:
            elapsed: Final = time.monotonic() - launch.started
            listening, ready_pid = _answer(client, launch)
            yield _Probe(elapsed, listening, not listening and _prisma_running(launch), ready_pid)
            assert time.monotonic() < deadline, "Owned proxy readiness deadline exceeded"
            time.sleep(0.1)


def _first(recorded: float | None, observed: bool, elapsed: float) -> float | None:
    return recorded if recorded is not None or not observed else elapsed


def _observe(phases: BootPhases, probe: _Probe) -> BootPhases:
    prisma_started: Final = _first(phases.prisma_started, probe.prisma, probe.elapsed)
    prisma_finished: Final = _first(phases.prisma_finished, prisma_started is not None, probe.elapsed)
    return BootPhases(
        prisma_started=prisma_started,
        prisma_finished=None if probe.prisma else prisma_finished,
        listening=_first(phases.listening, probe.listening, probe.elapsed),
        first_ready=_first(phases.first_ready, probe.ready_pid is not None, probe.elapsed),
        ready_pids=phases.ready_pids | frozenset(() if probe.ready_pid is None else (probe.ready_pid,)),
    )


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


def _serves_all(launch: _Launch, phases: BootPhases, workers: int) -> bool:
    if len(phases.ready_pids) < max(1, workers):
        return False
    members: Final = frozenset(
        pid for member in group_members(launch.process.pid) if (pid := _ready_worker_pid(member)) is not None
    )
    return len(phases.ready_pids & members) >= max(1, workers)


def _wait_until_ready(launch: _Launch, workers: int = 1) -> BootPhases | None:
    return next(
        (
            phases
            for phases in accumulate(_probes(launch, workers), _observe, initial=BootPhases())
            if _serves_all(launch, phases, workers)
        ),
        None,
    )


def _launch_until_ready(
    command: tuple[str, ...], root: Path, environment: Mapping[str, str], output: Path, workers: int
) -> _Launch:
    launch: Final = _launch(command, root, environment, output)
    try:
        phases: Final = _wait_until_ready(launch, workers)
        assert launch.process.poll() is None, "Owned proxy exited before readiness"
    except BaseException:
        _stop_launch(launch)
        raise
    assert phases is not None
    launch.log.with_suffix(".startup.json").write_text(
        json.dumps(
            {
                "root_pid": launch.process.pid,
                "readiness_seconds": time.monotonic() - launch.started,
                "logical_cpus": os.cpu_count(),
                "workers": workers,
                "spawned_at": _wall_clock(launch.started),
                "phases": {
                    "prisma_started": phases.prisma_started,
                    "prisma_finished": phases.prisma_finished,
                    "listening": phases.listening,
                    "first_ready": phases.first_ready,
                },
            }
        )
        + "\n"
    )
    return launch


def _proxy_root() -> Path:
    return Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])


def _configuration_copy(config: Path, directory: Path, prefix: str) -> Path:
    checkouts: Final = (_harness_root(), _proxy_root().resolve())
    inside_checkout: Final = any(config.resolve().is_relative_to(checkout) for checkout in checkouts)
    target: Final = directory if inside_checkout else config.parent
    target.mkdir(parents=True, exist_ok=True)
    return target / f"{prefix}-{uuid.uuid4().hex}.yaml"


def private_configuration(config: Path, directory: Path) -> Path:
    loaded: Final[object] = yaml.safe_load(config.read_text())
    if not isinstance(loaded, dict):
        return config
    settings: Final = CONFIGURATION.validate_python(loaded)
    router: Final = settings.get("router_settings") or {}
    if not isinstance(router, dict) or "num_retries" in router:
        return config
    destination: Final = _configuration_copy(config, directory, "private-config")
    destination.write_text(
        yaml.safe_dump(
            {**settings, "router_settings": {**CONFIGURATION.validate_python(router), "num_retries": 0}},
            sort_keys=False,
        )
    )
    return destination


def worker_configuration(config: Path, directory: Path, inherited: Mapping[str, str] = os.environ) -> Path:
    if inherited.get("INTEGRATION_WORKER_ISOLATION") != "1":
        return config

    def endpoint(match: re.Match[str]) -> str:
        if match.group(1) == "8190":
            return inherited["INTEGRATION_UPSTREAM_URL"]
        return f"http://127.0.0.1:{inherited['INTEGRATION_PROVIDER_PORT']}"

    contents: Final = re.sub(r"http://127[.]0[.]0[.]1:(8190|8191)(?![0-9])", endpoint, config.read_text())
    destination: Final = _configuration_copy(config, directory, "worker-config")
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
    gateway: Gateway,
    overrides: Mapping[str, str],
    remove_environment: tuple[str, ...],
    source: Mapping[str, str] = os.environ,
) -> Mapping[str, str]:
    inherited: Final = {**source, **proxy_database_environment(source)}
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
            "LITELLM_SALT_KEY": source.get("LITELLM_SALT_KEY", "sk-integration-salt"),
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
    inherited: Mapping[str, str] = os.environ,
    client_timeout: float = 15,
    state: OwnedState | None = None,
) -> Iterator[OwnedProxy]:
    assert "--timeout_worker_healthcheck" not in extra_arguments, "Choose the worker healthcheck with `healthcheck`"
    root: Final = _proxy_root()
    with owned_environment(directory, overrides, remove_environment, state, inherited) as owned:
        configured: Final = worker_configuration(
            config or root / "tests/integration/proxy_config.yaml", directory, inherited
        )
        owned_configuration: Final = (
            configured if owned.database_url is None else private_configuration(configured, directory)
        )
        environment: Final = _proxy_environment(
            gateway,
            {**owned.overrides, **({"COVERAGE_FILE": str(coverage_file)} if coverage_file else {})},
            remove_environment,
            inherited,
        )
        output: Final = Path(inherited.get("INTEGRATION_RESULTS_DIR", str(directory)))
        output.mkdir(parents=True, exist_ok=True)
        command: Final = (
            sys.executable,
            "-m",
            *(("coverage", "run", f"--rcfile={MCP_COVERAGE_CONFIG}", "-m") if coverage_file else ()),
            "integration._support.proxy",
            "--config",
            str(owned_configuration),
            "--host",
            "127.0.0.1",
            "--num_workers",
            str(workers),
            "--timeout_worker_healthcheck",
            str(healthcheck_seconds(healthcheck)),
            *database_setup,
            *extra_arguments,
        )
        launch: Final = _launch_until_ready(command, root, environment, output, workers)
        process: Final = launch.process
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{launch.port}",
                timeout=client_timeout,
                trust_env=False,
                limits=GATEWAY_LIMITS,
            ) as client:
                yield OwnedProxy(
                    Gateway(client, gateway.key, gateway.upstream_url, owned.database_url, workers),
                    process,
                    launch.log,
                    healthcheck,
                )
        finally:
            _stop_launch(launch)


@contextmanager
def configless_proxy_process(
    gateway: Gateway,
    directory: Path,
    overrides: Mapping[str, str],
    *,
    model: str,
    api_base: str,
    workers: int = 1,
) -> Iterator[OwnedProxy]:
    environment: Final = _proxy_environment(gateway, {**overrides, "STORE_MODEL_IN_DB": "False"}, ())
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(directory)))
    output.mkdir(parents=True, exist_ok=True)
    command: Final = (
        sys.executable,
        "-m",
        "integration._support.proxy",
        "--model",
        model,
        "--api_base",
        api_base,
        "--host",
        "127.0.0.1",
        "--num_workers",
        str(workers),
        "--timeout_worker_healthcheck",
        str(healthcheck_seconds(WorkerHealthcheck.STRICT)),
        *DB_PUSH,
    )
    launch: Final = _launch_until_ready(command, _proxy_root(), environment, output, workers)
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{launch.port}", timeout=15, trust_env=False, limits=GATEWAY_LIMITS
        ) as client:
            yield OwnedProxy(Gateway(client, gateway.key, gateway.upstream_url), launch.process, launch.log)
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
    launch: Final = _launch_until_ready(command, root, environment, output, workers)
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
    extra_arguments: tuple[str, ...] = (),
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
        *extra_arguments,
    )
    launch: Final = _launch(command, root, environment, output)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{launch.port}", timeout=15, trust_env=False) as client:
            deadline: Final = time.monotonic() + readiness_seconds(1)
            while launch.process.poll() is None:
                assert not (_owns_listener(launch) and _is_ready(client)), (
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

    __slots__ = ("certificate", "directory", "inherited", "port", "process", "root")

    def __init__(
        self,
        directory: Path,
        port: int,
        root: Path,
        certificate: UpstreamCertificate | None = None,
        inherited: Mapping[str, str] = os.environ,
    ) -> None:
        self.directory = directory
        self.port = port
        self.root = root
        self.certificate = certificate
        self.inherited = inherited
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
        output: Final = Path(self.inherited.get("INTEGRATION_RESULTS_DIR") or self.directory)
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
                env=dict(self.inherited),
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
def owned_upstream(
    directory: Path, *, tls: bool = False, port: int | None = None, inherited: Mapping[str, str] = os.environ
) -> Generator[UpstreamSlot]:
    certificate: Final = self_signed_certificate(directory) if tls else None
    slot: Final = UpstreamSlot(directory, port or allocate_port(), _harness_root(), certificate, inherited)
    try:
        slot.start()
        yield slot
    finally:
        if slot.process is not None:
            slot.stop()
