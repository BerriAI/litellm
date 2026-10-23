from __future__ import annotations

import ctypes
import http.client
import ipaddress
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import FrameType, MappingProxyType
from typing import Final, Protocol, cast
from urllib.parse import urlsplit

import click
from pydantic import TypeAdapter
from uvicorn.main import main as uvicorn_main

from gateway.launch import uvicorn_argv

RUST_GATEWAY_FLAG: Final = "LITELLM_RUST_GATEWAY"
RUST_GATEWAY_BINARY: Final = "LITELLM_RUST_GATEWAY_BINARY"
PYTHON_PORT_ENV: Final = "LITELLM_RUST_GATEWAY_PYTHON_PORT"
PYTHON_UPSTREAM_ENV: Final = "LITELLM_PYTHON_UPSTREAM"
TRUSTED_PROXIES_ENV: Final = "LITELLM_RUST_GATEWAY_TRUSTED_PROXIES"
METRICS_PORT_ENV: Final = "LITELLM_GATEWAY_METRICS_PORT"
TERMINATION_GRACE_ENV: Final = "LITELLM_RUST_GATEWAY_TERMINATION_GRACE_SECONDS"
DEFAULT_RUST_GATEWAY_BINARY: Final = "/usr/local/bin/litellm-gateway"
DEFAULT_PRIVATE_PORT: Final = 4002
DEFAULT_PYTHON_GRACE_SECONDS: Final = 5
DEFAULT_RUST_DRAIN_SECONDS: Final = 15
DEFAULT_STARTUP_TIMEOUT_SECONDS: Final = 120
DEFAULT_TERMINATION_GRACE_SECONDS: Final = 30
TERMINATION_CLEANUP_MARGIN_SECONDS: Final = 10
LIVENESS_PATH: Final = "/health/liveliness"


@dataclass(frozen=True, slots=True)
class ConfigurationFailure:
    reason: str


@dataclass(frozen=True, slots=True)
class SupervisorFailure:
    reason: str


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    python_command: tuple[str, ...]
    rust_command: tuple[str, ...]
    python_environment: Mapping[str, str]
    rust_environment: Mapping[str, str]
    public_probe_host: str
    public_port: int
    private_port: int
    startup_timeout: int
    rust_drain_timeout: int
    python_grace_timeout: int
    termination_grace_timeout: int = DEFAULT_TERMINATION_GRACE_SECONDS


class ChildProcess(Protocol):
    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class ProcessRunner(Protocol):
    def start(self, command: tuple[str, ...], environment: Mapping[str, str]) -> ChildProcess: ...


class SubprocessRunner:
    def start(self, command: tuple[str, ...], environment: Mapping[str, str]) -> ChildProcess:
        return subprocess.Popen(command, env=environment, start_new_session=True)


def gateway_enabled(environ: Mapping[str, str]) -> bool | ConfigurationFailure:
    value: Final = environ.get(RUST_GATEWAY_FLAG)
    if value is None or value == "false":
        return False
    if value == "true":
        return True
    return ConfigurationFailure(f"{RUST_GATEWAY_FLAG} must be exactly 'true' or 'false'")


def _port_setting(name: str, default: int, environ: Mapping[str, str]) -> int | ConfigurationFailure:
    value: Final = environ.get(name)
    if value is None:
        return default
    try:
        port: Final = int(value)
    except ValueError:
        return ConfigurationFailure(f"{name} must be an integer port")
    if not 1 <= port <= 65535:
        return ConfigurationFailure(f"{name} must be between 1 and 65535")
    return port


def _timeout_setting(name: str, default: int, environ: Mapping[str, str]) -> int | ConfigurationFailure:
    value: Final = environ.get(name)
    if value is None:
        return default
    try:
        timeout: Final = int(value)
    except ValueError:
        return ConfigurationFailure(f"{name} must be a positive integer")
    if timeout < 1:
        return ConfigurationFailure(f"{name} must be a positive integer")
    return timeout


def _boolean_setting(name: str, environ: Mapping[str, str]) -> bool | ConfigurationFailure:
    value: Final = environ.get(name)
    if value is None:
        return False
    normalized: Final = value.casefold()
    if normalized in ("true", "1", "yes", "y", "on", "t"):
        return True
    if normalized in ("false", "0", "no", "n", "off", "f"):
        return False
    return ConfigurationFailure(f"{name} must be a boolean")


def _collector_port(environ: Mapping[str, str]) -> int | ConfigurationFailure | None:
    enabled: Final = _boolean_setting("LITELLM_COLLECTOR_ENABLED", environ)
    if isinstance(enabled, ConfigurationFailure) or not enabled:
        return enabled if isinstance(enabled, ConfigurationFailure) else None
    address: Final = environ.get("LITELLM_COLLECTOR_ADDRESS")
    if address is None or not address.startswith("tcp://"):
        return None
    try:
        parsed: Final = urlsplit(address)
        port: Final = parsed.port
    except ValueError:
        return ConfigurationFailure("LITELLM_COLLECTOR_ADDRESS must contain a valid TCP port")
    if port is None:
        return ConfigurationFailure("LITELLM_COLLECTOR_ADDRESS must contain a valid TCP port")
    if not 1 <= port <= 65535:
        return ConfigurationFailure("LITELLM_COLLECTOR_ADDRESS must contain a valid TCP port")
    return port


def _trusted_proxies(environ: Mapping[str, str]) -> str | ConfigurationFailure:
    value: Final = environ.get(TRUSTED_PROXIES_ENV, "")
    if not value:
        return value
    try:
        networks: Final = tuple(ipaddress.ip_network(entry.strip(), strict=False) for entry in value.split(","))
    except ValueError:
        return ConfigurationFailure(f"{TRUSTED_PROXIES_ENV} must contain IP addresses or CIDRs")
    if any(network.prefixlen == 0 for network in networks):
        return ConfigurationFailure(f"{TRUSTED_PROXIES_ENV} must not trust every address")
    if any(not entry.strip() for entry in value.split(",")):
        return ConfigurationFailure(f"{TRUSTED_PROXIES_ENV} contains an empty entry")
    return value


def _worker_arguments(arguments: tuple[str, ...], private_port: int) -> tuple[str, ...]:
    value_options: Final = frozenset(("--host", "--port", "--forwarded-allow-ips"))
    boolean_options: Final = frozenset(("--proxy-headers",))
    consumed_values: Final = frozenset(
        index + 1 for index, argument in enumerate(arguments[:-1]) if argument in value_options
    )
    without_overrides: Final = tuple(
        argument
        for index, argument in enumerate(arguments)
        if index not in consumed_values
        and not any(
            argument == option or argument.startswith(f"{option}=") for option in value_options | boolean_options
        )
    )
    return (
        "--host",
        "127.0.0.1",
        "--port",
        str(private_port),
        "--proxy-headers",
        "--forwarded-allow-ips",
        "127.0.0.1",
        *without_overrides[1:],
    )


def _context_value(context: click.Context, name: str) -> object:
    parameters: Final = TypeAdapter(dict[str, object]).validate_python(context.params)
    value: Final = parameters[name]
    return value


def _first_port_collision(ports: tuple[tuple[str, int], ...]) -> tuple[str, str] | None:
    for index, current in enumerate(ports):
        for other in ports[index + 1 :]:
            if current[1] == other[1]:
                return current[0], other[0]
    return None


def build_configuration(
    argv: Sequence[str],
    environ: Mapping[str, str],
    python_executable: str,
) -> SupervisorConfig | ConfigurationFailure:
    arguments: Final = uvicorn_argv(argv, environ)
    try:
        context: Final = uvicorn_main.make_context(
            "uvicorn",
            list(arguments),
            allow_extra_args=True,
            ignore_unknown_options=True,
        )
    except click.ClickException as error:
        return ConfigurationFailure(f"invalid Uvicorn options: {error.format_message()}")

    if any(argument == "--no-proxy-headers" for argument in arguments):
        return ConfigurationFailure("--no-proxy-headers is unsupported with the Rust gateway")
    if any(argument.startswith("--ssl-") for argument in arguments):
        return ConfigurationFailure("Uvicorn TLS options are unsupported with the Rust gateway")
    if any(name.startswith("UVICORN_SSL_") for name in environ):
        return ConfigurationFailure("UVICORN_SSL_* settings are unsupported with the Rust gateway")
    if _context_value(context, "env_file") is not None:
        return ConfigurationFailure("--env-file is unsupported with the Rust gateway")
    if _context_value(context, "uds") is not None or _context_value(context, "fd") is not None:
        return ConfigurationFailure("Uvicorn Unix socket and file descriptor listeners are unsupported")
    if _context_value(context, "reload") is True:
        return ConfigurationFailure("Uvicorn reload is unsupported with the Rust gateway")
    if _context_value(context, "proxy_headers") is not True:
        return ConfigurationFailure("Uvicorn proxy headers must be enabled with the Rust gateway")

    protocol: Final = _context_value(context, "http")
    if not isinstance(protocol, str) or protocol not in ("auto", "h11", "httptools"):
        return ConfigurationFailure("the Rust gateway supports only Uvicorn HTTP/1.1 protocols")

    host: Final = _context_value(context, "host")
    port: Final = _context_value(context, "port")
    if (
        not isinstance(host, str)
        or not host
        or any(character.isspace() or character in "/[]\r\n" for character in host)
    ):
        return ConfigurationFailure("Uvicorn must have a non-empty public host")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        return ConfigurationFailure("Uvicorn public port must be between 1 and 65535")
    if "HOST" in environ and environ["HOST"] != host:
        return ConfigurationFailure("HOST conflicts with the effective Uvicorn host")
    if "PORT" in environ and environ["PORT"] != str(port):
        return ConfigurationFailure("PORT conflicts with the effective Uvicorn port")

    private_port: Final = _port_setting(PYTHON_PORT_ENV, DEFAULT_PRIVATE_PORT, environ)
    if isinstance(private_port, ConfigurationFailure):
        return private_port
    metrics_port: Final = _port_setting(METRICS_PORT_ENV, 0, environ) if METRICS_PORT_ENV in environ else None
    if isinstance(metrics_port, ConfigurationFailure):
        return metrics_port
    pgbouncer_enabled: Final = _boolean_setting("LITELLM_PGBOUNCER_ENABLED", environ)
    if isinstance(pgbouncer_enabled, ConfigurationFailure):
        return pgbouncer_enabled
    pgbouncer_port: Final = _port_setting("LITELLM_PGBOUNCER_PORT", 6432, environ) if pgbouncer_enabled else None
    if isinstance(pgbouncer_port, ConfigurationFailure):
        return pgbouncer_port
    collector_port: Final = _collector_port(environ)
    if isinstance(collector_port, ConfigurationFailure):
        return collector_port
    startup_timeout: Final = _timeout_setting(
        "LITELLM_RUST_GATEWAY_STARTUP_TIMEOUT_SECONDS",
        DEFAULT_STARTUP_TIMEOUT_SECONDS,
        environ,
    )
    if isinstance(startup_timeout, ConfigurationFailure):
        return startup_timeout
    rust_drain_timeout: Final = _timeout_setting(
        "LITELLM_RUST_GATEWAY_DRAIN_TIMEOUT_SECONDS",
        DEFAULT_RUST_DRAIN_SECONDS,
        environ,
    )
    if isinstance(rust_drain_timeout, ConfigurationFailure):
        return rust_drain_timeout
    python_grace_timeout: Final = _timeout_setting(
        "LITELLM_RUST_GATEWAY_PYTHON_GRACE_SECONDS",
        DEFAULT_PYTHON_GRACE_SECONDS,
        environ,
    )
    if isinstance(python_grace_timeout, ConfigurationFailure):
        return python_grace_timeout
    termination_grace_timeout: Final = _timeout_setting(
        TERMINATION_GRACE_ENV,
        DEFAULT_TERMINATION_GRACE_SECONDS,
        environ,
    )
    if isinstance(termination_grace_timeout, ConfigurationFailure):
        return termination_grace_timeout
    if rust_drain_timeout + python_grace_timeout + TERMINATION_CLEANUP_MARGIN_SECONDS > termination_grace_timeout:
        return ConfigurationFailure(f"{TERMINATION_GRACE_ENV} must allow Rust drain, Python grace, and cleanup margin")
    trusted_proxies: Final = _trusted_proxies(environ)
    if isinstance(trusted_proxies, ConfigurationFailure):
        return trusted_proxies

    collisions: Final = _first_port_collision(
        (
            ("public gateway", port),
            ("private Python upstream", private_port),
            *(() if metrics_port is None else (("metrics server", metrics_port),)),
            *(() if pgbouncer_port is None else (("PgBouncer", pgbouncer_port),)),
            *(() if collector_port is None else (("collector", collector_port),)),
        )
    )
    if collisions is not None:
        return ConfigurationFailure(f"{collisions[0]} and {collisions[1]} use the same TCP port")

    binary: Final = environ.get(RUST_GATEWAY_BINARY, DEFAULT_RUST_GATEWAY_BINARY)
    binary_path: Final = Path(binary)
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        return ConfigurationFailure(f"Rust gateway binary is not executable: {binary}")

    upstream: Final = f"http://127.0.0.1:{private_port}"
    configured_upstream: Final = environ.get(PYTHON_UPSTREAM_ENV)
    if configured_upstream is not None and configured_upstream != upstream:
        return ConfigurationFailure(f"{PYTHON_UPSTREAM_ENV} must equal {upstream}")

    python_environment: Final = MappingProxyType(
        {
            **environ,
            RUST_GATEWAY_FLAG: "false",
        }
    )
    rust_environment: Final = MappingProxyType(
        {
            **environ,
            RUST_GATEWAY_FLAG: "true",
            PYTHON_UPSTREAM_ENV: upstream,
            TRUSTED_PROXIES_ENV: trusted_proxies,
            "HOST": host,
            "PORT": str(port),
        }
    )
    probe_host: Final = "127.0.0.1" if host in ("0.0.0.0", "*") else "::1" if host == "::" else host
    python_command: Final = (
        python_executable,
        "-m",
        "gateway.launch",
        *_worker_arguments(arguments, private_port),
    )
    return SupervisorConfig(
        python_command=python_command,
        rust_command=(binary,),
        python_environment=python_environment,
        rust_environment=rust_environment,
        public_probe_host=probe_host,
        public_port=port,
        private_port=private_port,
        startup_timeout=startup_timeout,
        rust_drain_timeout=rust_drain_timeout,
        python_grace_timeout=python_grace_timeout,
        termination_grace_timeout=termination_grace_timeout,
    )


def _enable_subreaper() -> SupervisorFailure | None:
    if not sys.platform.startswith("linux"):
        return SupervisorFailure("the Rust gateway supervisor requires Linux")
    prctl: Final = ctypes.CDLL(None, use_errno=True).prctl
    if prctl(36, 1, 0, 0, 0) != 0:
        return SupervisorFailure("could not enable Linux child subreaper mode")
    return None


def _install_signal_handlers(
    requested: Event,
) -> tuple[tuple[signal.Signals, object], ...]:
    signals: Final = (signal.SIGTERM, signal.SIGINT)
    previous: Final = tuple((handled, signal.getsignal(handled)) for handled in signals)

    def request_shutdown(_signum: int, _frame: FrameType | None) -> None:
        requested.set()

    for handled in signals:
        signal.signal(handled, request_shutdown)
    return previous


def _restore_signal_handlers(
    previous: tuple[tuple[signal.Signals, object], ...],
) -> None:
    for handled, handler in previous:
        signal.signal(handled, cast(signal.Handlers | Callable[[int, FrameType | None], object] | None, handler))


def _tcp_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _http_ready(host: str, port: int) -> bool:
    connection: Final = http.client.HTTPConnection(host, port, timeout=0.3)
    try:
        connection.request("GET", LIVENESS_PATH)
        response: Final = connection.getresponse()
        response.read()
        return response.status == 200
    except (http.client.HTTPException, OSError):
        return False
    finally:
        connection.close()


def _group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_group(process_group: int, requested: signal.Signals) -> None:
    try:
        os.killpg(process_group, requested)
    except ProcessLookupError:
        pass


def _reap_adopted_children(excluded: frozenset[int]) -> None:
    children_path: Final = Path(f"/proc/self/task/{os.getpid()}/children")
    try:
        children: Final = tuple(int(pid) for pid in children_path.read_text().split() if int(pid) not in excluded)
    except (OSError, ValueError):
        return
    for child in children:
        try:
            os.waitpid(child, os.WNOHANG)
        except ChildProcessError:
            pass


def _adopted_children(excluded: frozenset[int]) -> tuple[int, ...]:
    children_path: Final = Path(f"/proc/self/task/{os.getpid()}/children")
    try:
        return tuple(int(pid) for pid in children_path.read_text().split() if int(pid) not in excluded)
    except (OSError, ValueError):
        return ()


def _signal_adopted_child(child: int, requested: signal.Signals) -> None:
    process_group: Final = os.getpgid(child)
    if process_group == os.getpgrp():
        os.kill(child, requested)
    else:
        os.killpg(process_group, requested)


def _terminate_adopted_children(excluded: frozenset[int]) -> None:
    children: Final = _adopted_children(excluded)
    for child in children:
        try:
            _signal_adopted_child(child, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline: Final = time.monotonic() + 1
    while time.monotonic() < deadline and _adopted_children(excluded):
        _reap_adopted_children(excluded)
        time.sleep(0.05)
    for child in _adopted_children(excluded):
        try:
            _signal_adopted_child(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
    final_deadline: Final = time.monotonic() + 1
    while time.monotonic() < final_deadline and _adopted_children(excluded):
        _reap_adopted_children(excluded)
        time.sleep(0.05)
    _reap_adopted_children(excluded)


def _wait_for_group(
    process: ChildProcess,
    deadline: float,
    protected_children: frozenset[int],
) -> bool:
    while time.monotonic() < deadline and _group_exists(process.pid):
        process.poll()
        _reap_adopted_children(protected_children)
        time.sleep(0.05)
    process.poll()
    _reap_adopted_children(protected_children)
    return not _group_exists(process.pid)


def _stop_process_group(
    process: ChildProcess,
    timeout: int,
    protected_children: frozenset[int],
) -> None:
    _signal_group(process.pid, signal.SIGTERM)
    if not _wait_for_group(process, time.monotonic() + timeout, protected_children):
        _signal_group(process.pid, signal.SIGKILL)
        _wait_for_group(process, time.monotonic() + 1, protected_children)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        _signal_group(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    _reap_adopted_children(protected_children)


def _wait_for_python(
    process: ChildProcess,
    port: int,
    deadline: float,
    requested: Event,
) -> bool | SupervisorFailure:
    protected_children: Final = frozenset((process.pid,))
    while not requested.is_set() and time.monotonic() < deadline:
        _reap_adopted_children(protected_children)
        if (status := process.poll()) is not None:
            return SupervisorFailure(f"Python gateway exited during startup with status {status}")
        if _tcp_ready("127.0.0.1", port):
            return True
        time.sleep(0.05)
    _reap_adopted_children(protected_children)
    return False


def _wait_for_rust(
    python: ChildProcess,
    rust: ChildProcess,
    host: str,
    port: int,
    deadline: float,
    requested: Event,
) -> bool | SupervisorFailure:
    protected_children: Final = frozenset((python.pid, rust.pid))
    while not requested.is_set() and time.monotonic() < deadline:
        _reap_adopted_children(protected_children)
        if (python_status := python.poll()) is not None:
            return SupervisorFailure(f"Python gateway exited during startup with status {python_status}")
        if (rust_status := rust.poll()) is not None:
            return SupervisorFailure(f"Rust gateway exited during startup with status {rust_status}")
        if _http_ready(host, port):
            return True
        time.sleep(0.05)
    _reap_adopted_children(protected_children)
    return False


def _monitor_children(python: ChildProcess, rust: ChildProcess, requested: Event) -> SupervisorFailure | None:
    protected_children: Final = frozenset((python.pid, rust.pid))
    while not requested.wait(0.1):
        _reap_adopted_children(protected_children)
        if (python_status := python.poll()) is not None:
            return SupervisorFailure(f"Python gateway exited unexpectedly with status {python_status}")
        if (rust_status := rust.poll()) is not None:
            return SupervisorFailure(f"Rust gateway exited unexpectedly with status {rust_status}")
    _reap_adopted_children(protected_children)
    return None


def supervise(config: SupervisorConfig, runner: ProcessRunner | None = None) -> SupervisorFailure | None:
    subreaper_failure: Final = _enable_subreaper()
    if subreaper_failure is not None:
        return subreaper_failure
    process_runner: Final = runner if runner is not None else SubprocessRunner()
    requested: Final = Event()
    previous_handlers: Final = _install_signal_handlers(requested)
    try:
        with ExitStack() as cleanup:
            try:
                python: Final = process_runner.start(config.python_command, config.python_environment)
            except OSError as error:
                return SupervisorFailure(f"could not start Python gateway: {error}")
            cleanup.callback(
                _stop_process_group,
                python,
                config.python_grace_timeout,
                frozenset((python.pid,)),
            )
            deadline: Final = time.monotonic() + config.startup_timeout
            python_ready: Final = _wait_for_python(python, config.private_port, deadline, requested)
            if isinstance(python_ready, SupervisorFailure):
                return python_ready
            if not python_ready:
                return None if requested.is_set() else SupervisorFailure("Python gateway startup timed out")
            if requested.is_set():
                return None
            try:
                rust: Final = process_runner.start(config.rust_command, config.rust_environment)
            except OSError as error:
                return SupervisorFailure(f"could not start Rust gateway: {error}")
            cleanup.callback(
                _stop_process_group,
                rust,
                config.rust_drain_timeout,
                frozenset((python.pid, rust.pid)),
            )
            rust_ready: Final = _wait_for_rust(
                python,
                rust,
                config.public_probe_host,
                config.public_port,
                deadline,
                requested,
            )
            if isinstance(rust_ready, SupervisorFailure):
                return rust_ready
            if not rust_ready:
                return None if requested.is_set() else SupervisorFailure("Rust gateway readiness timed out")
            return _monitor_children(python, rust, requested)
    finally:
        _terminate_adopted_children(frozenset())
        _restore_signal_handlers(previous_handlers)


if __name__ == "__main__":
    sys.exit("run the Rust gateway through python -m gateway.launch")
