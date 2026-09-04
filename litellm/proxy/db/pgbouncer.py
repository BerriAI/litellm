"""In-container PgBouncer shared by every proxy worker.

Each uvicorn worker owns a Prisma query engine with its own pool of
``connection_limit`` server connections, so the connections a pod holds open
against Postgres scale as ``workers * connection_limit`` and a database with a
fixed connection ceiling runs out of room as pods and workers are added.

When ``LITELLM_PGBOUNCER_ENABLED`` is set, the supervisor process starts one
PgBouncer next to the workers (no extra network hop: it listens on loopback
inside the pod) in transaction pooling mode, points ``DATABASE_URL`` at it
with ``pgbouncer=true`` so Prisma stops using server-side prepared statements,
and keeps it running for the life of the proxy. Every worker's pool then
becomes cheap client connections to PgBouncer while the upstream connection
count is capped at ``LITELLM_PGBOUNCER_MAX_DB_CONNECTIONS`` per pod, no matter
how many workers run.

Migrations and the schema diff run in the supervisor before the pooler is
started, so they always go straight to Postgres. ``DATABASE_URL_READ_REPLICA``
is left untouched.

The pooler holds the database password from startup, so it cannot be combined
with ``IAM_TOKEN_DB_AUTH`` or ``AZURE_POSTGRESQL_AUTH``: those rotate the
password inside every worker on their own schedule, and PgBouncer would keep
authenticating upstream with the expired token.
"""

from __future__ import annotations

import atexit
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.token_auth import AZURE_POSTGRESQL_AUTH_ENV_VAR, IAM_TOKEN_DB_AUTH_ENV_VAR

PGBOUNCER_ENV_PREFIX: Final = "LITELLM_PGBOUNCER_"
PGBOUNCER_LISTEN_ADDR: Final = "127.0.0.1"
PGBOUNCER_INI_NAME: Final = "pgbouncer.ini"
PGBOUNCER_USERLIST_NAME: Final = "userlist.txt"
PGBOUNCER_RESTART_DELAY_SECONDS: Final = 1.0
PGBOUNCER_READY_TIMEOUT_SECONDS: Final = 15.0
PGBOUNCER_STOP_GRACE_SECONDS: Final = 10.0
PGBOUNCER_UNPRIVILEGED_USER: Final = "nobody"
PGBOUNCER_MIN_VERSION: Final = (1, 19)
PGBOUNCER_VERSION_PATTERN: Final = re.compile(r"PgBouncer (\d+)\.(\d+)")
PGBOUNCER_TOKEN_AUTH_CONFLICT: Final = (
    f"the in-container pgbouncer cannot be combined with {IAM_TOKEN_DB_AUTH_ENV_VAR} or "
    f"{AZURE_POSTGRESQL_AUTH_ENV_VAR}: each worker rotates the database password on its own schedule and the pooler "
    "would keep using the expired token upstream. Disable the pooler or use a static database password"
)

# Prisma's client-side TLS params describe the hop to Postgres, which becomes
# PgBouncer's server side. They move into ``server_tls_*`` and must not stay on
# the loopback URL: the listener speaks plain TCP and Prisma would refuse it
# under ``sslmode=require``.
PRISMA_TLS_PARAM_KEYS: Final[frozenset[str]] = frozenset(
    {"sslmode", "sslcert", "sslaccept", "sslidentity", "sslpassword"}
)
POOLED_URL_DROPPED_KEYS: Final[frozenset[str]] = PRISMA_TLS_PARAM_KEYS | frozenset(("options", "pgbouncer"))
PGBOUNCER_SSLMODES: Final[frozenset[str]] = frozenset(
    {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
)


class PgBouncerSettings(BaseSettings):
    """``LITELLM_PGBOUNCER_*`` env vars, read once in the supervisor."""

    model_config = SettingsConfigDict(
        env_prefix=PGBOUNCER_ENV_PREFIX, case_sensitive=False, extra="ignore", frozen=True
    )

    enabled: bool = False
    port: int = Field(default=6432, ge=1, le=65535)
    max_db_connections: int = Field(default=20, ge=1)
    max_client_conn: int = Field(default=1000, ge=1)
    binary: str = "pgbouncer"


@dataclass(frozen=True, slots=True)
class PgBouncerPlan:
    ini: str
    userlist: str
    pooled_url: str


@dataclass(frozen=True, slots=True)
class PgBouncerError:
    reason: str


def _single_quoted(value: str) -> str:
    """Quote for SQL and for PgBouncer's ``[databases]`` connection string: both double a literal ``'``."""
    return "'" + value.replace("'", "''") + "'"


def _userlist_quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _option_settings(tokens: Sequence[str]) -> tuple[str, ...] | None:
    """The ``name=value`` settings in a libpq ``options`` string, or None if it holds anything else.

    Accepts ``-c name=value``, ``-cname=value`` and ``--name=value``; a
    detached ``-c`` is folded into the token that follows it first.
    """
    folded: Final = tuple(
        f"-c{tokens[index + 1]}" if token == "-c" and index + 1 < len(tokens) else token
        for index, token in enumerate(tokens)
        if index == 0 or tokens[index - 1] != "-c"
    )
    settings: Final = tuple(token[2:] for token in folded if token.startswith(("-c", "--")) and "=" in token[2:])
    return settings if len(settings) == len(folded) else None


def _connect_query(options: str) -> str | PgBouncerError:
    """Turn Prisma's ``options=-c name=value ...`` startup param into ``SET`` statements.

    PgBouncer rejects any ``-c`` setting in ``options`` that is not one of the
    handful it tracks (``statement_timeout`` and ``lock_timeout`` are not), so
    the settings are applied to each new server connection instead. Every
    client shares them, which is what the single ``DATABASE_URL`` gave anyway.
    """
    settings: Final = _option_settings(tuple(shlex.split(options)))
    if settings is None:
        return PgBouncerError(f"cannot translate the DATABASE_URL options {options!r} into PgBouncer settings")
    return "; ".join(
        f"SET {name.strip()} TO {_single_quoted(value.strip())}"
        for name, value in (setting.split("=", 1) for setting in settings)
    )


def _server_tls_settings(sslmode: str, sslcert: str, sslaccept: str) -> tuple[str, ...] | PgBouncerError:
    if sslmode not in PGBOUNCER_SSLMODES:
        return PgBouncerError(f"unsupported sslmode {sslmode!r} on DATABASE_URL")
    verify: Final = sslmode in ("verify-ca", "verify-full") or (sslmode == "require" and sslaccept == "strict")
    if verify and not sslcert:
        return PgBouncerError(
            "DATABASE_URL asks for a verified TLS connection but names no CA bundle; "
            "add sslcert=<ca.pem> (or sslrootcert=) so the in-container PgBouncer can verify Postgres"
        )
    mode: Final = "verify-full" if verify else sslmode
    return (f"server_tls_sslmode = {mode}", *((f"server_tls_ca_file = {sslcert}",) if sslcert else ()))


def plan_pgbouncer(
    upstream_url: str,
    settings: PgBouncerSettings,
    runtime_dir: Path,
    run_as_user: str | None,
) -> PgBouncerPlan | PgBouncerError:
    """Render the PgBouncer config for ``upstream_url`` and the loopback URL Prisma uses instead.

    Params describing Prisma's own pool (``connection_limit``, ``pool_timeout``,
    ...) stay on the pooled URL; the TLS params and ``options`` describe the hop
    to Postgres and move into the PgBouncer config. ``run_as_user`` is the
    unprivileged user PgBouncer drops to when the proxy runs as root, which
    PgBouncer itself refuses to do.
    """
    parsed: Final = urllib.parse.urlsplit(upstream_url)
    params: Final[Mapping[str, str]] = MappingProxyType(
        dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    )
    dbname: Final = urllib.parse.unquote(parsed.path.lstrip("/"))
    username: Final = urllib.parse.unquote(parsed.username or "")
    password: Final = None if parsed.password is None else urllib.parse.unquote(parsed.password)
    if not parsed.hostname or not username or password is None or not dbname:
        return PgBouncerError(
            "DATABASE_URL must carry a host, user, password and database name for the in-container PgBouncer"
        )
    if "sslidentity" in params:
        return PgBouncerError("client certificates (sslidentity) are not supported with the in-container PgBouncer")
    tls: Final = _server_tls_settings(
        params.get("sslmode", "prefer"), params.get("sslcert", ""), params.get("sslaccept", "")
    )
    if isinstance(tls, PgBouncerError):
        return tls
    connect_query: Final = _connect_query(params["options"]) if params.get("options") else ""
    if isinstance(connect_query, PgBouncerError):
        return connect_query
    upstream: Final = " ".join(
        (
            f"host={_single_quoted(parsed.hostname)}",
            f"port={parsed.port or 5432}",
            f"dbname={_single_quoted(dbname)}",
            f"user={_single_quoted(username)}",
            f"password={_single_quoted(password)}",
            *((f"connect_query={_single_quoted(connect_query)}",) if connect_query else ()),
        )
    )
    ini: Final = "\n".join(
        (
            "[databases]",
            f"{dbname} = {upstream}",
            "",
            "[pgbouncer]",
            f"listen_addr = {PGBOUNCER_LISTEN_ADDR}",
            f"listen_port = {settings.port}",
            f"unix_socket_dir = {runtime_dir}",
            f"auth_file = {runtime_dir / PGBOUNCER_USERLIST_NAME}",
            "auth_type = scram-sha-256",
            "pool_mode = transaction",
            f"max_client_conn = {settings.max_client_conn}",
            f"default_pool_size = {settings.max_db_connections}",
            f"max_db_connections = {settings.max_db_connections}",
            "ignore_startup_parameters = extra_float_digits",
            *tls,
            *((f"user = {run_as_user}",) if run_as_user else ()),
            "",
        )
    )
    userlist: Final = f"{_userlist_quote(username)} {_userlist_quote(password)}\n"
    pooled_query: Final = urllib.parse.urlencode(
        (*((key, value) for key, value in params.items() if key not in POOLED_URL_DROPPED_KEYS), ("pgbouncer", "true"))
    )
    credentials: Final = f"{urllib.parse.quote(username, safe='')}:{urllib.parse.quote(password, safe='')}"
    pooled_url: Final = urllib.parse.urlunsplit(
        parsed._replace(netloc=f"{credentials}@{PGBOUNCER_LISTEN_ADDR}:{settings.port}", query=pooled_query)
    )
    return PgBouncerPlan(ini=ini, userlist=userlist, pooled_url=pooled_url)


def write_pgbouncer_files(plan: PgBouncerPlan, runtime_dir: Path, run_as_user: str | None) -> Path:
    """Write the ini and userlist (both hold the password, so mode 0600) and return the ini path.

    ``run_as_user`` is the user PgBouncer drops to when started as root; it has
    to own the files it re-reads on reload and the socket directory.
    """
    ini_path: Final = runtime_dir / PGBOUNCER_INI_NAME
    userlist_path: Final = runtime_dir / PGBOUNCER_USERLIST_NAME
    for path, content in ((userlist_path, plan.userlist), (ini_path, plan.ini)):
        path.touch(mode=0o600)
        path.write_text(content, encoding="utf-8")
    if run_as_user is not None:
        runtime_dir.chmod(0o700)
        for path in (runtime_dir, ini_path, userlist_path):
            shutil.chown(path, user=run_as_user)
    return ini_path


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection((PGBOUNCER_LISTEN_ADDR, port), timeout=0.5):
            return True
    except OSError:
        return False


def _unix_socket_open(path: Path) -> bool:
    with socket.socket(socket.AF_UNIX) as probe:
        probe.settimeout(0.5)
        try:
            probe.connect(str(path))
        except OSError:
            return False
        return True


def unix_socket_path(runtime_dir: Path, port: int) -> Path:
    return runtime_dir / f".s.PGSQL.{port}"


def pgbouncer_version(binary: str) -> tuple[int, int] | PgBouncerError:
    """``(major, minor)`` from ``<binary> --version``.

    Readiness relies on PgBouncer exiting when it cannot bind its TCP port,
    which it does from 1.19 on. Older releases log a warning and serve the unix
    socket alone, so their socket would vouch for a port held by someone else.
    """
    try:
        output: Final = subprocess.run(
            (binary, "--version"), capture_output=True, text=True, check=False, timeout=10
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as run_error:
        return PgBouncerError(f"could not run {binary!r} --version: {run_error}")
    found: Final = PGBOUNCER_VERSION_PATTERN.search(output)
    if found is None:
        return PgBouncerError(f"{binary!r} --version did not report a PgBouncer version: {output.strip()!r}")
    return int(found[1]), int(found[2])


def _end(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=PGBOUNCER_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class PgBouncerProcess:
    """Runs ``argv`` as a foreground child and restarts it whenever it exits on its own.

    Prisma reconnects by itself after a failed query, so a PgBouncer crash
    costs the requests in flight plus one failed query per idle pooled
    connection the crash severed, and nothing else once the replacement is
    listening again. A replacement that cannot be spawned, finds its port
    taken, exits again or never starts listening is retried every
    ``restart_delay_seconds`` until ``stop`` is called.

    A connect probe of ``port`` cannot tell the child from another process
    that grabbed the port after the availability check, so readiness also
    needs ``socket_path``: the unix socket PgBouncer creates in the private
    runtime directory, which it only does once every TCP listener is bound
    (PgBouncer 1.19 or newer, see ``pgbouncer_version``).
    """

    def __init__(
        self,
        argv: Sequence[str],
        port: int,
        socket_path: Path,
        restart_delay_seconds: float = PGBOUNCER_RESTART_DELAY_SECONDS,
        ready_timeout_seconds: float = PGBOUNCER_READY_TIMEOUT_SECONDS,
    ) -> None:
        self.argv: Final = tuple(argv)
        self.port: Final = port
        self.socket_path: Final = socket_path
        self.restart_delay_seconds: Final = restart_delay_seconds
        self.ready_timeout_seconds: Final = ready_timeout_seconds
        self._stopping: Final = threading.Event()
        self._lock: Final = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def pid(self) -> int | None:
        with self._lock:
            return None if self._process is None else self._process.pid

    def _spawn(self) -> subprocess.Popen[bytes] | PgBouncerError | None:
        """Start a child, or None once ``stop`` ran; both take the lock so no child can slip in after a stop.

        The port has to be free first: a listener that is already there would
        pass the readiness check while the child fails to bind.
        """
        with self._lock:
            if self._stopping.is_set():
                return None
            if _port_open(self.port):
                return PgBouncerError(f"{PGBOUNCER_LISTEN_ADDR}:{self.port} is already in use by another process")
            try:
                process: Final = subprocess.Popen(self.argv)
            except OSError as spawn_error:
                return PgBouncerError(f"could not start {self.argv[0]!r}: {spawn_error}")
            self._process = process
            return process

    def _wait_ready(self, process: subprocess.Popen[bytes]) -> PgBouncerError | None:
        deadline: Final = time.monotonic() + self.ready_timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return PgBouncerError(f"pgbouncer exited with status {process.returncode} during startup")
            if _port_open(self.port) and _unix_socket_open(self.socket_path):
                return None
            time.sleep(0.1)
        if _port_open(self.port):
            return PgBouncerError(
                f"{PGBOUNCER_LISTEN_ADDR}:{self.port} is served by another process, not the pgbouncer that was started"
            )
        return PgBouncerError(
            f"pgbouncer did not start listening on {PGBOUNCER_LISTEN_ADDR}:{self.port} "
            f"within {self.ready_timeout_seconds:.0f}s"
        )

    def start(self) -> PgBouncerError | None:
        """Spawn PgBouncer, wait until it listens on port and unix socket, then supervise it from a daemon thread."""
        process: Final = self._spawn()
        if process is None:
            return PgBouncerError("pgbouncer was stopped before it started")
        if isinstance(process, PgBouncerError):
            return process
        not_ready: Final = self._wait_ready(process)
        if not_ready is not None:
            self.stop()
            return not_ready
        self._watch(process)
        return None

    def _watch(self, process: subprocess.Popen[bytes]) -> None:
        threading.Thread(
            target=self._supervise, args=(process,), daemon=True, name="litellm-pgbouncer-supervisor"
        ).start()

    def _supervise(self, process: subprocess.Popen[bytes]) -> None:
        status: Final = process.wait()
        if self._stopping.is_set():
            return
        verbose_proxy_logger.error(
            "In-container pgbouncer (pid %s) exited with status %s; restarting in %.1fs.",
            process.pid,
            status,
            self.restart_delay_seconds,
        )
        self._restart_after_delay()

    def _restart_after_delay(self) -> None:
        time.sleep(self.restart_delay_seconds)
        process: Final = self._spawn()
        if process is None:
            return
        if isinstance(process, PgBouncerError):
            self._retry_restart(process.reason)
            return
        not_ready: Final = self._wait_ready(process)
        if not_ready is None:
            self._watch(process)
            return
        _end(process)
        self._retry_restart(not_ready.reason)

    def _retry_restart(self, reason: str) -> None:
        if self._stopping.is_set():
            return
        verbose_proxy_logger.error(
            "In-container pgbouncer could not be restarted (%s); retrying in %.1fs.", reason, self.restart_delay_seconds
        )
        threading.Thread(target=self._restart_after_delay, daemon=True, name="litellm-pgbouncer-supervisor").start()

    def stop(self) -> None:
        with self._lock:
            self._stopping.set()
            process: Final = self._process
        if process is not None:
            _end(process)


def start_in_container_pgbouncer(
    settings: PgBouncerSettings, upstream_url: str, token_auth_enabled: bool = False
) -> str | PgBouncerError:
    """Start the pooler for ``upstream_url`` and return the loopback URL the workers must use.

    The pooler lives as long as this process: it is stopped from ``atexit``
    once the worker manager has returned. PgBouncer refuses to run as root, so
    a root proxy (the default image) has it drop to ``nobody``.
    """
    if token_auth_enabled:
        return PgBouncerError(PGBOUNCER_TOKEN_AUTH_CONFLICT)
    version: Final = pgbouncer_version(settings.binary)
    if isinstance(version, PgBouncerError):
        return version
    if version < PGBOUNCER_MIN_VERSION:
        return PgBouncerError(
            f"PgBouncer {version[0]}.{version[1]} keeps running after failing to bind its TCP port, so the proxy "
            f"cannot tell it apart from another listener; {PGBOUNCER_MIN_VERSION[0]}.{PGBOUNCER_MIN_VERSION[1]} "
            "or newer is required"
        )
    runtime_dir: Final = Path(tempfile.mkdtemp(prefix="litellm-pgbouncer-"))
    atexit.register(shutil.rmtree, runtime_dir, ignore_errors=True)
    run_as_user: Final = PGBOUNCER_UNPRIVILEGED_USER if os.geteuid() == 0 else None
    plan: Final = plan_pgbouncer(upstream_url, settings, runtime_dir, run_as_user)
    if isinstance(plan, PgBouncerError):
        return plan
    ini_path: Final = write_pgbouncer_files(plan, runtime_dir, run_as_user)
    pooler: Final = PgBouncerProcess(
        argv=(settings.binary, str(ini_path)),
        port=settings.port,
        socket_path=unix_socket_path(runtime_dir, settings.port),
    )
    failed: Final = pooler.start()
    if failed is not None:
        return failed
    atexit.register(pooler.stop)
    verbose_proxy_logger.info(
        "In-container pgbouncer (pid %s) listening on %s:%s; capping this pod at %s upstream database connections.",
        pooler.pid,
        PGBOUNCER_LISTEN_ADDR,
        settings.port,
        settings.max_db_connections,
    )
    return plan.pooled_url
