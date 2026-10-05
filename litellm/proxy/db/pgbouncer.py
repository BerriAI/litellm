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

The workers never hold the upstream credential: they log in to PgBouncer as
``litellm_pgbouncer`` with a random password made at startup, and PgBouncer
takes the database user's password from its auth file. Under
``IAM_TOKEN_DB_AUTH`` or ``AZURE_POSTGRESQL_AUTH`` that password is a
short-lived token, so the supervisor mints a new one before it expires,
rewrites the auth file and asks PgBouncer to reload; only new upstream
connections authenticate, so live ones are unaffected. The pooled
``DATABASE_URL`` then carries a static password, and the workers must not run
their own token refresh against it: ``LITELLM_PGBOUNCER_POOLED_DATABASE_URL``
tells them so, while a read replica keeps refreshing its own token.
"""

from __future__ import annotations

import atexit
import functools
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.token_auth import (
    DatabaseTokenAuth,
    IAMEndpoint,
    mint_database_token,
    parse_database_token_expiration,
    parse_iam_endpoint_from_url,
)

PGBOUNCER_ENV_PREFIX: Final = "LITELLM_PGBOUNCER_"
PGBOUNCER_POOLED_ENV_VAR: Final = "LITELLM_PGBOUNCER_POOLED_DATABASE_URL"
PGBOUNCER_LISTEN_ADDR: Final = "127.0.0.1"
PGBOUNCER_POOL_USER: Final = "litellm_pgbouncer"
PGBOUNCER_INI_NAME: Final = "pgbouncer.ini"
PGBOUNCER_USERLIST_NAME: Final = "userlist.txt"
PGBOUNCER_CA_NAME: Final = "server-ca.pem"
PGBOUNCER_RESTART_DELAY_SECONDS: Final = 1.0
PGBOUNCER_READY_TIMEOUT_SECONDS: Final = 15.0
PGBOUNCER_STOP_GRACE_SECONDS: Final = 10.0
PGBOUNCER_UNPRIVILEGED_USER: Final = "nobody"
PGBOUNCER_MIN_VERSION: Final = (1, 19)
PGBOUNCER_MAX_PASSWORD_BYTES: Final = 2048
PGBOUNCER_VERSION_PATTERN: Final = re.compile(r"PgBouncer (\d+)\.(\d+)")
PGBOUNCER_TOKEN_REFRESH_BUFFER_SECONDS: Final = 180.0
PGBOUNCER_TOKEN_FALLBACK_REFRESH_SECONDS: Final = 600.0
PGBOUNCER_TOKEN_RETRY_SECONDS: Final = 30.0

# Prisma's client-side TLS params describe the hop to Postgres, which becomes
# PgBouncer's server side. They move into ``server_tls_*`` and must not stay on
# the loopback URL: the listener speaks plain TCP and Prisma would refuse it
# under ``sslmode=require`` or ``channel_binding=require``.
PRISMA_TLS_PARAM_KEYS: Final[frozenset[str]] = frozenset(
    {"sslmode", "sslcert", "sslaccept", "sslidentity", "sslpassword", "channel_binding", "gssencmode"}
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
    pooled_url: str
    upstream_user: str
    upstream_password: str | None
    pool_password: str
    ca_source: str | None = None

    def userlist(self, upstream_password: str) -> str:
        return "".join(
            f"{_userlist_quote(user)} {_userlist_quote(password)}\n"
            for user, password in ((self.upstream_user, upstream_password), (PGBOUNCER_POOL_USER, self.pool_password))
        )


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


def _server_tls_settings(sslmode: str, sslcert: str, sslaccept: str, ca_path: Path) -> tuple[str, ...] | PgBouncerError:
    """``server_tls_*`` lines naming ``ca_path``, the runtime-dir copy of the bundle: the original (or the
    0600 root pinned by ``pin_bundle_root``) is often unreadable for the user PgBouncer drops to."""
    if sslmode not in PGBOUNCER_SSLMODES:
        return PgBouncerError(f"unsupported sslmode {sslmode!r} on DATABASE_URL")
    verify: Final = sslmode in ("verify-ca", "verify-full") or (sslmode == "require" and sslaccept == "strict")
    if verify and not sslcert:
        return PgBouncerError(
            "DATABASE_URL asks for a verified TLS connection but names no CA bundle; "
            "add sslcert=<ca.pem> (or sslrootcert=) so the in-container PgBouncer can verify Postgres"
        )
    mode: Final = "verify-full" if verify else sslmode
    return (f"server_tls_sslmode = {mode}", *((f"server_tls_ca_file = {ca_path}",) if sslcert else ()))


def plan_pgbouncer(
    upstream_url: str,
    settings: PgBouncerSettings,
    runtime_dir: Path,
    run_as_user: str | None,
) -> PgBouncerPlan | PgBouncerError:
    """Render the PgBouncer config for ``upstream_url`` and the loopback URL Prisma uses instead.

    Params describing Prisma's own pool (``connection_limit``, ``pool_timeout``,
    ...) stay on the pooled URL; the TLS params and ``options`` describe the hop
    to Postgres and move into the PgBouncer config. The upstream password is
    left out of the config on purpose: PgBouncer then takes it from the auth
    file, which can be rewritten while it runs. ``run_as_user`` is the
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
    if not parsed.hostname or not username or not dbname:
        return PgBouncerError("DATABASE_URL must carry a host, user and database name for the in-container PgBouncer")
    if username == PGBOUNCER_POOL_USER:
        return PgBouncerError(
            f"the database user cannot be named {PGBOUNCER_POOL_USER!r}: that is the user the workers log in to the "
            "in-container PgBouncer as, and PgBouncer keeps one password per user"
        )
    if "sslidentity" in params:
        return PgBouncerError("client certificates (sslidentity) are not supported with the in-container PgBouncer")
    tls: Final = _server_tls_settings(
        params.get("sslmode", "prefer"),
        params.get("sslcert", ""),
        params.get("sslaccept", ""),
        runtime_dir / PGBOUNCER_CA_NAME,
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
            f"stats_users = {PGBOUNCER_POOL_USER}",
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
    pooled_query: Final = urllib.parse.urlencode(
        (*((key, value) for key, value in params.items() if key not in POOLED_URL_DROPPED_KEYS), ("pgbouncer", "true"))
    )
    pool_password: Final = secrets.token_urlsafe(32)
    pooled_url: Final = urllib.parse.urlunsplit(
        parsed._replace(
            netloc=f"{PGBOUNCER_POOL_USER}:{pool_password}@{PGBOUNCER_LISTEN_ADDR}:{settings.port}", query=pooled_query
        )
    )
    return PgBouncerPlan(
        ini=ini,
        pooled_url=pooled_url,
        upstream_user=username,
        upstream_password=password,
        pool_password=pool_password,
        ca_source=params.get("sslcert") or None,
    )


def pooled_database_url(upstream_url: str, settings: PgBouncerSettings) -> str | PgBouncerError:
    """The loopback URL of a PgBouncer another container in the pod already runs for ``upstream_url``.

    Only the container that started PgBouncer knows the pool user's password, so
    this logs in as the upstream user, whom the auth file lists as well.
    """
    plan: Final = plan_pgbouncer(upstream_url, settings, runtime_dir=Path("/nonexistent"), run_as_user=None)
    if isinstance(plan, PgBouncerError):
        return plan
    password: Final = urllib.parse.urlsplit(upstream_url).password or ""
    credentials: Final = f"{urllib.parse.quote(plan.upstream_user, safe='')}:{password}"
    return urllib.parse.urlunsplit(
        urllib.parse.urlsplit(plan.pooled_url)._replace(netloc=f"{credentials}@{PGBOUNCER_LISTEN_ADDR}:{settings.port}")
    )


def _write_private(path: Path, content: str, run_as_user: str | None) -> None:
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as handle:
        handle.write(content)
    if run_as_user is not None:
        shutil.chown(path, user=run_as_user)


def write_pgbouncer_ini(plan: PgBouncerPlan, runtime_dir: Path, run_as_user: str | None) -> Path | PgBouncerError:
    """Write the ini (mode 0600) and the CA copy, and return the ini path.

    ``run_as_user`` is the user PgBouncer drops to when started as root; it has
    to own the files it re-reads on reload and the socket directory.
    """
    ini_path: Final = runtime_dir / PGBOUNCER_INI_NAME
    ca_path: Final = runtime_dir / PGBOUNCER_CA_NAME
    if plan.ca_source is not None:
        try:
            shutil.copyfile(plan.ca_source, ca_path)
        except OSError as error:
            return PgBouncerError(f"cannot read the CA bundle {plan.ca_source!r} named by sslcert: {error}")
    _write_private(ini_path, plan.ini, run_as_user)
    if run_as_user is not None:
        runtime_dir.chmod(0o700)
        for path in (runtime_dir, *((ca_path,) if plan.ca_source is not None else ())):
            shutil.chown(path, user=run_as_user)
    return ini_path


def write_userlist(userlist: str, runtime_dir: Path, run_as_user: str | None) -> Path:
    """Replace the auth file in one step, so a PgBouncer starting or reloading meanwhile reads the old or the new one whole."""
    userlist_path: Final = runtime_dir / PGBOUNCER_USERLIST_NAME
    staged_path: Final = runtime_dir / f".{PGBOUNCER_USERLIST_NAME}.next"
    _write_private(staged_path, userlist, run_as_user)
    os.replace(staged_path, userlist_path)
    return userlist_path


def export_pooled_database_url(pooled_url: str) -> None:
    os.environ["DATABASE_URL"] = pooled_url
    os.environ[PGBOUNCER_POOLED_ENV_VAR] = "true"


def database_url_is_pooled(environ: Mapping[str, str] = os.environ) -> bool:
    return environ.get(PGBOUNCER_POOLED_ENV_VAR) == "true"


@dataclass(frozen=True, slots=True)
class PgBouncerTokenSource:
    auth: DatabaseTokenAuth
    endpoint: IAMEndpoint

    def mint(self) -> str:
        """The token as Postgres expects it: ``mint_database_token`` returns it percent-encoded for a URL."""
        return urllib.parse.unquote(mint_database_token(self.auth, self.endpoint))

    def expires_at(self, token: str) -> datetime | None:
        return parse_database_token_expiration(self.auth, token)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PgBouncerTokenRefresher:
    """Keeps the token in PgBouncer's auth file current from a daemon thread.

    ``install`` gets each fresh token and is expected to rewrite the auth file
    and reload PgBouncer. The next refresh is due ``buffer_seconds`` before the
    token expires, or ``fallback_seconds`` later when the expiry cannot be read.
    A refresh that fails leaves the previous auth file in place and is retried
    after ``retry_seconds``: the old token stays good until it expires, so a
    transient credential-provider error costs nothing unless it persists.
    """

    def __init__(
        self,
        source: PgBouncerTokenSource,
        install: Callable[[str], None],
        *,
        buffer_seconds: float = PGBOUNCER_TOKEN_REFRESH_BUFFER_SECONDS,
        fallback_seconds: float = PGBOUNCER_TOKEN_FALLBACK_REFRESH_SECONDS,
        retry_seconds: float = PGBOUNCER_TOKEN_RETRY_SECONDS,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._source: Final = source
        self._install: Final = install
        self._buffer_seconds: Final = buffer_seconds
        self._fallback_seconds: Final = fallback_seconds
        self._retry_seconds: Final = retry_seconds
        self._now: Final = now
        self._stopping: Final = threading.Event()
        self._delay: float = 0.0
        self._thread: threading.Thread | None = None

    def refresh(self) -> float | PgBouncerError:
        label: Final = self._source.auth.label
        try:
            token: Final = self._source.mint()
        except Exception as mint_error:
            return PgBouncerError(f"could not mint a {label} for the in-container pgbouncer: {mint_error!r}")
        if len(token.encode()) >= PGBOUNCER_MAX_PASSWORD_BYTES:
            return PgBouncerError(
                f"the {label} is {len(token.encode())} bytes long, but PgBouncer's auth file holds passwords of at "
                f"most {PGBOUNCER_MAX_PASSWORD_BYTES - 1} bytes"
            )
        try:
            self._install(token)
        except OSError as install_error:
            return PgBouncerError(f"could not install the {label} into the pgbouncer auth file: {install_error}")
        expires_at: Final = self._source.expires_at(token)
        if expires_at is None:
            return self._fallback_seconds
        return max(self._retry_seconds, (expires_at - self._now()).total_seconds() - self._buffer_seconds)

    def start(self) -> PgBouncerError | None:
        primed: Final = self.refresh()
        if isinstance(primed, PgBouncerError):
            return primed
        self._delay = primed
        self._thread = threading.Thread(target=self._run, daemon=True, name="litellm-pgbouncer-token-refresh")
        self._thread.start()
        return None

    def _run(self) -> None:
        while not self._stopping.wait(self._delay):
            self._delay = self._refresh_and_report()

    def _refresh_and_report(self) -> float:
        outcome: Final = self.refresh()
        if isinstance(outcome, PgBouncerError):
            verbose_proxy_logger.error(
                "In-container pgbouncer keeps its current %s (%s); retrying in %.0fs.",
                self._source.auth.label,
                outcome.reason,
                self._retry_seconds,
            )
            return self._retry_seconds
        verbose_proxy_logger.info(
            "In-container pgbouncer picked up a fresh %s; the next one is due in %.0fs.",
            self._source.auth.label,
            outcome,
        )
        return outcome

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join()


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

    def reload(self) -> None:
        with self._lock:
            if self._process is not None:
                self._process.send_signal(signal.SIGHUP)

    def stop(self) -> None:
        with self._lock:
            self._stopping.set()
            process: Final = self._process
        if process is not None:
            _end(process)


def install_pgbouncer_token(
    plan: PgBouncerPlan, runtime_dir: Path, run_as_user: str | None, pooler: PgBouncerProcess, token: str
) -> None:
    write_userlist(plan.userlist(token), runtime_dir, run_as_user)
    pooler.reload()


def _install_upstream_password(
    plan: PgBouncerPlan,
    runtime_dir: Path,
    run_as_user: str | None,
    pooler: PgBouncerProcess,
    token_auth: DatabaseTokenAuth | None,
    upstream_url: str,
) -> PgBouncerTokenRefresher | None | PgBouncerError:
    if token_auth is None:
        if plan.upstream_password is None:
            return PgBouncerError(
                "DATABASE_URL carries no password and neither IAM_TOKEN_DB_AUTH nor AZURE_POSTGRESQL_AUTH is on, "
                "so the in-container PgBouncer has nothing to authenticate to Postgres with"
            )
        write_userlist(plan.userlist(plan.upstream_password), runtime_dir, run_as_user)
        return None
    refresher: Final = PgBouncerTokenRefresher(
        PgBouncerTokenSource(auth=token_auth, endpoint=parse_iam_endpoint_from_url(upstream_url)),
        functools.partial(install_pgbouncer_token, plan, runtime_dir, run_as_user, pooler),
    )
    failed: Final = refresher.start()
    if failed is not None:
        return failed
    return refresher


def _only_in_this_process(action: Callable[[], None]) -> Callable[[], None]:
    """An exit hook that does nothing in a forked child, which inherits the parent's ``atexit`` table."""
    owner_pid: Final = os.getpid()

    def run() -> None:
        if os.getpid() == owner_pid:
            action()

    return run


def start_in_container_pgbouncer(
    settings: PgBouncerSettings,
    upstream_url: str,
    token_auth: DatabaseTokenAuth | None = None,
    register_exit_hook: Callable[[Callable[[], None]], object] = atexit.register,
) -> str | PgBouncerError:
    """Start the pooler for ``upstream_url`` and return the loopback URL the workers must use.

    The pooler lives as long as this process: it is stopped from the exit hooks
    once the worker manager has returned, and only by the process that started
    it (gunicorn forks its workers, so they carry the hooks too). PgBouncer
    refuses to run as root, so a root proxy (the default image) has it drop to
    ``nobody``. With ``token_auth`` the password on ``upstream_url`` is ignored:
    the pooler mints its own tokens and renews them for as long as it runs.
    """
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
    register_exit_hook(_only_in_this_process(lambda: shutil.rmtree(runtime_dir, ignore_errors=True)))
    run_as_user: Final = PGBOUNCER_UNPRIVILEGED_USER if os.geteuid() == 0 else None
    plan: Final = plan_pgbouncer(upstream_url, settings, runtime_dir, run_as_user)
    if isinstance(plan, PgBouncerError):
        return plan
    ini_path: Final = write_pgbouncer_ini(plan, runtime_dir, run_as_user)
    if isinstance(ini_path, PgBouncerError):
        return ini_path
    pooler: Final = PgBouncerProcess(
        argv=(settings.binary, str(ini_path)),
        port=settings.port,
        socket_path=unix_socket_path(runtime_dir, settings.port),
    )
    refresher: Final = _install_upstream_password(plan, runtime_dir, run_as_user, pooler, token_auth, upstream_url)
    if isinstance(refresher, PgBouncerError):
        return refresher
    failed: Final = pooler.start()
    if failed is not None:
        if refresher is not None:
            refresher.stop()
        return failed
    register_exit_hook(_only_in_this_process(pooler.stop))
    if refresher is not None:
        register_exit_hook(_only_in_this_process(refresher.stop))
    verbose_proxy_logger.info(
        "In-container pgbouncer (pid %s) listening on %s:%s; capping this pod at %s upstream database connections%s.",
        pooler.pid,
        PGBOUNCER_LISTEN_ADDR,
        settings.port,
        settings.max_db_connections,
        "" if token_auth is None else f" and renewing its {token_auth.label} before each one expires",
    )
    return plan.pooled_url
