"""Settings for the native (SQLx) pool, resolved from the same URL Prisma connects with.

``DATABASE_URL`` is final once ``db_url_settings`` and the CLI have assembled it: token
minted, libpq TLS translated to Prisma's dialect, timeouts in ``options``, pool params
added. Reading that URL keeps one resolver for both pools instead of a second pass over
the ``DATABASE_*`` env vars.
"""

import asyncio
import shlex
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Final, Protocol

from litellm._logging import verbose_proxy_logger
from litellm._service_logger import ServiceTypes
from litellm.proxy.db.db_span import db_event_metadata
from litellm.proxy.db.pgbouncer import database_url_is_pooled, libpq_option_settings
from litellm.proxy.db.token_auth import (
    DEFAULT_POSTGRES_PORT,
    DatabaseTokenAuth,
    IAMEndpoint,
    mint_database_token,
    parse_database_token_expiration,
    parse_iam_endpoint_from_url,
    resolve_database_token_auth,
)
from litellm.rust_bridge.database import NativeDatabaseEndpoint, NativePoolLimits, SslMode

TOKEN_REFRESH_BUFFER_SECONDS: Final = 180.0
TOKEN_FALLBACK_REFRESH_SECONDS: Final = 600.0
TOKEN_RETRY_SECONDS: Final = 30.0
PRISMA_DEFAULT_SCHEMA: Final = "public"


def _ssl_mode(params: Mapping[str, str]) -> SslMode:
    sslmode: Final = params.get("sslmode", "prefer")
    if sslmode == "disable":
        return "disable"
    if params.get("sslaccept") == "strict":
        return "verify-full"
    if sslmode == "require":
        return "require"
    return "prefer"


def _session_options(params: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    raw: Final = params.get("options", "")
    settings: Final = libpq_option_settings(tuple(shlex.split(raw)))
    if settings is None:
        raise ValueError(f"cannot read the DATABASE_URL options {raw!r} as -c name=value settings")
    schema: Final = params.get("schema", PRISMA_DEFAULT_SCHEMA)
    search_path: Final = () if schema == PRISMA_DEFAULT_SCHEMA else (("search_path", f'"{schema}"'),)
    return (*search_path, *((name, value) for name, _, value in (setting.partition("=") for setting in settings)))


def _seconds(params: Mapping[str, str], key: str) -> float | None:
    value: Final = params.get(key)
    if value is None or float(value) == 0:
        return None
    return float(value)


def native_endpoint(url: str) -> NativeDatabaseEndpoint:
    """The native pool's endpoint for the Prisma URL ``url``.

    Prisma's ``sslaccept=strict`` verifies chain and hostname, so it is ``verify-full``;
    ``sslcert`` is the single root ``translate_libpq_ssl_params`` pinned. A
    ``pool_timeout`` of 0 means "wait forever" to Prisma and has no SQLx spelling, so
    it falls back to SQLx's default. Prisma's ``connect_timeout`` and ``socket_timeout``
    have no SQLx counterpart and are not carried over.
    """
    parsed: Final = urllib.parse.urlsplit(url)
    params: Final = MappingProxyType(dict(urllib.parse.parse_qsl(parsed.query)))
    if not parsed.hostname or not parsed.username:
        raise ValueError("DATABASE_URL must name a host and a user for the native pool")
    password: Final = urllib.parse.unquote(parsed.password) if parsed.password else None
    connection_limit: Final = params.get("connection_limit")
    return NativeDatabaseEndpoint(
        host=parsed.hostname,
        port=parsed.port or int(DEFAULT_POSTGRES_PORT),
        username=urllib.parse.unquote(parsed.username),
        password=password,
        database=urllib.parse.unquote(parsed.path.lstrip("/")) or "postgres",
        ssl_mode=_ssl_mode(params),
        ssl_root_cert=params.get("sslcert"),
        options=_session_options(params),
        statement_cache=params.get("pgbouncer") != "true",
        pool=NativePoolLimits(
            max_connections=int(connection_limit) if connection_limit is not None else None,
            acquire_timeout_seconds=_seconds(params, "pool_timeout"),
            idle_timeout_seconds=_seconds(params, "max_idle_connection_lifetime"),
        ),
    )


@dataclass(frozen=True, slots=True)
class TokenRotation:
    auth: DatabaseTokenAuth
    endpoint: IAMEndpoint
    mint_token: Callable[[DatabaseTokenAuth, IAMEndpoint], str] = mint_database_token

    def mint(self) -> str:
        return urllib.parse.unquote(self.mint_token(self.auth, self.endpoint))

    def seconds_until_refresh(self, token: str, now: datetime) -> float:
        expires_at: Final = parse_database_token_expiration(self.auth, token)
        if expires_at is None:
            return TOKEN_FALLBACK_REFRESH_SECONDS
        return max(TOKEN_RETRY_SECONDS, (expires_at - now).total_seconds() - TOKEN_REFRESH_BUFFER_SECONDS)


def writer_token_rotation(url: str) -> TokenRotation | None:
    """The token rotation the writer needs, or None for a static password.

    A URL pointed at the in-container PgBouncer carries PgBouncer's own credential,
    and PgBouncer renews the upstream token itself.
    """
    if database_url_is_pooled():
        return None
    auth: Final = resolve_database_token_auth()
    if auth is None:
        return None
    return TokenRotation(auth=auth, endpoint=parse_iam_endpoint_from_url(url))


def reader_token_rotation(url: str) -> TokenRotation | None:
    """The token rotation the read replica needs, or None for a static password.

    The reader never goes through the in-container PgBouncer, so only the token toggles
    decide, with the reader's own region, the same as its Prisma client.
    """
    auth: Final = resolve_database_token_auth(read_replica=True)
    if auth is None:
        return None
    return TokenRotation(auth=auth, endpoint=parse_iam_endpoint_from_url(url))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def _refresh(install: Callable[[str], None], rotation: TokenRotation, now: Callable[[], datetime]) -> float:
    try:
        token: Final = await asyncio.to_thread(rotation.mint)
    except Exception as error:  # noqa: BLE001  # boto3 and azure-identity raise many types; any failure means retry
        verbose_proxy_logger.error(
            "Native database pool keeps its current %s (%r); retrying in %.0fs.",
            rotation.auth.label,
            error,
            TOKEN_RETRY_SECONDS,
        )
        return TOKEN_RETRY_SECONDS
    install(token)
    return rotation.seconds_until_refresh(token, now())


async def keep_token_fresh(
    install: Callable[[str], None],
    rotation: TokenRotation,
    current_token: str,
    now: Callable[[], datetime] = _utcnow,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Hand the native pool a fresh token before the current one expires.

    Existing connections stay authenticated; only connections opened after a refresh
    use the new token, the same as a re-minted Prisma URL.
    """
    delay = rotation.seconds_until_refresh(current_token, now())  # rebind-ok: each refresh schedules the next
    while True:
        await sleep(delay)
        delay = await _refresh(install, rotation, now)


class DatabaseServiceHooks(Protocol):
    async def async_service_success_hook(
        self,
        service: ServiceTypes,
        call_type: str,
        duration: float,
        parent_otel_span: None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_metadata: dict[str, str] | None = None,  # mutable-ok: matches the ServiceLogging hook signature
    ) -> None: ...

    async def async_service_failure_hook(
        self,
        service: ServiceTypes,
        duration: float,
        error: Exception,
        call_type: str,
        parent_otel_span: None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_metadata: dict[str, str] | None = None,  # mutable-ok: matches the ServiceLogging hook signature
    ) -> None: ...


class NativeDatabaseError(Exception):
    """A native (SQLx) query failure, reported to service logging like a Prisma one."""


@dataclass(frozen=True, slots=True)
class ServiceLoggingEvents:
    """Turns native round trips into the ``ServiceTypes.DB`` events ``db_span`` emits.

    Rust calls ``record`` from its own threads once per round trip, so the hook is
    handed to the proxy's event loop rather than awaited here.
    """

    loop: asyncio.AbstractEventLoop
    service_logging: Callable[[], DatabaseServiceHooks | None]

    def record(
        self,
        call_type: str,
        table: str | None,
        operation: str | None,
        started_at: float,
        duration_seconds: float,
        failure: str | None,
    ) -> None:
        service_logging: Final = self.service_logging()
        if service_logging is None:
            return
        start_time: Final = datetime.fromtimestamp(started_at, tz=timezone.utc).astimezone().replace(tzinfo=None)
        hook: Final = (
            service_logging.async_service_success_hook(
                service=ServiceTypes.DB,
                call_type=call_type,
                parent_otel_span=None,
                duration=duration_seconds,
                start_time=start_time,
                end_time=start_time + timedelta(seconds=duration_seconds),
                event_metadata=db_event_metadata(table, operation),
            )
            if failure is None
            else service_logging.async_service_failure_hook(
                error=NativeDatabaseError(failure),
                service=ServiceTypes.DB,
                call_type=call_type,
                parent_otel_span=None,
                duration=duration_seconds,
                start_time=start_time,
                end_time=start_time + timedelta(seconds=duration_seconds),
                event_metadata=db_event_metadata(table, operation),
            )
        )
        _ = self.loop.call_soon_threadsafe(self.loop.create_task, hook)
