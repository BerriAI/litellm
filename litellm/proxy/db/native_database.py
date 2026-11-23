"""What the Python host adds to the native (SQLx) pool.

The pool's endpoint is resolved in Rust, by ``DatabaseEndpoint.resolve``, from the same URL
Prisma connects with. ``DATABASE_URL`` is final once ``db_url_settings`` and the CLI have
assembled it: token minted, libpq TLS translated to Prisma's dialect, timeouts in ``options``,
pool params added. The Rust gateway reads that URL with the same code, so no endpoint rule
lives here. This module keeps what only this host does: the PKCS#12 client identity and the
token refresh loop.

Credentials follow the host. Each process runs one token refresh loop, owned by whatever
hosts it, and feeds every client in that process: here the worker's loop re-mints Prisma's
URL and pushes the same kind of token into the Rust pool through ``set_writer_password``.
The Rust core only accepts a password and never mints one, so a Rust gateway process
brings its own loop. Two processes minting at once is fine, since IAM and Entra tokens
are independent. What must not exist twice is the logic that resolves region, endpoint
and user, so when the Rust gateway needs it, it moves into the Rust core and this
loop calls it.
"""

import asyncio
import urllib.parse
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, pkcs12

from litellm._internal_context import current_request_parent_otel_span
from litellm._logging import verbose_proxy_logger
from litellm._service_logger import ServiceTypes
from litellm.proxy.db.db_span import db_event_metadata
from litellm.proxy.db.pgbouncer import database_url_is_pooled
from litellm.proxy.db.token_auth import (
    DatabaseTokenAuth,
    IAMEndpoint,
    mint_database_token,
    parse_database_token_expiration,
    parse_iam_endpoint_from_url,
    resolve_database_token_auth,
)
from litellm.rust_bridge.database import NativeClientIdentity

if TYPE_CHECKING:
    from opentelemetry.trace import Span

TOKEN_REFRESH_BUFFER_SECONDS: Final = 180.0
TOKEN_FALLBACK_REFRESH_SECONDS: Final = 600.0
TOKEN_RETRY_SECONDS: Final = 30.0


def client_identity(url: str, read_file: Callable[[Path], bytes] = Path.read_bytes) -> NativeClientIdentity | None:
    """Prisma's ``sslidentity`` / ``sslpassword``, a PKCS#12 client certificate, as the PEM the native pool takes.

    Rust decides the endpoint's TLS, so the file is read whenever the URL names it without
    ``sslmode=disable``, and the pool drops the identity when a ``prefer`` server refuses TLS.
    """
    params: Final = MappingProxyType(dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query)))
    identity_path: Final = params.get("sslidentity")
    if identity_path is None or params.get("sslmode") == "disable":
        return None
    password: Final = params.get("sslpassword")
    key, cert, chain = pkcs12.load_key_and_certificates(
        read_file(Path(identity_path)), None if password is None else password.encode()
    )
    if key is None or cert is None:
        raise ValueError(f"sslidentity {identity_path!r} must hold a client certificate and its private key")
    return NativeClientIdentity(
        cert_pem=b"".join(c.public_bytes(Encoding.PEM) for c in (cert, *chain)).decode(),
        key_pem=key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode(),
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
        parent_otel_span: "Span | None" = None,
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
        parent_otel_span: "Span | None" = None,
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
    handed to the proxy's event loop rather than awaited here. A read a request made
    is recorded inside that request's ``contextvars`` context, so its event parents
    under the request span.
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
        parent_otel_span: Final = current_request_parent_otel_span()
        start_time: Final = datetime.fromtimestamp(started_at, tz=timezone.utc).astimezone().replace(tzinfo=None)
        hook: Final = (
            service_logging.async_service_success_hook(
                service=ServiceTypes.DB,
                call_type=call_type,
                parent_otel_span=parent_otel_span,
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
                parent_otel_span=parent_otel_span,
                duration=duration_seconds,
                start_time=start_time,
                end_time=start_time + timedelta(seconds=duration_seconds),
                event_metadata=db_event_metadata(table, operation),
            )
        )
        _ = self.loop.call_soon_threadsafe(self.loop.create_task, hook)
