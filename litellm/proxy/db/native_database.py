"""Settings for the native (SQLx) pool, resolved from the same URL Prisma connects with.

``DATABASE_URL`` is final once ``db_url_settings`` and the CLI have assembled it: token
minted, libpq TLS translated to Prisma's dialect, timeouts in ``options``, pool params
added. Reading that URL keeps one resolver for both pools instead of a second pass over
the ``DATABASE_*`` env vars.

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
import shlex
import struct
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final

from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, pkcs12

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.db_url_settings import NATIVE_APPLICATION_NAME, PRISMA_APPLICATION_NAME
from litellm.proxy.db.pgbouncer import database_url_is_pooled, fold_libpq_option_tokens, libpq_option_setting
from litellm.proxy.db.token_auth import (
    DEFAULT_POSTGRES_PORT,
    DatabaseTokenAuth,
    IAMEndpoint,
    mint_database_token,
    parse_database_token_expiration,
    parse_iam_endpoint_from_url,
    resolve_database_token_auth,
)
from litellm.rust_bridge.database import (
    NativeClientIdentity,
    NativeDatabaseEndpoint,
    NativePoolLimits,
    SslMode,
)

TOKEN_REFRESH_BUFFER_SECONDS: Final = 180.0
TOKEN_FALLBACK_REFRESH_SECONDS: Final = 600.0
TOKEN_RETRY_SECONDS: Final = 30.0
PRISMA_DEFAULT_SCHEMA: Final = "public"
PRISMA_DEFAULT_POOL_TIMEOUT_SECONDS: Final = 10.0
PRISMA_DEFAULT_STATEMENT_CACHE_SIZE: Final = 100
TLS_PROBE_TIMEOUT_SECONDS: Final = 5.0
_POSTGRES_SSL_REQUEST: Final = struct.pack("!ii", 8, 80877103)


@dataclass(frozen=True, slots=True)
class NativeEndpointSettings:
    endpoint: NativeDatabaseEndpoint
    ignored_options: tuple[str, ...]


def _params(url: str) -> Mapping[str, str]:
    return MappingProxyType(dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query)))


def needs_tls_probe(url: str) -> bool:
    """Prisma's ``sslmode=prefer`` with ``sslaccept=strict`` verifies TLS when the server offers it
    and falls back to plaintext when it does not; SQLx has no verified-or-plaintext mode, so the
    caller asks the server once and pins the answer."""
    params: Final = _params(url)
    return params.get("sslmode", "prefer") == "prefer" and params.get("sslaccept") == "strict" and "host" not in params


async def server_offers_tls(url: str, timeout: float = TLS_PROBE_TIMEOUT_SECONDS) -> bool:
    """Whether the server answers a Postgres ``SSLRequest`` with ``S``; an unreachable server counts as offering it."""
    parsed: Final = urllib.parse.urlsplit(url)
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(parsed.hostname, parsed.port or int(DEFAULT_POSTGRES_PORT)), timeout
        )
    except (OSError, asyncio.TimeoutError):
        return True
    try:
        writer.write(_POSTGRES_SSL_REQUEST)
        await writer.drain()
        answer: Final = await asyncio.wait_for(reader.read(1), timeout)
    except (OSError, asyncio.TimeoutError):
        return True
    finally:
        writer.close()
    return answer != b"N"


def _tls(params: Mapping[str, str], tls_offered: bool) -> tuple[SslMode, str | None]:
    sslmode: Final = params.get("sslmode", "prefer")
    strict: Final = params.get("sslaccept") == "strict"
    if sslmode == "disable" or (sslmode == "prefer" and strict and not tls_offered):
        return "disable", None
    if strict:
        return "verify-full", params.get("sslcert")
    return ("require" if sslmode == "require" else "prefer"), None


def _client_identity(
    params: Mapping[str, str], ssl_mode: SslMode, read_file: Callable[[Path], bytes]
) -> NativeClientIdentity | None:
    identity_path: Final = params.get("sslidentity")
    if identity_path is None or ssl_mode == "disable":
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


def _session_options(params: Mapping[str, str]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    raw: Final = params.get("options", "")
    try:
        tokens: Final[Sequence[str]] = fold_libpq_option_tokens(tuple(shlex.split(raw)))
    except ValueError:
        return (), (raw,)
    readable: Final = tuple((token, libpq_option_setting(token)) for token in tokens)
    schema: Final = params.get("schema", PRISMA_DEFAULT_SCHEMA)
    search_path: Final = () if schema == PRISMA_DEFAULT_SCHEMA else (("search_path", f'"{schema}"'),)
    settings: Final = tuple(
        (name, value) for name, _, value in (setting.partition("=") for _, setting in readable if setting is not None)
    )
    return (*search_path, *settings), tuple(token for token, setting in readable if setting is None)


def _seconds(params: Mapping[str, str], key: str, default: float | None = None) -> float | None:
    value: Final = params.get(key)
    if value is None:
        return default
    return None if float(value) == 0 else float(value)


def _statement_cache_capacity(params: Mapping[str, str]) -> int:
    if params.get("pgbouncer") == "true":
        return 0
    return int(params.get("statement_cache_size", PRISMA_DEFAULT_STATEMENT_CACHE_SIZE))


def _application_name(params: Mapping[str, str]) -> str:
    configured: Final = params.get("application_name", PRISMA_APPLICATION_NAME)
    return NATIVE_APPLICATION_NAME if configured == PRISMA_APPLICATION_NAME else configured


def native_endpoint(
    url: str,
    *,
    tls_offered: bool,
    max_connections: int,
    read_file: Callable[[Path], bytes] = Path.read_bytes,
) -> NativeEndpointSettings:
    """The native pool's endpoint for the Prisma URL ``url``, read the way Prisma reads it.

    ``host`` is Prisma's Unix socket directory. The pool takes ``max_connections``, the
    Rust share of the budget (``connection_budget``), and tags its connections
    ``NATIVE_APPLICATION_NAME``. An unset ``pool_timeout`` takes Prisma's default and
    0 waits forever.
    ``sslidentity`` / ``sslpassword`` are a PKCS#12 client certificate, handed over as PEM.
    ``options`` tokens that are not ``-c name=value`` settings cannot be sent by SQLx and
    come back in ``ignored_options`` instead of failing boot.
    """
    parsed: Final = urllib.parse.urlsplit(url)
    params: Final = _params(url)
    socket: Final = params.get("host")
    if not (parsed.hostname or socket) or not parsed.username:
        raise ValueError("DATABASE_URL must name a host and a user for the native pool")
    ssl_mode, ssl_root_cert = _tls(params, tls_offered)
    options, ignored = _session_options(params)
    return NativeEndpointSettings(
        endpoint=NativeDatabaseEndpoint(
            host=parsed.hostname or "localhost",
            port=parsed.port or int(DEFAULT_POSTGRES_PORT),
            socket=socket,
            username=urllib.parse.unquote(parsed.username),
            password=urllib.parse.unquote(parsed.password) if parsed.password else None,
            database=urllib.parse.unquote(parsed.path.lstrip("/")) or "postgres",
            ssl_mode=ssl_mode,
            ssl_root_cert=ssl_root_cert,
            ssl_client=_client_identity(params, ssl_mode, read_file),
            application_name=_application_name(params),
            options=options,
            statement_cache_capacity=_statement_cache_capacity(params),
            pool=NativePoolLimits(
                max_connections=max_connections,
                acquire_timeout_seconds=_seconds(params, "pool_timeout", PRISMA_DEFAULT_POOL_TIMEOUT_SECONDS),
                idle_timeout_seconds=_seconds(params, "max_idle_connection_lifetime"),
                socket_timeout_seconds=_seconds(params, "socket_timeout"),
            ),
        ),
        ignored_options=ignored,
    )


async def resolve_native_endpoint(url: str, max_connections: int) -> NativeEndpointSettings:
    tls_offered: Final = await server_offers_tls(url) if needs_tls_probe(url) else True
    return native_endpoint(url, tls_offered=tls_offered, max_connections=max_connections)


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
