from typing import Literal, TypeAlias

from typing_extensions import ReadOnly, TypedDict

SslMode: TypeAlias = Literal["disable", "prefer", "require", "verify-ca", "verify-full"]


class NativePoolLimits(TypedDict):
    max_connections: ReadOnly[int]
    acquire_timeout_seconds: ReadOnly[float | None]
    idle_timeout_seconds: ReadOnly[float | None]
    socket_timeout_seconds: ReadOnly[float | None]


class NativeClientIdentity(TypedDict):
    cert_pem: ReadOnly[str]
    key_pem: ReadOnly[str]


class NativeDatabaseEndpoint(TypedDict):
    host: ReadOnly[str]
    port: ReadOnly[int]
    socket: ReadOnly[str | None]
    username: ReadOnly[str]
    password: ReadOnly[str | None]
    database: ReadOnly[str]
    ssl_mode: ReadOnly[SslMode]
    ssl_root_cert: ReadOnly[str | None]
    ssl_client: ReadOnly[NativeClientIdentity | None]
    application_name: ReadOnly[str | None]
    options: ReadOnly[tuple[tuple[str, str], ...]]
    statement_cache_capacity: ReadOnly[int]
    pool: ReadOnly[NativePoolLimits]
