from typing import Literal, Protocol, TypeAlias

from typing_extensions import ReadOnly, TypedDict

SslMode: TypeAlias = Literal["disable", "prefer", "require", "verify-ca", "verify-full"]


class NativePoolLimits(TypedDict):
    max_connections: ReadOnly[int | None]
    acquire_timeout_seconds: ReadOnly[float | None]
    idle_timeout_seconds: ReadOnly[float | None]


class NativeDatabaseEndpoint(TypedDict):
    host: ReadOnly[str]
    port: ReadOnly[int]
    username: ReadOnly[str]
    password: ReadOnly[str | None]
    database: ReadOnly[str]
    ssl_mode: ReadOnly[SslMode]
    ssl_root_cert: ReadOnly[str | None]
    options: ReadOnly[tuple[tuple[str, str], ...]]
    statement_cache: ReadOnly[bool]
    pool: ReadOnly[NativePoolLimits]


class NativeDatabaseEvents(Protocol):
    def record(
        self,
        call_type: str,
        table: str | None,
        operation: str | None,
        started_at: float,
        duration_seconds: float,
        failure: str | None,
    ) -> None: ...
