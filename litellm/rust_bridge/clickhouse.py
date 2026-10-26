import os
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import ConfigDict, TypeAdapter

from litellm.constants import DEFAULT_AGENT_TRACING_RETENTION_DAYS, DEFAULT_CLICKHOUSE_DATABASE
from litellm.rust_bridge.loader import get_native_bridge


@dataclass(frozen=True, slots=True, repr=False)
class SpendStorageConfig:
    url: str
    database: str = DEFAULT_CLICKHOUSE_DATABASE
    retention_days: int = DEFAULT_AGENT_TRACING_RETENTION_DAYS


class RollupDimension(str, Enum):
    """Columns of the ClickHouse `spend_logs` table the Rust rollup can group or filter by."""

    TAG = "tag"
    API_KEY = "api_key"
    TEAM_ID = "team_id"
    ORGANIZATION_ID = "organization_id"
    USER = "user"
    END_USER = "end_user"
    MODEL = "model"
    MODEL_GROUP = "model_group"
    PROVIDER = "custom_llm_provider"
    STATUS = "status"
    CALL_TYPE = "call_type"


RollupBucket = Literal["1m", "5m", "15m", "1h", "1d"]
ROLLUP_BUCKETS: Final[tuple[RollupBucket, ...]] = ("1m", "5m", "15m", "1h", "1d")
ROLLUP_ROW_LIMIT: Final = 1000


def _no_filters() -> Mapping[RollupDimension, tuple[str, ...]]:
    return {}


@dataclass(frozen=True, slots=True)
class SpendRollupRequest:
    """One bounded GROUP BY over `spend_logs`; Rust owns the SQL, this only names the shape."""

    start: datetime
    end: datetime
    bucket: RollupBucket | None
    group_by: tuple[RollupDimension, ...] = ()
    filters: Mapping[RollupDimension, tuple[str, ...]] = field(default_factory=_no_filters)
    limit: int = ROLLUP_ROW_LIMIT

    def native(self) -> dict[str, object]:
        return {
            "start_ms": _epoch_ms(self.start),
            "end_ms": _epoch_ms(self.end),
            "bucket": self.bucket,
            "group_by": [dimension.value for dimension in self.group_by],
            "filters": {dimension.value: list(values) for dimension, values in self.filters.items()},
            "limit": self.limit,
        }


def _epoch_ms(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("rollup timestamps must be timezone-aware")
    return int(value.timestamp() * 1000)


class NativeSpendConfig(Protocol):
    def __init__(self, database: str, url: str, retention_days: int) -> None: ...


class NativeSpendStorage(Protocol):
    def __init__(self, config: NativeSpendConfig) -> None: ...
    def ensure_schema(self) -> Awaitable[None]: ...
    def insert_rows(self, rows: Sequence[Mapping[str, object]]) -> Awaitable[None]: ...
    def fetch_rollup(self, request: Mapping[str, object]) -> Awaitable[list[dict[str, object]]]: ...


@runtime_checkable
class NativeSpendBridge(Protocol):
    NativeClickHouseSpendConfig: type[NativeSpendConfig]
    NativeClickHouseSpendStorage: type[NativeSpendStorage]


_NATIVE: Final[TypeAdapter[NativeSpendBridge]] = TypeAdapter(
    NativeSpendBridge, config=ConfigDict(arbitrary_types_allowed=True)
)


def spend_storage_config(environ: Mapping[str, str] = os.environ) -> SpendStorageConfig:
    url: Final = environ.get("CLICKHOUSE_URL")
    if not url:
        raise ValueError("CLICKHOUSE_URL is required")
    value: Final = environ.get("AGENT_TRACING_RETENTION_DAYS", str(DEFAULT_AGENT_TRACING_RETENTION_DAYS))
    try:
        retention: Final = int(value)
    except ValueError as error:
        raise ValueError("AGENT_TRACING_RETENTION_DAYS must be a positive integer") from error
    if not 0 < retention <= 2**32 - 1:
        raise ValueError("AGENT_TRACING_RETENTION_DAYS must be a positive integer")
    return SpendStorageConfig(url, environ.get("CLICKHOUSE_DATABASE", DEFAULT_CLICKHOUSE_DATABASE), retention)


class ClickHouseSpendStorage:
    def __init__(self, config: SpendStorageConfig) -> None:
        bridge: Final = get_native_bridge()
        if bridge is None:
            raise RuntimeError("ClickHouse spend logging requires the Rust extension")
        native: Final = _NATIVE.validate_python(bridge)
        self._native: Final = native.NativeClickHouseSpendStorage(
            native.NativeClickHouseSpendConfig(config.database, config.url, config.retention_days)
        )

    async def ensure_schema(self) -> None:
        await self._native.ensure_schema()

    async def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> None:
        if table != "spend_logs":
            raise ValueError("The ClickHouse spend writer only accepts spend_logs")
        await self._native.insert_rows(rows)

    async def fetch_rollup(self, request: SpendRollupRequest) -> tuple[Mapping[str, object], ...]:
        return tuple(await self._native.fetch_rollup(request.native()))
