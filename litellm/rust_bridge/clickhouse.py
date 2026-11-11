import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from litellm.constants import DEFAULT_AGENT_TRACING_RETENTION_DAYS, DEFAULT_CLICKHOUSE_DATABASE
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True, repr=False)
class SpendStorageConfig:
    url: str
    database: str = DEFAULT_CLICKHOUSE_DATABASE
    retention_days: int = DEFAULT_AGENT_TRACING_RETENTION_DAYS


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
        self._native: Final = _native.NativeClickHouseSpendStorage(
            _native.NativeClickHouseSpendConfig(config.database, config.url, config.retention_days)
        )

    async def ensure_schema(self) -> None:
        await self._native.ensure_schema()

    async def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> None:
        if table != "spend_logs":
            raise ValueError("The ClickHouse spend writer only accepts spend_logs")
        await self._native.insert_rows(rows)
