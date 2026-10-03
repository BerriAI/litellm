import os
from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.constants import DEFAULT_AGENT_TRACING_RETENTION_DAYS, DEFAULT_CLICKHOUSE_DATABASE
from litellm.rust_bridge.trace.storage import TraceStorageConfig

STORE_SETTINGS: Final = TypeAdapter(dict[str, object])


def is_clickhouse_tracing_enabled(settings: object) -> bool:
    if not isinstance(settings, Mapping):
        return False
    typed_settings: Final = STORE_SETTINGS.validate_python(settings)
    store: Final = typed_settings.get("store")
    if not isinstance(store, Mapping):
        return False
    return STORE_SETTINGS.validate_python(store).get("type") == "clickhouse"


def _value(settings: Mapping[str, object], field: str, environ: Mapping[str, str], default: object) -> object:
    if field not in settings:
        return default
    supplied: Final = settings[field]
    resolved: Final = (
        environ.get(supplied.removeprefix("os.environ/"))
        if isinstance(supplied, str) and supplied.startswith("os.environ/")
        else supplied
    )
    if resolved is None:
        raise ValueError(f"tracing.store.{field} is set but resolved to no value")
    return resolved


def _retention_days(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("tracing.store.retention_days must be a positive integer")
    try:
        days: Final = int(value)
    except ValueError as error:
        raise ValueError("tracing.store.retention_days must be a positive integer") from error
    if not 0 < days <= 2**32 - 1:
        raise ValueError("tracing.store.retention_days must be a positive integer")
    return days


def _clickhouse_store(settings: Mapping[str, object]) -> Mapping[str, object]:
    raw_store: Final = settings.get("store")
    if raw_store is None:
        return {}
    if isinstance(raw_store, Mapping):
        store: Final = STORE_SETTINGS.validate_python(raw_store)
        if store.get("type") == "clickhouse":
            return store
    raise ValueError("tracing.store.type must be clickhouse")


def trace_storage_config(settings: Mapping[str, object], environ: Mapping[str, str] = os.environ) -> TraceStorageConfig:
    store: Final = _clickhouse_store(settings)
    unknown: Final = store.keys() - {"type", "url", "database", "retention_days"}
    if unknown:
        raise ValueError(f"unsupported tracing.store settings: {', '.join(sorted(unknown))}")
    url: Final = _value(store, "url", environ, environ.get("CLICKHOUSE_URL"))
    database: Final = _value(
        store, "database", environ, environ.get("CLICKHOUSE_DATABASE", DEFAULT_CLICKHOUSE_DATABASE)
    )
    if not isinstance(url, str) or not url:
        raise ValueError("tracing.store.url or CLICKHOUSE_URL is required")
    if not isinstance(database, str):
        raise ValueError("tracing.store.database must be a string")
    return TraceStorageConfig(
        url=url,
        database=database,
        retention_days=_retention_days(
            _value(
                store,
                "retention_days",
                environ,
                environ.get("AGENT_TRACING_RETENTION_DAYS", DEFAULT_AGENT_TRACING_RETENTION_DAYS),
            )
        ),
    )
