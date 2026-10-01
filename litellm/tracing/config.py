import os
from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.rust_bridge.traces import TraceStorageConfig

STORE_SETTINGS: Final = TypeAdapter(dict[str, object])


def is_clickhouse_tracing_enabled(settings: object) -> bool:
    if not isinstance(settings, Mapping):
        return False
    typed_settings: Final = STORE_SETTINGS.validate_python(settings)
    store: Final = typed_settings.get("store")
    if store == "clickhouse":
        return True
    if not isinstance(store, Mapping):
        return False
    return STORE_SETTINGS.validate_python(store).get("type") == "clickhouse"


def _value(settings: Mapping[str, object], field: str, environ: Mapping[str, str], env_name: str) -> object:
    supplied: Final = settings.get(field)
    if supplied is None:
        return environ.get(env_name)
    if isinstance(supplied, str) and supplied.startswith("os.environ/"):
        return environ.get(supplied.removeprefix("os.environ/"))
    return supplied


def _retention_days(value: object) -> int:
    if value is None:
        return 14
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("tracing.store.retention_days must be a positive integer")
    try:
        days: Final = int(value)
    except ValueError as error:
        raise ValueError("tracing.store.retention_days must be a positive integer") from error
    if not 0 < days <= 2**32 - 1:
        raise ValueError("tracing.store.retention_days must be a positive integer")
    return days


def _retention_value(settings: Mapping[str, object], environ: Mapping[str, str]) -> object:
    if "retention_days" in settings:
        return _value(settings, "retention_days", environ, "AGENT_TRACING_RETENTION_DAYS")
    trace_days: Final = environ.get("AGENT_TRACING_RETENTION_DAYS")
    spend_days: Final = environ.get("AGENT_TRACING_SPEND_LOG_RETENTION_DAYS")
    if trace_days is not None and spend_days is not None and trace_days != spend_days:
        raise ValueError("legacy tracing retention values differ; set tracing.store.retention_days")
    return trace_days if trace_days is not None else spend_days


def _clickhouse_store(settings: Mapping[str, object]) -> Mapping[str, object]:
    raw_store: Final = settings.get("store", "clickhouse")
    if raw_store == "clickhouse":
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
    url: Final = _value(store, "url", environ, "CLICKHOUSE_URL")
    configured_database: Final = _value(store, "database", environ, "CLICKHOUSE_DATABASE")
    database: Final = "litellm" if configured_database is None else configured_database
    if not isinstance(url, str) or not url:
        raise ValueError("tracing.store.url or CLICKHOUSE_URL is required")
    if not isinstance(database, str):
        raise ValueError("tracing.store.database must be a string")
    return TraceStorageConfig(
        url=url,
        database=database,
        retention_days=_retention_days(_retention_value(store, environ)),
    )
