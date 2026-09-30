from collections.abc import Awaitable, Mapping, Sequence
from typing import Final, Protocol, cast

from pydantic import BaseModel, ConfigDict, JsonValue

from litellm.rust_bridge.loader import get_native_bridge


class NativeTraces(Protocol):
    def trace_schema_statements(
        self, database: str, trace_retention_days: int, spend_log_retention_days: int
    ) -> list[str]: ...

    def trace_query(
        self,
        url: str,
        database: str,
        user: str,
        password: str,
        sql: str,
        parameters: Mapping[str, str | int | Sequence[str]],
    ) -> Awaitable[str]: ...


class QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: list[dict[str, JsonValue]]


def _native() -> NativeTraces:
    native: Final = get_native_bridge()
    if native is None:
        raise RuntimeError("Agent tracing requires the Rust extension")
    return cast(NativeTraces, native)


def schema_statements(database: str, trace_retention_days: int, spend_log_retention_days: int) -> list[str]:
    return _native().trace_schema_statements(database, trace_retention_days, spend_log_retention_days)


async def query(
    url: str,
    database: str,
    user: str,
    password: str,
    sql: str,
    parameters: Mapping[str, str | int | Sequence[str]],
) -> list[dict[str, JsonValue]]:
    result: Final = await _native().trace_query(url, database, user, password, sql, parameters)
    return QueryResponse.model_validate_json(result).data
