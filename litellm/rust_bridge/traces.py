from collections.abc import Awaitable, Mapping, Sequence
from typing import Final, Protocol, TypedDict, cast

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from typing_extensions import ReadOnly

from litellm.rust_bridge.loader import get_native_bridge


class DecodedEvent(TypedDict):
    name: ReadOnly[str]
    attributes: ReadOnly[dict[str, str]]


class DecodedSpan(TypedDict):
    trace_id: ReadOnly[str]
    span_id: ReadOnly[str]
    parent_span_id: ReadOnly[str]
    trace_state: ReadOnly[str]
    name: ReadOnly[str]
    kind: ReadOnly[str]
    resource_attributes: ReadOnly[dict[str, str]]
    scope_name: ReadOnly[str]
    scope_version: ReadOnly[str]
    attributes: ReadOnly[dict[str, str]]
    start_ns: ReadOnly[int]
    end_ns: ReadOnly[int]
    status_code: ReadOnly[str]
    status_message: ReadOnly[str]
    events: ReadOnly[list[DecodedEvent]]


class NativeStore(Protocol):
    def __init__(self, database: str, url: str, reader_url: str | None = None) -> None: ...

    def ensure_schema(self, trace_retention_days: int, spend_log_retention_days: int) -> Awaitable[None]: ...

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, JsonValue]]) -> Awaitable[None]: ...

    def query(self, sql: str, parameters: Mapping[str, str | int | Sequence[str]]) -> Awaitable[str]: ...


class NativeTraces(Protocol):
    NativeTraceStorage: type[NativeStore]

    def trace_decode_otlp(
        self,
        body: bytes,
        content_type: str | None,
        content_encoding: str | None,
        max_decompressed_bytes: int,
    ) -> list[DecodedSpan]: ...

class QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: list[dict[str, JsonValue]]


INSERT_ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])
QUERY_PARAMETERS: Final = TypeAdapter(dict[str, str | int | list[str]])


def _native() -> NativeTraces:
    native: Final = get_native_bridge()
    if native is None:
        raise RuntimeError("Agent tracing requires the Rust extension")
    return cast(NativeTraces, native)  # cast-ok: the native extension is validated against this protocol at call sites


def decode_otlp(
    body: bytes, content_type: str | None, content_encoding: str | None, max_decompressed_bytes: int
) -> list[DecodedSpan]:
    return _native().trace_decode_otlp(body, content_type, content_encoding, max_decompressed_bytes)


class TraceStorage:
    def __init__(self, database: str, url: str, reader_url: str | None = None) -> None:
        self._native: Final = _native().NativeTraceStorage(database, url, reader_url)

    async def ensure_schema(self, trace_retention_days: int, spend_log_retention_days: int) -> None:
        await self._native.ensure_schema(trace_retention_days, spend_log_retention_days)

    async def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> None:
        await self._native.insert_rows(table, INSERT_ROWS.validate_python(rows))

    async def query(self, sql: str, parameters: Mapping[str, object] | None = None) -> list[dict[str, JsonValue]]:
        result: Final = await self._native.query(sql, QUERY_PARAMETERS.validate_python(parameters or {}))
        return QueryResponse.model_validate_json(result).data
