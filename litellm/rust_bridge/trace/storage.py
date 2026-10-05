from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final, Literal, Protocol, TypeAlias, TypeVar, runtime_checkable

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.constants import AGENT_TRACING_LIST_PAGE_SIZE, OTLP_MAX_ATTRIBUTE_VALUE_BYTES
from litellm.rust_bridge.loader import get_native_bridge

from .generated.models import TraceQueryHelp
from .generated.types import (
    QueryScope,
    RunOrder,
    RunValues,
    SpanDetail,
    SpanErrorPage,
    SpanText,
    Trace,
    TraceHistogram,
    TracePage,
)
from .queries import TraceSQLResponse

SpanPart: TypeAlias = Literal["input", "output", "error", "attributes"]


@dataclass(frozen=True, slots=True)
class Tenant:
    """Who sent the spans. Always taken from auth, never from span attributes."""

    team_id: str
    api_key_hash: str
    org_id: str = ""
    user_id: str = ""


_EMPTY_TENANT: Final = Tenant("", "")

NEWEST: Final[RunOrder] = {"key": "start_ms", "descending": True}
BY_REFERENCE: Final[RunOrder] = {"key": "trace_ref", "descending": False}


class NativeStore(Protocol):
    def __init__(self, config: "NativeConfig") -> None: ...

    def ensure_schema(self) -> Awaitable[None]: ...

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> Awaitable[None]: ...

    def ingest(self, payload: bytes, content_type: str | None, tenant: Mapping[str, str]) -> Awaitable[int]: ...

    def list_traces(
        self,
        scope: QueryScope,
        start_ms: int,
        end_ms: int,
        q: str,
        cursor: str | None,
        limit: int,
        order: RunOrder,
        trace_refs: Sequence[str],
    ) -> Awaitable[JsonValue]: ...

    def count_traces(
        self, scope: QueryScope, start_ms: int, end_ms: int, q: str, trace_refs: Sequence[str]
    ) -> Awaitable[JsonValue]: ...

    def span_text(
        self,
        trace_id: str,
        trace_ref: str,
        span_ids: Sequence[str],
        part: SpanPart,
        scope: QueryScope,
        offset: int,
        max_chars: int | None,
        tail: bool,
        contains: str | None,
    ) -> Awaitable[JsonValue]: ...

    def trace_histogram(
        self, scope: QueryScope, start_ms: int, end_ms: int, q: str, buckets: int
    ) -> Awaitable[JsonValue]: ...

    def run_values(
        self, scope: QueryScope, start_ms: int, end_ms: int, q: str, field: str, contains: str, limit: int
    ) -> Awaitable[JsonValue]: ...

    def get_trace(
        self, trace_id: str, scope: QueryScope, trace_ref: str, cursor: str | None = None, page_size: int | None = None
    ) -> Awaitable[JsonValue]: ...

    def get_span(self, trace_id: str, span_id: str, scope: QueryScope, trace_ref: str) -> Awaitable[JsonValue]: ...

    def get_span_error(
        self, trace_id: str, span_id: str, scope: QueryScope, trace_ref: str, cursor: str | None
    ) -> Awaitable[JsonValue]: ...

    def query_sql(self, sql: str, scope: QueryScope, secret: str) -> Awaitable[str]: ...

    def query_help(self, scope: QueryScope, secret: str) -> Awaitable[JsonValue]: ...


@runtime_checkable
class NativeTraces(Protocol):
    NativeTraceConfig: type["NativeConfig"]
    NativeTraceStorage: type[NativeStore]

    def trace_encode_error(self, message: str) -> bytes: ...

    def trace_span_rows(
        self, body: bytes, content_type: str | None, tenant: Mapping[str, str], max_attribute_value_bytes: int
    ) -> list[dict[str, JsonValue]]: ...


_SQL_RESPONSE: Final = TypeAdapter(TraceSQLResponse)
_HELP_RESPONSE: Final = TypeAdapter(TraceQueryHelp)
_TRACE_PAGE: Final = TypeAdapter(TracePage)
_TRACE_HISTOGRAM: Final = TypeAdapter(TraceHistogram)
_RUN_VALUES: Final = TypeAdapter(RunValues)
_COUNT: Final = TypeAdapter(int)
_SPAN_TEXTS: Final = TypeAdapter(tuple[SpanText, ...])
_TRACE: Final[TypeAdapter[Trace | None]] = TypeAdapter(Trace | None)
_SPAN_DETAIL: Final[TypeAdapter[SpanDetail | None]] = TypeAdapter(SpanDetail | None)
_SPAN_ERROR_PAGE: Final[TypeAdapter[SpanErrorPage | None]] = TypeAdapter(SpanErrorPage | None)
_ResponseT: Final = TypeVar("_ResponseT")
_NATIVE_ADAPTER: Final[TypeAdapter[NativeTraces]] = TypeAdapter(
    NativeTraces, config=ConfigDict(arbitrary_types_allowed=True)
)


class NativeConfig(Protocol):
    def __init__(self, database: str, url: str, retention_days: int, max_attribute_value_bytes: int) -> None: ...


@dataclass(frozen=True, slots=True, repr=False)
class TraceStorageConfig:
    url: str
    database: str = "litellm"
    retention_days: int = 14
    max_attribute_value_bytes: int = OTLP_MAX_ATTRIBUTE_VALUE_BYTES


def _native() -> NativeTraces:
    native: Final = get_native_bridge()
    if native is None:
        raise RuntimeError("Agent tracing requires the Rust extension")
    return _NATIVE_ADAPTER.validate_python(native)


def span_rows(
    body: bytes,
    content_type: str | None,
    tenant: Tenant = _EMPTY_TENANT,
    max_attribute_value_bytes: int = OTLP_MAX_ATTRIBUTE_VALUE_BYTES,
) -> list[dict[str, JsonValue]]:
    """The `otel_traces` rows an OTLP export would be stored as, without writing them."""
    return _native().trace_span_rows(body, content_type, asdict(tenant), max_attribute_value_bytes)


def encode_error(message: str) -> bytes:
    if get_native_bridge() is None:
        return b""
    return _native().trace_encode_error(message)


def _decode_query_response(adapter: TypeAdapter[_ResponseT], body: str) -> _ResponseT:
    try:
        return adapter.validate_json(body)
    except ValidationError as error:
        raise RuntimeError("Native trace query returned an invalid response") from error


def _validate_query_response(adapter: TypeAdapter[_ResponseT], value: JsonValue) -> _ResponseT:
    try:
        return adapter.validate_python(value)
    except ValidationError as error:
        raise RuntimeError("Native trace query returned an invalid response") from error


class ClickHouseStorage:
    def __init__(self, config: TraceStorageConfig) -> None:
        native: Final = _native()
        validated: Final = native.NativeTraceConfig(
            config.database,
            config.url,
            config.retention_days,
            config.max_attribute_value_bytes,
        )
        self._native: Final = native.NativeTraceStorage(validated)

    async def ensure_schema(self) -> None:
        await self._native.ensure_schema()

    async def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> None:
        await self._native.insert_rows(table, rows)

    async def ingest(self, payload: bytes, content_type: str | None, tenant: Tenant) -> int:
        return await self._native.ingest(payload, content_type, asdict(tenant))

    async def list_traces(
        self,
        scope: QueryScope,
        start_ms: int,
        end_ms: int,
        q: str = "",
        cursor: str | None = None,
        limit: int = AGENT_TRACING_LIST_PAGE_SIZE,
        order: RunOrder = NEWEST,
        trace_refs: Sequence[str] = (),
    ) -> TracePage:
        result: Final = await self._native.list_traces(
            scope, start_ms, end_ms, q, cursor, limit, order, tuple(trace_refs)
        )
        return _validate_query_response(_TRACE_PAGE, result)

    async def count_traces(
        self, scope: QueryScope, start_ms: int, end_ms: int, q: str = "", trace_refs: Sequence[str] = ()
    ) -> int:
        result: Final = await self._native.count_traces(scope, start_ms, end_ms, q, tuple(trace_refs))
        return _validate_query_response(_COUNT, result)

    async def span_text(
        self,
        trace_id: str,
        trace_ref: str,
        span_ids: Sequence[str],
        part: SpanPart,
        scope: QueryScope,
        offset: int = 0,
        max_chars: int | None = None,
        tail: bool = False,
        contains: str | None = None,
    ) -> tuple[SpanText, ...]:
        result: Final = await self._native.span_text(
            trace_id, trace_ref, tuple(span_ids), part, scope, offset, max_chars, tail, contains
        )
        return _validate_query_response(_SPAN_TEXTS, result)

    async def trace_histogram(
        self, scope: QueryScope, start_ms: int, end_ms: int, q: str, buckets: int
    ) -> TraceHistogram:
        result: Final = await self._native.trace_histogram(scope, start_ms, end_ms, q, buckets)
        return _validate_query_response(_TRACE_HISTOGRAM, result)

    async def run_values(
        self, scope: QueryScope, start_ms: int, end_ms: int, q: str, field: str, contains: str, limit: int
    ) -> RunValues:
        result: Final = await self._native.run_values(scope, start_ms, end_ms, q, field, contains, limit)
        return _validate_query_response(_RUN_VALUES, result)

    async def get_trace(
        self,
        trace_id: str,
        scope: QueryScope,
        trace_ref: str = "",
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> Trace | None:
        result: Final = await self._native.get_trace(trace_id, scope, trace_ref, cursor, page_size)
        return _validate_query_response(_TRACE, result)

    async def get_span(self, trace_id: str, span_id: str, scope: QueryScope, trace_ref: str = "") -> SpanDetail | None:
        result: Final = await self._native.get_span(trace_id, span_id, scope, trace_ref)
        return _validate_query_response(_SPAN_DETAIL, result)

    async def get_span_error(
        self, trace_id: str, span_id: str, scope: QueryScope, trace_ref: str = "", cursor: str | None = None
    ) -> SpanErrorPage | None:
        result: Final = await self._native.get_span_error(trace_id, span_id, scope, trace_ref, cursor)
        return _validate_query_response(_SPAN_ERROR_PAGE, result)

    async def query_sql(self, sql: str, scope: QueryScope, secret: str) -> TraceSQLResponse:
        result: Final = await self._native.query_sql(sql, scope, secret)
        return _decode_query_response(_SQL_RESPONSE, result)

    async def query_help(self, scope: QueryScope, secret: str) -> TraceQueryHelp:
        result: Final = await self._native.query_help(scope, secret)
        return _validate_query_response(_HELP_RESPONSE, result)
