from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final, Protocol, TypeVar, runtime_checkable

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.constants import AGENT_TRACING_LIST_PAGE_SIZE, OTLP_MAX_ATTRIBUTE_VALUE_BYTES
from litellm.rust_bridge.loader import get_native_bridge
from litellm.rust_bridge.trace.generated.models import (
    ActivityAvailability,
    AgentRow,
    CountRow,
    ExecutionRow,
    LensAccessParams,
    LensContentParams,
    LensEvidenceParams,
    LensSampleParams,
    PartRow,
    TraceAgentRow,
    TraceAgentsParams,
)
from litellm.rust_bridge.trace.generated.responses import TraceSQLResponse
from litellm.rust_bridge.trace.generated.types import ReadQueryName
from litellm.rust_bridge.trace.queries import (
    LENS_AGENTS,
    LENS_AVAILABILITY,
    LENS_CONTENT,
    LENS_EVIDENCE,
    LENS_SAMPLE,
    TRACE_AGENTS,
    ClickHouseSQLEnvelope,
    ParamsT,
    ReadQuery,
    RowT,
)

from .generated.models import TraceQueryHelp
from .generated.types import (
    QueryScope,
    SpanDetail,
    SpanErrorPage,
    Trace,
    TracePage,
    TraceScope,
)


@dataclass(frozen=True, slots=True)
class Tenant:
    """Who sent the spans. Always taken from auth, never from span attributes."""

    team_id: str
    api_key_hash: str
    org_id: str = ""
    user_id: str = ""


_EMPTY_TENANT: Final = Tenant("", "")


class NativeStore(Protocol):
    def ensure_schema(self) -> Awaitable[None]: ...

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> Awaitable[None]: ...

    def ingest(
        self, payload: bytes, content_type: str | None, tenant: Mapping[str, str], logs: bool = False
    ) -> Awaitable[int]: ...

    def list_traces(
        self, scope: TraceScope, start_ms: int, end_ms: int, cursor: str | None, limit: int
    ) -> Awaitable[JsonValue]: ...

    def get_trace(
        self, trace_id: str, scope: TraceScope, trace_ref: str, cursor: str | None = None, page_size: int | None = None
    ) -> Awaitable[JsonValue]: ...

    def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str) -> Awaitable[JsonValue]: ...

    def get_span_error(
        self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str, cursor: str | None
    ) -> Awaitable[JsonValue]: ...

    def query_sql(self, sql: str, scope: QueryScope, secret: str) -> Awaitable[str]: ...

    def query_help(self, scope: QueryScope, secret: str) -> Awaitable[JsonValue]: ...

    def query(
        self, name: ReadQueryName, parameters: Mapping[str, str | int | float | Sequence[str]]
    ) -> Awaitable[str]: ...


@runtime_checkable
class NativeTraces(Protocol):
    NativeTraceConfig: type["NativeConfig"]
    NativeTraceStorage: type[NativeStore]

    def trace_encode_error(self, message: str) -> bytes: ...

    def trace_span_rows(
        self, body: bytes, content_type: str | None, tenant: Mapping[str, str], max_attribute_value_bytes: int
    ) -> list[dict[str, JsonValue]]: ...


QUERY_PARAMETERS: Final = TypeAdapter(dict[str, str | int | float | list[str]])
_SQL_ENVELOPE: Final = TypeAdapter(ClickHouseSQLEnvelope)
_HELP_RESPONSE: Final = TypeAdapter(TraceQueryHelp)
_TRACE_PAGE: Final = TypeAdapter(TracePage)
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
    def __init__(self, config: TraceStorageConfig | NativeStore) -> None:
        self._native: Final = self._transport(config)

    @staticmethod
    def _transport(config: TraceStorageConfig | NativeStore) -> NativeStore:
        if not isinstance(config, TraceStorageConfig):
            return config
        native: Final = _native()
        validated: Final = native.NativeTraceConfig(
            config.database,
            config.url,
            config.retention_days,
            config.max_attribute_value_bytes,
        )
        return native.NativeTraceStorage(validated)

    async def ensure_schema(self) -> None:
        await self._native.ensure_schema()

    async def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> None:
        await self._native.insert_rows(table, rows)

    async def ingest(self, payload: bytes, content_type: str | None, tenant: Tenant, logs: bool = False) -> int:
        return await self._native.ingest(payload, content_type, asdict(tenant), logs)

    async def list_traces(
        self,
        scope: TraceScope,
        start_ms: int,
        end_ms: int,
        cursor: str | None = None,
        limit: int = AGENT_TRACING_LIST_PAGE_SIZE,
    ) -> TracePage:
        result: Final = await self._native.list_traces(scope, start_ms, end_ms, cursor, limit)
        return _validate_query_response(_TRACE_PAGE, result)

    async def get_trace(
        self,
        trace_id: str,
        scope: TraceScope,
        trace_ref: str = "",
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> Trace | None:
        result: Final = await self._native.get_trace(trace_id, scope, trace_ref, cursor, page_size)
        return _validate_query_response(_TRACE, result)

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "") -> SpanDetail | None:
        result: Final = await self._native.get_span(trace_id, span_id, scope, trace_ref)
        return _validate_query_response(_SPAN_DETAIL, result)

    async def get_span_error(
        self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "", cursor: str | None = None
    ) -> SpanErrorPage | None:
        result: Final = await self._native.get_span_error(trace_id, span_id, scope, trace_ref, cursor)
        return _validate_query_response(_SPAN_ERROR_PAGE, result)

    async def query(self, query: ReadQuery[ParamsT, RowT], parameters: ParamsT) -> tuple[RowT, ...]:
        validated: Final = query.parameters.model_validate(parameters)
        result: Final = await self._native.query(query.name, QUERY_PARAMETERS.validate_python(validated.model_dump()))
        return _decode_query_response(query.response, result).data

    async def query_sql(self, sql: str, scope: QueryScope, secret: str) -> TraceSQLResponse:
        result: Final = await self._native.query_sql(sql, scope, secret)
        envelope: Final = _decode_query_response(_SQL_ENVELOPE, result)
        return TraceSQLResponse(data=envelope.data)

    async def query_help(self, scope: QueryScope, secret: str) -> TraceQueryHelp:
        result: Final = await self._native.query_help(scope, secret)
        return _validate_query_response(_HELP_RESPONSE, result)

    async def trace_agents(self, parameters: TraceAgentsParams) -> tuple[TraceAgentRow, ...]:
        return await self.query(TRACE_AGENTS, parameters)

    async def lens_sample(self, parameters: LensSampleParams) -> tuple[ExecutionRow, ...]:
        return await self.query(LENS_SAMPLE, parameters)

    async def lens_availability(self, parameters: LensAccessParams) -> tuple[ActivityAvailability, ...]:
        return await self.query(LENS_AVAILABILITY, parameters)

    async def lens_agents(self, parameters: LensAccessParams) -> tuple[AgentRow, ...]:
        return await self.query(LENS_AGENTS, parameters)

    async def lens_content(self, parameters: LensContentParams) -> tuple[PartRow, ...]:
        return await self.query(LENS_CONTENT, parameters)

    async def lens_evidence(self, parameters: LensEvidenceParams) -> tuple[CountRow, ...]:
        return await self.query(LENS_EVIDENCE, parameters)
