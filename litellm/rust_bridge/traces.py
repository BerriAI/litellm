from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypedDict, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError
from typing_extensions import ReadOnly

from litellm.rust_bridge.loader import get_native_bridge
from litellm.rust_bridge.trace_queries import (
    LENS_AGENTS,
    LENS_AVAILABILITY,
    LENS_CONTENT,
    LENS_EVIDENCE,
    LENS_SAMPLE,
    ActivityAvailability,
    AgentRow,
    CountRow,
    ExecutionRow,
    LensAccessParams,
    LensContentParams,
    LensEvidenceParams,
    LensSampleParams,
    ParamsT,
    PartRow,
    ReadQuery,
    ReadQueryName,
    RowT,
)
from litellm.rust_bridge.trace_query_responses import TraceQueryHelp, TraceSQLResponse


class DecodedEvent(TypedDict):
    name: ReadOnly[str]
    attributes: ReadOnly[dict[str, str]]


class NormalizedSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    observation_type: Literal["agent", "llm", "tool", "chain", "framework"]
    agent_name: str
    framework: str
    litellm_request_id: str
    model: str
    input_tokens: int = Field(ge=0, le=2**32 - 1)
    output_tokens: int = Field(ge=0, le=2**32 - 1)
    input: str
    output: str


class NormalizedFieldDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    clickhouse_column: str
    clickhouse_type: str
    meaning: str


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
    normalized: ReadOnly[NormalizedSpan]
    consumed_attributes: ReadOnly[tuple[str, str]]


class AllQueryScope(TypedDict):
    kind: ReadOnly[Literal["all"]]


class OwnedQueryScope(TypedDict):
    kind: ReadOnly[Literal["owned"]]
    user_id: ReadOnly[str]
    team_ids: ReadOnly[tuple[str, ...]]


QueryScope = AllQueryScope | OwnedQueryScope


class NativeStore(Protocol):
    def __init__(self, config: "NativeConfig") -> None: ...

    def ensure_schema(self) -> Awaitable[None]: ...

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> Awaitable[None]: ...

    def query_sql(self, sql: str, scope: QueryScope, secret: str) -> Awaitable[str]: ...

    def query_help(self, scope: QueryScope, secret: str) -> Awaitable[JsonValue]: ...

    def query(
        self, name: ReadQueryName, parameters: Mapping[str, str | int | float | Sequence[str]]
    ) -> Awaitable[str]: ...


@runtime_checkable
class NativeTraces(Protocol):
    NativeTraceConfig: type["NativeConfig"]
    NativeTraceStorage: type[NativeStore]

    def trace_decode_otlp(
        self,
        body: bytes,
        content_type: str | None,
    ) -> list[DecodedSpan]: ...

    def trace_encode_error(self, message: str) -> bytes: ...

    def trace_normalized_field_definitions(self) -> list[dict[str, str]]: ...


QUERY_PARAMETERS: Final = TypeAdapter(dict[str, str | int | float | list[str]])
_FIELD_DEFINITIONS_ADAPTER: Final = TypeAdapter(tuple[NormalizedFieldDefinition, ...])
_SQL_RESPONSE: Final = TypeAdapter(TraceSQLResponse)
_HELP_RESPONSE: Final = TypeAdapter(TraceQueryHelp)
_ResponseT: Final = TypeVar("_ResponseT")
_NATIVE_ADAPTER: Final[TypeAdapter[NativeTraces]] = TypeAdapter(
    NativeTraces, config=ConfigDict(arbitrary_types_allowed=True)
)


class NativeConfig(Protocol):
    def __init__(self, database: str, url: str, retention_days: int) -> None: ...


@dataclass(frozen=True, slots=True, repr=False)
class TraceStorageConfig:
    url: str
    database: str = "litellm"
    retention_days: int = 14


def _native() -> NativeTraces:
    native: Final = get_native_bridge()
    if native is None:
        raise RuntimeError("Agent tracing requires the Rust extension")
    return _NATIVE_ADAPTER.validate_python(native)


def decode_otlp(body: bytes, content_type: str | None) -> list[DecodedSpan]:
    return [
        {**span, "normalized": NormalizedSpan.model_validate(span["normalized"])}
        for span in _native().trace_decode_otlp(body, content_type)
    ]


def normalized_field_definitions() -> tuple[NormalizedFieldDefinition, ...]:
    fields: Final = _FIELD_DEFINITIONS_ADAPTER.validate_python(_native().trace_normalized_field_definitions())
    if frozenset(field.name for field in fields) != frozenset(NormalizedSpan.model_fields):
        raise ValueError("Rust and Python normalized trace fields disagree")
    return fields


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
        )
        self._native: Final = native.NativeTraceStorage(validated)

    async def ensure_schema(self) -> None:
        await self._native.ensure_schema()

    async def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> None:
        await self._native.insert_rows(table, rows)

    async def query(self, query: ReadQuery[ParamsT, RowT], parameters: ParamsT) -> tuple[RowT, ...]:
        validated: Final = query.parameters.model_validate(parameters)
        result: Final = await self._native.query(query.name, QUERY_PARAMETERS.validate_python(validated.model_dump()))
        return _decode_query_response(query.response, result).data

    async def query_sql(self, sql: str, scope: QueryScope, secret: str) -> TraceSQLResponse:
        result: Final = await self._native.query_sql(sql, scope, secret)
        return _decode_query_response(_SQL_RESPONSE, result)

    async def query_help(self, scope: QueryScope, secret: str) -> TraceQueryHelp:
        result: Final = await self._native.query_help(scope, secret)
        return _validate_query_response(_HELP_RESPONSE, result)

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
