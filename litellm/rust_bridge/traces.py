from collections.abc import Awaitable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from typing_extensions import ReadOnly

from litellm.rust_bridge.loader import get_native_bridge


class DecodedEvent(TypedDict):
    name: ReadOnly[str]
    attributes: ReadOnly[dict[str, str]]


class NormalizedSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    observation_type: Literal["agent", "llm", "tool", "chain", "framework"]
    agent_name: str
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


ReadQueryName = Literal["list_traces", "trace_spans", "span_detail", "span_error", "spend_by_response_ids"]


class NativeStore(Protocol):
    def __init__(self, database: str, url: str, reader_url: str | None = None) -> None: ...

    def ensure_schema(self, trace_retention_days: int, spend_log_retention_days: int) -> Awaitable[None]: ...

    def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> Awaitable[None]: ...

    def lens_query(self, name: str, parameters: Mapping[str, str | int | Sequence[str]]) -> Awaitable[str]: ...

    def query(self, name: ReadQueryName, parameters: Mapping[str, str | int | Sequence[str]]) -> Awaitable[str]: ...


class NativeTraces(Protocol):
    NativeTraceStorage: type[NativeStore]

    def trace_decode_otlp(
        self,
        body: bytes,
        content_type: str | None,
    ) -> list[DecodedSpan]: ...

    def trace_encode_error(self, message: str) -> bytes: ...

    def trace_normalized_field_definitions(self) -> list[dict[str, str]]: ...


class QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: list[dict[str, JsonValue]]


QUERY_PARAMETERS: Final = TypeAdapter(dict[str, str | int | list[str]])
_FIELD_DEFINITIONS_ADAPTER: Final = TypeAdapter(tuple[NormalizedFieldDefinition, ...])


def _native() -> NativeTraces:
    native: Final = get_native_bridge()
    if native is None:
        raise RuntimeError("Agent tracing requires the Rust extension")
    return cast(NativeTraces, native)  # cast-ok: the native extension is validated against this protocol at call sites


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


class ClickHouseStorage:
    def __init__(self, database: str, url: str, reader_url: str | None = None) -> None:
        self._native: Final = _native().NativeTraceStorage(database, url, reader_url)

    async def ensure_schema(self, trace_retention_days: int, spend_log_retention_days: int) -> None:
        await self._native.ensure_schema(trace_retention_days, spend_log_retention_days)

    async def insert_rows(self, table: str, rows: Sequence[Mapping[str, object]]) -> None:
        await self._native.insert_rows(table, rows)

    async def query(
        self, name: ReadQueryName, parameters: Mapping[str, object] | None = None
    ) -> list[dict[str, JsonValue]]:
        result: Final = await self._native.query(
            name, QUERY_PARAMETERS.validate_python(parameters or MappingProxyType({}))
        )
        return QueryResponse.model_validate_json(result).data

    async def _lens_query(self, name: str, parameters: Mapping[str, object]) -> list[dict[str, JsonValue]]:
        result: Final = await self._native.lens_query(name, QUERY_PARAMETERS.validate_python(parameters))
        return QueryResponse.model_validate_json(result).data

    async def lens_sample(self, parameters: Mapping[str, object]) -> list[dict[str, JsonValue]]:
        return await self._lens_query("sample", parameters)

    async def lens_availability(self, parameters: Mapping[str, object]) -> list[dict[str, JsonValue]]:
        return await self._lens_query("availability", parameters)

    async def lens_agents(self, parameters: Mapping[str, object]) -> list[dict[str, JsonValue]]:
        return await self._lens_query("agents", parameters)

    async def lens_content(self, parameters: Mapping[str, object]) -> list[dict[str, JsonValue]]:
        return await self._lens_query("content", parameters)

    async def lens_evidence(self, parameters: Mapping[str, object]) -> list[dict[str, JsonValue]]:
        return await self._lens_query("evidence", parameters)
