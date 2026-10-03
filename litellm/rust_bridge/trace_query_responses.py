from collections.abc import Mapping
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, JsonValue
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.rust_bridge.trace_queries import SpanType

_RESPONSE_CONFIG: Final = ConfigDict(frozen=True, extra="allow")
_HELP_CONFIG: Final = ConfigDict(frozen=True, extra="forbid")
TraceTableName = Literal["otel_traces", "agent_traces_by_key", "spend_logs"]
MetadataValueType = Literal["array", "boolean", "integer", "null", "number", "object", "string"]


class TraceQueryColumn(BaseModel):
    model_config = _RESPONSE_CONFIG
    name: str
    type: str


class TraceQueryStatistics(BaseModel):
    model_config = _RESPONSE_CONFIG
    elapsed: float
    rows_read: int | str
    bytes_read: int | str


class TraceSQLResponse(BaseModel):
    model_config = _RESPONSE_CONFIG
    meta: tuple[TraceQueryColumn, ...]
    data: tuple[Mapping[str, JsonValue], ...]
    rows: int | str
    statistics: TraceQueryStatistics


class TraceQueryTable(BaseModel):
    model_config = _HELP_CONFIG
    name: TraceTableName
    columns: tuple[TraceQueryColumn, ...]


class TraceQueryNormalizedField(BaseModel):
    model_config = _HELP_CONFIG
    table: TraceTableName
    name: str
    column: str
    type: str
    meaning: str


class TraceQueryMetadataField(BaseModel):
    model_config = _HELP_CONFIG
    path: tuple[str | int, ...]
    types: tuple[MetadataValueType, ...]
    expression: str


class TraceQueryMetadata(BaseModel):
    model_config = _HELP_CONFIG
    table: TraceTableName
    column: str
    fields: tuple[TraceQueryMetadataField, ...]
    sampled_rows: int
    invalid_json_rows: int
    truncated: bool
    sample_sql: str
    scope: str
    error: str | None = None


class TraceQueryAttributeField(BaseModel):
    model_config = _HELP_CONFIG
    key: str
    type: Literal["String"]
    expression: str


class TraceQueryAttributes(BaseModel):
    model_config = _HELP_CONFIG
    table: TraceTableName
    column: str
    fields: tuple[TraceQueryAttributeField, ...]
    truncated: bool
    discovery_sql: str
    scope: str
    error: str | None = None


class TraceQueryRelationship(BaseModel):
    model_config = _HELP_CONFIG
    left: str
    right: str
    additional_predicates: str
    meaning: str


class TraceQueryExample(BaseModel):
    model_config = _HELP_CONFIG
    name: str
    sql: str


class TraceQueryHelp(BaseModel):
    model_config = _HELP_CONFIG
    dialect: str
    access: str
    response: str
    tables: tuple[TraceQueryTable, ...]
    normalized_fields: tuple[TraceQueryNormalizedField, ...]
    metadata: TraceQueryMetadata
    attributes: tuple[TraceQueryAttributes, ...]
    relationships: tuple[TraceQueryRelationship, ...]
    examples: tuple[TraceQueryExample, ...]
    gotchas: tuple[str, ...]
    guide: str


SpanStatus = Literal["ok", "error", "unset"]
ChatRole: TypeAlias = Literal["system", "user", "assistant", "tool"]


class UIToolCall(TypedDict):
    name: ReadOnly[str]
    arguments: ReadOnly[str]


class UIMessage(TypedDict):
    role: ReadOnly[ChatRole]
    content: ReadOnly[str]
    name: ReadOnly[NotRequired[str]]
    tool_calls: ReadOnly[NotRequired[tuple[UIToolCall, ...]]]


class UIField(TypedDict):
    key: ReadOnly[str]
    value: ReadOnly[str]


class UIMessages(TypedDict):
    kind: ReadOnly[Literal["messages"]]
    messages: ReadOnly[tuple[UIMessage, ...]]


class UIFields(TypedDict):
    kind: ReadOnly[Literal["fields"]]
    fields: ReadOnly[tuple[UIField, ...]]


class UIText(TypedDict):
    kind: ReadOnly[Literal["text"]]
    text: ReadOnly[str]


UIContent: TypeAlias = UIMessages | UIFields | UIText


class Span(TypedDict):
    span_id: ReadOnly[str]
    parent_span_id: ReadOnly[str | None]
    name: ReadOnly[str]
    type: ReadOnly[SpanType]
    agent: ReadOnly[str]
    framework: ReadOnly[str]
    start_offset_ms: ReadOnly[float]
    duration_ms: ReadOnly[float]
    status: ReadOnly[SpanStatus]
    error: ReadOnly[str | None]
    error_truncated: ReadOnly[bool]
    input_preview: ReadOnly[str]
    model: ReadOnly[str | None]
    input_tokens: ReadOnly[int]
    output_tokens: ReadOnly[int]
    litellm_request_id: ReadOnly[str | None]
    spend: ReadOnly[float | None]


class AgentNode(TypedDict):
    """One distinct agent in a trace. 200 invocations of `researcher` = one node."""

    name: ReadOnly[str]
    parent_agent: ReadOnly[str | None]
    invocations: ReadOnly[int]
    llm_calls: ReadOnly[int]
    tool_calls: ReadOnly[int]
    duration_ms: ReadOnly[float]
    spend: ReadOnly[float | None]


class TraceSummary(TypedDict):
    trace_id: ReadOnly[str]
    trace_ref: ReadOnly[NotRequired[str]]
    name: ReadOnly[str]
    service: ReadOnly[str]
    agent_names: ReadOnly[NotRequired[tuple[str, ...]]]
    frameworks: ReadOnly[NotRequired[tuple[str, ...]]]
    input_preview: ReadOnly[str]
    start_time: ReadOnly[str]
    duration_ms: ReadOnly[float]
    status: ReadOnly[SpanStatus]
    span_count: ReadOnly[int]
    agent_count: ReadOnly[int]
    agent_invocations: ReadOnly[int]
    llm_calls: ReadOnly[int]
    tool_calls: ReadOnly[int]
    error_count: ReadOnly[int]
    input_tokens: ReadOnly[int]
    output_tokens: ReadOnly[int]
    models: ReadOnly[tuple[str, ...]]
    spend: ReadOnly[float | None]


class Trace(TypedDict):
    summary: ReadOnly[TraceSummary]
    agents: ReadOnly[tuple[AgentNode, ...]]
    spans: ReadOnly[tuple[Span, ...]]


class TracePage(TypedDict):
    data: ReadOnly[tuple[TraceSummary, ...]]
    next_cursor: ReadOnly[str | None]


class SpanDetail(TypedDict):
    span_id: ReadOnly[str]
    input: ReadOnly[str]
    output: ReadOnly[str]
    input_ui: ReadOnly[UIContent]
    output_ui: ReadOnly[UIContent]
    attributes: ReadOnly[dict[str, str]]


class SpanErrorPage(TypedDict):
    span_id: ReadOnly[str]
    message: ReadOnly[str]
    total_chars: ReadOnly[int]
    next_cursor: ReadOnly[str | None]
