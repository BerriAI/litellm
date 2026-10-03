from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Final, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict

ReadQueryName: TypeAlias = Literal[
    "list_traces",
    "trace_spans",
    "trace_identity",
    "span_detail",
    "span_error",
    "spend_by_response_ids",
    "availability",
    "agents",
    "sample",
    "content",
    "evidence",
]

Int64: TypeAlias = Annotated[int, Field(ge=-(2**63), le=2**63 - 1)]
UInt64: TypeAlias = Annotated[int, Field(ge=0, le=2**64 - 1)]
UInt32: TypeAlias = Annotated[int, Field(ge=0, le=2**32 - 1)]
_PARAMETERS_CONFIG: Final = ConfigDict(frozen=True, extra="forbid")


class ListTracesParams(BaseModel):
    model_config = _PARAMETERS_CONFIG
    all_teams: Literal[0, 1]
    user_id: str
    team_ids: tuple[str, ...]
    start_ms: Int64
    end_ms: Int64
    cursor_ms: Int64
    cursor_trace_id: str
    limit: UInt32


class TraceSpansParams(BaseModel):
    model_config = _PARAMETERS_CONFIG
    all_teams: Literal[0, 1]
    user_id: str
    team_ids: tuple[str, ...]
    trace_id: str
    trace_ref: str


class SpanDetailParams(BaseModel):
    model_config = _PARAMETERS_CONFIG
    all_teams: Literal[0, 1]
    user_id: str
    team_ids: tuple[str, ...]
    trace_id: str
    trace_ref: str
    span_id: str


class SpanErrorParams(BaseModel):
    model_config = _PARAMETERS_CONFIG
    all_teams: Literal[0, 1]
    user_id: str
    team_ids: tuple[str, ...]
    trace_id: str
    trace_ref: str
    span_id: str
    error_offset: UInt64
    error_version: str


class SpendByResponseIdsParams(BaseModel):
    model_config = _PARAMETERS_CONFIG
    all_teams: Literal[0, 1]
    user_id: str
    team_ids: tuple[str, ...]
    response_ids: tuple[str, ...]
    start_ms: Int64
    end_ms: Int64


class LensAccessParams(BaseModel):
    model_config = _PARAMETERS_CONFIG
    all_teams: Literal[0, 1]
    team: str
    key_hash: str


class LensSampleParams(BaseModel):
    model_config = _PARAMETERS_CONFIG
    all_teams: Literal[0, 1]
    team: str
    key_hash: str
    source: Literal["traces", "requests", "both"]
    start: UInt64
    end: UInt64
    agent_name: str
    service: str
    filter_keys: tuple[str, ...]
    filter_values: tuple[str, ...]
    selected_team: str
    execution_ids: tuple[str, ...]
    sample_cap: UInt64
    sample_percent: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    preview: Literal[0, 1]
    after: str
    limit: UInt32
    offset: UInt64


class LensContentParams(BaseModel):
    model_config = _PARAMETERS_CONFIG
    all_teams: Literal[0, 1]
    team: str
    key_hash: str
    source: Literal["traces", "requests"]
    id: str
    record_team: str
    trace_ref: str
    cursor: str
    offset: UInt32


class LensEvidenceParams(BaseModel):
    model_config = _PARAMETERS_CONFIG
    all_teams: Literal[0, 1]
    team: str
    key_hash: str
    source: Literal["traces", "requests"]
    id: str
    record_team: str
    trace_ref: str
    span: str
    quote: str


class TraceIdentityParams(BaseModel):
    model_config = _PARAMETERS_CONFIG
    all_teams: Literal[0, 1]
    user_id: str
    team_ids: tuple[str, ...]
    trace_id: str


class TraceIdentityRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    trace_ref: str


class ListTracesRow(TypedDict):
    trace_id: ReadOnly[str]
    trace_ref: ReadOnly[str]
    team_id: ReadOnly[str]
    api_key_hash: ReadOnly[str]
    user_id: ReadOnly[str]
    name: ReadOnly[str]
    service: ReadOnly[str]
    input_preview: ReadOnly[str]
    status: ReadOnly[str]
    start_ms: ReadOnly[int]
    duration_ms: ReadOnly[int]
    span_count: ReadOnly[int]
    agent_count: ReadOnly[int]
    agent_invocations: ReadOnly[int]
    agent_names: NotRequired[ReadOnly[tuple[str, ...]]]
    frameworks: NotRequired[ReadOnly[tuple[str, ...]]]
    llm_calls: ReadOnly[int]
    tool_calls: ReadOnly[int]
    input_tokens: ReadOnly[int]
    output_tokens: ReadOnly[int]
    models: ReadOnly[tuple[str, ...]]
    error_count: ReadOnly[int]
    request_ids: ReadOnly[tuple[str, ...]]


class TraceSpansRow(TypedDict):
    span_id: ReadOnly[str]
    parent_span_id: ReadOnly[str]
    name: ReadOnly[str]
    type: ReadOnly[Literal["agent", "llm", "tool", "chain", "framework"]]
    agent: ReadOnly[str]
    framework: NotRequired[ReadOnly[str]]
    status: ReadOnly[str]
    status_message: ReadOnly[str]
    error_truncated: ReadOnly[bool]
    start_ns: ReadOnly[int]
    duration_ns: ReadOnly[int]
    service: ReadOnly[str]
    input_preview: ReadOnly[str]
    model: ReadOnly[str]
    input_tokens: ReadOnly[int]
    output_tokens: ReadOnly[int]
    litellm_request_id: ReadOnly[str]
    team_id: ReadOnly[str]
    api_key_hash: ReadOnly[str]
    user_id: ReadOnly[str]


class SpanDetailRow(TypedDict):
    span_id: ReadOnly[str]
    input: ReadOnly[str]
    output: ReadOnly[str]
    attributes: ReadOnly[Mapping[str, str]]


class SpanErrorRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    span_id: str
    message: str
    total_chars: int
    version: str


class SpendRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    request_id: str
    response_id: str
    team_id: str
    api_key: str
    user: str
    spend: float
    start_ms: int


class ActivityAvailability(BaseModel):
    model_config = ConfigDict(frozen=True)
    traces: bool = False
    requests: bool = False


class ExecutionRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    selection_key: str = ""
    source: Literal["traces", "requests"]
    trace_id: str
    trace_ref: str = ""
    team_id: str
    name: str
    start_time: str
    span_count: int
    root_seen: int
    eligible: int
    selected: int = 0
    service: str = ""
    attributes: tuple[tuple[str, str], ...] = ()


class PartRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    span_id: str
    parent_span_id: str
    name: str
    kind: str
    content: str
    truncated: int


class CountRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    count: int


class AgentRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    agent_name: str


ParamsT: Final = TypeVar("ParamsT", bound=BaseModel)
RowT: Final = TypeVar("RowT")


class QueryResponse(BaseModel, Generic[RowT]):
    model_config = ConfigDict(frozen=True)
    data: tuple[RowT, ...]


@dataclass(frozen=True, slots=True)
class ReadQuery(Generic[ParamsT, RowT]):
    name: ReadQueryName
    parameters: type[ParamsT]
    response: TypeAdapter[QueryResponse[RowT]]


LIST_TRACES: Final[ReadQuery[ListTracesParams, ListTracesRow]] = ReadQuery(
    "list_traces", ListTracesParams, TypeAdapter(QueryResponse[ListTracesRow])
)
TRACE_SPANS: Final[ReadQuery[TraceSpansParams, TraceSpansRow]] = ReadQuery(
    "trace_spans", TraceSpansParams, TypeAdapter(QueryResponse[TraceSpansRow])
)
SPAN_DETAIL: Final[ReadQuery[SpanDetailParams, SpanDetailRow]] = ReadQuery(
    "span_detail", SpanDetailParams, TypeAdapter(QueryResponse[SpanDetailRow])
)
SPAN_ERROR: Final[ReadQuery[SpanErrorParams, SpanErrorRow]] = ReadQuery(
    "span_error", SpanErrorParams, TypeAdapter(QueryResponse[SpanErrorRow])
)
SPEND_BY_RESPONSE_IDS: Final[ReadQuery[SpendByResponseIdsParams, SpendRow]] = ReadQuery(
    "spend_by_response_ids", SpendByResponseIdsParams, TypeAdapter(QueryResponse[SpendRow])
)
LENS_AVAILABILITY: Final[ReadQuery[LensAccessParams, ActivityAvailability]] = ReadQuery(
    "availability", LensAccessParams, TypeAdapter(QueryResponse[ActivityAvailability])
)
LENS_AGENTS: Final[ReadQuery[LensAccessParams, AgentRow]] = ReadQuery(
    "agents", LensAccessParams, TypeAdapter(QueryResponse[AgentRow])
)
LENS_SAMPLE: Final[ReadQuery[LensSampleParams, ExecutionRow]] = ReadQuery(
    "sample", LensSampleParams, TypeAdapter(QueryResponse[ExecutionRow])
)
LENS_CONTENT: Final[ReadQuery[LensContentParams, PartRow]] = ReadQuery(
    "content", LensContentParams, TypeAdapter(QueryResponse[PartRow])
)
LENS_EVIDENCE: Final[ReadQuery[LensEvidenceParams, CountRow]] = ReadQuery(
    "evidence", LensEvidenceParams, TypeAdapter(QueryResponse[CountRow])
)

TRACE_IDENTITY: Final = ReadQuery("trace_identity", TraceIdentityParams, TypeAdapter(QueryResponse[TraceIdentityRow]))
