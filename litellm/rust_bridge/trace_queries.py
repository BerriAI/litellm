from dataclasses import dataclass
from typing import Annotated, Final, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SpanType: TypeAlias = Literal[
    "agent",
    "llm",
    "tool",
    "chain",
    "framework",
    "retriever",
    "embedding",
    "reranker",
    "guardrail",
    "evaluator",
    "prompt",
    "decision",
]

ReadQueryName: TypeAlias = Literal[
    "availability",
    "agents",
    "sample",
    "content",
    "evidence",
]

UInt64: TypeAlias = Annotated[int, Field(ge=0, le=2**64 - 1)]
UInt32: TypeAlias = Annotated[int, Field(ge=0, le=2**32 - 1)]
_PARAMETERS_CONFIG: Final = ConfigDict(frozen=True, extra="forbid")














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
