from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from .types import MetadataValueType, TraceTableName

UInt64: TypeAlias = Annotated[int, Field(ge=0, le=2**64 - 1)]

UInt32: TypeAlias = Annotated[int, Field(ge=0, le=2**32 - 1)]

_PARAMETERS_CONFIG: Final = ConfigDict(frozen=True, extra="forbid")

_HELP_CONFIG: Final = ConfigDict(frozen=True, extra="forbid")
_RESPONSE_CONFIG: Final = ConfigDict(frozen=True, extra="allow")

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

class TraceQueryColumn(BaseModel):
    model_config = _RESPONSE_CONFIG
    name: str
    type: str

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
