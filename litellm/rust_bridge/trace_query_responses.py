from collections.abc import Mapping
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

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
