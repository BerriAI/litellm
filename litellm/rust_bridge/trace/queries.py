from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.types.llms.base import LiteLLMBaseModel

from .generated.models import (
    ActivityAvailability,
    AgentRow,
    CountRow,
    ExecutionRow,
    LensAccessParams,
    LensContentParams,
    LensEvidenceParams,
    LensSampleParams,
    PartRow,
    TraceQueryColumn,
)
from .generated.types import ReadQueryName

_RESPONSE_CONFIG: Final = ConfigDict(frozen=True, extra="allow")


class TraceQueryStatistics(LiteLLMBaseModel):
    model_config = _RESPONSE_CONFIG
    elapsed: float
    rows_read: int | str
    bytes_read: int | str


class ClickHouseSQLEnvelope(LiteLLMBaseModel):
    model_config = _RESPONSE_CONFIG
    meta: tuple[TraceQueryColumn, ...]
    data: tuple[Mapping[str, JsonValue], ...]
    rows: int | str
    statistics: TraceQueryStatistics


ParamsT: Final = TypeVar("ParamsT", bound=BaseModel)
RowT: Final = TypeVar("RowT")


class QueryResponse(LiteLLMBaseModel, Generic[RowT]):
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
