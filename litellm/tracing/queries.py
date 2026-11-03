from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.types.llms.base import LiteLLMBaseModel

from .generated.models import TraceAgentRow, TraceAgentsParams, TraceQueryColumn
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


TRACE_AGENTS: Final[ReadQuery[TraceAgentsParams, TraceAgentRow]] = ReadQuery(
    "trace_agents", TraceAgentsParams, TypeAdapter(QueryResponse[TraceAgentRow])
)
