from typing import Final, TypeVar

from pydantic import TypeAdapter, ValidationError

from litellm.constants import AGENT_TRACING_LIST_PAGE_SIZE
from litellm.tracing.generated.models import TraceAgentRow, TraceAgentsParams, TraceQueryHelp
from litellm.tracing.generated.responses import TraceSQLResponse
from litellm.tracing.generated.types import QueryScope, SpanDetail, SpanErrorPage, Trace, TracePage, TraceScope
from litellm.tracing.queries import TRACE_AGENTS, ClickHouseSQLEnvelope, ParamsT, ReadQuery, RowT
from litellm.tracing.remote import RemoteTraceStore

QUERY_PARAMETERS: Final = TypeAdapter(dict[str, str | int | float | list[str]])
_SQL_ENVELOPE: Final = TypeAdapter(ClickHouseSQLEnvelope)
_HELP_RESPONSE: Final = TypeAdapter(TraceQueryHelp)
_TRACE_PAGE: Final = TypeAdapter(TracePage)
_TRACE: Final[TypeAdapter[Trace | None]] = TypeAdapter(Trace | None)
_SPAN_DETAIL: Final[TypeAdapter[SpanDetail | None]] = TypeAdapter(SpanDetail | None)
_SPAN_ERROR_PAGE: Final[TypeAdapter[SpanErrorPage | None]] = TypeAdapter(SpanErrorPage | None)
_ResponseT: Final = TypeVar("_ResponseT")


def _decode_query_response(adapter: TypeAdapter[_ResponseT], body: str) -> _ResponseT:
    try:
        return adapter.validate_json(body)
    except ValidationError as error:
        raise RuntimeError("Lens trace query returned an invalid response") from error


def _validate_query_response(adapter: TypeAdapter[_ResponseT], value: object) -> _ResponseT:
    try:
        return adapter.validate_python(value)
    except ValidationError as error:
        raise RuntimeError("Lens trace query returned an invalid response") from error


class LensTraceStorage:
    def __init__(self, remote: RemoteTraceStore) -> None:
        self._remote: Final = remote

    async def list_traces(
        self,
        scope: TraceScope,
        start_ms: int,
        end_ms: int,
        cursor: str | None = None,
        limit: int = AGENT_TRACING_LIST_PAGE_SIZE,
    ) -> TracePage:
        result: Final = await self._remote.list_traces(scope, start_ms, end_ms, cursor, limit)
        return _validate_query_response(_TRACE_PAGE, result)

    async def get_trace(
        self,
        trace_id: str,
        scope: TraceScope,
        trace_ref: str = "",
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> Trace | None:
        result: Final = await self._remote.get_trace(trace_id, scope, trace_ref, cursor, page_size)
        return _validate_query_response(_TRACE, result)

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "") -> SpanDetail | None:
        result: Final = await self._remote.get_span(trace_id, span_id, scope, trace_ref)
        return _validate_query_response(_SPAN_DETAIL, result)

    async def get_span_error(
        self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "", cursor: str | None = None
    ) -> SpanErrorPage | None:
        result: Final = await self._remote.get_span_error(trace_id, span_id, scope, trace_ref, cursor)
        return _validate_query_response(_SPAN_ERROR_PAGE, result)

    async def query(self, query: ReadQuery[ParamsT, RowT], parameters: ParamsT) -> tuple[RowT, ...]:
        validated: Final = query.parameters.model_validate(parameters)
        result: Final = await self._remote.query(query.name, QUERY_PARAMETERS.validate_python(validated.model_dump()))
        return _decode_query_response(query.response, result).data

    async def query_sql(self, sql: str, scope: QueryScope, secret: str) -> TraceSQLResponse:
        result: Final = await self._remote.query_sql(sql, scope, secret)
        envelope: Final = _decode_query_response(_SQL_ENVELOPE, result)
        return TraceSQLResponse(data=envelope.data)

    async def query_help(self, scope: QueryScope, secret: str) -> TraceQueryHelp:
        result: Final = await self._remote.query_help(scope, secret)
        return _validate_query_response(_HELP_RESPONSE, result)

    async def trace_agents(self, parameters: TraceAgentsParams) -> tuple[TraceAgentRow, ...]:
        return await self.query(TRACE_AGENTS, parameters)
