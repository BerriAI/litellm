"""ClickHouse-backed trace store: batched span writes and scoped reads."""

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm._logging import verbose_logger
from litellm.constants import AGENT_TRACING_LIST_PAGE_SIZE
from litellm.integrations.clickhouse.schema import (
    OTEL_TRACES_TABLE,
)
from litellm.rust_bridge.trace_queries import (
    LIST_TRACES,
    SPAN_DETAIL,
    SPAN_ERROR,
    SPEND_BY_RESPONSE_IDS,
    TRACE_IDENTITY,
    TRACE_PAGE_SPANS,
    TRACE_SPANS,
    ListTracesParams,
    ListTracesRow,
    SpanDetailParams,
    SpanErrorParams,
    SpendByResponseIdsParams,
    SpendRow,
    TraceIdentityParams,
    TracePageSpansParams,
    TraceSpansParams,
    TraceSpansRow,
)
from litellm.rust_bridge.traces import ClickHouseStorage
from litellm.tracing.attribution import resolve_trace, spend_lookup_ids
from litellm.tracing.types import (
    SpanDetail,
    SpanErrorPage,
    SpanRow,
    SpanStatus,
    Trace,
    TracePage,
    TraceScope,
    TraceSummary,
)
from litellm.tracing.ui_format import to_ui_content

NANOS_PER_MS: Final = 1_000_000
SPEND_WINDOW_MS: Final = 30 * 60 * 1000
_STATUS: Final[Mapping[str, SpanStatus]] = MappingProxyType({"STATUS_CODE_OK": "ok", "STATUS_CODE_ERROR": "error"})


TraceCursorParts: TypeAlias = tuple[Annotated[int, Field(gt=0)], Annotated[str, Field(min_length=1)]]
_TRACE_CURSOR: Final[TypeAdapter[TraceCursorParts]] = TypeAdapter(TraceCursorParts)


class _ErrorCursor(BaseModel):
    model_config = ConfigDict(frozen=True)
    offset: int = Field(ge=0, le=(1 << 63) - 1)
    version: str = Field(pattern=r"^[A-F0-9]{64}$")


class AmbiguousTraceError(ValueError):
    pass


def encode_cursor(start_ms: int, trace_id: str) -> str:
    return base64.urlsafe_b64encode(json.dumps((start_ms, trace_id)).encode()).decode()


def decode_cursor(cursor: str | None) -> tuple[int, str]:
    if not cursor:
        return 0, ""
    try:
        return _TRACE_CURSOR.validate_json(base64.b64decode(cursor, altchars=b"-_", validate=True), strict=True)
    except (ValueError, UnicodeError, binascii.Error) as error:
        raise ValueError("Invalid trace cursor") from error


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _status(code: str) -> SpanStatus:
    return _STATUS.get(code, "unset")


def trace_summary_from_row(row: ListTracesRow) -> TraceSummary:
    """A listed trace whose spans could not be read: rollup counts only, cost unknown."""
    return TraceSummary(
        trace_id=row["trace_id"],
        trace_ref=row.get("trace_ref", ""),
        name=row["name"],
        service=row["service"],
        agent_names=tuple(row.get("agent_names") or ()),
        frameworks=tuple(row.get("frameworks") or ()),
        input_preview=row["input_preview"],
        start_time=_iso(int(row["start_ms"])),
        duration_ms=float(row["duration_ms"]),
        status=_status(row["status"]),
        span_count=int(row["span_count"]),
        agent_count=int(row["agent_count"]),
        agent_invocations=int(row.get("agent_invocations") or row["agent_count"]),
        llm_calls=int(row["llm_calls"]),
        tool_calls=int(row["tool_calls"]),
        error_count=int(row.get("error_count") or 0),
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        models=tuple(row["models"]),
        spend=None,
    )


class TraceStore:
    """Stores spans and runs scoped trace reads."""

    def __init__(self, storage: ClickHouseStorage) -> None:
        self.storage = storage

    async def insert_spans(self, rows: Sequence[SpanRow]) -> None:
        await self.storage.insert_rows(OTEL_TRACES_TABLE, tuple(rows))

    async def _reference(self, trace_id: str, scope: TraceScope, trace_ref: str) -> str | None:
        if trace_ref:
            return trace_ref
        identities: Final = await self.storage.query(TRACE_IDENTITY, TraceIdentityParams(**scope, trace_id=trace_id))
        if len(identities) > 1:
            raise AmbiguousTraceError("Multiple traces have this ID; provide trace_ref")
        return identities[0].trace_ref if identities else None

    async def _spend_rows(self, scope: TraceScope, rows: Sequence[TraceSpansRow]) -> tuple[SpendRow, ...]:
        response_ids, request_ids, trace_ids = spend_lookup_ids(rows)
        if not (response_ids or request_ids or trace_ids):
            return ()
        start_ms: Final = min(int(row["start_ns"]) // NANOS_PER_MS for row in rows)
        end_ms: Final = max((int(row["start_ns"]) + int(row["duration_ns"])) // NANOS_PER_MS for row in rows)
        try:
            spend: Final = await self.storage.query(
                SPEND_BY_RESPONSE_IDS,
                SpendByResponseIdsParams(
                    **scope,
                    response_ids=response_ids,
                    request_ids=request_ids,
                    trace_ids=trace_ids,
                    start_ms=start_ms - SPEND_WINDOW_MS,
                    end_ms=end_ms + SPEND_WINDOW_MS,
                ),
            )
        except RuntimeError as error:
            verbose_logger.warning("Trace spend lookup unavailable: %s", error)
            return ()
        return tuple(spend)

    async def list_traces(
        self,
        scope: TraceScope,
        start_ms: int,
        end_ms: int,
        cursor: str | None = None,
        limit: int = AGENT_TRACING_LIST_PAGE_SIZE,
    ) -> TracePage:
        cursor_ms, cursor_trace_id = decode_cursor(cursor)
        page: Final = await self.storage.query(
            LIST_TRACES,
            ListTracesParams(
                **scope,
                start_ms=start_ms,
                end_ms=end_ms,
                cursor_ms=cursor_ms,
                cursor_trace_id=cursor_trace_id,
                limit=limit,
            ),
        )
        next_cursor: Final = (
            encode_cursor(int(page[-1]["start_ms"]), page[-1]["trace_ref"]) if len(page) == limit else None
        )
        if not page:
            return TracePage(data=(), next_cursor=next_cursor)
        span_rows: Final = await self.storage.query(
            TRACE_PAGE_SPANS,
            TracePageSpansParams(
                **scope,
                trace_refs=tuple(row["trace_ref"] for row in page),
                start_ms=min(int(row["start_ms"]) for row in page),
                end_ms=max(int(row["start_ms"]) + int(row["duration_ms"]) for row in page) + 1,
            ),
        )
        spend_rows: Final = await self._spend_rows(scope, span_rows)
        by_trace: dict[tuple[str, str, str], list[TraceSpansRow]] = {}  # mutable-ok: group page spans
        for span in span_rows:
            by_trace.setdefault((span["team_id"], span["api_key_hash"], span.get("trace_id", "")), []).append(span)

        def summary(row: ListTracesRow) -> TraceSummary:
            spans = by_trace.get((row["team_id"], row["api_key_hash"], row["trace_id"]), [])
            trace = resolve_trace(row["trace_id"], spans, row["trace_ref"], spend_rows)
            return trace["summary"] if trace is not None else trace_summary_from_row(row)

        return TracePage(data=tuple(summary(row) for row in page), next_cursor=next_cursor)

    async def get_trace(self, trace_id: str, scope: TraceScope, trace_ref: str = "") -> Trace | None:
        reference: Final = await self._reference(trace_id, scope, trace_ref)
        if reference is None:
            return None
        rows: Final = await self.storage.query(
            TRACE_SPANS, TraceSpansParams(**scope, trace_id=trace_id, trace_ref=reference)
        )
        if not rows:
            return None
        return resolve_trace(trace_id, rows, reference, await self._spend_rows(scope, rows))

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "") -> SpanDetail | None:
        reference: Final = await self._reference(trace_id, scope, trace_ref)
        if reference is None:
            return None
        rows: Final = await self.storage.query(
            SPAN_DETAIL,
            SpanDetailParams(**scope, trace_id=trace_id, span_id=span_id, trace_ref=reference),
        )
        if not rows:
            return None
        return SpanDetail(
            span_id=rows[0]["span_id"],
            input=rows[0]["input"],
            output=rows[0]["output"],
            input_ui=to_ui_content(rows[0]["input"]),
            output_ui=to_ui_content(rows[0]["output"]),
            attributes=dict(rows[0]["attributes"]),
        )

    async def get_span_error(
        self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "", cursor: str | None = None
    ) -> SpanErrorPage | None:
        try:
            position: Final = (
                _ErrorCursor.model_validate_json(base64.b64decode(cursor, altchars=b"-_", validate=True))
                if cursor
                else None
            )
        except (ValueError, binascii.Error) as error:
            raise ValueError("Invalid diagnostic cursor") from error
        reference: Final = await self._reference(trace_id, scope, trace_ref)
        if reference is None:
            return None
        rows: Final = await self.storage.query(
            SPAN_ERROR,
            SpanErrorParams(
                **scope,
                trace_id=trace_id,
                span_id=span_id,
                trace_ref=reference,
                error_offset=position.offset if position else 0,
                error_version=position.version if position else "",
            ),
        )
        if not rows:
            return None
        row: Final = rows[0]
        offset: Final = (position.offset if position else 0) + len(row.message)
        continuation: Final = _ErrorCursor(offset=offset, version=row.version) if offset < row.total_chars else None
        return SpanErrorPage(
            span_id=row.span_id,
            message=row.message,
            total_chars=row.total_chars,
            next_cursor=base64.urlsafe_b64encode(continuation.model_dump_json().encode()).decode()
            if continuation
            else None,
        )
