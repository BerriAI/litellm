"""ClickHouse-backed trace store: batched span writes and scoped reads."""

import asyncio
import base64
import binascii
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.constants import AGENT_TRACING_LIST_PAGE_SIZE
from litellm.integrations.clickhouse.schema import (
    OTEL_TRACES_TABLE,
)
from litellm.rust_bridge.traces import TraceStorage
from litellm.tracing.types import (
    AgentNode,
    Span,
    SpanDetail,
    SpanRow,
    SpanStatus,
    SpendDetails,
    SpendFallbackReason,
    SpendSource,
    SpendTotals,
    Trace,
    TracePage,
    TraceScope,
    TraceSummary,
)

NANOS_PER_MS: Final = 1_000_000
SPEND_WINDOW_MS: Final = 30 * 60 * 1000
_STATUS: Final = MappingProxyType({"STATUS_CODE_OK": "ok", "STATUS_CODE_ERROR": "error"})


class _SpendRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    response_id: str
    team_id: str
    api_key: str
    spend: float
    start_ms: int
    call_id: str = ""
    log_id: str = ""
    pricing_known: bool = False
    source: SpendSource = "clickhouse"
    fallback_reason: SpendFallbackReason | None = None


_SPEND_ROWS: Final = TypeAdapter(tuple[_SpendRow, ...])
SpendFallback = Callable[[TraceScope, Sequence[str], int, int], Awaitable[Sequence[Mapping[str, object]]]]


@dataclass(frozen=True, slots=True)
class _Attribution:
    source: SpendSource | None = None
    fallback_reason: SpendFallbackReason | None = None
    event_id: str | None = None
    log_id: str | None = None
    cost: float | None = None
    subtotal: float = 0.0
    reason: str | None = None


def _spend_for(
    response_id: str,
    call_id: str,
    team_id: str,
    api_key_hash: str,
    start_ms: int,
    end_ms: int,
    rows: Sequence[_SpendRow],
) -> _Attribution:
    if not response_id and not call_id:
        return _Attribution(reason="missing_identity")
    scoped: Final = tuple(
        row
        for row in rows
        if row.team_id == team_id
        and row.api_key == api_key_hash
        and start_ms - SPEND_WINDOW_MS <= row.start_ms < end_ms + SPEND_WINDOW_MS
    )
    matches: Final = tuple(
        row for row in scoped if (row.call_id == call_id if call_id else row.response_id == response_id)
    )
    if not matches:
        return _Attribution(reason="missing_spend")
    if len(matches) != 1:
        return _Attribution(reason="ambiguous_identity")
    match_row: Final = matches[0]
    if call_id and response_id and match_row.response_id and match_row.response_id != response_id:
        return _Attribution(reason="conflicting_identity")
    event_id: Final = hashlib.sha256(json.dumps((team_id, api_key_hash, match_row.request_id)).encode()).hexdigest()
    return _Attribution(
        source=match_row.source,
        fallback_reason=match_row.fallback_reason,
        event_id=event_id,
        log_id=match_row.log_id or match_row.request_id,
        cost=match_row.spend if match_row.pricing_known else None,
        subtotal=match_row.spend,
        reason=None if match_row.pricing_known else "unknown_pricing",
    )


def _totals(attributions: Sequence[_Attribution], missing_calls: int = 0) -> SpendTotals:
    matched: Final = MappingProxyType({item.event_id: item for item in attributions if item.event_id})
    unmatched: Final = sum(item.event_id is None for item in attributions) + missing_calls
    expected: Final = len(matched) + unmatched
    priced: Final = sum(item.cost is not None for item in matched.values())
    complete: Final = expected > 0 and priced == expected
    subtotal: Final = sum(item.subtotal for item in matched.values())
    reasons: Final = tuple(
        sorted(
            frozenset(
                tuple(item.reason for item in attributions if item.reason)
                + (("incomplete_spans",) if missing_calls else ())
                + (("no_model_calls",) if not expected else ())
            )
        )
    )
    return SpendTotals(
        spend=subtotal if complete else None,
        spend_details=SpendDetails(
            sources=tuple(sorted(frozenset(item.source for item in matched.values() if item.source))),
            status="complete" if complete else "partial" if matched else "unavailable",
            subtotal=subtotal,
            matched_calls=len(matched),
            priced_calls=priced,
            expected_calls=expected,
            reasons=reasons,
        ),
    )


def _span_attribution(span: Span) -> _Attribution:
    return _Attribution(
        source=span.get("spend_source"),
        fallback_reason=span.get("spend_fallback_reason"),
        event_id=span.get("spend_event_id"),
        log_id=span.get("spend_log_id"),
        cost=span["spend"],
        subtotal=span.get("spend_subtotal", 0.0),
        reason=span.get("spend_reason"),
    )


_LIST_CALLS: Final = TypeAdapter(tuple[tuple[str, str, str, int, int], ...])


def _list_calls(row: Mapping[str, object]) -> tuple[tuple[str, str, str, int, int], ...]:
    return _LIST_CALLS.validate_python(row.get("llm_spans") or ())


def _summary_spend(row: dict[str, Any], spend_rows: Sequence[_SpendRow]) -> SpendTotals:
    calls: Final = _list_calls(row)
    attributions: Final = tuple(
        _spend_for(
            response_id, call_id, row.get("team_id") or "", row.get("api_key_hash") or "", start, end, spend_rows
        )
        for _, response_id, call_id, start, end in calls
    )
    return _totals(attributions, max(int(row.get("llm_call_count", row["llm_calls"])) - len(calls), 0))


def encode_cursor(start_ms: int, trace_id: str) -> str:
    return base64.urlsafe_b64encode(json.dumps((start_ms, trace_id)).encode()).decode()


def decode_cursor(cursor: str | None) -> tuple[int, str]:
    if not cursor:
        return 0, ""
    try:
        value: Final = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], int)
            or isinstance(value[0], bool)
            or value[0] <= 0
            or not isinstance(value[1], str)
            or not value[1]
        ):
            raise ValueError("Invalid trace cursor")
        return value[0], value[1]
    except (ValueError, UnicodeError, binascii.Error) as error:
        raise ValueError("Invalid trace cursor") from error


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _status(code: str) -> SpanStatus:
    return _STATUS.get(code, "unset")


def trace_summary_from_row(row: dict[str, Any], spend_rows: Sequence[_SpendRow] = ()) -> TraceSummary:
    return TraceSummary(
        trace_id=row["trace_id"],
        trace_ref=row.get("trace_ref", ""),
        name=row["name"],
        service=row["service"],
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
        **_summary_spend(row, spend_rows),
    )


def span_from_row(row: dict[str, Any], trace_start_ns: int, spend_rows: Sequence[_SpendRow] = ()) -> Span:
    attribution: Final = (
        _spend_for(
            row["litellm_request_id"],
            row.get("litellm_call_id") or "",
            row.get("team_id") or "",
            row.get("api_key_hash") or "",
            int(row["start_ns"]) // NANOS_PER_MS,
            (int(row["start_ns"]) + int(row["duration_ns"])) // NANOS_PER_MS,
            spend_rows,
        )
        if row["type"] == "llm"
        else _Attribution()
    )
    return Span(
        span_id=row["span_id"],
        parent_span_id=row["parent_span_id"] or None,
        name=row["name"],
        type=row["type"],
        agent=row["agent"],
        start_offset_ms=(int(row["start_ns"]) - trace_start_ns) / NANOS_PER_MS,
        duration_ms=int(row["duration_ns"]) / NANOS_PER_MS,
        status=_status(row["status"]),
        error=row.get("status_message") or None,
        input_preview=row["input_preview"],
        model=row["model"] or None,
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        litellm_request_id=row["litellm_request_id"] or None,
        litellm_call_id=row.get("litellm_call_id") or None,
        spend=attribution.cost,
        spend_source=attribution.source,
        spend_fallback_reason=attribution.fallback_reason,
        spend_event_id=attribution.event_id,
        spend_log_id=attribution.log_id,
        spend_subtotal=attribution.subtotal,
        spend_reason=attribution.reason,
    )


def _parent_agent_of(span: Span, by_id: Mapping[str, Span]) -> str | None:
    parent_id = span["parent_span_id"]
    for _ in by_id:
        if parent_id is None or parent_id not in by_id or parent_id == span["span_id"]:
            return None
        parent = by_id[parent_id]
        if parent["type"] == "agent" and parent["name"] != span["name"]:
            return parent["name"]
        parent_id = parent["parent_span_id"]
    return None


def agent_nodes(spans: Sequence[Span]) -> tuple[AgentNode, ...]:
    """One node per distinct agent name (200 `researcher` invocations = 1 node), with who invoked it."""
    by_id: Final = MappingProxyType({s["span_id"]: s for s in spans})
    agents: dict[str, AgentNode] = {}  # mutable-ok: linear-time aggregation updates counters per agent
    for span in spans:
        if span["type"] != "agent":
            continue
        node = agents.setdefault(
            span["name"],
            AgentNode(
                name=span["name"],
                parent_agent=_parent_agent_of(span, by_id),
                invocations=0,
                llm_calls=0,
                tool_calls=0,
                duration_ms=0.0,
                spend=None,
            ),
        )
        node["invocations"] += 1
        node["duration_ms"] += span["duration_ms"]
    for span in spans:
        owner = agents.get(span["agent"])
        if owner is None:
            continue
        if span["type"] == "llm":
            owner["llm_calls"] += 1
        elif span["type"] == "tool":
            owner["tool_calls"] += 1
    return tuple(
        AgentNode(
            name=agent["name"],
            parent_agent=agent["parent_agent"],
            invocations=agent["invocations"],
            llm_calls=agent["llm_calls"],
            tool_calls=agent["tool_calls"],
            duration_ms=agent["duration_ms"],
            **_agent_spend(spans, agent["name"]),
        )
        for agent in agents.values()
    )


def _agent_spend(spans: Sequence[Span], agent_name: str) -> SpendTotals:
    return _totals(
        tuple(_span_attribution(span) for span in spans if span["type"] == "llm" and span["agent"] == agent_name)
    )


def trace_from_rows(
    trace_id: str, rows: list[dict[str, Any]], trace_ref: str = "", spend_rows: Sequence[_SpendRow] = ()
) -> Trace | None:
    if not rows:
        return None
    trace_start_ns: Final = min(int(r["start_ns"]) for r in rows)
    trace_end_ns: Final = max(int(r["start_ns"]) + int(r["duration_ns"]) for r in rows)
    spans: Final = tuple(span_from_row(r, trace_start_ns, spend_rows) for r in rows)
    root: Final = next((s for s in spans if s["parent_span_id"] is None), spans[0])
    agents: Final = agent_nodes(spans)
    llm_spans: Final = tuple(s for s in spans if s["type"] == "llm")
    return Trace(
        summary=TraceSummary(
            trace_id=trace_id,
            trace_ref=trace_ref,
            name=root["name"],
            service=rows[0]["service"],
            input_preview=root["input_preview"],
            start_time=_iso(trace_start_ns // NANOS_PER_MS),
            duration_ms=(trace_end_ns - trace_start_ns) / NANOS_PER_MS,
            status=root["status"],
            span_count=len(spans),
            agent_count=len(agents),
            agent_invocations=sum(a["invocations"] for a in agents),
            llm_calls=len(llm_spans),
            tool_calls=sum(1 for s in spans if s["type"] == "tool"),
            error_count=sum(1 for s in spans if s["status"] == "error"),
            input_tokens=sum(s["input_tokens"] for s in spans),
            output_tokens=sum(s["output_tokens"] for s in spans),
            models=tuple(sorted(frozenset(s["model"] for s in llm_spans if s["model"]))),
            **_totals(tuple(_span_attribution(span) for span in llm_spans)),
        ),
        agents=agents,
        spans=spans,
    )


class ClickHouseTraceStore:
    """Stores spans and runs scoped trace reads."""

    def __init__(self, storage: TraceStorage, spend_fallback: SpendFallback | None = None) -> None:
        self.storage = storage
        self.spend_fallback = spend_fallback

    async def insert_spans(self, rows: Sequence[SpanRow]) -> None:
        await self.storage.insert_rows(OTEL_TRACES_TABLE, tuple(rows))

    async def _spend_rows(
        self,
        scope: TraceScope,
        request_ids: Sequence[str],
        start_ms: int,
        end_ms: int,
        call_ids: Sequence[str] = (),
    ) -> tuple[_SpendRow, ...]:
        ids: Final = tuple(sorted(frozenset(request_id for request_id in request_ids if request_id)))
        calls: Final = tuple(sorted(frozenset(call_id for call_id in call_ids if call_id)))
        if not ids and not calls:
            return ()
        try:
            rows: Final = await self.storage.query(
                "spend_by_response_ids",
                MappingProxyType(
                    {
                        **scope,
                        "response_ids": ids,
                        "call_ids": calls,
                        "start_ms": start_ms - SPEND_WINDOW_MS,
                        "end_ms": end_ms + SPEND_WINDOW_MS,
                    }
                ),
            )
            stored: Final = _SPEND_ROWS.validate_python(rows)
        except (RuntimeError, ValidationError) as error:
            verbose_logger.warning("Trace spend lookup unavailable: %s", error)
            return await self._fallback_rows(scope, calls, start_ms, end_ms, "clickhouse_unavailable")
        missing: Final = tuple(call_id for call_id in calls if not any(row.call_id == call_id for row in stored))
        return stored + await self._fallback_rows(scope, missing, start_ms, end_ms, "clickhouse_rows_missing")

    async def _fallback_rows(
        self,
        scope: TraceScope,
        call_ids: Sequence[str],
        start_ms: int,
        end_ms: int,
        reason: SpendFallbackReason,
    ) -> tuple[_SpendRow, ...]:
        if self.spend_fallback is None or not call_ids:
            return ()
        try:
            rows: Final = await self.spend_fallback(
                scope, call_ids, start_ms - SPEND_WINDOW_MS, end_ms + SPEND_WINDOW_MS
            )
            recovered: Final = _SPEND_ROWS.validate_python(
                tuple({**row, "source": "postgres_fallback", "fallback_reason": reason} for row in rows)
            )
            verbose_logger.info("Trace spend fallback: reason=%s, candidate_rows=%s", reason, len(recovered))
            return recovered
        except (RuntimeError, ValidationError) as error:
            verbose_logger.warning("Postgres trace spend lookup unavailable: %s", error)
            return ()

    async def _summary(self, row: dict[str, Any]) -> TraceSummary:
        calls: Final = _list_calls(row)
        spend_rows: Final = await self._spend_rows(
            TraceScope(team_ids=(row.get("team_id") or "",), api_key_hash=row.get("api_key_hash") or ""),
            tuple(call[1] for call in calls),
            int(row["start_ms"]),
            int(row["start_ms"]) + int(row["duration_ms"]),
            tuple(call[2] for call in calls),
        )
        return trace_summary_from_row(row, spend_rows)

    async def list_traces(
        self,
        scope: TraceScope,
        start_ms: int,
        end_ms: int,
        cursor: str | None = None,
        limit: int = AGENT_TRACING_LIST_PAGE_SIZE,
    ) -> TracePage:
        cursor_ms, cursor_trace_id = decode_cursor(cursor)
        rows = await self.storage.query(
            "list_traces",
            MappingProxyType(
                {
                    **scope,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "cursor_ms": cursor_ms,
                    "cursor_trace_id": cursor_trace_id,
                    "limit": limit,
                }
            ),
        )
        summaries: Final = tuple(await asyncio.gather(*(self._summary(row) for row in rows)))
        next_cursor: Final = (
            encode_cursor(int(rows[-1]["start_ms"]), rows[-1]["trace_ref"]) if len(rows) == limit else None
        )
        return TracePage(data=summaries, next_cursor=next_cursor)

    async def get_trace(self, trace_id: str, scope: TraceScope, trace_ref: str = "") -> Trace | None:
        rows = await self.storage.query(
            "trace_spans", MappingProxyType({**scope, "trace_id": trace_id, "trace_ref": trace_ref})
        )
        row_scopes: Final = frozenset((row.get("team_id") or "", row.get("api_key_hash") or "") for row in rows)
        exact_scope: Final = (
            TraceScope(team_ids=(rows[0].get("team_id") or "",), api_key_hash=rows[0].get("api_key_hash") or "")
            if len(row_scopes) == 1
            else scope
        )
        spend_rows: Final = await self._spend_rows(
            exact_scope,
            tuple(row["litellm_request_id"] for row in rows),
            min((int(row["start_ns"]) // NANOS_PER_MS for row in rows), default=0),
            max(((int(row["start_ns"]) + int(row["duration_ns"])) // NANOS_PER_MS for row in rows), default=0),
            tuple(row.get("litellm_call_id") or "" for row in rows),
        )
        return trace_from_rows(trace_id, rows, trace_ref, spend_rows)

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "") -> SpanDetail | None:
        rows = await self.storage.query(
            "span_detail",
            MappingProxyType({**scope, "trace_id": trace_id, "span_id": span_id, "trace_ref": trace_ref}),
        )
        if not rows:
            return None
        return SpanDetail(
            span_id=rows[0]["span_id"],
            input=rows[0]["input"],
            output=rows[0]["output"],
            attributes=rows[0]["attributes"],
        )
