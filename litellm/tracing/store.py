"""ClickHouse-backed trace store: batched span writes and scoped reads."""

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

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
    Trace,
    TracePage,
    TraceScope,
    TraceSummary,
)
from litellm.tracing.ui_format import to_ui_content

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


_SPEND_ROWS: Final = TypeAdapter(tuple[_SpendRow, ...])


def _spend_for(request_id: str, team_id: str, api_key_hash: str, rows: Sequence[_SpendRow]) -> float | None:
    matches: Final = tuple(
        row for row in rows if row.response_id == request_id and row.team_id == team_id and row.api_key == api_key_hash
    )
    return matches[0].spend if len(matches) == 1 else None


def _trace_spend(
    request_ids: Sequence[str], team_id: str, api_key_hash: str, rows: Sequence[_SpendRow]
) -> float | None:
    ids: Final = frozenset(request_id for request_id in request_ids if request_id)
    costs: Final = tuple(_spend_for(request_id, team_id, api_key_hash, rows) for request_id in ids)
    return (
        sum(cost for cost in costs if cost is not None) if costs and all(cost is not None for cost in costs) else None
    )


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
        spend=_trace_spend(
            row.get("request_ids") or (), row.get("team_id") or "", row.get("api_key_hash") or "", spend_rows
        ),
    )


def span_from_row(row: dict[str, Any], trace_start_ns: int, spend_rows: Sequence[_SpendRow] = ()) -> Span:
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
        spend=(
            _spend_for(row["litellm_request_id"], row.get("team_id") or "", row.get("api_key_hash") or "", spend_rows)
            if row["litellm_request_id"]
            else None
        ),
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
            spend=_agent_spend(spans, agent["name"]),
        )
        for agent in agents.values()
    )


def _agent_spend(spans: Sequence[Span], agent_name: str) -> float | None:
    by_request: Final = MappingProxyType(
        {
            span["litellm_request_id"]: span["spend"]
            for span in spans
            if span["type"] == "llm" and span["agent"] == agent_name and span["litellm_request_id"]
        }
    )
    return (
        sum(cost for cost in by_request.values() if cost is not None)
        if by_request and all(cost is not None for cost in by_request.values())
        else None
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
            spend=_trace_spend(
                tuple(row["litellm_request_id"] for row in rows),
                rows[0].get("team_id") or "",
                rows[0].get("api_key_hash") or "",
                spend_rows,
            ),
        ),
        agents=agents,
        spans=spans,
    )


class ClickHouseTraceStore:
    """Stores spans and runs scoped trace reads."""

    def __init__(self, storage: TraceStorage) -> None:
        self.storage = storage

    async def insert_spans(self, rows: Sequence[SpanRow]) -> None:
        await self.storage.insert_rows(OTEL_TRACES_TABLE, tuple(rows))

    async def _spend_rows(
        self, scope: TraceScope, request_ids: Sequence[str], start_ms: int, end_ms: int
    ) -> tuple[_SpendRow, ...]:
        ids: Final = tuple(sorted(frozenset(request_id for request_id in request_ids if request_id)))
        if not ids:
            return ()
        try:
            rows: Final = await self.storage.query(
                "spend_by_response_ids",
                MappingProxyType(
                    {
                        **scope,
                        "response_ids": ids,
                        "start_ms": start_ms - SPEND_WINDOW_MS,
                        "end_ms": end_ms + SPEND_WINDOW_MS,
                    }
                ),
            )
        except RuntimeError as error:
            verbose_logger.warning("Trace spend lookup unavailable: %s", error)
            return ()
        return _SPEND_ROWS.validate_python(rows)

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
        spend_rows: Final = await self._spend_rows(
            scope,
            tuple(chain.from_iterable(row.get("request_ids") or () for row in rows)),
            min((int(row["start_ms"]) for row in rows), default=start_ms),
            max((int(row["start_ms"]) + int(row["duration_ms"]) for row in rows), default=end_ms),
        )
        next_cursor = encode_cursor(int(rows[-1]["start_ms"]), rows[-1]["trace_ref"]) if len(rows) == limit else None
        return TracePage(data=tuple(trace_summary_from_row(r, spend_rows) for r in rows), next_cursor=next_cursor)

    async def get_trace(self, trace_id: str, scope: TraceScope, trace_ref: str = "") -> Trace | None:
        rows = await self.storage.query(
            "trace_spans", MappingProxyType({**scope, "trace_id": trace_id, "trace_ref": trace_ref})
        )
        spend_rows: Final = await self._spend_rows(
            scope,
            tuple(row["litellm_request_id"] for row in rows),
            min((int(row["start_ns"]) // NANOS_PER_MS for row in rows), default=0),
            max(((int(row["start_ns"]) + int(row["duration_ns"])) // NANOS_PER_MS for row in rows), default=0),
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
            input_ui=to_ui_content(rows[0]["input"]),
            output_ui=to_ui_content(rows[0]["output"]),
            attributes=rows[0]["attributes"],
        )
