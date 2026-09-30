"""ClickHouse-backed trace store: batched span writes and scoped reads."""

import base64
import json
from datetime import UTC, datetime
from typing import Any, Final

from litellm.constants import AGENT_TRACING_LIST_PAGE_SIZE
from litellm.integrations.clickhouse.schema import (
    AGENT_TRACES_TABLE,
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

NANOS_PER_MS: Final = 1_000_000
_STATUS: Final[dict[str, SpanStatus]] = {"STATUS_CODE_OK": "ok", "STATUS_CODE_ERROR": "error"}

_SCOPE_OTEL: Final = (
    "(empty({team_ids:Array(String)}) OR TeamId IN {team_ids:Array(String)})"
    " AND ({api_key_hash:String} = '' OR ApiKeyHash = {api_key_hash:String})"
)
# Page of traces from the per-trace MV.
# The MV writes one partial row per insert, so root fields come from the partial that saw the root span.
# agent_traces has no ApiKeyHash, so key-scoped (team-less) reads filter trace ids through otel_traces.
LIST_TRACES_SQL: Final = f"""
SELECT t.TraceId AS trace_id, any(t.RootName) AS name, any(t.ServiceName) AS service,
       any(t.RootInput) AS input_preview, any(t.RootStatus) AS status,
       toUnixTimestamp64Milli(any(t.StartTs)) AS start_ms,
       dateDiff('millisecond', any(t.StartTs), any(t.EndTs)) AS duration_ms,
       any(t.SpanCount) AS span_count, length(any(t.AgentNames)) AS agent_count,
       any(t.AgentCount) AS agent_invocations,
       any(t.LlmCount) AS llm_calls, any(t.ToolCount) AS tool_calls,
       any(t.InputTokens) AS input_tokens, any(t.OutputTokens) AS output_tokens,
       any(t.Models) AS models, any(t.ErrorCount) AS error_count
FROM (
    SELECT TeamId, TraceId, min(StartTs) AS StartTs, max(EndTs) AS EndTs,
           any(ServiceName) AS ServiceName, anyLastIf(a.RootName, a.RootName != '') AS RootName,
           anyLastIf(a.RootInput, a.RootName != '') AS RootInput,
           anyLastIf(a.RootStatus, a.RootName != '') AS RootStatus,
           sum(SpanCount) AS SpanCount, sum(AgentCount) AS AgentCount, sum(LlmCount) AS LlmCount,
           sum(ToolCount) AS ToolCount, sum(ErrorCount) AS ErrorCount, sum(InputTokens) AS InputTokens,
           sum(OutputTokens) AS OutputTokens, groupUniqArrayArray(Models) AS Models,
           groupUniqArrayArray(AgentNames) AS AgentNames
    FROM {AGENT_TRACES_TABLE} AS a
    WHERE (empty({{team_ids:Array(String)}}) OR TeamId IN {{team_ids:Array(String)}})
      AND ({{api_key_hash:String}} = '' OR TraceId IN (
          SELECT TraceId FROM {OTEL_TRACES_TABLE} WHERE {_SCOPE_OTEL}))
    GROUP BY TeamId, TraceId
    HAVING StartTs >= fromUnixTimestamp64Milli({{start_ms:Int64}})
       AND StartTs < fromUnixTimestamp64Milli({{end_ms:Int64}})
       AND (({{cursor_ms:Int64}} = 0) OR (toUnixTimestamp64Milli(StartTs), TraceId)
            < ({{cursor_ms:Int64}}, {{cursor_trace_id:String}}))
    ORDER BY StartTs DESC, TraceId DESC
    LIMIT {{limit:UInt32}}
) AS t
GROUP BY t.TraceId
ORDER BY start_ms DESC, t.TraceId DESC
"""

TRACE_SPANS_SQL: Final = f"""
SELECT o.SpanId AS span_id, o.ParentSpanId AS parent_span_id, o.SpanName AS name,
       o.ObservationType AS type, o.AgentName AS agent, o.StatusCode AS status,
       o.StatusMessage AS status_message,
       toUnixTimestamp64Nano(o.Timestamp) AS start_ns, o.Duration AS duration_ns,
       o.ServiceName AS service, o.InputPreview AS input_preview, o.Model AS model,
       o.InputTokens AS input_tokens, o.OutputTokens AS output_tokens,
       o.LiteLLMRequestId AS litellm_request_id
FROM {OTEL_TRACES_TABLE} AS o
WHERE o.TraceId = {{trace_id:String}} AND {_SCOPE_OTEL}
ORDER BY o.Timestamp
LIMIT 1 BY o.SpanId
"""

TRACE_IO_SQL: Final = f"""
SELECT SpanId AS span_id, Input AS input, Output AS output
FROM {OTEL_TRACES_TABLE}
WHERE TraceId = {{trace_id:String}} AND {_SCOPE_OTEL}
"""

SPAN_DETAIL_SQL: Final = f"""
SELECT SpanId AS span_id, Input AS input, Output AS output, SpanAttributes AS attributes
FROM {OTEL_TRACES_TABLE}
WHERE TraceId = {{trace_id:String}} AND SpanId = {{span_id:String}} AND {_SCOPE_OTEL}
LIMIT 1
"""


def encode_cursor(start_ms: int, trace_id: str) -> str:
    return base64.urlsafe_b64encode(json.dumps([start_ms, trace_id]).encode()).decode()


def decode_cursor(cursor: str | None) -> tuple[int, str]:
    if not cursor:
        return 0, ""
    start_ms, trace_id = json.loads(base64.urlsafe_b64decode(cursor.encode()))
    return int(start_ms), str(trace_id)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


def _status(code: str) -> SpanStatus:
    return _STATUS.get(code, "unset")


def trace_summary_from_row(row: dict[str, Any]) -> TraceSummary:
    return TraceSummary(
        trace_id=row["trace_id"],
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
        models=list(row["models"]),
    )


def span_from_row(row: dict[str, Any], trace_start_ns: int) -> Span:
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
    )


def _parent_agent_of(span: Span, by_id: dict[str, Span]) -> str | None:
    parent_id = span["parent_span_id"]
    while parent_id is not None and parent_id in by_id:
        parent = by_id[parent_id]
        if parent["type"] == "agent" and parent["name"] != span["name"]:
            return parent["name"]
        parent_id = parent["parent_span_id"]
    return None


def agent_nodes(spans: list[Span]) -> list[AgentNode]:
    """One node per distinct agent name (200 `researcher` invocations = 1 node), with who invoked it."""
    by_id: Final = {s["span_id"]: s for s in spans}
    agents: dict[str, AgentNode] = {}
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
    return list(agents.values())


def trace_from_rows(trace_id: str, rows: list[dict[str, Any]]) -> Trace | None:
    if not rows:
        return None
    trace_start_ns: Final = min(int(r["start_ns"]) for r in rows)
    trace_end_ns: Final = max(int(r["start_ns"]) + int(r["duration_ns"]) for r in rows)
    spans: Final = [span_from_row(r, trace_start_ns) for r in rows]
    root: Final = next((s for s in spans if s["parent_span_id"] is None), spans[0])
    agents: Final = agent_nodes(spans)
    llm_spans: Final = [s for s in spans if s["type"] == "llm"]
    return Trace(
        summary=TraceSummary(
            trace_id=trace_id,
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
            models=sorted({s["model"] for s in llm_spans if s["model"]}),
        ),
        agents=agents,
        spans=spans,
    )


class ClickHouseTraceStore:
    """Stores spans and runs scoped trace reads."""

    def __init__(self, storage: TraceStorage):
        self.storage = storage

    async def insert_spans(self, rows: list[SpanRow]) -> None:
        await self.storage.insert_rows(OTEL_TRACES_TABLE, [dict(row) for row in rows])

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
            LIST_TRACES_SQL,
            {
                **scope,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "cursor_ms": cursor_ms,
                "cursor_trace_id": cursor_trace_id,
                "limit": limit,
            },
        )
        next_cursor = encode_cursor(int(rows[-1]["start_ms"]), rows[-1]["trace_id"]) if len(rows) == limit else None
        return TracePage(data=[trace_summary_from_row(r) for r in rows], next_cursor=next_cursor)

    async def get_trace(self, trace_id: str, scope: TraceScope) -> Trace | None:
        rows = await self.storage.query(TRACE_SPANS_SQL, {**scope, "trace_id": trace_id})
        return trace_from_rows(trace_id, rows)

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope) -> SpanDetail | None:
        rows = await self.storage.query(SPAN_DETAIL_SQL, {**scope, "trace_id": trace_id, "span_id": span_id})
        if not rows:
            return None
        return SpanDetail(
            span_id=rows[0]["span_id"],
            input=rows[0]["input"],
            output=rows[0]["output"],
            attributes=dict(rows[0]["attributes"]),
        )

    async def get_span_io(self, trace_id: str, scope: TraceScope) -> dict[str, tuple[str, str]]:
        """span_id -> (input, output) for every span in the trace, in one query (for exports)."""
        rows = await self.client.query(TRACE_IO_SQL, {**scope, "trace_id": trace_id})
        return {r["span_id"]: (r["input"], r["output"]) for r in rows}
