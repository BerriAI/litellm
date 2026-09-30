"""ClickHouse-backed trace store: batched span writes and scoped reads."""

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Final

from litellm.constants import AGENT_TRACING_LIST_PAGE_SIZE
from litellm.integrations.clickhouse.schema import (
    AGENT_TRACES_BY_KEY_TABLE,
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
_STATUS: Final = MappingProxyType({"STATUS_CODE_OK": "ok", "STATUS_CODE_ERROR": "error"})

_SCOPE_OTEL: Final = (
    "(empty({team_ids:Array(String)}) OR TeamId IN {team_ids:Array(String)})"
    " AND ({api_key_hash:String} = '' OR ApiKeyHash = {api_key_hash:String})"
)
_TRACE_REF_SQL: Final = "hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId)))"
LIST_TRACES_SQL: Final = f"""
SELECT TraceId AS trace_id, {_TRACE_REF_SQL} AS trace_ref,
       ifNull(any(RootName), '') AS name, any(ServiceName) AS service,
       ifNull(any(RootInput), '') AS input_preview, ifNull(any(RootStatus), '') AS status,
       toUnixTimestamp64Milli(min(StartTs)) AS start_ms,
       dateDiff('millisecond', min(StartTs), max(EndTs)) AS duration_ms,
       sum(SpanCount) AS span_count, length(groupUniqArrayArray(AgentNames)) AS agent_count,
       sum(AgentCount) AS agent_invocations,
       sum(LlmCount) AS llm_calls, sum(ToolCount) AS tool_calls,
       sum(InputTokens) AS input_tokens, sum(OutputTokens) AS output_tokens,
       groupUniqArrayArray(Models) AS models, sum(ErrorCount) AS error_count
FROM {AGENT_TRACES_BY_KEY_TABLE}
WHERE (empty({{team_ids:Array(String)}}) OR TeamId IN {{team_ids:Array(String)}})
  AND ({{api_key_hash:String}} = '' OR ApiKeyHash = {{api_key_hash:String}})
GROUP BY TeamId, ApiKeyHash, TraceId
HAVING min(StartTs) >= fromUnixTimestamp64Milli({{start_ms:Int64}})
   AND min(StartTs) < fromUnixTimestamp64Milli({{end_ms:Int64}})
   AND ({{cursor_ms:Int64}} = 0 OR (toUnixTimestamp64Milli(min(StartTs)), trace_ref)
        < ({{cursor_ms:Int64}}, {{cursor_trace_id:String}}))
ORDER BY start_ms DESC, trace_ref DESC
LIMIT {{limit:UInt32}}
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
  AND ({{trace_ref:String}} = '' OR {_TRACE_REF_SQL} = {{trace_ref:String}})
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
  AND ({{trace_ref:String}} = '' OR {_TRACE_REF_SQL} = {{trace_ref:String}})
LIMIT 1
"""


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


def trace_summary_from_row(row: dict[str, Any]) -> TraceSummary:
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
    return tuple(agents.values())


def trace_from_rows(trace_id: str, rows: list[dict[str, Any]], trace_ref: str = "") -> Trace | None:
    if not rows:
        return None
    trace_start_ns: Final = min(int(r["start_ns"]) for r in rows)
    trace_end_ns: Final = max(int(r["start_ns"]) + int(r["duration_ns"]) for r in rows)
    spans: Final = tuple(span_from_row(r, trace_start_ns) for r in rows)
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
        next_cursor = encode_cursor(int(rows[-1]["start_ms"]), rows[-1]["trace_ref"]) if len(rows) == limit else None
        return TracePage(data=tuple(trace_summary_from_row(r) for r in rows), next_cursor=next_cursor)

    async def get_trace(self, trace_id: str, scope: TraceScope, trace_ref: str = "") -> Trace | None:
        rows = await self.storage.query(
            TRACE_SPANS_SQL, MappingProxyType({**scope, "trace_id": trace_id, "trace_ref": trace_ref})
        )
        return trace_from_rows(trace_id, rows, trace_ref)

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "") -> SpanDetail | None:
        rows = await self.storage.query(
            SPAN_DETAIL_SQL,
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

    async def get_span_io(self, trace_id: str, scope: TraceScope) -> dict[str, tuple[str, str]]:
        """span_id -> (input, output) for every span in the trace, in one query (for exports)."""
        rows = await self.client.query(TRACE_IO_SQL, {**scope, "trace_id": trace_id})
        return {r["span_id"]: (r["input"], r["output"]) for r in rows}
