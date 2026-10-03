"""Trace attribution: resolves the evidence normalization recorded on each span against the whole trace.

Normalization reads one span at a time, so it can only say what a span claims to be. Whether an
unnamed root agent only wraps the agents below it, which agent owns a model call, and which spend
records a call accounts for depend on the span graph, and parents and children can arrive in
separate exports. Trace detail and the trace list both resolve through here.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final

from litellm.rust_bridge.trace_queries import SpendRow, TraceSpansRow
from litellm.tracing.types import AgentNode, Span, SpanStatus, SpanType, Trace, TraceSummary

NANOS_PER_MS: Final = 1_000_000
_STATUS: Final[Mapping[str, SpanStatus]] = MappingProxyType({"STATUS_CODE_OK": "ok", "STATUS_CODE_ERROR": "error"})
PROVIDER_RESPONSE: Final = "provider_response"
LITELLM_REQUEST: Final = "litellm_request"
TRANSPORT: Final = "transport"


@dataclass(frozen=True, slots=True)
class Ownership:
    """Who a trace's spend records must belong to."""

    team_id: str
    api_key_hash: str
    user_id: str

    def owns(self, row: SpendRow) -> bool:
        return row.team_id == self.team_id and (
            bool(self.user_id and row.user == self.user_id)
            or bool(self.api_key_hash and row.api_key == self.api_key_hash)
        )


@dataclass(frozen=True, slots=True)
class CallKey:
    kind: str
    id: str

    @staticmethod
    def parse(encoded: str) -> "CallKey":
        kind, _, identifier = encoded.partition(":")
        return CallKey(kind, identifier)


def call_keys(row: TraceSpansRow) -> tuple[CallKey, ...]:
    """The span's recorded keys; rows stored before call evidence keep one response id."""
    keys: Final = tuple(CallKey.parse(key) for key in row.get("call_keys") or ())
    if keys or not row["litellm_request_id"]:
        return keys
    return (CallKey(PROVIDER_RESPONSE, row["litellm_request_id"]),)


def call_evidence(row: TraceSpansRow) -> str:
    recorded: Final = row.get("call_evidence", "")
    if recorded:
        return recorded
    return "complete" if row["litellm_request_id"] else "unknown"


def spend_lookup_ids(rows: Sequence[TraceSpansRow]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """(provider response ids, LiteLLM request ids, trace ids with transport spans) to fetch spend for."""
    keys: Final = tuple((row, key) for row in rows for key in call_keys(row))
    return (
        tuple(sorted({key.id for _, key in keys if key.kind == PROVIDER_RESPONSE and key.id})),
        tuple(sorted({key.id for _, key in keys if key.kind == LITELLM_REQUEST and key.id})),
        tuple(sorted({row.get("trace_id", "") for row, key in keys if key.kind == TRANSPORT} - {""})),
    )


def _status(code: str) -> SpanStatus:
    return _STATUS.get(code, "unset")


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


class _Graph:
    def __init__(self, rows: Sequence[TraceSpansRow]) -> None:
        self.rows: Final = tuple(rows)
        self.by_id: Final = MappingProxyType({row["span_id"]: row for row in rows})
        children: dict[str, list[TraceSpansRow]] = {}  # mutable-ok: built once, read-only after
        for row in rows:
            if row["parent_span_id"] in self.by_id and row["parent_span_id"] != row["span_id"]:
                children.setdefault(row["parent_span_id"], []).append(row)
        self.children: Final = MappingProxyType({key: tuple(value) for key, value in children.items()})

    def parent(self, row: TraceSpansRow) -> TraceSpansRow | None:
        parent: Final = self.by_id.get(row["parent_span_id"])
        return None if parent is None or parent["span_id"] == row["span_id"] else parent

    def ancestors(self, row: TraceSpansRow) -> Iterator[TraceSpansRow]:
        seen: Final[set[str]] = {row["span_id"]}  # mutable-ok: cycle guard
        current = self.parent(row)
        while current is not None and current["span_id"] not in seen:
            seen.add(current["span_id"])
            yield current
            current = self.parent(current)

    def descendants(self, row: TraceSpansRow) -> Iterator[TraceSpansRow]:
        seen: Final[set[str]] = {row["span_id"]}  # mutable-ok: cycle guard
        stack: Final = list(self.children.get(row["span_id"], ()))  # mutable-ok: traversal stack
        while stack:
            current = stack.pop()
            if current["span_id"] in seen:
                continue
            seen.add(current["span_id"])
            yield current
            stack.extend(self.children.get(current["span_id"], ()))


def _agent_label(row: TraceSpansRow) -> str:
    return row["agent"] or row["name"]


class _Resolution:
    """Roles, ownership and calls of one trace's spans."""

    def __init__(self, rows: Sequence[TraceSpansRow], spend_rows: Sequence[SpendRow]) -> None:
        self.graph: Final = _Graph(rows)
        self.ownership: Final = Ownership(
            team_id=rows[0].get("team_id") or "",
            api_key_hash=rows[0].get("api_key_hash") or "",
            user_id=rows[0].get("user_id") or "",
        )
        self.spend_rows: Final = tuple(spend_rows)
        named_agents: Final = frozenset(row["agent"] for row in rows if row["agent"])
        self.types: Final[Mapping[str, SpanType]] = MappingProxyType({row["span_id"]: self._resolved_type(row, named_agents) for row in rows})
        self.model_calls: Final = tuple(
            row
            for row in rows
            if self.types[row["span_id"]] == "llm"
            and not any(self.types[d["span_id"]] == "llm" for d in self.graph.descendants(row))
        )

    def _resolved_type(self, row: TraceSpansRow, named_agents: frozenset[str]) -> SpanType:
        if not row.get("wrapper_candidate") or row["type"] != "agent":
            return row["type"]
        # An unnamed agent wraps the run when the trace names its agents; a named one repeats the
        # agent it runs inside.
        if not row["agent"]:
            return "chain" if named_agents else "agent"
        nearest: Final = next((a for a in self.graph.ancestors(row) if a["type"] == "agent"), None)
        return "chain" if nearest is not None and _agent_label(nearest) == row["agent"] else "agent"

    def is_agent(self, row: TraceSpansRow) -> bool:
        return self.types[row["span_id"]] == "agent"

    def nearest_agent(self, row: TraceSpansRow) -> TraceSpansRow | None:
        return next((a for a in self.graph.ancestors(row) if self.is_agent(a)), None)

    def owner(self, row: TraceSpansRow) -> str:
        """The agent a call or tool runs for: its own, else the agent it runs inside."""
        if row["agent"]:
            return row["agent"]
        nearest: Final = self.nearest_agent(row)
        return _agent_label(nearest) if nearest is not None else ""

    def _matches(self, key: CallKey, row: TraceSpansRow) -> tuple[SpendRow, ...]:
        def matches(spend: SpendRow) -> bool:
            match key.kind:
                case "provider_response":
                    return bool(key.id) and key.id in (spend.response_id, spend.upstream_response_id)
                case "litellm_request":
                    return bool(key.id) and spend.request_id == key.id
                case "transport":
                    return spend.trace_id == row.get("trace_id", "") and spend.span_id == row["span_id"]
                case _:
                    return False

        found: Final = {s.request_id: s for s in self.spend_rows if self.ownership.owns(s) and matches(s)}
        return tuple(found.values())

    def requests(self, row: TraceSpansRow) -> tuple[tuple[SpendRow, ...], bool]:
        """Spend records the span's keys resolve to, and whether they are all of its requests."""
        resolved: Final = tuple(self._matches(key, row) for key in call_keys(row))
        rows: Final = {spend.request_id: spend for matches in resolved for spend in matches}
        complete: Final = (
            call_evidence(row) == "complete" and bool(resolved) and all(len(matches) == 1 for matches in resolved)
        )
        return tuple(rows.values()), complete

    def call_requests(self, call: TraceSpansRow) -> tuple[SpendRow, ...] | None:
        """All spend records behind a model call, or None when they cannot all be known.

        Evidence comes from the call span, LLM spans that only wrap this call, and the HTTP
        requests made inside it.
        """
        wrappers: Final = tuple(
            a
            for a in self.graph.ancestors(call)
            if self.types[a["span_id"]] == "llm"
            and all(
                d["span_id"] == call["span_id"] or self.types[d["span_id"]] != "llm" for d in self.graph.descendants(a)
            )
        )
        transports: Final = tuple(
            d for d in self.graph.descendants(call) if any(k.kind == TRANSPORT for k in call_keys(d))
        )
        found: dict[str, SpendRow] = {}  # mutable-ok: dedupe by request id
        known = False
        for source in (call, *wrappers):
            rows, complete = self.requests(source)
            found.update({row.request_id: row for row in rows})
            known = known or complete
        if transports:
            outcomes = tuple(self.requests(transport) for transport in transports)
            found.update({row.request_id: row for rows, _ in outcomes for row in rows})
            known = known or all(complete for _, complete in outcomes)
        return tuple(found.values()) if known else None


def _total(calls: Sequence[tuple[SpendRow, ...] | None]) -> float | None:
    if not calls or any(requests is None for requests in calls):
        return None
    unique: Final = {row.request_id: row.spend for requests in calls for row in requests or ()}
    return sum(unique.values())


def _span(row: TraceSpansRow, resolution: _Resolution, trace_start_ns: int) -> Span:
    requests, complete = resolution.requests(row)
    return Span(
        span_id=row["span_id"],
        parent_span_id=row["parent_span_id"] or None,
        name=row["name"],
        type=resolution.types[row["span_id"]],
        agent=row["agent"],
        framework=row.get("framework") or "",
        start_offset_ms=(int(row["start_ns"]) - trace_start_ns) / NANOS_PER_MS,
        duration_ms=int(row["duration_ns"]) / NANOS_PER_MS,
        status=_status(row["status"]),
        error=row.get("status_message") or None,
        error_truncated=bool(row.get("error_truncated", False)),
        input_preview=row["input_preview"],
        model=row["model"] or None,
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        litellm_request_id=row["litellm_request_id"] or None,
        spend=sum(r.spend for r in requests) if complete else None,
    )


def _agents(resolution: _Resolution) -> tuple[AgentNode, ...]:
    """One node per distinct agent: agent spans, and agents named only by their calls and tools."""
    graph: Final = resolution.graph
    entries: dict[str, list[TraceSpansRow]] = {}  # mutable-ok: first-seen order of each agent's invocations
    for row in graph.rows:
        if resolution.is_agent(row):
            entries.setdefault(_agent_label(row), []).append(row)
    explicit: Final = frozenset(entries)
    for row in graph.rows:
        parent = graph.parent(row)
        if row["agent"] and row["agent"] not in explicit and (parent is None or parent["agent"] != row["agent"]):
            entries.setdefault(row["agent"], []).append(row)

    def parent_agent(name: str, entry: TraceSpansRow) -> str | None:
        return next(
            (
                label
                for ancestor in graph.ancestors(entry)
                if resolution.is_agent(ancestor) and (label := _agent_label(ancestor)) != name
            ),
            None,
        )

    calls: Final = tuple((resolution.owner(call), resolution.call_requests(call)) for call in resolution.model_calls)
    tools: Final = _unique_tools(resolution)
    return tuple(
        AgentNode(
            name=name,
            parent_agent=parent_agent(name, spans[0]),
            invocations=len(spans),
            llm_calls=sum(1 for owner, _ in calls if owner == name),
            tool_calls=sum(1 for row in tools if resolution.owner(row) == name),
            duration_ms=sum(int(span["duration_ns"]) for span in spans) / NANOS_PER_MS,
            spend=_total(tuple(requests for owner, requests in calls if owner == name)),
        )
        for name, spans in entries.items()
    )


def _unique_tools(resolution: _Resolution) -> tuple[TraceSpansRow, ...]:
    """Tool spans, once per call: overlapping instrumentations record one call under one id."""
    by_call: dict[str, TraceSpansRow] = {}  # mutable-ok: first span per tool call
    for row in resolution.graph.rows:
        if resolution.types[row["span_id"]] == "tool":
            by_call.setdefault(row.get("tool_call_id") or row["span_id"], row)
    return tuple(by_call.values())


def resolve_trace(
    trace_id: str, rows: Sequence[TraceSpansRow], trace_ref: str = "", spend_rows: Sequence[SpendRow] = ()
) -> Trace | None:
    if not rows:
        return None
    resolution: Final = _Resolution(rows, spend_rows)
    trace_start_ns: Final = min(int(r["start_ns"]) for r in rows)
    trace_end_ns: Final = max(int(r["start_ns"]) + int(r["duration_ns"]) for r in rows)
    spans: Final = tuple(_span(row, resolution, trace_start_ns) for row in rows)
    root: Final = next(
        (s for s in spans if s["parent_span_id"] is None or s["parent_span_id"] not in resolution.graph.by_id), spans[0]
    )
    agents: Final = _agents(resolution)
    calls: Final = resolution.model_calls
    counted: Final = calls or tuple(rows)
    first_input: Final = next(
        (
            s["input_preview"]
            for s in sorted(spans, key=lambda s: s["start_offset_ms"])
            if s["input_preview"] and s["type"] in ("agent", "llm")
        ),
        "",
    )
    return Trace(
        summary=TraceSummary(
            trace_id=trace_id,
            trace_ref=trace_ref,
            name=root["name"],
            service=rows[0]["service"],
            agent_names=tuple(sorted(agent["name"] for agent in agents)),
            frameworks=tuple(sorted(frozenset(s["framework"] for s in spans if s["framework"]))),
            input_preview=root["input_preview"] or first_input,
            start_time=_iso(trace_start_ns // NANOS_PER_MS),
            duration_ms=(trace_end_ns - trace_start_ns) / NANOS_PER_MS,
            status=root["status"],
            span_count=len(spans),
            agent_count=len(agents),
            agent_invocations=sum(agent["invocations"] for agent in agents),
            llm_calls=len(calls),
            tool_calls=len(_unique_tools(resolution)),
            error_count=sum(1 for s in spans if s["status"] == "error"),
            # Agent spans repeat their calls' usage, so totals count model calls when there are any.
            input_tokens=sum(int(row["input_tokens"]) for row in counted),
            output_tokens=sum(int(row["output_tokens"]) for row in counted),
            models=tuple(sorted(frozenset(row["model"] for row in calls if row["model"]))),
            spend=_total(tuple(resolution.call_requests(call) for call in calls)),
        ),
        agents=agents,
        spans=spans,
    )
