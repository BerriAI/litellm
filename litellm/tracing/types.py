"""
Agent tracing types.

A trace is one agent run. It's made of spans (agent / llm / tool / chain / framework).
    Trace
    ├── summary: TraceSummary
    ├── agents:  list[AgentNode]      one per distinct agent name (for the agent graph)
    └── spans:   list[Span]           flat, linked by parent_span_id

"""

from typing import Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict

SpanType = Literal["agent", "llm", "tool", "chain", "framework"]
SpanStatus = Literal["ok", "error", "unset"]


class Span(TypedDict):
    span_id: ReadOnly[str]
    parent_span_id: ReadOnly[str | None]
    name: ReadOnly[str]
    type: ReadOnly[SpanType]
    agent: ReadOnly[str]  # the agent this span runs inside, e.g. "researcher"
    start_offset_ms: ReadOnly[float]  # relative to trace start
    duration_ms: ReadOnly[float]
    status: ReadOnly[SpanStatus]
    error: ReadOnly[str | None]  # exception message when status == "error"
    input_preview: ReadOnly[str]
    model: ReadOnly[str | None]
    input_tokens: ReadOnly[int]
    output_tokens: ReadOnly[int]
    litellm_request_id: ReadOnly[str | None]


class AgentNode(TypedDict):
    """One distinct agent in a trace. 200 invocations of `researcher` = one node."""

    name: ReadOnly[str]
    parent_agent: ReadOnly[str | None]
    invocations: int
    llm_calls: int
    tool_calls: int
    duration_ms: float


class TraceSummary(TypedDict):
    trace_id: ReadOnly[str]
    trace_ref: ReadOnly[NotRequired[str]]
    name: ReadOnly[str]
    service: ReadOnly[str]
    input_preview: ReadOnly[str]
    start_time: ReadOnly[str]  # ISO 8601
    duration_ms: ReadOnly[float]
    status: ReadOnly[SpanStatus]
    span_count: ReadOnly[int]
    agent_count: ReadOnly[int]  # distinct agent names (researcher x200 counts once)
    agent_invocations: ReadOnly[int]  # agent spans (researcher x200 counts 200)
    llm_calls: ReadOnly[int]
    tool_calls: ReadOnly[int]
    error_count: ReadOnly[int]  # spans with an error status; > 0 means the run shows as failed
    input_tokens: ReadOnly[int]
    output_tokens: ReadOnly[int]
    models: ReadOnly[tuple[str, ...]]


class Trace(TypedDict):
    summary: ReadOnly[TraceSummary]
    agents: ReadOnly[tuple[AgentNode, ...]]
    spans: ReadOnly[tuple[Span, ...]]


class TracePage(TypedDict):
    data: ReadOnly[tuple[TraceSummary, ...]]
    next_cursor: ReadOnly[str | None]


class SpanDetail(TypedDict):
    span_id: ReadOnly[str]
    input: ReadOnly[str]
    output: ReadOnly[str]
    attributes: ReadOnly[dict[str, str]]


class TraceScope(TypedDict):
    """Who is asking. Empty team_ids = all teams (admins only)."""

    team_ids: ReadOnly[tuple[str, ...]]
    api_key_hash: ReadOnly[str]


class SpanRow(TypedDict):
    """One stored span (ClickHouse `otel_traces` row). Produced by `litellm.tracing.decode`."""

    Timestamp: ReadOnly[int]  # unix ns
    TraceId: ReadOnly[str]
    SpanId: ReadOnly[str]
    ParentSpanId: ReadOnly[str]
    TraceState: ReadOnly[str]
    SpanName: ReadOnly[str]
    SpanKind: ReadOnly[str]
    ServiceName: ReadOnly[str]
    ResourceAttributes: dict[str, str]
    ScopeName: ReadOnly[str]
    ScopeVersion: ReadOnly[str]
    SpanAttributes: dict[str, str]
    Duration: ReadOnly[int]  # ns
    StatusCode: ReadOnly[str]
    StatusMessage: ReadOnly[str]
    TeamId: str
    ApiKeyHash: str
    ObservationType: SpanType
    AgentName: str
    LiteLLMRequestId: str
    Model: str
    InputTokens: int
    OutputTokens: int
    Input: str
    Output: str
