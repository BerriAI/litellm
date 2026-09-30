"""
Agent tracing types.

A trace is one agent run. It's made of spans (agent / llm / tool / chain / framework).
    Trace
    ├── summary: TraceSummary
    ├── agents:  list[AgentNode]      one per distinct agent name (for the agent graph)
    └── spans:   list[Span]           flat, linked by parent_span_id

See `litellm/tracing/README.md` for the full contract with example JSON.
"""

from typing import Literal

from typing_extensions import TypedDict

SpanType = Literal["agent", "llm", "tool", "chain", "framework"]
SpanStatus = Literal["ok", "error", "unset"]


class Span(TypedDict):
    span_id: str
    parent_span_id: str | None
    name: str
    type: SpanType
    agent: str  # the agent this span runs inside, e.g. "researcher"
    start_offset_ms: float  # relative to trace start
    duration_ms: float
    status: SpanStatus
    error: str | None  # exception message when status == "error"
    input_preview: str
    model: str | None
    input_tokens: int
    output_tokens: int
    litellm_request_id: str | None


class AgentNode(TypedDict):
    """One distinct agent in a trace. 200 invocations of `researcher` = one node."""

    name: str
    parent_agent: str | None
    invocations: int
    llm_calls: int
    tool_calls: int
    duration_ms: float


class TraceSummary(TypedDict):
    trace_id: str
    name: str
    service: str
    input_preview: str
    start_time: str  # ISO 8601
    duration_ms: float
    status: SpanStatus
    span_count: int
    agent_count: int  # distinct agent names (researcher x200 counts once)
    agent_invocations: int  # agent spans (researcher x200 counts 200)
    llm_calls: int
    tool_calls: int
    error_count: int  # spans with an error status; > 0 means the run shows as failed
    input_tokens: int
    output_tokens: int
    models: list[str]


class Trace(TypedDict):
    summary: TraceSummary
    agents: list[AgentNode]
    spans: list[Span]


class TracePage(TypedDict):
    data: list[TraceSummary]
    next_cursor: str | None


class SpanDetail(TypedDict):
    span_id: str
    input: str
    output: str
    attributes: dict[str, str]


class TraceScope(TypedDict):
    """Who is asking. Empty team_ids = all teams (admins only)."""

    team_ids: list[str]
    api_key_hash: str


class SpanRow(TypedDict):
    """One stored span (ClickHouse `otel_traces` row). Produced by `litellm.tracing.decode`."""

    Timestamp: int  # unix ns
    TraceId: str
    SpanId: str
    ParentSpanId: str
    TraceState: str
    SpanName: str
    SpanKind: str
    ServiceName: str
    ResourceAttributes: dict[str, str]
    ScopeName: str
    ScopeVersion: str
    SpanAttributes: dict[str, str]
    Duration: int  # ns
    StatusCode: str
    StatusMessage: str
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
