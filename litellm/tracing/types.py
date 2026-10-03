"""
Agent tracing types.

A trace is one agent run. It's made of spans (agent / llm / tool / chain / framework).
    Trace
    ├── summary: TraceSummary
    ├── agents:  list[AgentNode]      one per distinct agent name (for the agent graph)
    └── spans:   list[Span]           flat, linked by parent_span_id

"""

from collections.abc import Mapping, Sequence
from typing import Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.tracing.ui_format import UIContent

SpanType = Literal["agent", "llm", "tool", "chain", "framework"]
SpanStatus = Literal["ok", "error", "unset"]


class Span(TypedDict):
    span_id: ReadOnly[str]
    parent_span_id: ReadOnly[str | None]
    name: ReadOnly[str]
    type: ReadOnly[SpanType]
    agent: ReadOnly[str]  # the agent this span runs inside, e.g. "researcher"
    framework: ReadOnly[str]  # SDK that emitted the span, e.g. "claude-agent-sdk"; "" when unknown
    start_offset_ms: ReadOnly[float]  # relative to trace start
    duration_ms: ReadOnly[float]
    status: ReadOnly[SpanStatus]
    error: ReadOnly[str | None]
    error_truncated: ReadOnly[bool]
    input_preview: ReadOnly[str]
    model: ReadOnly[str | None]
    input_tokens: ReadOnly[int]
    output_tokens: ReadOnly[int]
    litellm_request_id: ReadOnly[str | None]
    spend: ReadOnly[float | None]


class AgentNode(TypedDict):
    """One distinct agent in a trace. 200 invocations of `researcher` = one node."""

    name: ReadOnly[str]
    parent_agent: ReadOnly[str | None]
    invocations: int
    llm_calls: int
    tool_calls: int
    duration_ms: float
    spend: ReadOnly[float | None]


class TraceSummary(TypedDict):
    trace_id: ReadOnly[str]
    trace_ref: ReadOnly[NotRequired[str]]
    name: ReadOnly[str]
    service: ReadOnly[str]
    agent_names: ReadOnly[NotRequired[tuple[str, ...]]]
    frameworks: ReadOnly[NotRequired[tuple[str, ...]]]
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
    spend: ReadOnly[float | None]


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
    input_ui: ReadOnly[UIContent]
    output_ui: ReadOnly[UIContent]
    attributes: ReadOnly[dict[str, str]]


class SpanErrorPage(TypedDict):
    span_id: ReadOnly[str]
    message: ReadOnly[str]
    total_chars: ReadOnly[int]
    next_cursor: ReadOnly[str | None]


class TraceScope(TypedDict):
    """Authenticated request-log visibility."""

    all_teams: ReadOnly[Literal[0, 1]]
    user_id: ReadOnly[str]
    team_ids: ReadOnly[tuple[str, ...]]


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
    ResourceAttributes: ReadOnly[Mapping[str, str]]
    ScopeName: ReadOnly[str]
    ScopeVersion: ReadOnly[str]
    SpanAttributes: ReadOnly[Mapping[str, str]]
    Duration: ReadOnly[int]  # ns
    StatusCode: ReadOnly[str]
    StatusMessage: ReadOnly[str]
    TeamId: ReadOnly[str]
    ApiKeyHash: ReadOnly[str]
    UserId: ReadOnly[str]
    ObservationType: SpanType
    AgentName: str
    Framework: ReadOnly[str]
    LiteLLMRequestId: str
    Model: str
    InputTokens: int
    OutputTokens: int
    Input: str
    Output: str


class SpendLogRecord(TypedDict):
    """One LiteLLM request, as written by the `clickhouse` logging callback."""

    request_id: ReadOnly[str]
    response_id: ReadOnly[str]
    call_type: ReadOnly[str]
    api_key: ReadOnly[str]
    key_alias: ReadOnly[str]
    team_id: ReadOnly[str]
    team_alias: ReadOnly[str]
    organization_id: ReadOnly[str]
    user: ReadOnly[str]
    end_user: ReadOnly[str]
    model: ReadOnly[str]
    model_group: ReadOnly[str]
    model_id: ReadOnly[str]
    custom_llm_provider: ReadOnly[str]
    api_base: ReadOnly[str]
    spend: ReadOnly[float]
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]
    total_tokens: ReadOnly[int]
    cache_read_tokens: ReadOnly[int]
    cache_write_tokens: ReadOnly[int]
    start_time: ReadOnly[int]  # unix ms
    end_time: ReadOnly[int]  # unix ms
    completion_start_time: ReadOnly[int | None]
    status: ReadOnly[str]
    error_str: ReadOnly[str]
    cache_hit: ReadOnly[bool]
    session_id: ReadOnly[str]
    trace_id: ReadOnly[str]  # from an incoming W3C traceparent, if any
    span_id: ReadOnly[str]
    request_tags: ReadOnly[Sequence[str]]
    metadata: ReadOnly[str]
    messages: ReadOnly[str]
    response: ReadOnly[str]
