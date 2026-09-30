"""
Agent tracing types.

A trace is one agent run. It's made of spans (agent / llm / tool / chain / framework).
LLM spans that went through LiteLLM carry a `LiteLLMRequest`: the spend-log row for that
call, joined on the provider response id.

    Trace
    ├── summary: TraceSummary
    ├── agents:  list[AgentNode]      one per distinct agent name (for the agent graph)
    └── spans:   list[Span]           flat, linked by parent_span_id
                 └── litellm: LiteLLMRequest | None     (llm spans only)

"""

from typing import Literal

from typing_extensions import TypedDict

SpanType = Literal["agent", "llm", "tool", "chain", "framework"]
SpanStatus = Literal["ok", "error", "unset"]


class LiteLLMRequest(TypedDict):
    """The LiteLLM side of an LLM span: joined from spend logs by response id."""

    request_id: str
    model: str
    model_group: str
    provider: str
    key_alias: str
    team_alias: str
    spend: float
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_ms: int
    ttft_ms: int | None
    status: str


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
    litellm: LiteLLMRequest | None


class AgentNode(TypedDict):
    """One distinct agent in a trace. 200 invocations of `researcher` = one node."""

    name: str
    parent_agent: str | None
    invocations: int
    llm_calls: int
    tool_calls: int
    spend: float | None
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
    spend: float | None
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


class SpendLogRecord(TypedDict):
    """One LiteLLM request, as written by the `clickhouse` logging callback."""

    request_id: str
    response_id: str
    call_type: str
    api_key: str
    key_alias: str
    team_id: str
    team_alias: str
    organization_id: str
    user: str
    end_user: str
    model: str
    model_group: str
    model_id: str
    custom_llm_provider: str
    api_base: str
    spend: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    start_time: int  # unix ms
    end_time: int  # unix ms
    completion_start_time: int | None
    status: str
    error_str: str
    cache_hit: bool
    session_id: str
    trace_id: str  # from an incoming W3C traceparent, if any
    span_id: str
    request_tags: list[str]
    metadata: str
    messages: str
    response: str
