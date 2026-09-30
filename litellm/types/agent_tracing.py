"""
Types for LiteLLM agent tracing.

Two contracts live here:
- `SpanRecord`: one normalized span, as stored by a TraceStore (ClickHouse `otel_traces` row).
- `TraceResponse` / `TraceListResponse`: what `GET /v1/traces*` returns to the UI and API users.
"""

from typing import Literal

from typing_extensions import TypedDict

ObservationType = Literal["agent", "llm", "tool", "chain", "framework"]


class SpanRecord(TypedDict):
    """One normalized span. Standard OTLP fields + LiteLLM-derived fields."""

    # standard OTLP fields (column-compatible with the OTel Collector clickhouseexporter)
    Timestamp: int  # start, unix nanoseconds
    TraceId: str  # 32 hex chars
    SpanId: str  # 16 hex chars
    ParentSpanId: str  # "" for root spans
    TraceState: str
    SpanName: str
    SpanKind: str
    ServiceName: str
    ResourceAttributes: dict[str, str]
    ScopeName: str
    ScopeVersion: str
    SpanAttributes: dict[str, str]
    Duration: int  # nanoseconds
    StatusCode: str
    StatusMessage: str
    # LiteLLM-derived fields (set by a SpanNormalizer + tenant stamping)
    TeamId: str
    ApiKeyHash: str
    ObservationType: ObservationType
    AgentName: str  # nearest enclosing agent, e.g. "researcher"
    LiteLLMRequestId: str  # provider response id == spend_logs.response_id
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


class LiteLLMRequestView(TypedDict):
    """The LiteLLM side of an LLM span: joined from spend logs."""

    request_id: str
    model: str
    model_group: str
    provider: str
    api_base: str
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


class SpanView(TypedDict):
    span_id: str
    parent_span_id: str | None
    name: str
    type: ObservationType
    agent: str
    start_offset_ms: float
    duration_ms: float
    status: str
    input_preview: str
    model: str | None
    input_tokens: int
    output_tokens: int
    litellm: LiteLLMRequestView | None


class AgentView(TypedDict):
    """One agent (root or subagent) inside a trace, for the multi-agent graph."""

    name: str
    parent_agent: str | None
    invocations: int
    llm_calls: int
    tool_calls: int
    spend: float
    duration_ms: float


class TraceSummary(TypedDict):
    trace_id: str
    name: str
    service: str
    input_preview: str
    start_time: str  # ISO8601
    duration_ms: float
    status: str
    span_count: int
    agent_count: int
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    spend: float
    models: list[str]


class TraceResponse(TypedDict):
    trace: TraceSummary
    agents: list[AgentView]
    spans: list[SpanView]


class TraceListResponse(TypedDict):
    data: list[TraceSummary]
    next_cursor: str | None


class SpanDetailResponse(TypedDict):
    span_id: str
    input: str
    output: str
    attributes: dict[str, str]
