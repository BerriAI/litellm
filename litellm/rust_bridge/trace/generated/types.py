from typing import Literal, TypeAlias

from typing_extensions import NotRequired, ReadOnly, TypedDict

SpanType: TypeAlias = Literal[
    "agent",
    "llm",
    "tool",
    "chain",
    "framework",
    "retriever",
    "embedding",
    "reranker",
    "guardrail",
    "evaluator",
    "prompt",
    "decision",
]

ReadQueryName: TypeAlias = Literal[
    "availability",
    "agents",
    "sample",
    "content",
    "evidence",
]

class TraceScope(TypedDict):
    """Authenticated request-log visibility."""

    all_teams: ReadOnly[Literal[0, 1]]
    user_id: ReadOnly[str]
    team_ids: ReadOnly[tuple[str, ...]]

class AllQueryScope(TypedDict):
    kind: ReadOnly[Literal["all"]]

class OwnedQueryScope(TypedDict):
    kind: ReadOnly[Literal["owned"]]
    user_id: ReadOnly[str]
    team_ids: ReadOnly[tuple[str, ...]]

QueryScope = AllQueryScope | OwnedQueryScope

TraceTableName = Literal["otel_traces", "agent_traces_by_key", "spend_logs"]

MetadataValueType = Literal["array", "boolean", "integer", "null", "number", "object", "string"]

SpanStatus = Literal["ok", "error", "unset"]

ChatRole: TypeAlias = Literal["system", "user", "assistant", "tool"]

class UIToolCall(TypedDict):
    name: ReadOnly[str]
    arguments: ReadOnly[str]

class UIMessage(TypedDict):
    role: ReadOnly[ChatRole]
    content: ReadOnly[str]
    name: ReadOnly[NotRequired[str]]
    tool_calls: ReadOnly[NotRequired[tuple[UIToolCall, ...]]]

class UIField(TypedDict):
    key: ReadOnly[str]
    value: ReadOnly[str]

class UIMessages(TypedDict):
    kind: ReadOnly[Literal["messages"]]
    messages: ReadOnly[tuple[UIMessage, ...]]

class UIFields(TypedDict):
    kind: ReadOnly[Literal["fields"]]
    fields: ReadOnly[tuple[UIField, ...]]

class UIText(TypedDict):
    kind: ReadOnly[Literal["text"]]
    text: ReadOnly[str]

UIContent: TypeAlias = UIMessages | UIFields | UIText

class Span(TypedDict):
    span_id: ReadOnly[str]
    parent_span_id: ReadOnly[str | None]
    name: ReadOnly[str]
    type: ReadOnly[SpanType]
    agent: ReadOnly[str]
    framework: ReadOnly[str]
    start_offset_ms: ReadOnly[float]
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
    invocations: ReadOnly[int]
    llm_calls: ReadOnly[int]
    tool_calls: ReadOnly[int]
    duration_ms: ReadOnly[float]
    spend: ReadOnly[float | None]

class TraceSummary(TypedDict):
    trace_id: ReadOnly[str]
    trace_ref: ReadOnly[NotRequired[str]]
    name: ReadOnly[str]
    service: ReadOnly[str]
    agent_names: ReadOnly[NotRequired[tuple[str, ...]]]
    frameworks: ReadOnly[NotRequired[tuple[str, ...]]]
    input_preview: ReadOnly[str]
    start_time: ReadOnly[str]
    duration_ms: ReadOnly[float]
    status: ReadOnly[SpanStatus]
    span_count: ReadOnly[int]
    agent_count: ReadOnly[int]
    agent_invocations: ReadOnly[int]
    llm_calls: ReadOnly[int]
    tool_calls: ReadOnly[int]
    error_count: ReadOnly[int]
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
