"""
Payloads for Datadog LLM Observability Service (LLMObs)

API Reference: https://docs.datadoghq.com/llm_observability/setup/api/?tab=example#api-standards
"""

from collections.abc import Sequence
from typing import Any, Literal

from typing_extensions import ReadOnly, TypedDict

from litellm.types.integrations.custom_logger import StandardCustomLoggerInitParams


class ToolCall(TypedDict, total=False):
    """A tool call on a message, as LLM Obs names its fields."""

    name: ReadOnly[str]
    arguments: ReadOnly[dict[str, Any] | str]  # parsed object, or the raw string when it will not parse to one
    tool_id: ReadOnly[str]
    type: ReadOnly[str]


class ToolResult(TypedDict, total=False):
    """The result of a tool call, as LLM Obs names its fields."""

    name: ReadOnly[str]
    result: ReadOnly[str]
    tool_id: ReadOnly[str]
    type: ReadOnly[str]


class ToolDefinition(TypedDict, total=False):
    """A tool the model was offered on the request."""

    name: ReadOnly[str]
    description: ReadOnly[str]
    schema: ReadOnly[dict[str, Any]]


class Message(TypedDict, total=False):
    """A message on a span, as LLM Obs names its fields."""

    content: ReadOnly[str]
    role: ReadOnly[str]
    reasoning_content: ReadOnly[str]
    tool_calls: ReadOnly[Sequence[ToolCall]]
    tool_results: ReadOnly[Sequence[ToolResult]]


class InputMeta(TypedDict):
    messages: Sequence[
        Message | dict[str, Any]  # changed to fit with tool calls
    ]  # Relevant Issue: https://github.com/BerriAI/litellm/issues/9494


class OutputMeta(TypedDict):
    messages: Sequence[Any]


class DDLLMObsError(TypedDict, total=False):
    """Error information on the span according to DD LLM Obs API spec"""

    message: str  # The error message
    stack: str | None  # The stack trace
    type: str | None  # The error type


class Meta(TypedDict, total=False):
    # The span kind: "agent", "workflow", "llm", "tool", "task", "embedding", or "retrieval".
    kind: Literal["llm", "tool", "task", "embedding", "retrieval"]
    input: InputMeta  # The span's input information.
    output: OutputMeta  # The span's output information.
    metadata: dict[str, Any]
    error: DDLLMObsError | None  # Error information on the span
    tool_definitions: ReadOnly[Sequence[ToolDefinition]]  # The tools offered to the model on this request


class LLMMetrics(TypedDict, total=False):
    input_tokens: float
    output_tokens: float
    total_tokens: float
    time_to_first_token: float
    time_per_output_token: float
    total_cost: float
    cache_read_input_tokens: ReadOnly[float]
    cache_write_input_tokens: ReadOnly[float]
    non_cached_input_tokens: ReadOnly[float]
    reasoning_output_tokens: ReadOnly[float]


class LLMObsPayload(TypedDict, total=False):
    parent_id: str
    trace_id: str
    apm_id: str
    span_id: str
    name: str
    meta: Meta
    start_ns: int
    duration: int
    metrics: LLMMetrics
    tags: list
    status: Literal["ok", "error"]  # Error status ("ok" or "error"). Defaults to "ok".


class DDSpanAttributes(TypedDict):
    ml_app: str
    tags: list[str]
    spans: list[LLMObsPayload]


class DDIntakePayload(TypedDict):
    type: str
    attributes: DDSpanAttributes


class DatadogLLMObsInitParams(StandardCustomLoggerInitParams):
    """
    Params for initializing a DatadogLLMObs logger on litellm
    """


class DDLLMObsLatencyMetrics(TypedDict, total=False):
    time_to_first_token_ms: float
    litellm_overhead_time_ms: float
    guardrail_overhead_time_ms: float


class DDLLMObsSpendMetrics(TypedDict, total=False):
    response_cost: float
    user_api_key_spend: float
    user_api_key_max_budget: float
    user_api_key_budget_reset_at: str
