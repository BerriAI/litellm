"""
Payloads for Datadog LLM Observability Service (LLMObs)

API Reference: https://docs.datadoghq.com/llm_observability/setup/api/?tab=example#api-standards
"""
from typing import Any, Dict, List, Literal, Optional, TypedDict

from litellm.types.integrations.custom_logger import StandardCustomLoggerInitParams


class InputMeta(TypedDict):
    messages: List[
        Dict[str, str]
    ]  # Relevant Issue: https://github.com/BerriAI/litellm/issues/9494


class OutputMeta(TypedDict):
    messages: List[Any]


class DDLLMObsError(TypedDict, total=False):
    """Error information on the span according to DD LLM Obs API spec"""
    message: str  # The error message
    stack: Optional[str]  # The stack trace
    type: Optional[str]  # The error type


class Meta(TypedDict, total=False):
    # The span kind: "agent", "workflow", "llm", "tool", "task", "embedding", or "retrieval".
    kind: Literal["llm", "tool", "task", "embedding", "retrieval"]
    input: InputMeta  # The span's input information.
    output: OutputMeta  # The span's output information.
    metadata: Dict[str, Any]
    error: Optional[DDLLMObsError]  # Error information on the span


class LLMMetrics(TypedDict, total=False):
    input_tokens: float
    output_tokens: float
    total_tokens: float
    time_to_first_token: float
    time_per_output_token: float
    total_cost: float


class LLMObsPayload(TypedDict, total=False):
    parent_id: str
    trace_id: str
    span_id: str
    name: str
    meta: Meta
    start_ns: int
    duration: int
    metrics: LLMMetrics
    tags: List
    status: Literal["ok", "error"] # Error status ("ok" or "error"). Defaults to "ok".


class DDSpanAttributes(TypedDict):
    ml_app: str
    tags: List[str]
    spans: List[LLMObsPayload]


class DDIntakePayload(TypedDict):
    type: str
    attributes: DDSpanAttributes


class DatadogLLMObsInitParams(StandardCustomLoggerInitParams):
    """
    Params for initializing a DatadogLLMObs logger on litellm
    """
    pass


class DDLLMObsLatencyMetrics(TypedDict, total=False):
    time_to_first_token_ms: float
    litellm_overhead_time_ms: float
    guardrail_overhead_time_ms: float