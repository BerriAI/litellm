"""
Payloads for Datadog LLM Observability Service (LLMObs)

API Reference: https://docs.datadoghq.com/llm_observability/setup/api/?tab=example#api-standards
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict


class InputMeta(TypedDict):
    messages: List[Any]


class OutputMeta(TypedDict):
    messages: List[Any]


class Meta(TypedDict):
    # The span kind: "agent", "workflow", "llm", "tool", "task", "embedding", or "retrieval".
    kind: Literal["llm", "tool", "task", "embedding", "retrieval"]
    input: InputMeta  # The span’s input information.
    output: OutputMeta  # The span’s output information.
    metadata: Dict[str, Any]


class LLMMetrics(TypedDict, total=False):
    input_tokens: float
    output_tokens: float
    total_tokens: float
    time_to_first_token: float
    time_per_output_token: float


class LLMObsPayload(TypedDict):
    parent_id: str
    trace_id: str
    span_id: str
    name: str
    meta: Meta
    start_ns: int
    duration: int
    metrics: LLMMetrics
    tags: List


class DDSpanAttributes(TypedDict):
    ml_app: str
    tags: List[str]
    spans: List[LLMObsPayload]


class DDIntakePayload(TypedDict):
    type: str
    attributes: DDSpanAttributes
