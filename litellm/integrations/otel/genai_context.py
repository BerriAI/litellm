"""OpenTelemetry GenAI client context suppression & attribute propagation.

Integrates ``OpenTelemetryV2`` with ``opentelemetry-util-genai``'s context-sharing
contract (see ``open-telemetry/semantic-conventions-genai#682`` and
``open-telemetry/opentelemetry-python-genai#859``):

When LiteLLM is called inside an outer GenAI instrumentation that attached
``InferenceData``, ``EmbeddingData``, ``RetrievalData``, or
``FetchResponseData`` under ``CLIENT_<OP>_CONTEXT_KEY`` in
``opentelemetry.context``, LiteLLM:
- Suppresses its duplicate client span, ``gen_ai.client.*`` metrics, and
  operation-exception log events.
- Publishes its mapped span attributes and ``metric_attributes`` into the
  active context data container.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, cast

from opentelemetry.context import Context, get_value
from opentelemetry.util.genai._invocation import _ContextData
from opentelemetry.util.genai.invocation import (
    CLIENT_EMBEDDING_CONTEXT_KEY,
    CLIENT_FETCH_RESPONSE_CONTEXT_KEY,
    CLIENT_INFERENCE_CONTEXT_KEY,
    CLIENT_RETRIEVAL_CONTEXT_KEY,
    EmbeddingData,
    FetchResponseData,
    InferenceData,
    RetrievalData,
)
from opentelemetry.util.types import AttributeValue

from litellm.integrations.otel.model.payloads import LLMCallSpanData
from litellm.integrations.otel.model.semconv import Error, GenAIOperation

__all__ = [
    "CLIENT_EMBEDDING_CONTEXT_KEY",
    "CLIENT_FETCH_RESPONSE_CONTEXT_KEY",
    "CLIENT_INFERENCE_CONTEXT_KEY",
    "CLIENT_RETRIEVAL_CONTEXT_KEY",
    "EmbeddingData",
    "FetchResponseData",
    "InferenceData",
    "RetrievalData",
    "context_spec_for_operation",
    "get_active_genai_context_data",
    "publish_to_genai_context",
]

_INFERENCE_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        GenAIOperation.CHAT.value,
        GenAIOperation.TEXT_COMPLETION.value,
        GenAIOperation.GENERATE_CONTENT.value,
    }
)

_FETCH_RESPONSE_CALL_TYPES: Final[frozenset[str]] = frozenset(
    {
        "get_responses",
        "aget_responses",
    }
)


def context_spec_for_operation(
    operation: GenAIOperation | str | None,
    call_type: str | None = None,
) -> tuple[str, type[_ContextData]] | None:
    """Return ``(context_key, dataclass_type)`` for a GenAI client operation, or ``None``."""
    op_str: Final = operation.value if isinstance(operation, GenAIOperation) else (operation or "")
    if call_type and call_type.lower() in _FETCH_RESPONSE_CALL_TYPES:
        return CLIENT_FETCH_RESPONSE_CONTEXT_KEY, FetchResponseData
    if op_str == "fetch_response":
        return CLIENT_FETCH_RESPONSE_CONTEXT_KEY, FetchResponseData
    if op_str in _INFERENCE_OPERATIONS:
        return CLIENT_INFERENCE_CONTEXT_KEY, InferenceData
    if op_str == GenAIOperation.EMBEDDINGS.value:
        return CLIENT_EMBEDDING_CONTEXT_KEY, EmbeddingData
    if op_str == GenAIOperation.RETRIEVAL.value:
        return CLIENT_RETRIEVAL_CONTEXT_KEY, RetrievalData
    return None


def get_active_genai_context_data(
    operation: GenAIOperation | str | None,
    *,
    call_type: str | None = None,
    context: Context | None = None,
) -> _ContextData | None:
    """Return the active outer ``<Op>Data`` from ``opentelemetry.context`` if present."""
    spec: Final = context_spec_for_operation(operation, call_type=call_type)
    if spec is None:
        return None
    context_key, data_cls = spec
    for candidate_ctx in (context, None):
        value = get_value(context_key, context=candidate_ctx)
        if isinstance(value, data_cls):
            return value
        if candidate_ctx is None:
            break
    return None


def publish_to_genai_context(
    ctx_data: _ContextData,
    data: LLMCallSpanData,
    span_attrs: Mapping[str, object],
    metric_attrs: Mapping[str, object] | None = None,
) -> None:
    """Publish LiteLLM's mapped span and metric attributes into an outer ``ctx_data``."""
    for key, val in span_attrs.items():
        if val is not None and isinstance(val, (str, bool, int, float, list, tuple)):
            ctx_data.attributes[key] = cast(AttributeValue, val)

    if data.error is not None:
        ctx_data.attributes[Error.TYPE] = data.error.error_type
        if data.error.message:
            ctx_data.attributes[Error.MESSAGE] = data.error.message

    if metric_attrs:
        for key, val in metric_attrs.items():
            if val is not None and isinstance(val, (str, bool, int, float)):
                ctx_data.metric_attributes[key] = val
