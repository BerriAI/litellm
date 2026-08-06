"""Mapper for the older semantic-convention attribute vocabulary.

Emits attributes under the semconv-ai / Traceloop key names (e.g.
``gen_ai.system``, ``gen_ai.usage.prompt_tokens``, ``llm.is_streaming``) plus a
few bare, unprefixed service keys (``service``, ``call_type``, ``error``), for
backends that consume those names.

Like ``GenAIMapper``, each span kind declares its schema as a flat
``attribute key -> extractor`` table: one lambda per mapping operation.
"""

from collections.abc import Callable
from typing import Final

from litellm.integrations.otel.mappers.base import AttributeMap, AttrValue, SpanData
from litellm.integrations.otel.mappers.utils import (
    MAX_TOOL_DEFINITION_ATTRS_PER_SPAN,
    collect,
    tool_definition_attrs,
)
from litellm.integrations.otel.model.payloads import (
    LLMCallSpanData,
    ServiceSpanData,
    ToolDefinition,
)

# Attribute keys in the semconv-ai / Traceloop vocabulary.
_LEGACY_SYSTEM: Final = "gen_ai.system"
_LEGACY_PROMPT_TOKENS: Final = "gen_ai.usage.prompt_tokens"
_LEGACY_COMPLETION_TOKENS: Final = "gen_ai.usage.completion_tokens"
_LEGACY_TOTAL_TOKENS: Final = "gen_ai.usage.total_tokens"
_LEGACY_IS_STREAMING: Final = "llm.is_streaming"
_LEGACY_TOP_K: Final = "llm.top_k"
_LEGACY_FREQUENCY_PENALTY: Final = "llm.frequency_penalty"
_LEGACY_PRESENCE_PENALTY: Final = "llm.presence_penalty"
_LEGACY_STOP_SEQUENCES: Final = "llm.chat.stop_sequences"
_LEGACY_SERVICE: Final = "service"
_LEGACY_CALL_TYPE: Final = "call_type"
_LEGACY_ERROR: Final = "error"


class LegacyMapper:
    """Emits LLM-call and service attributes under the older key names."""

    _LLM_CALL_ATTRS: dict[str, Callable[[LLMCallSpanData], AttrValue | None]] = {
        _LEGACY_SYSTEM: lambda d: d.provider or None,
        _LEGACY_PROMPT_TOKENS: lambda d: d.usage.input_tokens,
        _LEGACY_COMPLETION_TOKENS: lambda d: d.usage.output_tokens,
        _LEGACY_TOTAL_TOKENS: lambda d: d.usage.total_tokens,
        _LEGACY_IS_STREAMING: lambda d: d.is_streaming,
        _LEGACY_TOP_K: lambda d: d.request_params.top_k,
        _LEGACY_FREQUENCY_PENALTY: lambda d: d.request_params.frequency_penalty,
        _LEGACY_PRESENCE_PENALTY: lambda d: d.request_params.presence_penalty,
        _LEGACY_STOP_SEQUENCES: lambda d: (
            list(d.request_params.stop_sequences) if d.request_params.stop_sequences else None
        ),
    }

    _TOOL_ATTRS: dict[str, Callable[[ToolDefinition], AttrValue | None]] = {
        "name": lambda t: t.name,
        "description": lambda t: t.description or None,
        "parameters": lambda t: t.parameters_json or None,
    }

    _SERVICE_ATTRS: dict[str, Callable[[ServiceSpanData], AttrValue | None]] = {
        _LEGACY_SERVICE: lambda d: d.service_name,
        _LEGACY_CALL_TYPE: lambda d: d.call_type,
        _LEGACY_ERROR: lambda d: d.error.message if d.error is not None and d.error.message else None,
    }

    def __init__(self, tool_attr_budget: int = MAX_TOOL_DEFINITION_ATTRS_PER_SPAN) -> None:
        self._tool_attr_budget = tool_attr_budget

    def map(self, data: SpanData) -> AttributeMap:
        match data:
            case LLMCallSpanData():
                return self._llm_call(data)
            case ServiceSpanData():
                return self._service(data)
            case _:
                return {}

    def _llm_call(self, data: LLMCallSpanData) -> AttributeMap:
        attrs: Final = collect(self._LLM_CALL_ATTRS, data)
        attrs.update(
            tool_definition_attrs(
                lambda idx, suffix: f"llm.request.functions.{idx}.{suffix}",
                data.tools,
                self._TOOL_ATTRS,
                self._tool_attr_budget,
            )
        )
        return attrs

    @classmethod
    def _service(cls, data: ServiceSpanData) -> AttributeMap:
        attrs: Final = collect(cls._SERVICE_ATTRS, data)
        attrs.update(dict(data.event_metadata))
        return attrs
