"""Dual-emit mapper: deprecated attribute keys kept for backward compatibility.

These keys are intentionally NOT in ``semconv.py`` — that module holds only the
canonical keys. This is the single place legacy/deprecated strings live, so they
can be deleted wholesale once the deprecation window closes.
"""

from typing import Final

from litellm.integrations.otel.mappers.base import AttributeMap
from litellm.integrations.otel.payloads import LLMCallSpanData

# Deprecated keys (semconv-ai / Traceloop era).
_LEGACY_SYSTEM: Final = "gen_ai.system"
_LEGACY_PROMPT_TOKENS: Final = "gen_ai.usage.prompt_tokens"
_LEGACY_COMPLETION_TOKENS: Final = "gen_ai.usage.completion_tokens"
_LEGACY_TOTAL_TOKENS: Final = "gen_ai.usage.total_tokens"
_LEGACY_IS_STREAMING: Final = "llm.is_streaming"
_LEGACY_TOP_K: Final = "llm.top_k"
_LEGACY_FREQUENCY_PENALTY: Final = "llm.frequency_penalty"
_LEGACY_PRESENCE_PENALTY: Final = "llm.presence_penalty"
_LEGACY_STOP_SEQUENCES: Final = "llm.chat.stop_sequences"


class LegacyMapper:
    """Re-emits canonical values under their deprecated key names."""

    def map_llm_call(self, data: LLMCallSpanData) -> AttributeMap:
        attrs: AttributeMap = {}
        if data.provider:
            attrs[_LEGACY_SYSTEM] = data.provider
        if data.usage.input_tokens is not None:
            attrs[_LEGACY_PROMPT_TOKENS] = data.usage.input_tokens
        if data.usage.output_tokens is not None:
            attrs[_LEGACY_COMPLETION_TOKENS] = data.usage.output_tokens
        if data.usage.total_tokens is not None:
            attrs[_LEGACY_TOTAL_TOKENS] = data.usage.total_tokens
        if data.is_streaming is not None:
            attrs[_LEGACY_IS_STREAMING] = data.is_streaming

        rp = data.request_params
        if rp.top_k is not None:
            attrs[_LEGACY_TOP_K] = rp.top_k
        if rp.frequency_penalty is not None:
            attrs[_LEGACY_FREQUENCY_PENALTY] = rp.frequency_penalty
        if rp.presence_penalty is not None:
            attrs[_LEGACY_PRESENCE_PENALTY] = rp.presence_penalty
        if rp.stop_sequences:
            attrs[_LEGACY_STOP_SEQUENCES] = list(rp.stop_sequences)

        return attrs
