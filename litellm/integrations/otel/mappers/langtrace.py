"""Langtrace attribute mapper.

Produces Langtrace's attribute vocabulary so a span can be ingested by a
Langtrace backend. Compose it alongside other mappers like any other
vocabulary.

Scalar attributes are declared as a flat ``key -> extractor`` table (one lambda
per mapping operation); the prompt/completion blobs are serialized as a tail.
"""

from typing import Callable

from litellm.integrations.otel.mappers.base import AttributeMap, AttrValue, SpanData
from litellm.integrations.otel.mappers.utils import (
    collect,
    json_or_none,
    output_messages,
)
from litellm.integrations.otel.model.payloads import LLMCallSpanData


class LangtraceMapper:
    _LLM_CALL_ATTRS: dict[str, Callable[[LLMCallSpanData], AttrValue | None]] = {
        "gen_ai.operation.name": lambda d: "chat",
        "langtrace.service.name": lambda d: d.provider or None,
        "llm.model": lambda d: d.request_model or None,
        "gen_ai.response.model": lambda d: d.response_model or None,
        "gen_ai.response_id": lambda d: d.response_id or None,
        "gen_ai.system_fingerprint": lambda d: d.system_fingerprint or None,
        "llm.temperature": lambda d: d.request_params.temperature,
        "llm.top_p": lambda d: d.request_params.top_p,
        "llm.top_k": lambda d: d.request_params.top_k,
        "llm.max_tokens": lambda d: d.request_params.max_tokens,
        "llm.frequency_penalty": lambda d: d.request_params.frequency_penalty,
        "llm.presence_penalty": lambda d: d.request_params.presence_penalty,
        "llm.stream": lambda d: d.is_streaming,
        "llm.token.counts.prompt": lambda d: d.usage.input_tokens,
        "llm.token.counts.completion": lambda d: d.usage.output_tokens,
        "llm.token.counts.total": lambda d: d.usage.total_tokens,
    }

    _BLOB_ATTRS: dict[str, Callable[[LLMCallSpanData], AttrValue | None]] = {
        "llm.prompts": lambda d: json_or_none(list(d.messages_in)) if d.messages_in else None,
        "llm.completions": lambda d: json_or_none(output_messages(d)) if d.choices_out else None,
    }

    def map(self, data: SpanData) -> AttributeMap:
        match data:
            case LLMCallSpanData():
                return self._llm_call(data)
            case _:
                return {}

    @classmethod
    def _llm_call(cls, data: LLMCallSpanData) -> AttributeMap:
        return {
            **collect(cls._LLM_CALL_ATTRS, data),
            **collect(cls._BLOB_ATTRS, data),
        }
