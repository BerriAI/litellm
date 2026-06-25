"""Langfuse OTLP attribute mapper.

Langfuse ingests OTLP spans and reads from its own vendor namespace
(``langfuse.observation.*``, ``langfuse.trace.*``). Compose this mapper after
``GenAIMapper`` to send canonical + Langfuse-flavored spans simultaneously.

Every attribute is declared as a ``key -> extractor`` table entry (one callable
per mapping operation): ``_LLM_CALL_ATTRS`` for scalars and ``_BLOB_ATTRS`` for
the JSON-serialized payloads. ``_llm_call`` just applies both tables.
"""

import json
from typing import Callable

from litellm.integrations.otel.mappers.base import AttributeMap, AttrValue, SpanData
from litellm.integrations.otel.mappers.utils import (
    collect,
    json_if,
    output_messages,
    serialize_messages,
)
from litellm.integrations.otel.model.payloads import (
    LLMCallSpanData,
    LLMRequestParams,
    LLMUsage,
)


class LangfuseMapper:
    _LLM_CALL_ATTRS: dict[str, Callable[[LLMCallSpanData], AttrValue | None]] = {
        "langfuse.observation.type": lambda d: "generation",
        "langfuse.observation.model.name": lambda d: d.request_model or None,
        "langfuse.observation.metadata.provider": lambda d: d.provider or None,
        "langfuse.observation.id": lambda d: d.identity.call_id or None,
        "langfuse.trace.metadata.team_id": lambda d: d.identity.team_id or None,
        "langfuse.trace.metadata.team_alias": lambda d: d.identity.team_alias or None,
    }

    # Sub-tables folded into their respective JSON blobs.
    _MODEL_PARAMS: dict[str, Callable[[LLMRequestParams], AttrValue | None]] = {
        "temperature": lambda rp: rp.temperature,
        "top_p": lambda rp: rp.top_p,
        "max_tokens": lambda rp: rp.max_tokens,
        "frequency_penalty": lambda rp: rp.frequency_penalty,
        "presence_penalty": lambda rp: rp.presence_penalty,
        "seed": lambda rp: rp.seed,
    }
    _USAGE_FIELDS: dict[str, Callable[[LLMUsage], AttrValue | None]] = {
        "input": lambda u: u.input_tokens,
        "output": lambda u: u.output_tokens,
        "total": lambda u: u.total_tokens,
    }

    # JSON-payload attributes: each builder returns the serialized blob or None.
    _BLOB_ATTRS: dict[str, Callable[[LLMCallSpanData], AttrValue | None]] = {
        "langfuse.observation.model.parameters": lambda d: json_if(
            collect(LangfuseMapper._MODEL_PARAMS, d.request_params)
        ),
        "langfuse.observation.input": lambda d: serialize_messages(d.messages_in),
        "langfuse.observation.output": lambda d: serialize_messages(output_messages(d)),
        "langfuse.observation.usage_details": lambda d: json_if(
            collect(LangfuseMapper._USAGE_FIELDS, d.usage)
        ),
        "langfuse.observation.cost_details": lambda d: (
            json.dumps({"total": d.response_cost})
            if d.response_cost is not None
            else None
        ),
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
