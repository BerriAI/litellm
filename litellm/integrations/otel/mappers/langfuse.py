"""Langfuse OTLP attribute mapper.

Langfuse ingests OTLP spans and reads from its own vendor namespace
(``langfuse.observation.*``, ``langfuse.trace.*``). Compose this mapper after
``GenAIMapper`` to send canonical + Langfuse-flavored spans simultaneously.

Every attribute is declared as a ``key -> extractor`` table entry (one callable
per mapping operation): ``_LLM_CALL_ATTRS`` for scalars, ``_TRACE_ATTRS`` for the
caller's trace controls (shared with the root observation), and ``_BLOB_ATTRS`` for
the JSON-serialized payloads. ``_llm_call`` just applies the three tables.
"""

import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

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
from litellm.integrations.otel.model.trace_controls import TraceControls

LANGFUSE_OBSERVATION_INPUT: Final = "langfuse.observation.input"
LANGFUSE_OBSERVATION_OUTPUT: Final = "langfuse.observation.output"
LANGFUSE_TRACE_NAME: Final = "langfuse.trace.name"
LANGFUSE_TRACE_USER_ID: Final = "user.id"
LANGFUSE_TRACE_SESSION_ID: Final = "session.id"
LANGFUSE_TRACE_TAGS: Final = "langfuse.trace.tags"


class LangfuseMapper:
    _LLM_CALL_ATTRS: dict[str, Callable[[LLMCallSpanData], AttrValue | None]] = {
        "langfuse.observation.type": lambda d: "generation",
        "langfuse.observation.model.name": lambda d: d.request_model or None,
        "langfuse.observation.metadata.provider": lambda d: d.provider or None,
        "langfuse.observation.id": lambda d: d.identity.call_id or None,
        "langfuse.trace.metadata.team_id": lambda d: d.identity.team_id or None,
        "langfuse.trace.metadata.team_alias": lambda d: d.identity.team_alias or None,
    }

    _TRACE_ATTRS: Mapping[str, Callable[[TraceControls], AttrValue | None]] = MappingProxyType(
        {
            LANGFUSE_TRACE_NAME: lambda t: t.name or None,
            LANGFUSE_TRACE_USER_ID: lambda t: t.user_id or None,
            LANGFUSE_TRACE_SESSION_ID: lambda t: t.session_id or None,
            LANGFUSE_TRACE_TAGS: lambda t: t.tags or None,
        }
    )

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
        LANGFUSE_OBSERVATION_INPUT: lambda d: serialize_messages(d.messages_in),
        LANGFUSE_OBSERVATION_OUTPUT: lambda d: serialize_messages(output_messages(d)),
        "langfuse.observation.usage_details": lambda d: json_if(collect(LangfuseMapper._USAGE_FIELDS, d.usage)),
        "langfuse.observation.cost_details": lambda d: (
            json.dumps({"total": d.response_cost}) if d.response_cost is not None else None
        ),
    }

    def map(self, data: SpanData) -> AttributeMap:
        match data:
            case LLMCallSpanData():
                return self._llm_call(data)
            case _:
                return {}

    @classmethod
    def trace_attributes(cls, trace: TraceControls) -> AttributeMap:
        return collect(cls._TRACE_ATTRS, trace)

    @classmethod
    def _llm_call(cls, data: LLMCallSpanData) -> AttributeMap:
        return {
            **collect(cls._LLM_CALL_ATTRS, data),
            **cls.trace_attributes(data.trace),
            **collect(cls._BLOB_ATTRS, data),
        }
