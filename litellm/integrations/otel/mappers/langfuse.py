"""Langfuse OTLP attribute mapper.

Langfuse ingests OTLP spans and reads from its own vendor namespace
(``langfuse.observation.*``, ``langfuse.trace.*``). Compose this mapper after
``GenAIMapper`` to send canonical + Langfuse-flavored spans simultaneously.

Every attribute is declared as a ``key -> extractor`` table entry (one callable
per mapping operation): ``_LLM_CALL_ATTRS`` for scalars and ``_BLOB_ATTRS`` for
the JSON-serialized payloads. ``_llm_call`` applies both tables, plus the
caller's allowlisted metadata (``langfuse.trace.metadata.<key>``), which is
keyed per deployment and so can't live in a class-level table.

The trace-level controls (``user.id``, ``session.id``, ``langfuse.trace.name``,
``langfuse.trace.tags``) ride the generation span rather than a separate trace
span: Langfuse derives a trace from whichever observation carries them, which is
why they are repeated on every observation of the request.
"""

import json
from typing import Callable, Iterable

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


TRACE_METADATA_PREFIX = "langfuse.trace.metadata."


class LangfuseMapper:
    def __init__(self, trace_metadata_keys: Iterable[str] = ()) -> None:
        self._trace_metadata_keys = frozenset(trace_metadata_keys)

    _LLM_CALL_ATTRS: dict[str, Callable[[LLMCallSpanData], AttrValue | None]] = {
        "langfuse.observation.type": lambda d: "generation",
        "langfuse.observation.model.name": lambda d: d.request_model or None,
        "langfuse.observation.metadata.provider": lambda d: d.provider or None,
        "langfuse.observation.id": lambda d: d.identity.call_id or None,
        "user.id": lambda d: d.annotations.user_id or d.identity.end_user or None,
        "session.id": lambda d: d.annotations.session_id or None,
        "langfuse.trace.name": lambda d: d.annotations.trace_name or None,
        "langfuse.trace.tags": lambda d: list(d.annotations.tags) or None,
        f"{TRACE_METADATA_PREFIX}team_id": lambda d: d.identity.team_id or None,
        f"{TRACE_METADATA_PREFIX}team_alias": lambda d: d.identity.team_alias or None,
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

    def _llm_call(self, data: LLMCallSpanData) -> AttributeMap:
        return {
            **collect(self._LLM_CALL_ATTRS, data),
            **collect(self._BLOB_ATTRS, data),
            **self._trace_metadata(data),
        }

    def _trace_metadata(self, data: LLMCallSpanData) -> AttributeMap:
        """The caller's metadata, restricted to the operator's allowlist.

        Empty unless a deployment allowlists keys, so a request can never push
        arbitrary metadata of its own into the backend.
        """
        return {
            f"{TRACE_METADATA_PREFIX}{key}": value
            for key, value in data.annotations.requester_metadata.items()
            if key in self._trace_metadata_keys
        }
