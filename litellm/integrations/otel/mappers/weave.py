"""Weave (W&B) attribute mapper.

Weave consumes OpenInference + a small set of Weave-specific keys (display
name, thread id, output value). This mapper layers the latter on top of
OpenInference's vocabulary — compose ``["genai", "openinference", "weave"]``
to feed a Weave backend.
"""

from typing import Callable

from litellm.integrations.otel.mappers.base import AttributeMap, AttrValue, SpanData
from litellm.integrations.otel.mappers.utils import collect, json_or_none
from litellm.integrations.otel.model.payloads import LLMCallSpanData


class WeaveMapper:
    """Maps ``LLMCallSpanData`` to Weave's vendor attributes."""

    _LLM_CALL_ATTRS: dict[str, Callable[[LLMCallSpanData], AttrValue | None]] = {
        # ``display_name`` has the form ``"{operation} {model}"``. The span
        # name already covers that, but Weave reads this attribute too.
        "weave.display_name": lambda d: f"{d.operation.value} {d.request_model}" if d.request_model else None,
        "weave.call_id": lambda d: d.identity.call_id or None,
    }

    # JSON-payload attributes: each builder returns the serialized blob or None.
    _BLOB_ATTRS: dict[str, Callable[[LLMCallSpanData], AttrValue | None]] = {
        # Weave treats the response choices as the "output" payload.
        "weave.output": lambda d: json_or_none(list(d.choices_out)) if d.choices_out else None,
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
