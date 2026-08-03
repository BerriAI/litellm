"""OpenInference attribute mapper (Arize + Arize-Phoenix shared vocabulary).

Spec: https://github.com/Arize-ai/openinference/tree/main/spec — the standard
both Arize and Phoenix consume. Composing this mapper after ``GenAIMapper``
gives the same span both vocabularies, so a single trace lights up Arize +
Phoenix + any other OpenInference-aware backend simultaneously.
"""

import json
from typing import Callable, Sequence

from litellm.integrations.otel.mappers.base import AttributeMap, AttrValue, SpanData
from litellm.integrations.otel.mappers.utils import (
    collect,
    drop_none,
    MAX_TOOL_DEFINITION_ATTRS_PER_SPAN,
    json_if,
    message_content,
    output_messages,
    tool_definition_attrs,
)
from litellm.integrations.otel.model.payloads import (
    LLMCallSpanData,
    LLMRequestParams,
    ToolDefinition,
)


class OpenInferenceMapper:
    """Emits OpenInference attributes for LLM_CALL spans.

    Key families (per the OpenInference spec):
    - ``openinference.span.kind`` — discriminator (``"LLM"`` here)
    - ``llm.model_name`` / ``llm.provider`` / ``llm.invocation_parameters``
    - ``llm.input_messages.{i}.message.role`` / ``...content``
    - ``llm.output_messages.{i}.message.role`` / ``...content``
    - ``llm.token_count.prompt`` / ``...completion`` / ``...total``
    - ``input.value`` / ``output.value`` — JSON-serialized request / response
    """

    _LLM_CALL_ATTRS: dict[str, Callable[[LLMCallSpanData], AttrValue | None]] = {
        "openinference.span.kind": lambda d: "LLM",
        "llm.model_name": lambda d: d.request_model or None,
        "llm.provider": lambda d: d.provider or None,
        "llm.token_count.prompt": lambda d: d.usage.input_tokens,
        "llm.token_count.completion": lambda d: d.usage.output_tokens,
        "llm.token_count.total": lambda d: d.usage.total_tokens,
    }

    # Folded into the ``llm.invocation_parameters`` JSON blob.
    _INVOCATION_PARAMS: dict[str, Callable[[LLMRequestParams], AttrValue | None]] = {
        "temperature": lambda rp: rp.temperature,
        "top_p": lambda rp: rp.top_p,
        "top_k": lambda rp: rp.top_k,
        "max_tokens": lambda rp: rp.max_tokens,
        "frequency_penalty": lambda rp: rp.frequency_penalty,
        "presence_penalty": lambda rp: rp.presence_penalty,
        "seed": lambda rp: rp.seed,
    }

    # Per-tool extractors, keyed by the ``llm.tools.{idx}.*`` suffix.
    _TOOL_ATTRS: dict[str, Callable[[ToolDefinition], AttrValue | None]] = {
        "tool.name": lambda t: t.name,
        "tool.description": lambda t: t.description or None,
        "tool.json_schema": lambda t: t.parameters_json or None,
    }

    # JSON-payload attributes: each builder returns the serialized blob or None.
    _BLOB_ATTRS: dict[str, Callable[[LLMCallSpanData], AttrValue | None]] = {
        "llm.invocation_parameters": lambda d: json_if(
            collect(OpenInferenceMapper._INVOCATION_PARAMS, d.request_params)
        ),
    }

    def __init__(self, tool_attr_budget: int = MAX_TOOL_DEFINITION_ATTRS_PER_SPAN) -> None:
        self._tool_attr_budget = tool_attr_budget

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
            **self._messages("llm.input_messages", "input.value", data.messages_in),
            **self._messages("llm.output_messages", "output.value", output_messages(data)),
            **self._tools(data),
        }

    @staticmethod
    def _messages(prefix: str, value_key: str, messages: Sequence[object]) -> AttributeMap:
        """Per-message ``{prefix}.{idx}.message.*`` keys + the ``value_key`` blob."""
        parsed = [(m.get("role") if isinstance(m, dict) else None, message_content(m)) for m in messages]
        attrs = drop_none(
            {
                key: value
                for idx, (role, content) in enumerate(parsed)
                for key, value in (
                    (
                        f"{prefix}.{idx}.message.role",
                        role if isinstance(role, str) else None,
                    ),
                    (f"{prefix}.{idx}.message.content", content),
                )
            }
        )
        if parsed:
            attrs[value_key] = json.dumps([{"role": role, "content": content} for role, content in parsed])
        return attrs

    def _tools(self, data: LLMCallSpanData) -> AttributeMap:
        return tool_definition_attrs(
            lambda idx, suffix: f"llm.tools.{idx}.{suffix}",
            data.tools,
            self._TOOL_ATTRS,
            self._tool_attr_budget,
        )
