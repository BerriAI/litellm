"""OpenInference attribute mapper (Arize + Arize-Phoenix shared vocabulary).

Spec: https://github.com/Arize-ai/openinference/tree/main/spec — the standard
both Arize and Phoenix consume. Composing this mapper after ``GenAIMapper``
gives the same span both vocabularies, so a single trace lights up Arize +
Phoenix + any other OpenInference-aware backend simultaneously.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from itertools import accumulate, chain, groupby
from types import MappingProxyType
from typing import Final

from litellm.integrations.otel.mappers.base import AttributeMap, AttrValue, SpanData
from litellm.integrations.otel.mappers.utils import (
    MAX_TOOL_DEFINITION_ATTRS_PER_SPAN,
    collect,
    drop_none,
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

_INPUT_MESSAGES: Final = "llm.input_messages"
_OUTPUT_MESSAGES: Final = "llm.output_messages"
_MESSAGE_FAMILIES: Final = (_INPUT_MESSAGES, _OUTPUT_MESSAGES)


def _message_key_groups(attrs: Mapping[str, AttrValue]) -> Mapping[tuple[str, int], tuple[str, ...]]:
    """Per-index message keys in ``attrs`` grouped by ``(family, index)``."""
    tagged: Final = sorted(
        (family, int(key.split(".")[2]), key)
        for key in attrs
        for family in _MESSAGE_FAMILIES
        if key.startswith(f"{family}.")
    )
    return MappingProxyType(
        {group: tuple(key for _, _, key in keys) for group, keys in groupby(tagged, key=lambda tag: tag[:2])}
    )


def _shed_order(groups: Mapping[tuple[str, int], tuple[str, ...]]) -> tuple[tuple[str, int], ...]:
    """Message groups least valuable first: middle prompt turns, extra choices, then the opener, the newest turn
    and the first choice."""
    inputs: Final = sorted(idx for family, idx in groups if family == _INPUT_MESSAGES)
    outputs: Final = sorted(idx for family, idx in groups if family == _OUTPUT_MESSAGES)
    pinned_inputs: Final = tuple(dict.fromkeys((*inputs[:1], *inputs[-1:])))
    return (
        *((_INPUT_MESSAGES, idx) for idx in inputs[1:-1]),
        *((_OUTPUT_MESSAGES, idx) for idx in reversed(outputs[1:])),
        *((_INPUT_MESSAGES, idx) for idx in pinned_inputs),
        *((_OUTPUT_MESSAGES, idx) for idx in outputs[:1]),
    )


def fit_indexed_messages(attrs: Mapping[str, AttrValue], budget: int | None) -> Mapping[str, AttrValue]:
    """``attrs`` with whole per-index messages shed, least valuable first, until at most ``budget`` keys remain.

    ``None`` means the span has no attribute count limit. Every message still rides the ``input.value`` and
    ``output.value`` blobs, so shedding a per-index pair loses no content.
    """
    if budget is None or len(attrs) <= budget:
        return attrs
    groups: Final = _message_key_groups(attrs)
    order: Final = _shed_order(groups)
    running: Final = tuple(accumulate(len(groups[group]) for group in order))
    excess: Final = len(attrs) - budget
    shed_count: Final = next((n + 1 for n, total in enumerate(running) if total >= excess), len(order))
    shed: Final = frozenset(chain.from_iterable(groups[group] for group in order[:shed_count]))
    return MappingProxyType({key: value for key, value in attrs.items() if key not in shed})


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
            **self._messages(_INPUT_MESSAGES, "input.value", data.messages_in),
            **self._messages(_OUTPUT_MESSAGES, "output.value", output_messages(data)),
            **self._tools(data),
        }

    @staticmethod
    def _messages(prefix: str, value_key: str, messages: Sequence[object]) -> AttributeMap:
        """``{prefix}.{idx}.message.*`` keys for every message + the ``value_key`` blob of all of them."""
        parsed: Final = [(m.get("role") if isinstance(m, dict) else None, message_content(m)) for m in messages]
        attrs: Final = drop_none(
            {
                key: value
                for idx, (role, content) in enumerate(parsed)
                for key, value in (
                    (f"{prefix}.{idx}.message.role", role if isinstance(role, str) else None),
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
