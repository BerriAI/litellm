"""OpenInference attribute mapper (Arize + Arize-Phoenix shared vocabulary).

Spec: https://github.com/Arize-ai/openinference/tree/main/spec — the standard
both Arize and Phoenix consume. Composing this mapper after ``GenAIMapper``
gives the same span both vocabularies, so a single trace lights up Arize +
Phoenix + any other OpenInference-aware backend simultaneously.
"""

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import accumulate, chain, groupby
from types import MappingProxyType
from typing import Final

from litellm.integrations.otel.mappers.base import AttributeMap, AttrValue, SpanData
from litellm.integrations.otel.mappers.utils import (
    MAX_TOOL_DEFINITION_ATTRS_PER_SPAN,
    MessageToolCall,
    collect,
    drop_none_pairs,
    json_if,
    message_content,
    message_tool_calls,
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

_ParsedMessage = tuple[object, str | None, tuple[MessageToolCall, ...]]


def _parse_message(message: object) -> _ParsedMessage:
    role: Final = message.get("role") if isinstance(message, dict) else None
    return role, message_content(message), message_tool_calls(message)


def _tool_call_attribute_pairs(
    prefix: str, idx: int, tool_calls: tuple[MessageToolCall, ...]
) -> Iterator[tuple[str, str | None]]:
    for tool_idx, tool_call in enumerate(tool_calls):
        yield f"{prefix}.{idx}.message.tool_calls.{tool_idx}.tool_call.id", tool_call.id
        yield f"{prefix}.{idx}.message.tool_calls.{tool_idx}.tool_call.function.name", tool_call.name
        yield f"{prefix}.{idx}.message.tool_calls.{tool_idx}.tool_call.function.arguments", tool_call.arguments


def _message_attribute_pairs(
    prefix: str,
    messages: Sequence[_ParsedMessage],
    *,
    with_tool_call_attrs: bool,
) -> Iterator[tuple[str, str | None]]:
    for idx, (role, content, tool_calls) in enumerate(messages):
        yield f"{prefix}.{idx}.message.role", role if isinstance(role, str) else None
        yield f"{prefix}.{idx}.message.content", content
        if with_tool_call_attrs:
            yield from _tool_call_attribute_pairs(prefix, idx, tool_calls)


def _message_value(messages: Sequence[_ParsedMessage]) -> str:
    return json.dumps(
        [
            {
                "role": role,
                "content": content,
                **({"tool_calls": [tool_call.to_openai_dict() for tool_call in tool_calls]} if tool_calls else {}),
            }
            for role, content, tool_calls in messages
        ]
    )


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
    - ``llm.output_messages.{i}.message.tool_calls.{j}.tool_call.*``
    - ``llm.token_count.prompt`` / ``...completion`` / ``...total``
    - ``metadata`` — JSON object of allowlisted promoted request metadata
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
        "metadata": lambda d: json_if(dict(sorted(d.promoted_metadata.items()))),
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
            **self._messages(
                _INPUT_MESSAGES,
                "input.value",
                data.messages_in,
                with_tool_call_attrs=False,
            ),
            **self._messages(
                _OUTPUT_MESSAGES,
                "output.value",
                output_messages(data),
                with_tool_call_attrs=True,
            ),
            **self._tools(data),
        }

    @staticmethod
    def _messages(
        prefix: str,
        value_key: str,
        messages: Sequence[object],
        *,
        with_tool_call_attrs: bool,
    ) -> AttributeMap:
        """``{prefix}.{idx}.message.*`` keys for every message + the ``value_key`` blob of all of them."""
        parsed: Final = tuple(_parse_message(message) for message in messages)
        attrs: Final = drop_none_pairs(
            _message_attribute_pairs(prefix, parsed, with_tool_call_attrs=with_tool_call_attrs)
        )
        if parsed:
            attrs[value_key] = _message_value(parsed)
        return attrs

    def _tools(self, data: LLMCallSpanData) -> AttributeMap:
        return tool_definition_attrs(
            lambda idx, suffix: f"llm.tools.{idx}.{suffix}",
            data.tools,
            self._TOOL_ATTRS,
            self._tool_attr_budget,
        )
