"""Shared helpers for the attribute mappers.

Small, mapper-agnostic utilities — JSON serialization, message extraction, and
extractor-table application — pulled out of the individual mapper modules so
they live in one place.
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from litellm.integrations.otel.mappers.base import AttributeMap, AttrValue
from litellm.integrations.otel.model.payloads import LLMCallSpanData, ToolDefinition


@dataclass(frozen=True, slots=True)
class MessageToolCall:
    id: str | None
    type: str
    name: str | None
    arguments: str | None

    def to_openai_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {"name": self.name, "arguments": self.arguments},
        }


DEFAULT_SPAN_ATTRIBUTE_LIMIT: Final = 128
"""The OTel SDK's default per-span attribute count limit."""

MAX_TOOL_DEFINITION_ATTRS_PER_SPAN: Final = DEFAULT_SPAN_ATTRIBUTE_LIMIT // 4
"""Span-wide ceiling on attributes spent spelling out declared tool definitions.

Tool definitions are an unbounded attribute family: one entry per declared
tool, per field, per active vocabulary. Agentic clients declare hundreds, which
overruns the span attribute limit. That limit evicts oldest-first, so an
uncapped family silently destroys the core ``gen_ai.*`` attributes written
before it.

The ceiling is span-wide rather than per-mapper because several vocabularies
can be active at once and each spells the same tools out under its own keys, so
a per-mapper allowance multiplies by the number of vocabularies and reaches the
limit again. Reserving a quarter of the span for tool detail leaves the rest to
core telemetry no matter how many vocabularies are configured.
"""


def tool_attr_budget(vocabularies: int) -> int:
    """Split the span-wide tool-definition ceiling across active vocabularies."""
    return MAX_TOOL_DEFINITION_ATTRS_PER_SPAN // max(vocabularies, 1)


def drop_none(values: Mapping[str, AttrValue | None]) -> AttributeMap:
    """Return ``values`` with ``None``-valued entries removed."""
    return drop_none_pairs(values.items())


def drop_none_pairs(pairs: Iterable[tuple[str, AttrValue | None]]) -> AttributeMap:
    """Return ``pairs`` as a map with ``None``-valued entries removed."""
    return {k: v for k, v in pairs if v is not None}


def tool_definition_attrs(
    key_for: Callable[[int, str], str],
    tools: Sequence[ToolDefinition],
    extractors: Mapping[str, Callable[[ToolDefinition], AttrValue | None]],
    attr_budget: int,
) -> AttributeMap:
    """Per-index attributes for as many tools as ``attr_budget`` affords.

    ``key_for`` builds a vocabulary's key from the tool's index and the field
    name, so each mapper keeps its own naming while sharing the budget. One tool
    always keeps its detail, so the family stays legible even when many
    vocabularies split the ceiling.
    """
    max_tools: Final = max(attr_budget // max(len(extractors), 1), 1)
    return drop_none(
        {
            key_for(idx, suffix): extract(tool)
            for idx, tool in enumerate(tools[:max_tools])
            for suffix, extract in extractors.items()
        }
    )


def collect(table: Mapping[str, Callable], source: object) -> AttributeMap:
    """Apply an extractor table to ``source``, dropping ``None`` results."""
    return drop_none({key: extract(source) for key, extract in table.items()})


def json_if(payload: Mapping[str, object]) -> str | None:
    """JSON-serialize ``payload`` only when it's non-empty; else ``None``."""
    return json.dumps(payload) if payload else None


def json_or_none(value: object) -> str | None:
    """JSON-serialize ``value`` (falling back to ``str``); ``None`` on failure."""
    try:
        return json.dumps(value, default=str)
    except Exception:
        return None


def stringify_message(message: object) -> str | None:
    """JSON-serialize a chat message dict; ``None`` if not a dict or on failure."""
    if not isinstance(message, dict):
        return None
    try:
        return json.dumps(message, default=str)
    except Exception:
        return None


def serialize_messages(messages: Sequence[object]) -> str | None:
    """Round-trip a sequence of message dicts through ``stringify_message``."""
    serialized: Final = [json.loads(s) for s in (stringify_message(m) for m in messages) if s is not None]
    return json.dumps(serialized) if serialized else None


def message_content(message: object) -> str | None:
    """Extract the textual ``content`` from a chat message dict."""
    if not isinstance(message, dict):
        return None
    content: Final = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # multimodal: concatenate text parts only
        parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        return "".join(p for p in parts if isinstance(p, str)) or None
    return None


def output_messages(data: LLMCallSpanData) -> list:
    """The ``message`` payload of each response choice."""
    return [c.get("message") for c in data.choices_out if isinstance(c, dict)]


def message_tool_calls(message: object) -> tuple[MessageToolCall, ...]:
    if not isinstance(message, dict):
        return ()
    tool_calls: Final = message.get("tool_calls")
    if not isinstance(tool_calls, (list, tuple)) or not tool_calls:
        return ()
    return tuple(tool_call for value in tool_calls if (tool_call := _message_tool_call(value)) is not None)


def _message_tool_call(value: object) -> MessageToolCall | None:
    if not isinstance(value, dict):
        return None
    function: Final = value.get("function")
    function_data: Final = function if isinstance(function, dict) else {}
    raw_type: Final = value.get("type")
    raw_arguments: Final = function_data.get("arguments")
    arguments: Final = (
        raw_arguments
        if isinstance(raw_arguments, str)
        else json.dumps(raw_arguments, default=str)
        if raw_arguments is not None
        else None
    )
    name: Final = function_data.get("name")
    identifier: Final = value.get("id")
    return MessageToolCall(
        id=identifier if isinstance(identifier, str) else None,
        type=raw_type if isinstance(raw_type, str) else "function",
        name=name if isinstance(name, str) else None,
        arguments=arguments,
    )
