"""Parse an Anthropic Messages API response (the ``/v1/messages`` route) into what OpenTelemetry records.

Chat completions carry ``choices`` and the Responses API carries ``output``; an Anthropic Messages
response carries a top-level ``content`` list of blocks (text / thinking / tool_use) and a
``stop_reason``, so it needs its own reading.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import ConfigDict, TypeAdapter

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.types.utils import ChatCompletionMessageToolCall, Function

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_ITEMS: Final = TypeAdapter(list[object], config=ConfigDict(hide_input_in_errors=True))


@dataclass(frozen=True, slots=True)
class AnthropicMessageOutput:
    role: str
    parts: tuple[Mapping[str, object], ...]
    tool_calls: tuple[ChatCompletionMessageToolCall, ...]
    stop_reason: str | None


def _part(block: Mapping[str, object]) -> Mapping[str, object] | None:
    block_type: Final = block.get("type")
    if block_type == "text":
        return {"type": "text", "content": block.get("text", "")}
    if block_type == "thinking":
        return {"type": "text", "content": block.get("thinking", "")}
    if block_type == "tool_use":
        return {
            "type": "tool_call",
            "id": block.get("id", ""),
            "name": block.get("name", ""),
            "arguments": block.get("input") or {},
        }
    return None


def _tool_call(block: Mapping[str, object]) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        function=Function(arguments=safe_dumps(block.get("input") or {}), name=str(block.get("name", ""))),
        id=str(block.get("id", "")),
        type="function",
    )


def parse_anthropic_message(response_obj: object) -> AnthropicMessageOutput | None:
    """The output side of an Anthropic Messages response, or None when ``response_obj`` is not one.

    A response with no recognised blocks still comes back (empty ``parts``), so its ``stop_reason``
    is recorded; blocks of a type LiteLLM does not know are skipped rather than emitted blank.
    """
    if not isinstance(response_obj, Mapping):
        return None
    message: Final = _JSON_OBJECT.validate_python(response_obj)
    content: Final = message.get("content")
    if not isinstance(content, list) or not content:
        return None
    blocks: Final = [
        _JSON_OBJECT.validate_python(block)
        for block in _JSON_ITEMS.validate_python(content)
        if isinstance(block, Mapping)
    ]
    stop_reason: Final = message.get("stop_reason")
    return AnthropicMessageOutput(
        role=str(message.get("role") or "assistant"),
        parts=tuple(part for block in blocks if (part := _part(block)) is not None),
        tool_calls=tuple(_tool_call(block) for block in blocks if block.get("type") == "tool_use"),
        stop_reason=stop_reason if isinstance(stop_reason, str) and stop_reason else None,
    )
