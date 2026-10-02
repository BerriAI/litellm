import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

ChatRole: TypeAlias = Literal["system", "user", "assistant", "tool"]

MESSAGE_ROLES: Final[Mapping[str, ChatRole]] = MappingProxyType(
    {"human": "user", "user": "user", "ai": "assistant", "assistant": "assistant", "system": "system", "tool": "tool"}
)


class _ContentBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    type: str = ""
    text: str | None = None
    content: JsonValue = None
    result: JsonValue = None
    name: JsonValue = None
    arguments: JsonValue = None
    args: JsonValue = None
    input: JsonValue = None


@dataclass(frozen=True, slots=True)
class ContentToolCall:
    name: str
    arguments: JsonValue


_CONTENT_BLOCKS: Final = TypeAdapter(tuple[_ContentBlock, ...])
_NON_TEXT_BLOCKS: Final = frozenset(
    {"reasoning", "thinking", "redacted_thinking", "function_call", "tool_use", "tool_call"}
)
_TOOL_BLOCKS: Final = frozenset({"function_call", "tool_use", "tool_call"})


def _content_blocks(content: object) -> tuple[_ContentBlock, ...] | None:
    try:
        return _CONTENT_BLOCKS.validate_python(content)
    except ValidationError:
        return None


def _block_text(block: _ContentBlock) -> str | None:
    if block.type in _NON_TEXT_BLOCKS:
        return None
    if block.text is not None:
        return block.text
    if block.type == "text" and isinstance(block.content, str):
        return block.content
    if block.type == "tool_call_response" and "result" in block.model_fields_set:
        return block.result if isinstance(block.result, str) else json.dumps(block.result)
    return None


def _tool_arguments(block: _ContentBlock) -> JsonValue:
    if block.type == "tool_use":
        return block.input
    return block.arguments if block.args is None else block.args


def _is_tool_call(block: _ContentBlock) -> bool:
    if not isinstance(block.name, str) or not block.name:
        return False
    if block.type == "tool_use":
        return "input" in block.model_fields_set
    if block.type == "function_call":
        return "arguments" in block.model_fields_set
    if block.type == "tool_call":
        return "args" in block.model_fields_set or "arguments" in block.model_fields_set
    return False


def _hidden_block(block: _ContentBlock) -> bool:
    return _is_tool_call(block) if block.type in _TOOL_BLOCKS else block.type in _NON_TEXT_BLOCKS


def content_tool_calls(content: object) -> tuple[ContentToolCall, ...]:
    blocks: Final = _content_blocks(content)
    if blocks is None:
        return ()
    return tuple(
        ContentToolCall(block.name, _tool_arguments(block))
        for block in blocks
        if isinstance(block.name, str) and _is_tool_call(block)
    )


def content_text(content: object) -> str:
    """Message content as display text: Responses-style block lists keep only their text blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    blocks: Final = _content_blocks(content)
    if blocks is None:
        return json.dumps(content)
    texts: Final = tuple(_block_text(block) for block in blocks)
    if not all(text is not None or _hidden_block(block) for block, text in zip(blocks, texts, strict=True)):
        return json.dumps(content)
    return "\n\n".join(text for text in texts if text is not None)
