import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

ChatRole: TypeAlias = Literal["system", "user", "assistant", "tool"]

MESSAGE_ROLES: Final[Mapping[str, ChatRole]] = MappingProxyType(
    {"human": "user", "user": "user", "ai": "assistant", "assistant": "assistant", "system": "system", "tool": "tool"}
)


class _ContentBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    type: str = ""
    text: str | None = None


_CONTENT_BLOCKS: Final = TypeAdapter(tuple[_ContentBlock, ...])
_NON_TEXT_BLOCKS: Final = frozenset(
    {"reasoning", "thinking", "redacted_thinking", "function_call", "tool_use", "tool_call"}
)


def content_text(content: object) -> str:
    """Message content as display text: Responses-style block lists keep only their text blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        blocks: Final = _CONTENT_BLOCKS.validate_python(content)
    except ValidationError:
        return json.dumps(content)
    if not all(block.text is not None or block.type in _NON_TEXT_BLOCKS for block in blocks):
        return json.dumps(content)
    return "\n\n".join(block.text for block in blocks if block.text is not None)
