from collections import Counter
from collections.abc import Iterator
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.telemetry.records import BlockCounts, BlockType

_PART_TYPES: Final = {
    "text": BlockType.TEXT,
    "input_text": BlockType.TEXT,
    "image": BlockType.IMAGE,
    "image_url": BlockType.IMAGE,
    "input_image": BlockType.IMAGE,
    "input_audio": BlockType.AUDIO,
    "audio": BlockType.AUDIO,
    "file": BlockType.FILE,
    "document": BlockType.FILE,
    "input_file": BlockType.FILE,
    "tool_use": BlockType.TOOL_USE,
    "tool_result": BlockType.TOOL_RESULT,
    "thinking": BlockType.THINKING,
    "redacted_thinking": BlockType.THINKING,
}


class _Part(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    type: str


class _Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    role: str
    content: str | tuple[_Part, ...] | None = None
    tool_calls: tuple[object, ...] | None = None


_MESSAGES: Final = TypeAdapter(tuple[_Message, ...])


def _message_blocks(message: _Message) -> Iterator[BlockType]:
    match message.content:
        case str() if message.role == "tool":
            yield BlockType.TOOL_RESULT
        case str():
            yield BlockType.TEXT
        case tuple() as parts:
            yield from (_PART_TYPES.get(part.type, BlockType.OTHER) for part in parts)
        case None:
            pass
    yield from (BlockType.TOOL_USE for _ in message.tool_calls or ())


def _all_blocks(messages: tuple[_Message, ...]) -> Iterator[BlockType]:
    for message in messages:
        yield from _message_blocks(message)


def count_blocks(messages: object) -> BlockCounts | None:
    """Content block counts for OpenAI or Anthropic style ``messages``, or None when they are not messages"""
    try:
        parsed: Final = _MESSAGES.validate_python(messages)
    except ValidationError:
        return None
    counts: Final = Counter(_all_blocks(parsed))
    return BlockCounts(total=counts.total(), by_type=tuple(sorted(counts.items())))
