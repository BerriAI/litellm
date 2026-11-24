from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class TextBlock(_Frozen):
    type: Literal["text"]
    text: str


class ToolUseBlock(_Frozen):
    type: Literal["tool_use"]
    name: str


class OtherBlock(_Frozen):
    type: str | None = None


ContentBlock = Annotated[TextBlock | ToolUseBlock | OtherBlock, Field(union_mode="left_to_right")]


class AssistantMessage(_Frozen):
    content: tuple[ContentBlock, ...] = ()


class AssistantEvent(_Frozen):
    type: Literal["assistant"]
    message: AssistantMessage


class ResultEvent(_Frozen):
    type: Literal["result"]
    stop_reason: str | None = None
    num_turns: int | None = None
    result: str | None = None


class OtherEvent(_Frozen):
    type: str | None = None


StreamEvent = Annotated[AssistantEvent | ResultEvent | OtherEvent, Field(union_mode="left_to_right")]

_EVENTS: Final = TypeAdapter(tuple[StreamEvent, ...])


def _assistant_blocks(events: tuple[StreamEvent, ...]) -> Iterator[ContentBlock]:
    for event in events:
        if isinstance(event, AssistantEvent):
            yield from event.message.content


def describe_run(events: Sequence[Mapping[str, object]], *, reply_chars: int = 300) -> str:
    parsed: Final = _EVENTS.validate_python(events)
    tool_calls: Final = tuple(block.name for block in _assistant_blocks(parsed) if isinstance(block, ToolUseBlock))
    calls: Final = ", ".join(tool_calls) or "none"
    result: Final = next((event for event in parsed if isinstance(event, ResultEvent)), None)
    if result is None:
        return f"tool calls: {calls}; no result event"
    reply: Final = result.result or ""
    shown: Final = reply if len(reply) <= reply_chars else f"{reply[:reply_chars]}..."
    return f"tool calls: {calls}; stop_reason={result.stop_reason}; turns={result.num_turns}; reply={shown!r}"
