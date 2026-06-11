"""Frozen intermediate representation for chat-shaped requests.

This is the hub of the hub-and-spoke: inbound parsers map each accepted
request schema into these types, and provider serializers map these types
onto a wire format. Everything here is immutable. Product types are frozen
dataclasses, sum types are Expression tagged unions matched with ``match``,
and collections are Expression ``Block``/``Map`` rather than ``list``/``dict``.
Nothing in this module performs I/O or imports a provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Union

from expression import Option, case, tag, tagged_union
from expression.collections import Block, Map

Json = Union[None, bool, int, float, str, "Block[Json]", "Map[str, Json]"]

PlainJson = Union[
    None, bool, int, float, str, "list[PlainJson]", "dict[str, PlainJson]"
]

Body = Dict[str, PlainJson]


@dataclass(frozen=True)
class Unit:
    """The single inhabited value for tagged-union cases that carry no data."""


UNIT = Unit()


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Image:
    media_type: str
    data: str


@dataclass(frozen=True)
class ToolUse:
    id: str
    name: str
    arguments: Map[str, Json]


@dataclass(frozen=True)
class ToolResult:
    tool_use_id: str
    content: str


@tagged_union(frozen=True)
class ContentBlock:
    tag: Literal["text", "image", "tool_use", "tool_result"] = tag()

    text: Text = case()
    image: Image = case()
    tool_use: ToolUse = case()
    tool_result: ToolResult = case()

    @staticmethod
    def of_text(value: Text) -> "ContentBlock":
        return ContentBlock(text=value)

    @staticmethod
    def of_image(value: Image) -> "ContentBlock":
        return ContentBlock(image=value)

    @staticmethod
    def of_tool_use(value: ToolUse) -> "ContentBlock":
        return ContentBlock(tool_use=value)

    @staticmethod
    def of_tool_result(value: ToolResult) -> "ContentBlock":
        return ContentBlock(tool_result=value)


Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: Block[ContentBlock]


@dataclass(frozen=True)
class SystemText:
    text: str


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: Option[str]
    parameters: Map[str, Json]


@tagged_union(frozen=True)
class ToolChoice:
    tag: Literal["auto", "required", "none", "specific"] = tag()

    auto: Unit = case()
    required: Unit = case()
    none: Unit = case()
    specific: str = case()

    @staticmethod
    def of_auto() -> "ToolChoice":
        return ToolChoice(auto=UNIT)

    @staticmethod
    def of_required() -> "ToolChoice":
        return ToolChoice(required=UNIT)

    @staticmethod
    def of_none() -> "ToolChoice":
        return ToolChoice(none=UNIT)

    @staticmethod
    def of_specific(name: str) -> "ToolChoice":
        return ToolChoice(specific=name)


@dataclass(frozen=True)
class InferenceParams:
    max_tokens: Option[int]
    temperature: Option[float]
    top_p: Option[float]
    stop: Block[str]


@dataclass(frozen=True)
class ChatRequest:
    model: str
    system: Block[SystemText]
    messages: Block[Message]
    tools: Block[ToolDef]
    tool_choice: Option[ToolChoice]
    params: InferenceParams
    stream: bool


def has_tool_blocks(messages: Block[Message]) -> bool:
    """True when any message carries a tool_use or tool_result content block."""
    return any(
        block.tag in ("tool_use", "tool_result")
        for message in messages
        for block in message.content
    )
