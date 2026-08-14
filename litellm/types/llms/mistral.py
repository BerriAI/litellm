from typing import Literal

from typing_extensions import TypedDict


class FunctionCall(TypedDict):
    name: str | None
    arguments: str | dict | None


class MistralToolCallMessage(TypedDict):
    id: str | None
    type: Literal["function"]
    function: FunctionCall | None


class MistralTextBlock(TypedDict):
    type: Literal["text"]
    text: str


class MistralThinkingBlock(TypedDict):
    type: Literal["thinking"]
    thinking: list[MistralTextBlock]
