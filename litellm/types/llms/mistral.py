from typing import List, Literal, Optional, Union

from typing_extensions import TypedDict


class FunctionCall(TypedDict):
    name: Optional[str]
    arguments: Optional[Union[str, dict]]


class MistralToolCallMessage(TypedDict):
    id: Optional[str]
    type: Literal["function"]
    function: Optional[FunctionCall]


class MistralTextBlock(TypedDict):
    type: Literal["text"]
    text: str


class MistralThinkingBlock(TypedDict):
    type: Literal["thinking"]
    thinking: List[MistralTextBlock]
