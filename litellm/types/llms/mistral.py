from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict
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


class MistralConversationContentChunk(BaseModel):
    """A single chunk of a Conversations API ``message.output`` content list.

    Text chunks carry ``text``; ``tool_reference`` chunks (web search sources)
    carry ``title``/``url``. Modelled permissively so unknown chunk types from
    the API don't break parsing.
    """

    model_config = ConfigDict(extra="allow")
    type: str | None = None
    text: str | None = None
    title: str | None = None
    url: str | None = None


class MistralConversationOutput(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str | None = None
    name: str | None = None
    content: str | tuple[MistralConversationContentChunk, ...] | None = None


class MistralConversationUsage(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    connectors: Mapping[str, int] | None = None


class MistralConversationsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    conversation_id: str | None = None
    outputs: tuple[MistralConversationOutput, ...] = ()
    usage: MistralConversationUsage | None = None
