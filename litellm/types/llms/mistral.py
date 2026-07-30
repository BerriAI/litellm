from typing import List, Literal, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict
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


class MistralConversationContentChunk(BaseModel):
    """A single chunk of a Conversations API ``message.output`` content list.

    Text chunks carry ``text``; ``tool_reference`` chunks (web search sources)
    carry ``title``/``url``. Modelled permissively so unknown chunk types from
    the API don't break parsing.
    """

    model_config = ConfigDict(extra="allow")
    type: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None


class MistralConversationOutput(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Optional[str] = None
    name: Optional[str] = None
    content: Optional[Union[str, Tuple[MistralConversationContentChunk, ...]]] = None


class MistralConversationUsage(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    connectors: Optional[Mapping[str, int]] = None


class MistralConversationsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    conversation_id: Optional[str] = None
    outputs: Tuple[MistralConversationOutput, ...] = ()
    usage: Optional[MistralConversationUsage] = None
