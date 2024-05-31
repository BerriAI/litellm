from typing import List, Optional, Union, Iterable

from pydantic import BaseModel, validator

from typing_extensions import Literal, Required, TypedDict


class AnthropicMessagesToolChoice(TypedDict, total=False):
    type: Required[Literal["auto", "any", "tool"]]
    name: str


class AnthopicMessagesAssistantMessageTextContentParam(TypedDict, total=False):
    type: Required[Literal["text"]]

    text: str


class AnthopicMessagesAssistantMessageToolCallParam(TypedDict, total=False):
    type: Required[Literal["tool_use"]]

    id: str

    name: str

    input: dict


AnthropicMessagesAssistantMessageValues = Union[
    AnthopicMessagesAssistantMessageTextContentParam,
    AnthopicMessagesAssistantMessageToolCallParam,
]


class AnthopicMessagesAssistantMessageParam(TypedDict, total=False):
    content: Required[Union[str, Iterable[AnthropicMessagesAssistantMessageValues]]]
    """The contents of the system message."""

    role: Required[Literal["assistant"]]
    """The role of the messages author, in this case `author`."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the same
    role.
    """
