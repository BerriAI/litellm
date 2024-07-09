from typing import Iterable, List, Optional, Union

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


class ContentTextBlockDelta(TypedDict):
    """
    'delta': {'type': 'text_delta', 'text': 'Hello'}
    """

    type: str
    text: str


class ContentJsonBlockDelta(TypedDict):
    """
    "delta": {"type": "input_json_delta","partial_json": "{\"location\": \"San Fra"}}
    """

    type: str
    partial_json: str


class ContentBlockDelta(TypedDict):
    type: str
    index: int
    delta: Union[ContentTextBlockDelta, ContentJsonBlockDelta]


class ToolUseBlock(TypedDict):
    """
    "content_block":{"type":"tool_use","id":"toolu_01T1x1fJ34qAmk2tNTrN7Up6","name":"get_weather","input":{}}
    """

    id: str

    input: dict

    name: str

    type: Literal["tool_use"]


class TextBlock(TypedDict):
    text: str

    type: Literal["text"]


class ContentBlockStart(TypedDict):
    """
    event: content_block_start
    data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01T1x1fJ34qAmk2tNTrN7Up6","name":"get_weather","input":{}}}
    """

    type: str
    index: int
    content_block: Union[ToolUseBlock, TextBlock]


class MessageDelta(TypedDict, total=False):
    stop_reason: Optional[str]


class UsageDelta(TypedDict, total=False):
    input_tokens: int
    output_tokens: int


class MessageBlockDelta(TypedDict):
    """
    Anthropic
    chunk = {'type': 'message_delta', 'delta': {'stop_reason': 'max_tokens', 'stop_sequence': None}, 'usage': {'output_tokens': 10}}
    """

    type: Literal["message_delta"]
    delta: MessageDelta
    usage: UsageDelta


class MessageChunk(TypedDict, total=False):
    id: str
    type: str
    role: str
    model: str
    content: List
    stop_reason: Optional[str]
    stop_sequence: Optional[str]
    usage: UsageDelta


class MessageStartBlock(TypedDict):
    """
        Anthropic
        chunk = {
        "type": "message_start",
        "message": {
            "id": "msg_vrtx_011PqREFEMzd3REdCoUFAmdG",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [],
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {
                "input_tokens": 270,
                "output_tokens": 1
            }
        }
    }
    """

    type: Literal["message_start"]
    message: MessageChunk
