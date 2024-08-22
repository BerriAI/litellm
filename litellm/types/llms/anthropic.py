from typing import Any, Dict, Iterable, List, Optional, Union

from pydantic import BaseModel, validator
from typing_extensions import Literal, Required, TypedDict


class AnthropicMessagesToolChoice(TypedDict, total=False):
    type: Required[Literal["auto", "any", "tool"]]
    name: str


class AnthropicMessagesTool(TypedDict, total=False):
    name: Required[str]
    description: str
    input_schema: Required[dict]


class AnthropicMessagesTextParam(TypedDict, total=False):
    type: Literal["text"]
    text: str
    cache_control: Optional[dict]


class AnthropicMessagesToolUseParam(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict


AnthropicMessagesAssistantMessageValues = Union[
    AnthropicMessagesTextParam,
    AnthropicMessagesToolUseParam,
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


class AnthropicImageParamSource(TypedDict):
    type: Literal["base64"]
    media_type: str
    data: str


class AnthropicMessagesImageParam(TypedDict, total=False):
    type: Literal["image"]
    source: AnthropicImageParamSource
    cache_control: Optional[dict]


class AnthropicMessagesToolResultContent(TypedDict):
    type: Literal["text"]
    text: str


class AnthropicMessagesToolResultParam(TypedDict, total=False):
    type: Required[Literal["tool_result"]]
    tool_use_id: Required[str]
    is_error: bool
    content: Union[
        str,
        Iterable[
            Union[AnthropicMessagesToolResultContent, AnthropicMessagesImageParam]
        ],
    ]


AnthropicMessagesUserMessageValues = Union[
    AnthropicMessagesTextParam,
    AnthropicMessagesImageParam,
    AnthropicMessagesToolResultParam,
]


class AnthropicMessagesUserMessageParam(TypedDict, total=False):
    role: Required[Literal["user"]]
    content: Required[Union[str, Iterable[AnthropicMessagesUserMessageValues]]]


class AnthropicMetadata(TypedDict, total=False):
    user_id: str


class AnthropicSystemMessageContent(TypedDict, total=False):
    type: str
    text: str
    cache_control: Optional[dict]


class AnthropicMessagesRequest(TypedDict, total=False):
    model: Required[str]
    messages: Required[
        List[
            Union[
                AnthropicMessagesUserMessageParam,
                AnthopicMessagesAssistantMessageParam,
            ]
        ]
    ]
    max_tokens: Required[int]
    metadata: AnthropicMetadata
    stop_sequences: List[str]
    stream: bool
    system: Union[str, List]
    temperature: float
    tool_choice: AnthropicMessagesToolChoice
    tools: List[AnthropicMessagesTool]
    top_k: int
    top_p: float

    # litellm param - used for tracking litellm proxy metadata in the request
    litellm_metadata: dict


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
    type: Literal["content_block_delta"]
    index: int
    delta: Union[ContentTextBlockDelta, ContentJsonBlockDelta]


class ContentBlockStop(TypedDict):
    type: Literal["content_block_stop"]
    index: int


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


class AnthropicResponseContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class AnthropicResponseContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict


class AnthropicResponseUsageBlock(BaseModel):
    input_tokens: int
    output_tokens: int


AnthropicFinishReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]


class AnthropicResponse(BaseModel):
    id: str
    """Unique object identifier."""

    type: Literal["message"]
    """For Messages, this is always "message"."""

    role: Literal["assistant"]
    """Conversational role of the generated message. This will always be "assistant"."""

    content: List[
        Union[AnthropicResponseContentBlockText, AnthropicResponseContentBlockToolUse]
    ]
    """Content generated by the model."""

    model: str
    """The model that handled the request."""

    stop_reason: Optional[AnthropicFinishReason]
    """The reason that we stopped."""

    stop_sequence: Optional[str]
    """Which custom stop sequence was generated, if any."""

    usage: AnthropicResponseUsageBlock
    """Billing and rate-limit usage."""
