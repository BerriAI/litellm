from typing import Any, Dict, Iterable, List, Optional, Union

from pydantic import BaseModel, validator
from typing_extensions import Literal, Required, TypedDict

from .openai import ChatCompletionCachedContent, ChatCompletionThinkingBlock


class AnthropicMessagesToolChoice(TypedDict, total=False):
    type: Required[Literal["auto", "any", "tool"]]
    name: str
    disable_parallel_tool_use: bool  # default is false


class AnthropicInputSchema(TypedDict, total=False):
    type: Optional[str]
    properties: Optional[dict]
    additionalProperties: Optional[bool]


class AnthropicMessagesTool(TypedDict, total=False):
    name: Required[str]
    description: str
    input_schema: Optional[AnthropicInputSchema]
    type: Literal["custom"]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class AnthropicComputerTool(TypedDict, total=False):
    display_width_px: Required[int]
    display_height_px: Required[int]
    display_number: int
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    type: Required[str]
    name: Required[str]


class AnthropicWebSearchUserLocation(TypedDict, total=False):
    city: Optional[str]
    country: Optional[str]
    region: Optional[str]
    timezone: Optional[str]
    type: Required[Literal["approximate"]]


class AnthropicWebSearchTool(TypedDict, total=False):
    name: Required[Literal["web_search"]]
    type: Required[str]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    max_uses: Optional[int]
    user_location: Optional[AnthropicWebSearchUserLocation]


class AnthropicHostedTools(TypedDict, total=False):  # for bash_tool and text_editor
    type: Required[str]
    name: Required[str]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class AnthropicCodeExecutionTool(TypedDict, total=False):
    type: Required[str]
    name: Required[Literal["code_execution"]]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


AllAnthropicToolsValues = Union[
    AnthropicComputerTool,
    AnthropicHostedTools,
    AnthropicMessagesTool,
    AnthropicWebSearchTool,
    AnthropicCodeExecutionTool,
]


class AnthropicMessagesTextParam(TypedDict, total=False):
    type: Required[Literal["text"]]
    text: Required[str]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class AnthropicMessagesToolUseParam(TypedDict, total=False):
    type: Required[Literal["tool_use"]]
    id: str
    name: str
    input: dict
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


AnthropicMessagesAssistantMessageValues = Union[
    AnthropicMessagesTextParam,
    AnthropicMessagesToolUseParam,
    ChatCompletionThinkingBlock,
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


class AnthropicContentParamSource(TypedDict):
    type: Literal["base64"]
    media_type: str
    data: str


class AnthropicContentParamSourceFileId(TypedDict):
    type: Literal["file"]
    file_id: str


class AnthropicMessagesContainerUploadParam(TypedDict, total=False):
    type: Required[Literal["container_upload"]]
    file_id: str
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class AnthropicMessagesImageParam(TypedDict, total=False):
    type: Required[Literal["image"]]
    source: Required[
        Union[AnthropicContentParamSource, AnthropicContentParamSourceFileId]
    ]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class CitationsObject(TypedDict):
    enabled: bool


class AnthropicMessagesDocumentParam(TypedDict, total=False):
    type: Required[Literal["document"]]
    source: Required[
        Union[AnthropicContentParamSource, AnthropicContentParamSourceFileId]
    ]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    title: str
    context: str
    citations: Optional[CitationsObject]


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
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


AnthropicMessagesUserMessageValues = Union[
    AnthropicMessagesTextParam,
    AnthropicMessagesImageParam,
    AnthropicMessagesToolResultParam,
    AnthropicMessagesDocumentParam,
    AnthropicMessagesContainerUploadParam,
]


class AnthropicMessagesUserMessageParam(TypedDict, total=False):
    role: Required[Literal["user"]]
    content: Required[Union[str, Iterable[AnthropicMessagesUserMessageValues]]]


class AnthropicMetadata(TypedDict, total=False):
    user_id: str


class AnthropicSystemMessageContent(TypedDict, total=False):
    type: str
    text: str
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


AllAnthropicMessageValues = Union[
    AnthropicMessagesUserMessageParam, AnthopicMessagesAssistantMessageParam
]


class AnthropicMessagesRequestOptionalParams(TypedDict, total=False):
    max_tokens: Optional[int]
    metadata: Optional[Union[AnthropicMetadata, Dict]]
    stop_sequences: Optional[List[str]]
    stream: Optional[bool]
    system: Optional[Union[str, List]]
    temperature: Optional[float]
    thinking: Optional[Dict]
    tool_choice: Optional[Union[AnthropicMessagesToolChoice, Dict]]
    tools: Optional[List[Union[AllAnthropicToolsValues, Dict]]]
    top_k: Optional[int]
    top_p: Optional[float]


class AnthropicMessagesRequest(AnthropicMessagesRequestOptionalParams, total=False):
    model: Required[str]
    messages: Required[Union[List[AllAnthropicMessageValues], List[Dict]]]
    # litellm param - used for tracking litellm proxy metadata in the request
    litellm_metadata: dict


class ContentTextBlockDelta(TypedDict):
    """
    'delta': {'type': 'text_delta', 'text': 'Hello'}
    """

    type: str
    text: str


class ContentCitationsBlockDelta(TypedDict):
    type: Literal["citations"]
    citation: dict


class ContentJsonBlockDelta(TypedDict):
    """
    "delta": {"type": "input_json_delta","partial_json": "{\"location\": \"San Fra"}}
    """

    type: str
    partial_json: str


class ContentBlockDelta(TypedDict):
    type: Literal["content_block_delta"]
    index: int
    delta: Union[
        ContentTextBlockDelta, ContentJsonBlockDelta, ContentCitationsBlockDelta
    ]


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


from .openai import ChatCompletionUsageBlock


class AnthropicChatCompletionUsageBlock(ChatCompletionUsageBlock, total=False):
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


ANTHROPIC_API_HEADERS = {
    "anthropic-version",
    "anthropic-beta",
}

ANTHROPIC_API_ONLY_HEADERS = {  # fails if calling anthropic on vertex ai / bedrock
    "anthropic-beta",
}


class AnthropicThinkingParam(TypedDict, total=False):
    type: Literal["enabled"]
    budget_tokens: int
