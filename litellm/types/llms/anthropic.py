from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Union

from pydantic import BaseModel
from typing_extensions import Literal, Required, TypedDict

from .openai import (
    ChatCompletionCachedContent,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
)


class AnthropicMessagesToolChoice(TypedDict, total=False):
    type: Required[Literal["auto", "any", "tool", "none"]]
    name: str
    disable_parallel_tool_use: bool  # default is false


AnthropicInputSchema = TypedDict(
    "AnthropicInputSchema",
    {
        "type": Optional[str],
        "properties": Optional[dict],
        "additionalProperties": Optional[bool],
        "required": Optional[List[str]],
        "$defs": Optional[Dict],
        "strict": Optional[bool],
    },
    total=False,
)


class AnthropicOutputSchema(TypedDict, total=False):
    type: Required[Literal["json_schema"]]
    schema: Required[dict]


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


class AnthropicMemoryTool(TypedDict, total=False):
    type: Required[str]
    name: Required[Literal["memory"]]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


AllAnthropicToolsValues = Union[
    AnthropicComputerTool,
    AnthropicHostedTools,
    AnthropicMessagesTool,
    AnthropicWebSearchTool,
    AnthropicCodeExecutionTool,
    AnthropicMemoryTool,
]


class AnthropicMcpServerToolConfiguration(TypedDict, total=False):
    allowed_tools: Optional[List[str]]


class AnthropicMcpServerTool(TypedDict, total=False):
    type: Required[Literal["url"]]
    url: Required[str]
    name: Required[str]
    tool_configuration: AnthropicMcpServerToolConfiguration
    authorization_token: str


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
    ChatCompletionRedactedThinkingBlock,
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


class AnthropicContentParamSourceUrl(TypedDict):
    type: Literal["url"]
    url: str


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
        Union[
            AnthropicContentParamSource,
            AnthropicContentParamSourceFileId,
            AnthropicContentParamSourceUrl,
        ]
    ]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class CitationsObject(TypedDict):
    enabled: bool


class AnthropicCitationPageLocation(TypedDict, total=False):
    """
    Anthropic citation for page-based references.
    Used when citing from documents with page numbers.
    """

    type: Literal["page_location"]
    cited_text: str  # The exact text being cited (not counted towards output tokens)
    document_index: int  # Index referencing the cited document
    document_title: Optional[str]  # Title of the cited document
    start_page_number: int  # 1-indexed starting page
    end_page_number: int  # Exclusive ending page


class AnthropicCitationCharLocation(TypedDict, total=False):
    """
    Anthropic citation for character-based references.
    Used when citing from text with character positions.
    """

    type: Literal["char_location"]
    cited_text: str  # The exact text being cited (not counted towards output tokens)
    document_index: int  # Index referencing the cited document
    document_title: Optional[str]  # Title of the cited document
    start_char_index: int  # Starting character index for the citation
    end_char_index: int  # Ending character index for the citation


# Union type for all citation formats
AnthropicCitation = Union[AnthropicCitationPageLocation, AnthropicCitationCharLocation]


class AnthropicMessagesDocumentParam(TypedDict, total=False):
    type: Required[Literal["document"]]
    source: Required[
        Union[
            AnthropicContentParamSource,
            AnthropicContentParamSourceFileId,
            AnthropicContentParamSourceUrl,
        ]
    ]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    title: str
    context: str
    citations: Optional[CitationsObject]


class AnthropicMessagesToolResultContent(TypedDict):
    type: Literal["text"]
    text: str
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


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
    mcp_servers: Optional[List[AnthropicMcpServerTool]]
    context_management: Optional[Dict[str, Any]]


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


class ContentThinkingBlockDelta(TypedDict):
    """
    "delta": {"type": "thinking_delta", "thinking": "Let me solve this step by step:"}}
    """

    type: Literal["thinking_delta"]
    thinking: str


class ContentThinkingSignatureBlockDelta(TypedDict):
    """
    "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds..."}}
    """

    type: Literal["signature_delta"]
    signature: str


class ContentBlockDelta(TypedDict):
    type: Literal["content_block_delta"]
    index: int
    delta: Union[
        ContentTextBlockDelta,
        ContentJsonBlockDelta,
        ContentCitationsBlockDelta,
        ContentThinkingBlockDelta,
        ContentThinkingSignatureBlockDelta,
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


class ContentBlockStartToolUse(TypedDict):
    type: Literal["content_block_start"]
    id: str
    name: str
    input: dict
    content_block: ToolUseBlock


class ContentBlockStartText(TypedDict):
    type: Literal["content_block_start"]
    index: int
    content_block: TextBlock


ContentBlockContentBlockDict = Union[
    ToolUseBlock, TextBlock, ChatCompletionThinkingBlock
]

ContentBlockStart = Union[ContentBlockStartToolUse, ContentBlockStartText]


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
    provider_specific_fields: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"  # Allow provider_specific_fields


class AnthropicResponseContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: Optional[str]


class AnthropicResponseContentBlockRedactedThinking(BaseModel):
    type: Literal["redacted_thinking"]
    data: str


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
        Union[
            AnthropicResponseContentBlockText,
            AnthropicResponseContentBlockToolUse,
            AnthropicResponseContentBlockThinking,
            AnthropicResponseContentBlockRedactedThinking,
        ]
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


class ANTHROPIC_HOSTED_TOOLS(str, Enum):
    WEB_SEARCH = "web_search"
    BASH = "bash"
    TEXT_EDITOR = "text_editor"
    CODE_EXECUTION = "code_execution"
    WEB_FETCH = "web_fetch"
    MEMORY = "memory"


class ANTHROPIC_BETA_HEADER_VALUES(str, Enum):
    """
    Known beta header values for Anthropic.
    """

    WEB_FETCH_2025_09_10 = "web-fetch-2025-09-10"
    CONTEXT_MANAGEMENT_2025_06_27 = "context-management-2025-06-27"
    STRUCTURED_OUTPUT_2025_09_25 = "structured-outputs-2025-11-13"
