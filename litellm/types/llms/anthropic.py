from collections.abc import Iterable
from enum import Enum
from typing import Any, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict
from typing_extensions import NotRequired, ReadOnly, Required, TypedDict

from .openai import (
    ChatCompletionCachedContent,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    PromptCacheBreakpoint,
)


class AnthropicMessagesToolChoice(TypedDict, total=False):
    type: Required[Literal["auto", "any", "tool", "none"]]
    name: str
    disable_parallel_tool_use: bool  # default is false


AnthropicInputSchema = TypedDict(
    "AnthropicInputSchema",
    {
        "type": str | None,
        "properties": dict | None,
        "additionalProperties": bool | None,
        "required": list[str] | None,
        "$defs": dict | None,
        "strict": bool | None,
    },
    total=False,
)


class AnthropicOutputSchema(TypedDict, total=False):
    type: Required[Literal["json_schema"]]
    schema: Required[dict]
    strict: ReadOnly[bool]


class AnthropicOutputConfig(TypedDict, total=False):
    """Configuration for controlling Claude's output behavior."""

    effort: Literal["high", "medium", "low", "xhigh", "max"]
    format: AnthropicOutputSchema


class AnthropicMessagesTool(TypedDict, total=False):
    name: Required[str]
    description: str
    input_schema: AnthropicInputSchema | None
    strict: ReadOnly[bool]
    type: Literal["custom"]
    cache_control: dict | ChatCompletionCachedContent | None
    defer_loading: bool
    allowed_callers: list[str] | None
    input_examples: list[dict[str, Any]] | None


class AnthropicComputerTool(TypedDict, total=False):
    display_width_px: Required[int]
    display_height_px: Required[int]
    display_number: int
    cache_control: dict | ChatCompletionCachedContent | None
    type: Required[str]
    name: Required[str]


class AnthropicWebSearchUserLocation(TypedDict, total=False):
    city: str | None
    country: str | None
    region: str | None
    timezone: str | None
    type: Required[Literal["approximate"]]


class AnthropicWebSearchTool(TypedDict, total=False):
    name: Required[Literal["web_search"]]
    type: Required[str]
    cache_control: dict | ChatCompletionCachedContent | None
    max_uses: int | None
    user_location: AnthropicWebSearchUserLocation | None
    defer_loading: bool | None
    allowed_callers: list[str] | None
    input_examples: list[dict[str, Any]] | None


class AnthropicHostedTools(TypedDict, total=False):  # for bash_tool and text_editor
    type: Required[str]
    name: Required[str]
    cache_control: dict | ChatCompletionCachedContent | None
    defer_loading: bool | None
    allowed_callers: list[str] | None
    input_examples: list[dict[str, Any]] | None


class AnthropicCodeExecutionTool(TypedDict, total=False):
    type: Required[str]
    name: Required[Literal["code_execution"]]
    cache_control: dict | ChatCompletionCachedContent | None
    defer_loading: bool | None
    allowed_callers: list[str] | None
    input_examples: list[dict[str, Any]] | None


class AnthropicMemoryTool(TypedDict, total=False):
    type: Required[str]
    name: Required[Literal["memory"]]
    cache_control: dict | ChatCompletionCachedContent | None
    defer_loading: bool | None
    allowed_callers: list[str] | None
    input_examples: list[dict[str, Any]] | None


class AnthropicToolSearchToolRegex(TypedDict, total=False):
    """Tool search tool using regex patterns for tool discovery."""

    type: Required[Literal["tool_search_tool_regex_20251119"]]
    name: Required[str]


class AnthropicToolSearchToolBM25(TypedDict, total=False):
    """Tool search tool using BM25 algorithm for tool discovery."""

    type: Required[Literal["tool_search_tool_bm25_20251119"]]
    name: Required[str]
    cache_control: dict | ChatCompletionCachedContent | None
    defer_loading: bool | None
    allowed_callers: list[str] | None
    input_examples: list[dict[str, Any]] | None


ANTHROPIC_ADVISOR_TOOL_TYPE: Final = "advisor_20260301"


class AnthropicAdvisorTool(TypedDict, total=False):
    """Advisor tool — pairs a fast executor model with a high-intelligence advisor model."""

    type: Required[Literal["advisor_20260301"]]
    name: Required[Literal["advisor"]]
    model: Required[str]
    max_uses: int | None
    caching: dict | None


class ToolReference(TypedDict, total=False):
    """Reference to a tool that should be expanded from deferred tools."""

    type: Required[Literal["tool_reference"]]
    tool_name: Required[str]


class DirectToolCaller(TypedDict, total=False):
    """Indicates a tool was called directly by Claude."""

    type: Required[Literal["direct"]]


class CodeExecutionToolCaller(TypedDict, total=False):
    """Indicates a tool was called programmatically from code execution."""

    type: Required[Literal["code_execution_20250825"]]
    tool_id: Required[str]  # ID of the code execution tool that made the call


ToolCaller = DirectToolCaller | CodeExecutionToolCaller


class AnthropicContainer(TypedDict, total=False):
    """Container metadata for code execution."""

    id: Required[str]
    expires_at: str | None  # ISO 8601 timestamp


AllAnthropicToolsValues = (
    AnthropicComputerTool
    | AnthropicHostedTools
    | AnthropicMessagesTool
    | AnthropicWebSearchTool
    | AnthropicCodeExecutionTool
    | AnthropicMemoryTool
    | AnthropicToolSearchToolRegex
    | AnthropicToolSearchToolBM25
    | AnthropicAdvisorTool
)


class AnthropicMcpServerToolConfiguration(TypedDict, total=False):
    allowed_tools: list[str] | None


class AnthropicMcpServerTool(TypedDict, total=False):
    type: Required[Literal["url"]]
    url: Required[str]
    name: Required[str]
    tool_configuration: AnthropicMcpServerToolConfiguration
    authorization_token: str


class AnthropicMessagesTextParam(TypedDict, total=False):
    type: Required[Literal["text"]]
    text: Required[str]
    cache_control: dict | ChatCompletionCachedContent | None
    prompt_cache_breakpoint: ReadOnly[PromptCacheBreakpoint]


class AnthropicMessagesToolUseParam(TypedDict, total=False):
    type: Required[Literal["tool_use"]]
    id: str
    name: str
    input: dict
    cache_control: dict | ChatCompletionCachedContent | None
    caller: ToolCaller | None


AnthropicMessagesAssistantMessageValues = (
    AnthropicMessagesTextParam
    | AnthropicMessagesToolUseParam
    | ChatCompletionThinkingBlock
    | ChatCompletionRedactedThinkingBlock
)


class AnthopicMessagesAssistantMessageParam(TypedDict, total=False):
    content: Required[str | Iterable[AnthropicMessagesAssistantMessageValues]]
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
    cache_control: dict | ChatCompletionCachedContent | None


class AnthropicMessagesImageParam(TypedDict, total=False):
    type: Required[Literal["image"]]
    source: Required[AnthropicContentParamSource | AnthropicContentParamSourceFileId | AnthropicContentParamSourceUrl]
    cache_control: dict | ChatCompletionCachedContent | None
    prompt_cache_breakpoint: ReadOnly[PromptCacheBreakpoint]


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
    document_title: str | None  # Title of the cited document
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
    document_title: str | None  # Title of the cited document
    start_char_index: int  # Starting character index for the citation
    end_char_index: int  # Ending character index for the citation


# Union type for all citation formats
AnthropicCitation = AnthropicCitationPageLocation | AnthropicCitationCharLocation


class AnthropicMessagesDocumentParam(TypedDict, total=False):
    type: Required[Literal["document"]]
    source: Required[AnthropicContentParamSource | AnthropicContentParamSourceFileId | AnthropicContentParamSourceUrl]
    cache_control: dict | ChatCompletionCachedContent | None
    title: str
    context: str
    citations: CitationsObject | None


class AnthropicMessagesToolResultContent(TypedDict, total=False):
    type: Required[Literal["text"]]
    text: Required[str]
    cache_control: dict | ChatCompletionCachedContent | None


class AnthropicMessagesToolResultParam(TypedDict, total=False):
    type: Required[Literal["tool_result"]]
    tool_use_id: Required[str]
    is_error: bool
    content: (
        str
        | Iterable[AnthropicMessagesToolResultContent | AnthropicMessagesImageParam | AnthropicMessagesDocumentParam]
    )
    cache_control: dict | ChatCompletionCachedContent | None


AnthropicMessagesUserMessageValues = (
    AnthropicMessagesTextParam
    | AnthropicMessagesImageParam
    | AnthropicMessagesToolResultParam
    | AnthropicMessagesDocumentParam
    | AnthropicMessagesContainerUploadParam
)


class AnthropicMessagesUserMessageParam(TypedDict, total=False):
    role: Required[Literal["user"]]
    content: Required[str | Iterable[AnthropicMessagesUserMessageValues]]


class AnthropicMetadata(TypedDict, total=False):
    user_id: str


class AnthropicSystemMessageContent(TypedDict, total=False):
    type: str
    text: str
    cache_control: dict | ChatCompletionCachedContent | None
    prompt_cache_breakpoint: ReadOnly[PromptCacheBreakpoint]


class AnthropicMessagesSystemMessageParam(TypedDict, total=False):
    role: Required[Literal["system"]]
    content: Required[str | Iterable[AnthropicSystemMessageContent]]


AllAnthropicMessageValues = AnthropicMessagesUserMessageParam | AnthopicMessagesAssistantMessageParam

# System is not a native Anthropic message role; only pass-through adapters use this union.
AllAnthropicPassThroughMessageValues: TypeAlias = (
    AnthropicMessagesUserMessageParam | AnthopicMessagesAssistantMessageParam | AnthropicMessagesSystemMessageParam
)


class AnthropicMessagesRequestOptionalParams(TypedDict, total=False):
    max_tokens: int | None
    metadata: AnthropicMetadata | dict | None
    stop_sequences: list[str] | None
    stream: bool | None
    system: str | list | None
    temperature: float | None
    thinking: dict | None
    tool_choice: AnthropicMessagesToolChoice | dict | None
    tools: list[AllAnthropicToolsValues | dict] | None
    top_k: int | None
    inference_geo: str | None
    top_p: float | None
    mcp_servers: list[AnthropicMcpServerTool] | None
    context_management: dict[str, Any] | None
    container: dict[str, Any] | None  # Container config with skills for code execution
    output_format: AnthropicOutputSchema | None  # Structured outputs support
    speed: str | None  # Fast mode support for Opus models
    output_config: AnthropicOutputConfig | None  # Configuration for Claude's output behavior
    cache_control: dict[str, Any] | None  # Automatic prompt caching
    reasoning_effort: str | None


class AnthropicMessagesRequest(AnthropicMessagesRequestOptionalParams, total=False):
    model: Required[str]
    messages: Required[list[AllAnthropicMessageValues] | list[dict]]
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


StreamingContentBlockDeltaType = Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"]


class ContentBlockDelta(TypedDict):
    type: Literal["content_block_delta"]
    index: int
    delta: (
        ContentTextBlockDelta
        | ContentJsonBlockDelta
        | ContentCitationsBlockDelta
        | ContentThinkingBlockDelta
        | ContentThinkingSignatureBlockDelta
    )


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
    caller: ToolCaller | None


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


ContentBlockContentBlockDict = ToolUseBlock | TextBlock | ChatCompletionThinkingBlock

ContentBlockStart = ContentBlockStartToolUse | ContentBlockStartText


class MessageDelta(TypedDict, total=False):
    stop_reason: str | None


class ServerToolUsage(TypedDict, total=False):
    web_search_requests: ReadOnly[int]


class UsageDelta(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    server_tool_use: ReadOnly[ServerToolUsage]


class AppliedEdit(TypedDict, total=False):
    """One applied context_management edit (Anthropic response shape)."""

    type: str
    cleared_input_tokens: int
    cleared_tool_uses: int
    cleared_thinking_turns: int
    # compact_20260112 fields
    summary_input_tokens: int
    summary_output_tokens: int
    error: str
    warnings: list[str]


class ContextManagementResponse(TypedDict, total=False):
    """Response ``context_management`` with ``applied_edits``."""

    applied_edits: list[AppliedEdit]


class CompactionBlock(TypedDict, total=False):
    """Synthesized ``compaction`` content block (compact_20260112)."""

    type: Required[Literal["compaction"]]
    content: str | None


class UsageIteration(TypedDict, total=False):
    """One sampling iteration's token usage (compact_20260112)."""

    type: Required[Literal["compaction", "message"]]
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
    context_management: NotRequired[ContextManagementResponse]


class MessageChunk(TypedDict, total=False):
    id: str
    type: str
    role: str
    model: str
    content: list
    stop_reason: str | None
    stop_sequence: str | None
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
    provider_specific_fields: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")  # Allow provider_specific_fields


class AnthropicResponseContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: str | None


class AnthropicResponseContentBlockRedactedThinking(BaseModel):
    type: Literal["redacted_thinking"]
    data: str


class AnthropicResponseUsageBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int
    output_tokens: int


class AnthropicOutputTokensDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    thinking_tokens: int | None = None


AnthropicFinishReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]


class AnthropicResponse(BaseModel):
    id: str
    """Unique object identifier."""

    type: Literal["message"]
    """For Messages, this is always "message"."""

    role: Literal["assistant"]
    """Conversational role of the generated message. This will always be "assistant"."""

    content: list[
        AnthropicResponseContentBlockText
        | AnthropicResponseContentBlockToolUse
        | AnthropicResponseContentBlockThinking
        | AnthropicResponseContentBlockRedactedThinking
    ]
    """Content generated by the model."""

    model: str
    """The model that handled the request."""

    stop_reason: AnthropicFinishReason | None
    """The reason that we stopped."""

    stop_sequence: str | None
    """Which custom stop sequence was generated, if any."""

    usage: AnthropicResponseUsageBlock
    """Billing and rate-limit usage."""


from .openai import ChatCompletionUsageBlock


class AnthropicChatCompletionUsageBlock(ChatCompletionUsageBlock, total=False):
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


ANTHROPIC_API_HEADERS: Final = {
    "anthropic-version",
    "anthropic-beta",
}

ANTHROPIC_API_ONLY_HEADERS: Final = {  # fails if calling anthropic on vertex ai / bedrock
    "anthropic-beta",
}


class AnthropicThinkingParam(TypedDict, total=False):
    type: ReadOnly[Literal["enabled", "adaptive", "disabled"]]
    budget_tokens: int
    display: ReadOnly[Literal["summarized", "omitted"]]


class ANTHROPIC_HOSTED_TOOLS(str, Enum):
    WEB_SEARCH = "web_search"
    BASH = "bash"
    TEXT_EDITOR = "text_editor"
    CODE_EXECUTION = "code_execution"
    WEB_FETCH = "web_fetch"
    MEMORY = "memory"
    TOOL_SEARCH_TOOL = "tool_search_tool"


class ANTHROPIC_BETA_HEADER_VALUES(str, Enum):
    """
    Known beta header values for Anthropic.
    """

    WEB_FETCH_2025_09_10 = "web-fetch-2025-09-10"
    WEB_SEARCH_2025_03_05 = "web-search-2025-03-05"
    CONTEXT_MANAGEMENT_2025_06_27 = "context-management-2025-06-27"
    COMPACT_2026_01_12 = "compact-2026-01-12"
    STRUCTURED_OUTPUT_2025_09_25 = "structured-outputs-2025-11-13"
    ADVANCED_TOOL_USE_2025_11_20 = "advanced-tool-use-2025-11-20"
    FAST_MODE_2026_02_01 = "fast-mode-2026-02-01"
    ADVISOR_TOOL_2026_03_01 = "advisor-tool-2026-03-01"


# Tool search beta header constant (for Anthropic direct API and Microsoft Foundry)
ANTHROPIC_TOOL_SEARCH_BETA_HEADER: Final = "advanced-tool-use-2025-11-20"

# Effort beta header constant
ANTHROPIC_EFFORT_BETA_HEADER: Final = "effort-2025-11-24"

# OAuth constants
ANTHROPIC_OAUTH_TOKEN_PREFIX: Final = "sk-ant-oat"
ANTHROPIC_OAUTH_BETA_HEADER: Final = "oauth-2025-04-20"

ANTHROPIC_PROMPT_CACHING_SCOPE_BETA_HEADER: Final = "prompt-caching-scope-2026-01-05"
