from typing import Any, Literal, TypeAlias

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.types.llms.anthropic import (
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockToolUse,
    ContextManagementResponse,
    ServerToolUsage,
)


class AnthropicResponseTextBlock(TypedDict, total=False):
    """
    Anthropic Response Text Block: https://docs.anthropic.com/en/api/messages
    """

    citations: list[dict[str, Any]] | None
    text: str
    type: Literal["text"]


class AnthropicResponseToolUseBlock(TypedDict, total=False):
    """
    Anthropic Response Tool Use Block: https://docs.anthropic.com/en/api/messages
    """

    id: str | None
    input: str | None
    name: str | None
    type: Literal["tool_use"]


class AnthropicResponseThinkingBlock(TypedDict, total=False):
    """
    Anthropic Response Thinking Block: https://docs.anthropic.com/en/api/messages
    """

    signature: str | None
    thinking: str | None
    type: Literal["thinking"]


class AnthropicResponseRedactedThinkingBlock(TypedDict, total=False):
    """
    Anthropic Response Redacted Thinking Block: https://docs.anthropic.com/en/api/messages
    """

    data: str | None
    type: Literal["redacted_thinking"]


AnthropicResponseContentBlock: TypeAlias = (
    AnthropicResponseTextBlock
    | AnthropicResponseToolUseBlock
    | AnthropicResponseThinkingBlock
    | AnthropicResponseRedactedThinkingBlock
)


class AnthropicUsage(TypedDict, total=False):
    """
    Input and output tokens used in the request
    """

    input_tokens: int
    output_tokens: int

    """
    Cache Tokens Used
    """
    cache_creation_input_tokens: int
    cache_read_input_tokens: int

    """
    Server-side tool usage (e.g. web search request counts)
    """
    server_tool_use: NotRequired[ReadOnly[ServerToolUsage]]


class AnthropicMessagesResponse(TypedDict, total=False):
    """
    Anthropic Messages API Response: https://docs.anthropic.com/en/api/messages
    """

    content: (
        list[AnthropicResponseContentBlock | AnthropicResponseContentBlockText | AnthropicResponseContentBlockToolUse]
        | None
    )
    id: str
    model: str | None  # This represents the Model type from Anthropic
    role: Literal["assistant"] | None
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    stop_sequence: str | None
    type: Literal["message"] | None
    usage: AnthropicUsage | None
    context_management: NotRequired[ContextManagementResponse]
