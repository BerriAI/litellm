from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from typing_extensions import TypeAlias

from litellm.types.llms.anthropic import (
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockToolUse,
)


class AnthropicResponseTextBlock(TypedDict, total=False):
    """
    Anthropic Response Text Block: https://docs.anthropic.com/en/api/messages
    """

    citations: Optional[List[Dict[str, Any]]]
    text: str
    type: Literal["text"]


class AnthropicResponseToolUseBlock(TypedDict, total=False):
    """
    Anthropic Response Tool Use Block: https://docs.anthropic.com/en/api/messages
    """

    id: Optional[str]
    input: Optional[str]
    name: Optional[str]
    type: Literal["tool_use"]


class AnthropicResponseThinkingBlock(TypedDict, total=False):
    """
    Anthropic Response Thinking Block: https://docs.anthropic.com/en/api/messages
    """

    signature: Optional[str]
    thinking: Optional[str]
    type: Literal["thinking"]


class AnthropicResponseRedactedThinkingBlock(TypedDict, total=False):
    """
    Anthropic Response Redacted Thinking Block: https://docs.anthropic.com/en/api/messages
    """

    data: Optional[str]
    type: Literal["redacted_thinking"]


AnthropicResponseContentBlock: TypeAlias = Union[
    AnthropicResponseTextBlock,
    AnthropicResponseToolUseBlock,
    AnthropicResponseThinkingBlock,
    AnthropicResponseRedactedThinkingBlock,
]


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


class AnthropicMessagesResponse(TypedDict, total=False):
    """
    Anthropic Messages API Response: https://docs.anthropic.com/en/api/messages
    """

    content: Optional[
        List[
            Union[
                AnthropicResponseContentBlock,
                AnthropicResponseContentBlockText,
                AnthropicResponseContentBlockToolUse,
            ]
        ]
    ]
    id: str
    model: Optional[str]  # This represents the Model type from Anthropic
    role: Optional[Literal["assistant"]]
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ]
    stop_sequence: Optional[str]
    type: Optional[Literal["message"]]
    usage: Optional[AnthropicUsage]
