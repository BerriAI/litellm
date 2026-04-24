from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import TypeAlias, TypedDict

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


class AnthropicUsageIteration(TypedDict, total=False):
    """
    Per-iteration token usage for advisor-orchestrated requests.

    Emitted in ``AnthropicUsage.iterations`` when LiteLLM runs the advisor
    orchestration loop so callers can see the breakdown of executor vs.
    advisor sub-calls.
    """

    type: Literal["message", "advisor_message"]
    model: Optional[str]
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


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
    Per-iteration breakdown for advisor-orchestrated requests.

    Populated by LiteLLM's advisor orchestration loop; absent for normal
    (non-advisor) requests. Each entry describes a single executor or advisor
    sub-call that contributed to the aggregated usage above.
    """
    iterations: List[AnthropicUsageIteration]


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
