"""
Type definitions for AWS Bedrock AgentCore API.

https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agentcore_InvokeAgentRuntime.html
"""

from typing import Literal

from typing_extensions import TypedDict


# Request Types
class AgentCoreRequestPayload(TypedDict):
    """Payload for AgentCore request."""

    prompt: str


class AgentCoreRequest(TypedDict, total=False):
    """Complete request structure for AgentCore API (internal use)."""

    payload: str  # JSON-encoded AgentCoreRequestPayload


# Response SSE Event Types
class AgentCoreMessageRole(TypedDict):
    """Message role information."""

    role: Literal["assistant"]


class AgentCoreMessageStart(TypedDict):
    """Message start event."""

    role: Literal["assistant"]


class AgentCoreContentBlockDelta(TypedDict):
    """Content delta information."""

    text: str


class AgentCoreContentBlockDeltaEvent(TypedDict):
    """Content block delta event."""

    delta: AgentCoreContentBlockDelta
    contentBlockIndex: int


class AgentCoreContentBlockStop(TypedDict):
    """Content block stop event."""

    contentBlockIndex: int


class AgentCoreMessageStop(TypedDict):
    """Message stop event."""

    stopReason: Literal["end_turn", "max_tokens", "stop_sequence"]


class AgentCoreUsage(TypedDict):
    """Token usage information."""

    inputTokens: int
    outputTokens: int
    totalTokens: int


class AgentCoreMetrics(TypedDict):
    """Response metrics."""

    latencyMs: int


class AgentCoreMetadata(TypedDict):
    """Metadata event payload."""

    usage: AgentCoreUsage
    metrics: AgentCoreMetrics


class AgentCoreEventPayload(TypedDict, total=False):
    """Union payload for different event types."""

    # messageStart event
    messageStart: AgentCoreMessageStart | None

    # contentBlockDelta event
    contentBlockDelta: AgentCoreContentBlockDeltaEvent | None

    # contentBlockStop event
    contentBlockStop: AgentCoreContentBlockStop | None

    # messageStop event
    messageStop: AgentCoreMessageStop | None

    # metadata event
    metadata: AgentCoreMetadata | None


class AgentCoreEvent(TypedDict, total=False):
    """SSE event structure from AgentCore."""

    event: AgentCoreEventPayload | None


class AgentCoreContentBlock(TypedDict):
    """Content block in final message."""

    text: str


class AgentCoreMessage(TypedDict):
    """Complete message structure."""

    role: Literal["assistant"]
    content: list[AgentCoreContentBlock]


class AgentCoreFinalMessage(TypedDict):
    """Final message event containing complete response."""

    message: AgentCoreMessage


# Response parsing result (internal use)
class AgentCoreParsedResponse(TypedDict):
    """Parsed response from SSE stream."""

    content: str
    usage: AgentCoreUsage | None
    final_message: AgentCoreMessage | None
