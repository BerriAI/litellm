"""
Type definitions for AWS Bedrock Invoke Agent API responses.

https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_InvokeAgent.html
"""

from typing import Any, Final

from typing_extensions import TypedDict


class InvokeAgentEventHeaders(TypedDict, total=False):
    """Headers for AWS Invoke Agent events."""

    event_type: str  # 'trace', 'chunk', etc.
    content_type: str  # 'application/json', etc.
    message_type: str  # 'event', etc.


class InvokeAgentUsage(TypedDict):
    """Token usage information from trace events."""

    inputTokens: int
    outputTokens: int
    model: str | None


class InvokeAgentMetadata(TypedDict, total=False):
    """Metadata from model invocation."""

    clientRequestId: str | None
    endTime: str | None
    startTime: str | None
    totalTimeMs: int | None
    usage: InvokeAgentUsage | None


class InvokeAgentModelInvocationInput(TypedDict, total=False):
    """Model invocation input details."""

    foundationModel: str | None
    inferenceConfiguration: dict[str, Any] | None
    text: str | None
    traceId: str | None
    type: str | None


class InvokeAgentModelInvocationOutput(TypedDict, total=False):
    """Model invocation output details."""

    metadata: InvokeAgentMetadata | None
    parsedResponse: dict[str, Any] | None
    rawResponse: dict[str, Any] | None
    reasoningContent: dict[str, Any] | None
    traceId: str | None


class InvokeAgentOrchestrationTrace(TypedDict, total=False):
    """Orchestration trace information."""

    modelInvocationInput: InvokeAgentModelInvocationInput | None
    modelInvocationOutput: InvokeAgentModelInvocationOutput | None


class InvokeAgentPreProcessingTrace(TypedDict, total=False):
    """Pre-processing trace information."""

    modelInvocationInput: InvokeAgentModelInvocationInput | None
    modelInvocationOutput: InvokeAgentModelInvocationOutput | None


class InvokeAgentTrace(TypedDict, total=False):
    """Trace information container."""

    orchestrationTrace: InvokeAgentOrchestrationTrace | None
    preProcessingTrace: InvokeAgentPreProcessingTrace | None


class InvokeAgentCallerChain(TypedDict, total=False):
    """Caller chain information."""

    agentAliasArn: str


class InvokeAgentTracePayload(TypedDict, total=False):
    """Payload for trace events."""

    agentAliasId: str
    agentId: str
    agentVersion: str
    callerChain: list[InvokeAgentCallerChain]
    eventTime: str
    sessionId: str
    trace: InvokeAgentTrace


class InvokeAgentChunkPayload(TypedDict, total=False):
    """Payload for chunk events containing response content."""

    bytes: str  # Base64 encoded response content


class InvokeAgentEventPayload(TypedDict, total=False):
    """Union type for different event payload types."""

    # Trace event fields
    agentAliasId: str | None
    agentId: str | None
    agentVersion: str | None
    callerChain: list[InvokeAgentCallerChain] | None
    eventTime: str | None
    sessionId: str | None
    trace: InvokeAgentTrace | None

    # Chunk event fields
    bytes: str | None


class InvokeAgentEvent(TypedDict, total=False):
    """Complete event structure for AWS Invoke Agent responses."""

    headers: InvokeAgentEventHeaders
    payload: InvokeAgentEventPayload | None


# Type aliases for convenience
InvokeAgentEventList = list[InvokeAgentEvent]
InvokeAgentTraceEvent: Final = InvokeAgentEvent  # When headers.event_type == 'trace'
InvokeAgentChunkEvent: Final = InvokeAgentEvent  # When headers.event_type == 'chunk'
