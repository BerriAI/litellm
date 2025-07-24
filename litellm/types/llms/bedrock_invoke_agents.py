"""
Type definitions for AWS Bedrock Invoke Agent API responses.

https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_InvokeAgent.html
"""

from typing import Any, Dict, List, Optional, TypedDict, Union


class InvokeAgentEventHeaders(TypedDict, total=False):
    """Headers for AWS Invoke Agent events."""

    event_type: str  # 'trace', 'chunk', etc.
    content_type: str  # 'application/json', etc.
    message_type: str  # 'event', etc.


class InvokeAgentUsage(TypedDict):
    """Token usage information from trace events."""

    inputTokens: int
    outputTokens: int
    model: Optional[str]


class InvokeAgentMetadata(TypedDict, total=False):
    """Metadata from model invocation."""

    clientRequestId: Optional[str]
    endTime: Optional[str]
    startTime: Optional[str]
    totalTimeMs: Optional[int]
    usage: Optional[InvokeAgentUsage]


class InvokeAgentModelInvocationInput(TypedDict, total=False):
    """Model invocation input details."""

    foundationModel: Optional[str]
    inferenceConfiguration: Optional[Dict[str, Any]]
    text: Optional[str]
    traceId: Optional[str]
    type: Optional[str]


class InvokeAgentModelInvocationOutput(TypedDict, total=False):
    """Model invocation output details."""

    metadata: Optional[InvokeAgentMetadata]
    parsedResponse: Optional[Dict[str, Any]]
    rawResponse: Optional[Dict[str, Any]]
    reasoningContent: Optional[Dict[str, Any]]
    traceId: Optional[str]


class InvokeAgentOrchestrationTrace(TypedDict, total=False):
    """Orchestration trace information."""

    modelInvocationInput: Optional[InvokeAgentModelInvocationInput]
    modelInvocationOutput: Optional[InvokeAgentModelInvocationOutput]


class InvokeAgentPreProcessingTrace(TypedDict, total=False):
    """Pre-processing trace information."""

    modelInvocationInput: Optional[InvokeAgentModelInvocationInput]
    modelInvocationOutput: Optional[InvokeAgentModelInvocationOutput]


class InvokeAgentTrace(TypedDict, total=False):
    """Trace information container."""

    orchestrationTrace: Optional[InvokeAgentOrchestrationTrace]
    preProcessingTrace: Optional[InvokeAgentPreProcessingTrace]


class InvokeAgentCallerChain(TypedDict, total=False):
    """Caller chain information."""

    agentAliasArn: str


class InvokeAgentTracePayload(TypedDict, total=False):
    """Payload for trace events."""

    agentAliasId: str
    agentId: str
    agentVersion: str
    callerChain: List[InvokeAgentCallerChain]
    eventTime: str
    sessionId: str
    trace: InvokeAgentTrace


class InvokeAgentChunkPayload(TypedDict, total=False):
    """Payload for chunk events containing response content."""

    bytes: str  # Base64 encoded response content


class InvokeAgentEventPayload(TypedDict, total=False):
    """Union type for different event payload types."""

    # Trace event fields
    agentAliasId: Optional[str]
    agentId: Optional[str]
    agentVersion: Optional[str]
    callerChain: Optional[List[InvokeAgentCallerChain]]
    eventTime: Optional[str]
    sessionId: Optional[str]
    trace: Optional[InvokeAgentTrace]

    # Chunk event fields
    bytes: Optional[str]


class InvokeAgentEvent(TypedDict, total=False):
    """Complete event structure for AWS Invoke Agent responses."""

    headers: InvokeAgentEventHeaders
    payload: Optional[InvokeAgentEventPayload]


# Type aliases for convenience
InvokeAgentEventList = List[InvokeAgentEvent]
InvokeAgentTraceEvent = InvokeAgentEvent  # When headers.event_type == 'trace'
InvokeAgentChunkEvent = InvokeAgentEvent  # When headers.event_type == 'chunk'
