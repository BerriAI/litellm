"""
Type definitions for AWS Bedrock AgentCore Runtime API responses.

https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Amazon_Bedrock_Agent_Runtime.html
"""

from typing import Any, Dict, List, Optional, TypedDict


class AgentCoreMetadata(TypedDict, total=False):
    """Metadata from AgentCore agent response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    session_id: Optional[str]
    agent_version: Optional[str]
    custom_metadata: Optional[Dict[str, Any]]


class AgentCoreResponse(TypedDict, total=False):
    """Response from AgentCore agent invocation.

    AgentCore can return either:
    1. Plain string (when using BedrockAgentCoreApp)
    2. Dictionary with response and metadata (legacy format)
    """

    response: str
    metadata: Optional[AgentCoreMetadata]


class AgentCoreStreamChunk(TypedDict, total=False):
    """Streaming chunk from AgentCore SSE stream."""

    token: str
    finish_reason: Optional[str]
    index: int


class AgentCoreMediaItem(TypedDict):
    """Multi-modal media item (image, video, audio, document)."""

    type: str  # "image", "video", "audio", "document"
    format: str  # "jpeg", "png", "mp4", "mp3", "pdf", etc.
    data: str  # Base64-encoded content


class AgentCoreRequestPayload(TypedDict, total=False):
    """Request payload for AgentCore agent invocation."""

    prompt: str
    context: Optional[str]
    media: Optional[AgentCoreMediaItem | List[AgentCoreMediaItem]]
    runtimeSessionId: Optional[str]
    # Additional custom fields can be added


class AgentCoreInvokeParams(TypedDict, total=False):
    """Boto3 invoke parameters for AgentCore Runtime API."""

    agentRuntimeArn: str
    payload: str  # JSON-encoded string
    runtimeSessionId: Optional[str]
    qualifier: str  # Version or endpoint (defaults to "DEFAULT")


# Type aliases for convenience
AgentCoreResponseUnion = AgentCoreResponse | str
AgentCoreMediaList = List[AgentCoreMediaItem]
