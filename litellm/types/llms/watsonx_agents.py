"""
Type definitions for IBM watsonx.ai Orchestrate Agent API responses.

API Reference: https://developer.watson-orchestrate.ibm.com/apis/orchestrate-agent/chat-with-agents
"""

from typing import Any, Dict, List, Optional, Union

from typing_extensions import TypedDict


class WatsonxAgentMessageContent(TypedDict, total=False):
    """Content structure for agent messages."""

    response_type: Optional[str]  # e.g., "conversational_search"
    json_schema: Optional[Dict[str, Any]]
    ui_schema: Optional[Dict[str, Any]]
    form_data: Optional[Dict[str, Any]]
    id: Optional[str]
    form_operation: Optional[str]
    sub_type: Optional[str]
    event_type: Optional[str]
    dps_payload_id: Optional[str]


class WatsonxAgentMessage(TypedDict):
    """Message structure for watsonx agent requests."""

    role: str
    content: Union[str, List[WatsonxAgentMessageContent]]


class WatsonxAgentAdditionalParameters(TypedDict, total=False):
    """Additional parameters for agent requests."""

    pass  # Can be extended based on specific needs


class WatsonxAgentContext(TypedDict, total=False):
    """Context dictionary for agent requests."""

    pass  # Optional context information


class WatsonxAgentRequestBody(TypedDict):
    """Request body for watsonx agent chat completions."""

    messages: List[WatsonxAgentMessage]
    additional_parameters: WatsonxAgentAdditionalParameters
    context: WatsonxAgentContext
    stream: bool


class WatsonxAgentChoice(TypedDict, total=False):
    """Choice structure in agent response."""

    index: Optional[int]
    message: Optional[Dict[str, Any]]
    finish_reason: Optional[str]


class WatsonxAgentResponse(TypedDict):
    """Response structure from watsonx agent API."""

    id: str
    object: str
    created: int
    model: str
    choices: List[WatsonxAgentChoice]
    thread_id: str


class WatsonxAgentStreamChunk(TypedDict, total=False):
    """Streaming response chunk from watsonx agent API."""

    id: str
    object: str
    created: int
    model: str
    choices: List[WatsonxAgentChoice]
    thread_id: Optional[str]


class WatsonxAgentCredentials(TypedDict):
    """Credentials for watsonx agent authentication."""

    api_key: str
    api_base: str
    token: Optional[str]


class WatsonxAgentParams(TypedDict, total=False):
    """Parameters for watsonx agent API calls."""

    agent_id: str
    thread_id: Optional[str]
