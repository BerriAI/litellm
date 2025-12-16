"""
Type definitions for Google Interactions API

Based on https://ai.google.dev/static/api/interactions.openapi.json

The Google Interactions API provides endpoints for:
- Create interaction: POST /{api_version}/interactions
- Get interaction: GET /{api_version}/interactions/{interaction_id}
- Delete interaction: DELETE /{api_version}/interactions/{interaction_id}
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import PrivateAttr
from typing_extensions import TypedDict

from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject

# ============================================================
# Input Types (per OpenAPI spec)
# ============================================================

class InteractionPart(TypedDict, total=False):
    """Part of content - text, inline_data, function_call, etc."""
    text: Optional[str]
    inline_data: Optional[Dict[str, Any]]
    file_data: Optional[Dict[str, Any]]
    function_call: Optional[Dict[str, Any]]
    function_response: Optional[Dict[str, Any]]


class InteractionContent(TypedDict, total=False):
    """Content with role and parts."""
    role: str
    parts: List[InteractionPart]


class InteractionTurn(TypedDict, total=False):
    """A turn in conversation."""
    role: str
    parts: List[InteractionPart]


# Input can be string, Content, Content[], or Turn[]
InteractionInput = Union[str, InteractionContent, List[InteractionContent], List[InteractionTurn]]


# ============================================================
# Tool Types (per OpenAPI spec)
# ============================================================

class InteractionFunction(TypedDict, total=False):
    """Function tool."""
    type: Literal["function"]
    name: str
    description: Optional[str]
    parameters: Optional[Dict[str, Any]]


class InteractionGoogleSearch(TypedDict, total=False):
    """Google Search tool."""
    type: Literal["google_search"]


class InteractionCodeExecution(TypedDict, total=False):
    """Code Execution tool."""
    type: Literal["code_execution"]


class InteractionUrlContext(TypedDict, total=False):
    """URL Context tool."""
    type: Literal["url_context"]


class InteractionComputerUse(TypedDict, total=False):
    """Computer Use tool."""
    type: Literal["computer_use"]


class InteractionMcpServer(TypedDict, total=False):
    """MCP Server tool."""
    type: Literal["mcp_server"]
    url: str
    allowed_tools: Optional[List[str]]


class InteractionFileSearch(TypedDict, total=False):
    """File Search tool."""
    type: Literal["file_search"]


InteractionTool = Union[
    InteractionFunction,
    InteractionGoogleSearch,
    InteractionCodeExecution,
    InteractionUrlContext,
    InteractionComputerUse,
    InteractionMcpServer,
    InteractionFileSearch,
    Dict[str, Any],
]


class InteractionToolChoiceConfig(TypedDict, total=False):
    """Tool choice configuration."""
    allowed_tools: Optional[Dict[str, Any]]


class InteractionToolConfig(TypedDict, total=False):
    """Legacy tool config."""
    function_calling_config: Optional[Dict[str, Any]]


# ============================================================
# Config Types (per OpenAPI spec)
# ============================================================

class InteractionGenerationConfig(TypedDict, total=False):
    """Generation configuration."""
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    max_output_tokens: Optional[int]
    stop_sequences: Optional[List[str]]


class InteractionSafetySettings(TypedDict, total=False):
    """Safety settings."""
    category: str
    threshold: str


class InteractionAgentConfig(TypedDict, total=False):
    """Agent configuration."""
    type: str
    tools: Optional[List[InteractionTool]]
    tool_choice: Optional[InteractionToolChoiceConfig]
    instructions: Optional[str]


class InteractionInputContent(TypedDict, total=False):
    """System instruction content."""
    role: str
    parts: List[InteractionPart]


# ============================================================
# Request Types
# ============================================================

class InteractionsAPIOptionalRequestParams(TypedDict, total=False):
    """Optional request parameters."""
    tools: Optional[List[InteractionTool]]
    tool_choice: Optional[InteractionToolChoiceConfig]
    tool_config: Optional[InteractionToolConfig]
    instructions: Optional[str]
    agent_config: Optional[Dict[str, Any]]
    stream: Optional[bool]


# ============================================================
# Response Types (per OpenAPI spec)
# ============================================================

class InteractionOutputText(BaseLiteLLMOpenAIResponseObject):
    """Text output."""
    type: Literal["text"] = "text"
    text: Optional[str] = None


class InteractionOutputFunctionCall(BaseLiteLLMOpenAIResponseObject):
    """Function call output."""
    type: Literal["function_call"] = "function_call"
    function_call: Optional[Dict[str, Any]] = None


InteractionOutput = Union[InteractionOutputText, InteractionOutputFunctionCall, Dict[str, Any]]


class InteractionInputTokensByModality(BaseLiteLLMOpenAIResponseObject):
    """Input tokens by modality."""
    modality: Optional[str] = None
    tokens: Optional[int] = None


class InteractionUsage(BaseLiteLLMOpenAIResponseObject):
    """Usage information per OpenAPI spec."""
    input_tokens_by_modality: Optional[List[Dict[str, Any]]] = None
    total_cached_tokens: Optional[int] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    total_reasoning_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    total_tool_use_tokens: Optional[int] = None


# ============================================================
# Main Response Types (per OpenAPI spec)
# ============================================================

class InteractionsAPIResponse(BaseLiteLLMOpenAIResponseObject):
    """
    Response from the Google Interactions API per OpenAPI spec.
    """
    # Per OpenAPI spec
    id: Optional[str] = None
    object: Optional[str] = "interaction"
    created: Optional[str] = None
    updated: Optional[str] = None
    model: Optional[str] = None
    agent: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    outputs: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None
    
    # Hidden params for LiteLLM internal use
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class InteractionsAPIStreamingResponse(BaseLiteLLMOpenAIResponseObject):
    """
    Streaming response chunk per OpenAPI spec.
    
    Event types: interaction.start, interaction.status_update, 
    interaction.complete, content.start, content.delta, content.stop, error
    """
    event_type: Optional[str] = None
    id: Optional[str] = None
    object: Optional[str] = "interaction"
    created: Optional[str] = None
    updated: Optional[str] = None
    model: Optional[str] = None
    agent: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    outputs: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None
    delta: Optional[Dict[str, Any]] = None
    
    _hidden_params: dict = PrivateAttr(default_factory=dict)


# ============================================================
# Delete/Cancel Result Types
# ============================================================

class DeleteInteractionResult(BaseLiteLLMOpenAIResponseObject):
    """Result of deleting an interaction."""
    success: bool = True
    id: Optional[str] = None
    
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class CancelInteractionResult(BaseLiteLLMOpenAIResponseObject):
    """Result of cancelling an interaction."""
    id: Optional[str] = None
    status: Optional[str] = None
    
    _hidden_params: dict = PrivateAttr(default_factory=dict)
