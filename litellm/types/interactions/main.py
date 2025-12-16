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
# Input Content Types (per OpenAPI spec)
# ============================================================


class InteractionInputPart(TypedDict, total=False):
    """
    A part of the input content for an interaction.
    
    Based on Google AI API Part schema.
    """
    text: Optional[str]
    inline_data: Optional[Dict[str, Any]]  # For blob/media data
    file_data: Optional[Dict[str, Any]]  # For file references
    function_call: Optional[Dict[str, Any]]
    function_response: Optional[Dict[str, Any]]
    executable_code: Optional[Dict[str, Any]]
    code_execution_result: Optional[Dict[str, Any]]


class InteractionInputContent(TypedDict, total=False):
    """
    Content for an interaction - represents a message in the conversation.
    
    Based on Google AI API Content schema.
    """
    role: str  # "user" or "model"
    parts: List[InteractionInputPart]


class InteractionTurn(TypedDict, total=False):
    """
    A turn in a multi-turn conversation.
    
    Per OpenAPI spec: Turn object with role and parts.
    """
    role: str
    parts: List[InteractionInputPart]


# Type alias for input - can be a string, single content, list of contents, or list of turns
InteractionInput = Union[str, InteractionInputContent, List[InteractionInputContent], List[InteractionTurn]]


# ============================================================
# Tool Configuration Types (per OpenAPI spec)
# ============================================================


class FunctionDeclaration(TypedDict, total=False):
    """
    Function declaration for tool use.
    """
    name: str
    description: Optional[str]
    parameters: Optional[Dict[str, Any]]  # JSON Schema


class InteractionFunction(TypedDict, total=False):
    """
    Function tool type per OpenAPI spec.
    """
    type: Literal["function"]
    name: str
    description: Optional[str]
    parameters: Optional[Dict[str, Any]]


class InteractionGoogleSearch(TypedDict, total=False):
    """
    Google Search tool type per OpenAPI spec.
    """
    type: Literal["google_search"]


class InteractionCodeExecution(TypedDict, total=False):
    """
    Code Execution tool type per OpenAPI spec.
    """
    type: Literal["code_execution"]


class InteractionUrlContext(TypedDict, total=False):
    """
    URL Context tool type per OpenAPI spec.
    """
    type: Literal["url_context"]


class InteractionComputerUse(TypedDict, total=False):
    """
    Computer Use tool type per OpenAPI spec.
    """
    type: Literal["computer_use"]


class InteractionMcpServer(TypedDict, total=False):
    """
    MCP Server tool type per OpenAPI spec.
    """
    type: Literal["mcp_server"]
    url: str
    allowed_tools: Optional[List[str]]


class InteractionFileSearch(TypedDict, total=False):
    """
    File Search tool type per OpenAPI spec.
    """
    type: Literal["file_search"]


# Union of all tool types per OpenAPI spec
InteractionTool = Union[
    InteractionFunction,
    InteractionGoogleSearch,
    InteractionCodeExecution,
    InteractionUrlContext,
    InteractionComputerUse,
    InteractionMcpServer,
    InteractionFileSearch,
    Dict[str, Any],  # For backwards compatibility
]


class InteractionAllowedTools(TypedDict, total=False):
    """
    Allowed tools configuration per OpenAPI spec.
    """
    tool_names: Optional[List[str]]


class InteractionToolChoiceConfig(TypedDict, total=False):
    """
    Tool choice configuration per OpenAPI spec.
    """
    allowed_tools: Optional[InteractionAllowedTools]


class InteractionToolConfig(TypedDict, total=False):
    """
    Configuration for tool usage in interactions.
    
    Based on Google AI API ToolConfig schema.
    """
    function_calling_config: Optional[Dict[str, Any]]


# ============================================================
# Generation Configuration Types
# ============================================================


class InteractionGenerationConfig(TypedDict, total=False):
    """
    Generation configuration for the interaction.
    
    Based on Google AI API GenerationConfig schema.
    """
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    candidate_count: Optional[int]
    max_output_tokens: Optional[int]
    stop_sequences: Optional[List[str]]
    response_mime_type: Optional[str]
    response_schema: Optional[Dict[str, Any]]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    response_logprobs: Optional[bool]
    logprobs: Optional[int]


# ============================================================
# Safety Settings Types
# ============================================================


class InteractionSafetySettings(TypedDict, total=False):
    """
    Safety settings for the interaction.
    
    Based on Google AI API SafetySetting schema.
    """
    category: str  # HarmCategory enum value
    threshold: str  # HarmBlockThreshold enum value


# ============================================================
# Agent Configuration Types (per OpenAPI spec)
# ============================================================


class InteractionDynamicAgentConfig(TypedDict, total=False):
    """
    Dynamic agent configuration per OpenAPI spec.
    """
    type: Literal["dynamic"]
    tools: Optional[List[InteractionTool]]
    tool_choice: Optional[InteractionToolChoiceConfig]
    instructions: Optional[str]


class InteractionDeepResearchAgentConfig(TypedDict, total=False):
    """
    Deep Research agent configuration per OpenAPI spec.
    """
    type: Literal["deep_research"]


InteractionAgentConfig = Union[InteractionDynamicAgentConfig, InteractionDeepResearchAgentConfig]


# ============================================================
# Request Parameter Types (per OpenAPI spec)
# ============================================================


class InteractionsAPIOptionalRequestParams(TypedDict, total=False):
    """
    Optional request parameters for the Interactions API.
    
    These match the Google AI Interactions API OpenAPI specification.
    """
    # Generation configuration
    generation_config: Optional[InteractionGenerationConfig]
    
    # Safety settings
    safety_settings: Optional[List[InteractionSafetySettings]]
    
    # Tools (for model interactions)
    tools: Optional[List[InteractionTool]]
    tool_config: Optional[InteractionToolConfig]
    tool_choice: Optional[InteractionToolChoiceConfig]
    
    # System instruction
    system_instruction: Optional[InteractionInputContent]
    instructions: Optional[str]  # Alternative to system_instruction
    
    # Caching
    cached_content: Optional[str]
    
    # Streaming
    stream: Optional[bool]
    
    # Agent configuration (for agent interactions)
    agent_config: Optional[InteractionAgentConfig]


class CreateModelInteractionParams(TypedDict, total=False):
    """
    Parameters for creating a model interaction per OpenAPI spec.
    
    POST /{api_version}/interactions with model parameter.
    """
    model: str  # Required: The model to use (e.g., "gemini-2.5-flash")
    input: InteractionInput  # Required: The input content
    tools: Optional[List[InteractionTool]]
    tool_choice: Optional[InteractionToolChoiceConfig]
    instructions: Optional[str]
    generation_config: Optional[InteractionGenerationConfig]
    safety_settings: Optional[List[InteractionSafetySettings]]


class CreateAgentInteractionParams(TypedDict, total=False):
    """
    Parameters for creating an agent interaction per OpenAPI spec.
    
    POST /{api_version}/interactions with agent parameter.
    """
    agent: str  # Required: The agent to use (e.g., "deep-research-pro-preview-12-2025")
    input: InteractionInput  # Required: The input content
    agent_config: Optional[InteractionAgentConfig]


# ============================================================
# Response Types - Output Content (per OpenAPI spec)
# ============================================================


class InteractionOutputText(BaseLiteLLMOpenAIResponseObject):
    """
    Text output from an interaction per OpenAPI spec.
    """
    type: Literal["text"] = "text"
    text: Optional[str] = None


class InteractionOutputFunctionCall(BaseLiteLLMOpenAIResponseObject):
    """
    Function call output from an interaction per OpenAPI spec.
    """
    type: Literal["function_call"] = "function_call"
    function_call: Optional[Dict[str, Any]] = None


InteractionOutput = Union[InteractionOutputText, InteractionOutputFunctionCall, Dict[str, Any]]


# ============================================================
# Response Types - Usage (per OpenAPI spec)
# ============================================================


class InteractionInputTokensByModality(BaseLiteLLMOpenAIResponseObject):
    """
    Input tokens broken down by modality per OpenAPI spec.
    """
    modality: Optional[str] = None  # "text", "image", "audio", "video"
    tokens: Optional[int] = None


class InteractionUsage(BaseLiteLLMOpenAIResponseObject):
    """
    Usage information for an interaction per OpenAPI spec.
    """
    input_tokens_by_modality: Optional[List[InteractionInputTokensByModality]] = None
    total_cached_tokens: Optional[int] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    total_reasoning_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    total_tool_use_tokens: Optional[int] = None


# ============================================================
# Response Types - Safety Ratings
# ============================================================


class InteractionSafetyRating(BaseLiteLLMOpenAIResponseObject):
    """
    Safety rating for content in the response.
    
    Based on Google AI API SafetyRating schema.
    """
    category: Optional[str] = None
    probability: Optional[str] = None
    blocked: Optional[bool] = None


# ============================================================
# Response Types - Citation Metadata
# ============================================================


class InteractionCitationSource(BaseLiteLLMOpenAIResponseObject):
    """
    Citation source information.
    """
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    uri: Optional[str] = None
    license: Optional[str] = None


class InteractionCitationMetadata(BaseLiteLLMOpenAIResponseObject):
    """
    Citation metadata for the response.
    """
    citation_sources: Optional[List[InteractionCitationSource]] = None


# ============================================================
# Response Types - Grounding Metadata
# ============================================================


class InteractionGroundingMetadata(BaseLiteLLMOpenAIResponseObject):
    """
    Grounding metadata for search-augmented responses.
    """
    grounding_chunks: Optional[List[Dict[str, Any]]] = None
    grounding_supports: Optional[List[Dict[str, Any]]] = None
    web_search_queries: Optional[List[str]] = None
    search_entry_point: Optional[Dict[str, Any]] = None
    retrieval_metadata: Optional[Dict[str, Any]] = None


# ============================================================
# Response Types - Content Parts (legacy format)
# ============================================================


class InteractionPart(BaseLiteLLMOpenAIResponseObject):
    """
    A part of the response content.
    
    Based on Google AI API Part schema.
    """
    text: Optional[str] = None
    inline_data: Optional[Dict[str, Any]] = None
    file_data: Optional[Dict[str, Any]] = None
    function_call: Optional[Dict[str, Any]] = None
    function_response: Optional[Dict[str, Any]] = None
    executable_code: Optional[Dict[str, Any]] = None
    code_execution_result: Optional[Dict[str, Any]] = None


class InteractionContent(BaseLiteLLMOpenAIResponseObject):
    """
    Content in the response.
    
    Based on Google AI API Content schema.
    """
    role: Optional[str] = None
    parts: Optional[List[InteractionPart]] = None


# ============================================================
# Response Types - Candidate (legacy format)
# ============================================================


class InteractionCandidate(BaseLiteLLMOpenAIResponseObject):
    """
    A candidate response from the model.
    
    Based on Google AI API Candidate schema.
    """
    content: Optional[InteractionContent] = None
    finish_reason: Optional[str] = None
    safety_ratings: Optional[List[InteractionSafetyRating]] = None
    citation_metadata: Optional[InteractionCitationMetadata] = None
    grounding_metadata: Optional[InteractionGroundingMetadata] = None
    token_count: Optional[int] = None
    index: Optional[int] = None
    avg_logprobs: Optional[float] = None
    logprobs_result: Optional[Dict[str, Any]] = None


# ============================================================
# Response Types - Usage Metadata (legacy format)
# ============================================================


class InteractionUsageMetadata(BaseLiteLLMOpenAIResponseObject):
    """
    Usage metadata for the interaction (legacy format).
    
    Based on Google AI API UsageMetadata schema.
    """
    prompt_token_count: Optional[int] = None
    candidates_token_count: Optional[int] = None
    total_token_count: Optional[int] = None
    cached_content_token_count: Optional[int] = None


# ============================================================
# Response Types - Prompt Feedback
# ============================================================


class InteractionPromptFeedback(BaseLiteLLMOpenAIResponseObject):
    """
    Feedback about the prompt.
    
    Based on Google AI API PromptFeedback schema.
    """
    block_reason: Optional[str] = None
    safety_ratings: Optional[List[InteractionSafetyRating]] = None


# ============================================================
# Main Response Types (per OpenAPI spec)
# ============================================================


class InteractionsAPIResponse(BaseLiteLLMOpenAIResponseObject):
    """
    Response from the Google Interactions API per OpenAPI spec.
    
    Based on the Interaction schema from:
    https://ai.google.dev/static/api/interactions.openapi.json
    """
    # Interaction metadata (per OpenAPI spec)
    id: Optional[str] = None  # Interaction ID
    object: Literal["interaction"] = "interaction"
    created: Optional[str] = None  # ISO 8601 timestamp
    updated: Optional[str] = None  # ISO 8601 timestamp
    
    # Model or Agent
    model: Optional[str] = None
    agent: Optional[str] = None
    
    # Role
    role: Optional[str] = None  # "model"
    
    # Status per OpenAPI spec
    status: Optional[str] = None  # "completed", "requires_action", "in_progress", "failed"
    
    # Outputs per OpenAPI spec (new format)
    outputs: Optional[List[InteractionOutput]] = None
    
    # Usage per OpenAPI spec (new format)
    usage: Optional[InteractionUsage] = None
    
    # Legacy fields for backwards compatibility
    name: Optional[str] = None  # Interaction resource name (interactions/{id})
    interaction_id: Optional[str] = None  # Extracted interaction ID (deprecated, use 'id')
    candidates: Optional[List[InteractionCandidate]] = None
    prompt_feedback: Optional[InteractionPromptFeedback] = None
    usage_metadata: Optional[InteractionUsageMetadata] = None
    state: Optional[str] = None  # Legacy: ACTIVE, COMPLETED, CANCELLED, FAILED
    create_time: Optional[str] = None  # Legacy timestamp format
    update_time: Optional[str] = None  # Legacy timestamp format
    model_version: Optional[str] = None
    
    # Hidden params for LiteLLM internal use
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class InteractionsAPIStreamingResponse(BaseLiteLLMOpenAIResponseObject):
    """
    Streaming response chunk from the Google Interactions API.
    
    Per OpenAPI spec, streaming uses SSE with event types:
    - interaction.start
    - interaction.status_update
    - interaction.complete
    - content.start
    - content.delta
    - content.stop
    - error
    """
    # Event type
    event_type: Optional[str] = None
    
    # Interaction metadata
    id: Optional[str] = None
    object: Literal["interaction"] = "interaction"
    created: Optional[str] = None
    updated: Optional[str] = None
    
    # Model or Agent
    model: Optional[str] = None
    agent: Optional[str] = None
    
    # Role
    role: Optional[str] = None
    
    # Status
    status: Optional[str] = None
    
    # Outputs (for content events)
    outputs: Optional[List[InteractionOutput]] = None
    
    # Usage
    usage: Optional[InteractionUsage] = None
    
    # Content delta (for content.delta events)
    delta: Optional[Dict[str, Any]] = None
    
    # Legacy fields
    name: Optional[str] = None
    interaction_id: Optional[str] = None
    candidates: Optional[List[InteractionCandidate]] = None
    prompt_feedback: Optional[InteractionPromptFeedback] = None
    usage_metadata: Optional[InteractionUsageMetadata] = None
    is_final: Optional[bool] = None
    
    # Hidden params
    _hidden_params: dict = PrivateAttr(default_factory=dict)


# ============================================================
# Delete Result Type
# ============================================================


class DeleteInteractionResult(BaseLiteLLMOpenAIResponseObject):
    """
    Result of deleting an interaction.
    
    Google AI API returns an empty response on successful delete.
    """
    success: bool = True
    interaction_id: Optional[str] = None
    id: Optional[str] = None  # Alias for interaction_id
    
    # Hidden params
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class CancelInteractionResult(BaseLiteLLMOpenAIResponseObject):
    """
    Result of cancelling an interaction.
    
    Returns the interaction resource with updated state.
    """
    id: Optional[str] = None
    name: Optional[str] = None
    interaction_id: Optional[str] = None  # Legacy
    status: Optional[str] = None  # "cancelled"
    state: Optional[str] = None  # Legacy: Should be CANCELLED
    
    # Hidden params
    _hidden_params: dict = PrivateAttr(default_factory=dict)
