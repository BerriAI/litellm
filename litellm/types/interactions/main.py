"""
Type definitions for Google Interactions API

Based on https://ai.google.dev/api/interactions-api

The Google Interactions API provides endpoints for:
- Create interaction: POST /v1beta/{model}:generateContent (with stored session)
- Get interaction: GET /v1beta/interactions/{interaction_id}
- Delete interaction: DELETE /v1beta/interactions/{interaction_id}
- Cancel interaction: POST /v1beta/interactions/{interaction_id}:cancel
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import PrivateAttr
from typing_extensions import TypedDict

from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject


# ============================================================
# Input Content Types
# ============================================================


class InteractionInputPart(TypedDict, total=False):
    """
    A part of the input content for an interaction.
    
    Based on Google AI API Part schema:
    https://ai.google.dev/api/rest/v1beta/Content#Part
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
    
    Based on Google AI API Content schema:
    https://ai.google.dev/api/rest/v1beta/Content
    """
    role: str  # "user" or "model"
    parts: List[InteractionInputPart]


# Type alias for input - can be a string, single content, or list of contents
InteractionInput = Union[str, InteractionInputContent, List[InteractionInputContent]]


# ============================================================
# Tool Configuration Types
# ============================================================


class FunctionDeclaration(TypedDict, total=False):
    """
    Function declaration for tool use.
    """
    name: str
    description: Optional[str]
    parameters: Optional[Dict[str, Any]]  # JSON Schema


class InteractionTool(TypedDict, total=False):
    """
    Tool configuration for the interaction.
    
    Based on Google AI API Tool schema:
    https://ai.google.dev/api/rest/v1beta/Tool
    """
    function_declarations: Optional[List[FunctionDeclaration]]
    google_search_retrieval: Optional[Dict[str, Any]]
    code_execution: Optional[Dict[str, Any]]


class InteractionToolConfig(TypedDict, total=False):
    """
    Configuration for tool usage in interactions.
    
    Based on Google AI API ToolConfig schema:
    https://ai.google.dev/api/rest/v1beta/ToolConfig
    """
    function_calling_config: Optional[Dict[str, Any]]


# ============================================================
# Generation Configuration Types
# ============================================================


class InteractionGenerationConfig(TypedDict, total=False):
    """
    Generation configuration for the interaction.
    
    Based on Google AI API GenerationConfig schema:
    https://ai.google.dev/api/rest/v1beta/GenerationConfig
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
    
    Based on Google AI API SafetySetting schema:
    https://ai.google.dev/api/rest/v1beta/SafetySetting
    """
    category: str  # HarmCategory enum value
    threshold: str  # HarmBlockThreshold enum value


# ============================================================
# Request Parameter Types
# ============================================================


class InteractionsAPIOptionalRequestParams(TypedDict, total=False):
    """
    Optional request parameters for the Interactions API.
    
    These match the Google AI Interactions API specification.
    """
    # Generation configuration
    generation_config: Optional[InteractionGenerationConfig]
    
    # Safety settings
    safety_settings: Optional[List[InteractionSafetySettings]]
    
    # Tools
    tools: Optional[List[InteractionTool]]
    tool_config: Optional[InteractionToolConfig]
    
    # System instruction
    system_instruction: Optional[InteractionInputContent]
    
    # Caching
    cached_content: Optional[str]
    
    # Streaming
    stream: Optional[bool]
    
    # Session/Interaction management
    interaction_id: Optional[str]  # For continuing an interaction


class InteractionsAPIRequestParams(InteractionsAPIOptionalRequestParams, total=False):
    """
    Full request parameters for the Interactions API.
    """
    model: str  # Required: The model to use
    contents: List[InteractionInputContent]  # Required: The conversation contents


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
# Response Types - Content Parts
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
# Response Types - Candidate
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
# Response Types - Usage Metadata
# ============================================================


class InteractionUsageMetadata(BaseLiteLLMOpenAIResponseObject):
    """
    Usage metadata for the interaction.
    
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
# Main Response Types
# ============================================================


class InteractionsAPIResponse(BaseLiteLLMOpenAIResponseObject):
    """
    Response from the Google Interactions API.
    
    Based on Google AI API GenerateContentResponse and Interaction schemas.
    https://ai.google.dev/api/rest/v1beta/models/generateContent
    """
    # Interaction metadata
    name: Optional[str] = None  # Interaction resource name (interactions/{id})
    interaction_id: Optional[str] = None  # Extracted interaction ID
    model: Optional[str] = None
    
    # Response content
    candidates: Optional[List[InteractionCandidate]] = None
    prompt_feedback: Optional[InteractionPromptFeedback] = None
    usage_metadata: Optional[InteractionUsageMetadata] = None
    
    # Status
    state: Optional[str] = None  # ACTIVE, COMPLETED, CANCELLED, FAILED
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    
    # Model version
    model_version: Optional[str] = None
    
    # Hidden params for LiteLLM internal use
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class InteractionsAPIStreamingResponse(BaseLiteLLMOpenAIResponseObject):
    """
    Streaming response chunk from the Google Interactions API.
    """
    # Interaction metadata
    name: Optional[str] = None
    interaction_id: Optional[str] = None
    model: Optional[str] = None
    
    # Response content
    candidates: Optional[List[InteractionCandidate]] = None
    prompt_feedback: Optional[InteractionPromptFeedback] = None
    usage_metadata: Optional[InteractionUsageMetadata] = None
    
    # Streaming metadata
    is_final: Optional[bool] = None
    
    # Hidden params
    _hidden_params: dict = PrivateAttr(default_factory=dict)


# ============================================================
# Delete/Cancel Result Types
# ============================================================


class DeleteInteractionResult(BaseLiteLLMOpenAIResponseObject):
    """
    Result of deleting an interaction.
    
    Google AI API returns an empty response on successful delete.
    """
    success: bool = True
    interaction_id: Optional[str] = None
    
    # Hidden params
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class CancelInteractionResult(BaseLiteLLMOpenAIResponseObject):
    """
    Result of cancelling an interaction.
    
    Returns the interaction resource with updated state.
    """
    name: Optional[str] = None
    interaction_id: Optional[str] = None
    state: Optional[str] = None  # Should be CANCELLED
    
    # Hidden params
    _hidden_params: dict = PrivateAttr(default_factory=dict)
