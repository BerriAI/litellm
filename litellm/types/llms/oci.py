from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

OCIRoles = Literal["SYSTEM", "USER", "ASSISTANT", "TOOL"]


class OCIVendors(Enum):
    """
    A class to hold the vendor names for OCI models.
    This is used to map model names to their respective vendors.
    """

    COHERE = "COHERE"
    GEMINI = "GEMINI"
    GENERIC = "GENERIC"


# --- Base Models and Content Parts ---


class OCIContentPart(BaseModel):
    """Base model for content parts in an OCI message."""

    pass


class OCITextContentPart(OCIContentPart):
    """Text content part for the OCI API."""

    type: Literal["TEXT"] = "TEXT"
    text: str


class OCIImageContentPart(OCIContentPart):
    """Image content part for the OCI API."""

    type: Literal["IMAGE"] = "IMAGE"
    imageUrl: str


OCIContentPartUnion = Union[OCITextContentPart, OCIImageContentPart]

# --- Models for Tools and Tool Calls ---


class OCIToolCall(BaseModel):
    """Represents a tool call made by the model."""

    id: str
    type: Literal["FUNCTION"] = "FUNCTION"
    name: str
    arguments: str  # Arguments should be a JSON-serialized string


class OCIToolDefinition(BaseModel):
    """Defines a tool that can be used by the model."""

    type: Literal["FUNCTION"] = "FUNCTION"
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[dict] = None


# --- Message Models (Request and Response) ---


class OCIMessage(BaseModel):
    """Model for a single message in the request/response payload."""

    role: OCIRoles
    content: Optional[List[OCIContentPartUnion]] = None
    toolCalls: Optional[List[OCIToolCall]] = None
    toolCallId: Optional[str] = None


# --- Request Payload Models ---


class OCIChatRequestPayload(BaseModel):
    """Internal 'chatRequest' payload for the OCI API."""

    apiFormat: str
    messages: List[OCIMessage]
    tools: Optional[List[OCIToolDefinition]] = None
    isStream: bool = False
    numGenerations: Optional[int] = None
    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    frequencyPenalty: Optional[float] = None
    presencePenalty: Optional[float] = None


class OCIServingMode(BaseModel):
    """Defines the serving mode and the model to be used."""

    servingType: str
    endpointId: Optional[str] = None
    modelId: Optional[str] = None

class OCICompletionPayload(BaseModel):
    """Pydantic model for the complete OCI chat request body."""

    compartmentId: str
    servingMode: OCIServingMode
    chatRequest: Union[OCIChatRequestPayload, CohereChatRequest]


# --- API Response Models (Non-streaming) ---


class OCICompletionTokenDetails(BaseModel):
    """Completion token details in the OCI response."""

    acceptedPredictionTokens: int
    reasoningTokens: int


class OCIPromptTokensDetails(BaseModel):
    """Prompt token details in the OCI response."""

    cachedTokens: int


class OCIResponseUsage(BaseModel):
    """Token usage in the OCI response."""

    promptTokens: int
    completionTokens: int
    totalTokens: int
    completionTokensDetails: Optional[OCICompletionTokenDetails] = None
    promptTokensDetails: Optional[OCIPromptTokensDetails] = None


class OCIResponseChoice(BaseModel):
    """A completion choice in the OCI response."""

    index: int
    message: OCIMessage
    finishReason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class OCIChatResponse(BaseModel):
    """The 'chatResponse' object in the OCI response."""

    apiFormat: str
    timeCreated: str
    choices: List[OCIResponseChoice]
    usage: OCIResponseUsage


class OCICompletionResponse(BaseModel):
    """Model for the complete non-streaming OCI response body."""

    modelId: str
    modelVersion: str
    chatResponse: OCIChatResponse


# --- API Response Models (Streaming) ---


class OCIStreamDelta(BaseModel):
    """The content delta in a streaming chunk."""

    content: Optional[List[OCIContentPartUnion]] = None
    role: Optional[str] = None
    toolCalls: Optional[List[OCIToolCall]] = None


class OCIStreamChunk(BaseModel):
    """Model for a single SSE event chunk from OCI."""

    finishReason: Optional[str] = None
    message: Optional[OCIStreamDelta] = None
    pad: Optional[str] = None
    index: Optional[int] = None


# --- Cohere-Specific Models ---

class CohereStreamChunk(BaseModel):
    """Model for a single SSE event chunk from OCI Cohere API."""

    apiFormat: str
    text: Optional[str] = None
    chatHistory: Optional[List[CohereMessage]] = None
    finishReason: Optional[str] = None
    pad: Optional[str] = None
    index: Optional[int] = None

class CohereMessage(BaseModel):
    """Base model for Cohere messages."""
    
    role: str
    message: str
    toolCalls: Optional[List[CohereToolCall]] = None


class CohereUserMessage(CohereMessage):
    """User message in Cohere chat."""
    
    role: Literal["USER"] = "USER"


class CohereChatBotMessage(CohereMessage):
    """Chatbot message in Cohere chat."""
    
    role: Literal["CHATBOT"] = "CHATBOT"


class CohereSystemMessage(CohereMessage):
    """System message in Cohere chat."""
    
    role: Literal["SYSTEM"] = "SYSTEM"


class CohereToolMessage(CohereMessage):
    """Tool message in Cohere chat."""
    
    role: Literal["TOOL"] = "TOOL"
    toolCallId: str


class CohereParameterDefinition(BaseModel):
    """Parameter definition for Cohere tools."""
    
    description: str
    type: str
    isRequired: bool = False


class CohereTool(BaseModel):
    """Tool definition for Cohere."""
    
    name: str
    description: str
    parameterDefinitions: Dict[str, CohereParameterDefinition]


class CohereToolCall(BaseModel):
    """Tool call made by Cohere model."""
    
    name: str
    parameters: Dict[str, Any]


class CohereToolResult(BaseModel):
    """Result of a tool call."""
    
    callId: str
    result: str


class CohereResponseFormat(BaseModel):
    """Response format for Cohere."""
    
    type: str


class CohereResponseTextFormat(CohereResponseFormat):
    """Text response format for Cohere."""
    
    type: Literal["text"] = "text"



class CohereChatRequest(BaseModel):
    """Cohere chat request model."""
    
    # Required fields
    message: str
    apiFormat: Literal["COHERE"] = "COHERE"
    
    # Optional fields
    chatHistory: Optional[List[CohereMessage]] = None
    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    topK: Optional[int] = None
    frequencyPenalty: Optional[float] = None
    presencePenalty: Optional[float] = None
    stopSequences: Optional[List[str]] = None
    seed: Optional[int] = None
    tools: Optional[List[CohereTool]] = None
    toolChoice: Optional[Union[str, Dict[str, Any]]] = None
    responseFormat: Optional[CohereResponseFormat] = None
    preambleOverride: Optional[str] = None
    documents: Optional[List[Dict[str, Any]]] = None
    searchQueriesOnly: Optional[bool] = None
    searchEntryPoint: Optional[str] = None
    grounding: Optional[Dict[str, Any]] = None
    isEcho: Optional[bool] = None
    isSearchQueriesOnly: Optional[bool] = None
    isRawPrompting: Optional[bool] = None
    isForceSingleStep: Optional[bool] = None
    promptTruncation: Optional[str] = None
    safetyMode: Optional[str] = None
    citationQuality: Optional[str] = None
    maxInputTokens: Optional[int] = None
    isStream: Optional[bool] = None
    streamOptions: Optional[Dict[str, Any]] = None


class CohereUsage(BaseModel):
    """Usage information for Cohere response."""
    
    promptTokens: int
    completionTokens: int
    totalTokens: int
    promptTokensDetails: Optional[Dict[str, Any]] = None
    completionTokensDetails: Optional[Dict[str, Any]] = None


class CohereCitation(BaseModel):
    """Citation in Cohere response."""
    
    start: int
    end: int
    text: str
    document_ids: List[str]


class CohereSearchQuery(BaseModel):
    """Search query generated by Cohere."""
    
    text: str
    generation_id: str


class CohereChatResponse(BaseModel):
    """Cohere chat response model."""
    
    # Required fields
    text: str
    apiFormat: Literal["COHERE"] = "COHERE"
    finishReason: Literal["COMPLETE", "ERROR_TOXIC", "ERROR_LIMIT", "ERROR", "USER_CANCEL", "MAX_TOKENS"]
    
    # Optional fields
    chatHistory: Optional[List[CohereMessage]] = None
    citations: Optional[List[CohereCitation]] = None
    documents: Optional[List[Dict[str, Any]]] = None
    errorMessage: Optional[str] = None
    isSearchRequired: Optional[bool] = None
    prompt: Optional[str] = None
    searchQueries: Optional[List[CohereSearchQuery]] = None
    toolCalls: Optional[List[CohereToolCall]] = None
    usage: Optional[CohereUsage] = None


class CohereChatDetails(BaseModel):
    """Chat details for Cohere request."""
    
    compartmentId: str
    servingMode: OCIServingMode
    chatRequest: CohereChatRequest


class CohereChatResult(BaseModel):
    """Complete Cohere chat result."""
    
    modelId: str
    modelVersion: str
    chatResponse: CohereChatResponse

