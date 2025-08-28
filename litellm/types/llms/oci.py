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
    modelId: str


class OCICompletionPayload(BaseModel):
    """Pydantic model for the complete OCI chat request body."""

    compartmentId: str
    servingMode: OCIServingMode
    chatRequest: OCIChatRequestPayload


# --- API Response Models (Non-streaming) ---


class OCICompletionTokenDetails(BaseModel):
    """Completion token details in the OCI response."""

    acceptedPredictionTokens: int
    reasoningTokens: int


class OCIPropmtTokensDetails(BaseModel):
    """Prompt token details in the OCI response."""

    cachedTokens: int


class OCIResponseUsage(BaseModel):
    """Token usage in the OCI response."""

    promptTokens: int
    completionTokens: int
    totalTokens: int
    completionTokensDetails: OCICompletionTokenDetails
    promptTokensDetails: OCIPropmtTokensDetails


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
