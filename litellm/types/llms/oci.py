from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel
from enum import Enum

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

    type: str


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
    name: str | None = None
    description: str | None = None
    parameters: dict | None = None


# --- Message Models (Request and Response) ---


class OCIMessage(BaseModel):
    """Model for a single message in the request/response payload."""

    role: OCIRoles
    content: list[OCIContentPartUnion] | None = None
    toolCalls: list[OCIToolCall] | None = None
    toolCallId: str | None = None


# --- Request Payload Models ---


class OCIChatRequestPayload(BaseModel):
    """Internal 'chatRequest' payload for the OCI API."""

    apiFormat: str
    messages: list[OCIMessage]
    tools: list[OCIToolDefinition] | None = None
    isStream: bool = False
    numGenerations: int | None = None
    maxTokens: int | None = None
    temperature: float | None = None
    topP: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    frequencyPenalty: float | None = None
    presencePenalty: float | None = None


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
    finishReason: str | None
    logprobs: dict[str, Any] | None = None


class OCIChatResponse(BaseModel):
    """The 'chatResponse' object in the OCI response."""

    apiFormat: str
    timeCreated: str
    choices: list[OCIResponseChoice]
    usage: OCIResponseUsage


class OCICompletionResponse(BaseModel):
    """Model for the complete non-streaming OCI response body."""

    modelId: str
    modelVersion: str
    chatResponse: OCIChatResponse


# --- API Response Models (Streaming) ---


class OCIStreamDelta(BaseModel):
    """The content delta in a streaming chunk."""

    content: list[OCIContentPartUnion] | None = None
    role: str | None = None
    toolCalls: list[OCIToolCall] | None = None


class OCIStreamChunk(BaseModel):
    """Model for a single SSE event chunk from OCI."""

    finishReason: str | None = None
    message: OCIStreamDelta | None = None
    pad: str | None = None
    index: int | None = None
