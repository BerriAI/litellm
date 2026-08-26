from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, SerializeAsAny

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


class OCITextContentPart(OCIContentPart):
    """Text content part for the OCI API."""

    type: Literal["TEXT"] = "TEXT"
    text: str


class OCIImageUrl(BaseModel):
    """ImageUrl object for OCI API. See: https://docs.oracle.com/en-us/iaas/tools/python/latest/api/generative_ai_inference/models/oci.generative_ai_inference.models.ImageUrl.html"""

    url: str
    detail: Literal["AUTO", "HIGH", "LOW"] | None = None


class OCIImageContentPart(OCIContentPart):
    """Image content part for the OCI API."""

    type: Literal["IMAGE"] = "IMAGE"
    imageUrl: OCIImageUrl


OCIContentPartUnion = OCITextContentPart | OCIImageContentPart

# --- Models for Tools and Tool Calls ---


class OCIToolCall(BaseModel):
    """Represents a tool call made by the model."""

    id: str | None = None  # absent in some provider responses (e.g. Google via OCI)
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
    # GPT-5+ on OCI rejects maxTokens and requires maxCompletionTokens.
    maxCompletionTokens: int | None = None
    temperature: float | None = None
    topP: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    frequencyPenalty: float | None = None
    presencePenalty: float | None = None
    # Reasoning-token budget knob (OCI: NONE/MINIMAL/LOW/MEDIUM/HIGH).
    # Honoured by GPT-5 family, Gemini 2.5, Grok reasoning variants,
    # Cohere Command-A-Reasoning. Ignored by non-reasoning models.
    reasoningEffort: str | None = None
    responseFormat: dict[str, Any] | None = None
    toolChoice: str | dict[str, Any] | None = None
    logitBias: dict[str, Any] | None = None
    logProbs: int | None = None


class OCIServingMode(BaseModel):
    """Defines the serving mode and the model to be used."""

    servingType: str
    endpointId: str | None = None
    modelId: str | None = None


class OCICompletionPayload(BaseModel):
    """Pydantic model for the complete OCI chat request body."""

    compartmentId: str
    servingMode: OCIServingMode
    chatRequest: OCIChatRequestPayload | CohereChatRequest


# --- API Response Models (Non-streaming) ---


class OCICompletionTokenDetails(BaseModel):
    """Completion token details in the OCI response."""

    acceptedPredictionTokens: int | None = None
    reasoningTokens: int | None = None


class OCIPromptTokensDetails(BaseModel):
    """Prompt token details in the OCI response."""

    cachedTokens: int | None = None


class OCIResponseUsage(BaseModel):
    """Token usage in the OCI response."""

    promptTokens: int
    # completionTokens may be absent for reasoning models when all the output
    # budget is consumed by reasoning tokens before any visible content is produced.
    completionTokens: int | None = None
    totalTokens: int
    completionTokensDetails: OCICompletionTokenDetails | None = None
    promptTokensDetails: OCIPromptTokensDetails | None = None


class OCIResponseChoice(BaseModel):
    """A completion choice in the OCI response."""

    index: int
    # message is absent when a reasoning model exhausts max_tokens in the
    # reasoning phase without producing any visible content.
    message: OCIMessage | None = None
    finishReason: str | None = None
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


# --- Cohere-Specific Models ---


class CohereStreamChunk(BaseModel):
    """Model for a single SSE event chunk from OCI Cohere API."""

    apiFormat: str
    text: str | None = None
    chatHistory: list[CohereMessage] | None = None
    finishReason: str | None = None
    toolCalls: list[CohereToolCall] | None = None
    pad: str | None = None
    index: int | None = None


class CohereMessage(BaseModel):
    """Base model for Cohere messages."""

    role: str
    message: str | None = None
    toolCalls: list[CohereToolCall] | None = None


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
    """Tool message in Cohere chat.

    The OCI Cohere API represents tool results via a ``toolResults`` list on the
    TOOL-role history entry — not via a ``toolCallId`` string.
    """

    role: Literal["TOOL"] = "TOOL"
    toolResults: list[CohereToolResult]


class CohereParameterDefinition(BaseModel):
    """Parameter definition for Cohere tools."""

    description: str
    type: str
    isRequired: bool = False


class CohereTool(BaseModel):
    """Tool definition for Cohere."""

    name: str
    description: str
    parameterDefinitions: dict[str, CohereParameterDefinition]


class CohereToolCall(BaseModel):
    """Tool call made by Cohere model."""

    name: str
    parameters: dict[str, Any]


class CohereToolResult(BaseModel):
    """Result of a tool call.

    Matches the OCI SDK's CohereToolResult: each result carries the originating
    tool call (name + parameters) and a list of output objects.
    """

    call: CohereToolCall
    outputs: list[dict[str, Any]]


class CohereChatRequest(BaseModel):
    """Cohere chat request model."""

    # Required fields
    message: str
    apiFormat: Literal["COHERE"] = "COHERE"

    # Optional fields
    # ``SerializeAsAny`` preserves subclass-specific fields (e.g. ``toolResults``
    # on ``CohereToolMessage``) when this request is serialized via ``model_dump``.
    # Without it, Pydantic v2 would serialize each element using the declared
    # ``CohereMessage`` schema and silently drop subclass fields.
    chatHistory: list[SerializeAsAny[CohereMessage]] | None = None
    maxTokens: int | None = None
    temperature: float | None = None
    topP: float | None = None
    topK: int | None = None
    frequencyPenalty: float | None = None
    presencePenalty: float | None = None
    stopSequences: list[str] | None = None
    seed: int | None = None
    tools: list[CohereTool] | None = None
    # NOTE: OCI's Cohere chat endpoint does not accept ``toolChoice`` — see
    # ``OCIChatConfig.openai_to_oci_cohere_param_map`` which marks
    # ``tool_choice`` as unsupported. The field is intentionally absent here
    # so it isn't silently dropped or surfaced as a supported feature.
    # OCI Cohere responseFormat is {"type": "TEXT" | "JSON_OBJECT", "schema"?: ...};
    # there is no JSON_SCHEMA type. The shape is built in
    # OCIChatConfig._normalize_response_format.
    responseFormat: dict[str, Any] | None = None
    preambleOverride: str | None = None
    documents: list[dict[str, Any]] | None = None
    searchQueriesOnly: bool | None = None
    searchEntryPoint: str | None = None
    grounding: dict[str, Any] | None = None
    isEcho: bool | None = None
    isSearchQueriesOnly: bool | None = None
    isRawPrompting: bool | None = None
    isForceSingleStep: bool | None = None
    promptTruncation: str | None = None
    safetyMode: str | None = None
    citationQuality: str | None = None
    maxInputTokens: int | None = None
    isStream: bool | None = None
    streamOptions: dict[str, Any] | None = None


class CohereUsage(BaseModel):
    """Usage information for Cohere response."""

    promptTokens: int
    completionTokens: int
    totalTokens: int
    promptTokensDetails: dict[str, Any] | None = None
    completionTokensDetails: dict[str, Any] | None = None


class CohereCitation(BaseModel):
    """Citation in Cohere response."""

    start: int
    end: int
    text: str
    document_ids: list[str]


class CohereSearchQuery(BaseModel):
    """Search query generated by Cohere."""

    text: str
    generation_id: str


class CohereChatResponse(BaseModel):
    """Cohere chat response model."""

    # Required fields
    text: str
    apiFormat: Literal["COHERE"] = "COHERE"
    # Accept any string (with ``None`` for absent) so unknown finish reasons
    # — e.g. a value OCI adds in a future API revision — degrade gracefully
    # via ``handle_cohere_response``'s ``elif oci_finish_reason is not None``
    # fallback instead of crashing Pydantic validation. Mirrors
    # ``CohereStreamChunk.finishReason`` which has always been ``Optional[str]``.
    finishReason: str | None = None

    # Optional fields
    chatHistory: list[CohereMessage] | None = None
    citations: list[CohereCitation] | None = None
    documents: list[dict[str, Any]] | None = None
    errorMessage: str | None = None
    isSearchRequired: bool | None = None
    prompt: str | None = None
    searchQueries: list[CohereSearchQuery] | None = None
    toolCalls: list[CohereToolCall] | None = None
    usage: CohereUsage | None = None


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


# ---------------------------------------------------------------------------
# OCI Embed types
# ---------------------------------------------------------------------------


class OCIEmbedRequest(BaseModel):
    """Request body for POST /20231130/actions/embedText."""

    compartmentId: str
    servingMode: OCIServingMode
    inputs: list[str]
    inputType: str | None = None  # SEARCH_DOCUMENT | SEARCH_QUERY | CLASSIFICATION | CLUSTERING | IMAGE
    truncate: str | None = "END"  # NONE | START | END
    outputDimensions: int | None = None  # cohere.embed-v4.0+; valid: 256, 512, 1024, 1536


class OCIEmbedUsage(BaseModel):
    promptTokens: int
    totalTokens: int


class OCIEmbedResponse(BaseModel):
    """Response body from POST /20231130/actions/embedText."""

    id: str | None = None  # present in the official SDK response
    embeddings: list[list[float]]
    modelId: str
    modelVersion: str
    # OCI returns per-input token counts in inputTextTokenCounts (summed for total usage)
    inputTextTokenCounts: list[int] | None = None
    # Some deployments may return a usage object instead
    usage: OCIEmbedUsage | None = None
