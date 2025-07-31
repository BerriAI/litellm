from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field
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
    """Modelo base para partes de conteúdo em uma mensagem OCI."""
    type: str

class OCITextContentPart(OCIContentPart):
    """Parte de conteúdo de texto para a API OCI."""
    type: Literal["TEXT"] = "TEXT"
    text: str

class OCIImageContentPart(OCIContentPart):
    """Parte de conteúdo de imagem para a API OCI."""
    type: Literal["IMAGE"] = "IMAGE"
    imageUrl: str

OCIContentPartUnion = Union[OCITextContentPart, OCIImageContentPart]

# --- Modelos para Tools e Tool Calls ---

# class OCIFunction(BaseModel):
#     """Define uma função que o modelo pode chamar."""
#     name: str
#     arguments: str # Argumentos devem ser um JSON serializado em string

# class OCIToolCall(BaseModel):
#     """Representa uma chamada de ferramenta feita pelo modelo."""
#     id: str
#     type: Literal["FUNCTION"] = "FUNCTION"
#     function: OCIFunction

class OCIToolCall(BaseModel):
    """Representa uma chamada de ferramenta feita pelo modelo."""
    id: str
    type: Literal["FUNCTION"] = "FUNCTION"
    name: str
    arguments: str  # Argumentos devem ser um JSON serializado em string

class OCIToolDefinition(BaseModel):
    """Define uma ferramenta que pode ser usada pelo modelo."""
    type: Literal["FUNCTION"] = "FUNCTION"
    name: str | None = None
    description: str | None = None
    parameters: dict | None = None

# --- Modelos de Mensagem (Request e Response) ---

class OCIMessage(BaseModel):
    """Modelo para uma única mensagem no payload de request/response."""
    role: OCIRoles
    content: list[OCIContentPartUnion] | None = None
    toolCalls: list[OCIToolCall] | None = None
    toolCallId: str | None = None

# --- Modelos para o Payload de Request ---

class OCIChatRequestPayload(BaseModel):
    """Payload interno do 'chatRequest' para a API OCI."""
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
    """Define o modo de serviço e o modelo a ser usado."""
    servingType: str
    modelId: str

class OCICompletionPayload(BaseModel):
    """Modelo de Pydantic para o corpo completo da requisição de chat da OCI."""
    compartmentId: str
    servingMode: OCIServingMode
    chatRequest: OCIChatRequestPayload

# --- Modelos para a Resposta da API (Non-streaming) ---

class OCICompletionTokenDetails(BaseModel):
    """Detalhes dos tokens de conclusão na resposta da OCI."""
    acceptedPredictionTokens: int
    reasoningTokens: int

class OCIPropmtTokensDetails(BaseModel):
    """Detalhes dos tokens de prompt na resposta da OCI."""
    cachedTokens: int

class OCIResponseUsage(BaseModel):
    """Uso de tokens na resposta da OCI."""
    promptTokens: int
    completionTokens: int
    totalTokens: int
    completionTokensDetails: OCICompletionTokenDetails
    promptTokensDetails: OCIPropmtTokensDetails

class OCIResponseChoice(BaseModel):
    """Uma escolha de conclusão na resposta da OCI."""
    index: int
    message: OCIMessage
    finishReason: str | None
    logprobs: dict[str, Any] | None = None

class OCIChatResponse(BaseModel):
    """O objeto 'chatResponse' na resposta da OCI."""
    apiFormat: str
    timeCreated: str
    choices: list[OCIResponseChoice]
    usage: OCIResponseUsage

class OCICompletionResponse(BaseModel):
    """Modelo para o corpo completo da resposta non-streaming da OCI."""
    modelId: str
    modelVersion: str
    chatResponse: OCIChatResponse

# --- Modelos para a Resposta da API (Streaming) ---

class OCIStreamDelta(BaseModel):
    """O delta de conteúdo em um chunk de streaming."""
    content: str | None = None
    role: str | None = None
    toolCalls: list[OCIToolCall] | None

class OCIStreamChoice(BaseModel):
    """Uma escolha em um chunk de streaming."""
    index: int
    delta: OCIStreamDelta
    finishReason: str | None

class OCIStreamChatResponse(BaseModel):
    """O objeto 'chatResponse' em um chunk de streaming."""
    choices: list[OCIStreamChoice]

class OCIStreamChunk(BaseModel):
    """Modelo para um único chunk de evento SSE da OCI."""
    chatResponse: OCIStreamChatResponse
    modelId: str | None
