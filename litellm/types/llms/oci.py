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
    image_url: str = Field(..., alias="imageUrl")

OCIContentPartUnion = Union[OCITextContentPart, OCIImageContentPart]

# --- Modelos para Tools e Tool Calls ---

class OCIFunction(BaseModel):
    """Define uma função que o modelo pode chamar."""
    name: str
    arguments: str # Argumentos devem ser um JSON serializado em string

class OCIToolCall(BaseModel):
    """Representa uma chamada de ferramenta feita pelo modelo."""
    id: str
    type: Literal["FUNCTION"] = "FUNCTION"
    function: OCIFunction

# --- Modelos de Mensagem (Request e Response) ---

class OCIMessage(BaseModel):
    """Modelo para uma única mensagem no payload de request/response."""
    role: OCIRoles
    content: list[OCIContentPartUnion] | None = None
    tool_calls: list[OCIToolCall] | None = Field(None, alias="toolCalls")
    tool_call_id: str | None = Field(None, alias="toolCallId")

# --- Modelos para o Payload de Request ---

class OCIChatRequestPayload(BaseModel):
    """Payload interno do 'chatRequest' para a API OCI."""
    api_format: str = Field(..., alias="apiFormat")
    messages: list[OCIMessage]
    tools: list[Any] | None = None # A estrutura de 'tools' pode variar
    is_stream: bool = Field(False, alias="isStream")
    num_generations: int | None = Field(None, alias="numGenerations")
    max_tokens: int | None = Field(None, alias="maxTokens")
    temperature: float | None = None
    top_p: float | None = Field(None, alias="topP")
    stop: list[str] | None = None
    seed: int | None = None
    frequency_penalty: float | None = Field(None, alias="frequencyPenalty")
    presence_penalty: float | None = Field(None, alias="presencePenalty")

class OCIServingMode(BaseModel):
    """Define o modo de serviço e o modelo a ser usado."""
    serving_type: str = Field(..., alias="servingType")
    model_id: str = Field(..., alias="modelId")

class OCICompletionPayload(BaseModel):
    """Modelo de Pydantic para o corpo completo da requisição de chat da OCI."""
    compartment_id: str = Field(..., alias="compartmentId")
    serving_mode: OCIServingMode = Field(..., alias="servingMode")
    chat_request: OCIChatRequestPayload = Field(..., alias="chatRequest")

# --- Modelos para a Resposta da API (Non-streaming) ---

class OCIResponseUsage(BaseModel):
    """Uso de tokens na resposta da OCI."""
    prompt_tokens: int = Field(..., alias="promptTokens")
    completion_tokens: int = Field(..., alias="completionTokens")
    total_tokens: int = Field(..., alias="totalTokens")

class OCIResponseChoice(BaseModel):
    """Uma escolha de conclusão na resposta da OCI."""
    index: int
    message: OCIMessage
    finish_reason: str | None = Field(None, alias="finishReason")

class OCIChatResponse(BaseModel):
    """O objeto 'chatResponse' na resposta da OCI."""
    choices: list[OCIResponseChoice]
    usage: OCIResponseUsage
    model_id: str = Field(..., alias="modelId")
    time_created: str = Field(..., alias="timeCreated")

class OCICompletionResponse(BaseModel):
    """Modelo para o corpo completo da resposta non-streaming da OCI."""
    chat_response: OCIChatResponse = Field(..., alias="chatResponse")

# --- Modelos para a Resposta da API (Streaming) ---

class OCIStreamDelta(BaseModel):
    """O delta de conteúdo em um chunk de streaming."""
    content: str | None = None
    role: str | None = None
    tool_calls: list[OCIToolCall] | None = Field(None, alias="toolCalls")

class OCIStreamChoice(BaseModel):
    """Uma escolha em um chunk de streaming."""
    index: int
    delta: OCIStreamDelta
    finish_reason: str | None = Field(None, alias="finishReason")

class OCIStreamChatResponse(BaseModel):
    """O objeto 'chatResponse' em um chunk de streaming."""
    choices: list[OCIStreamChoice]

class OCIStreamChunk(BaseModel):
    """Modelo para um único chunk de evento SSE da OCI."""
    chat_response: OCIStreamChatResponse = Field(..., alias="chatResponse")
    model_id: str | None = Field(None, alias="modelId")
