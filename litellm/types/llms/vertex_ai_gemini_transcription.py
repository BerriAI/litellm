from typing import Literal

from pydantic import ConfigDict
from typing_extensions import ReadOnly, TypedDict

from litellm.types.llms.base import LiteLLMBaseModel


class VertexGeminiTranscriptionInlineData(TypedDict):
    mimeType: ReadOnly[str]
    data: ReadOnly[str]


class VertexGeminiTranscriptionPart(TypedDict):
    inlineData: ReadOnly[VertexGeminiTranscriptionInlineData]


class VertexGeminiTranscriptionContent(TypedDict):
    role: ReadOnly[Literal["user"]]
    parts: ReadOnly[tuple[VertexGeminiTranscriptionPart, ...]]


class VertexGeminiTranscriptionAudioConfig(TypedDict, total=False):
    languageCodes: ReadOnly[tuple[str, ...]]


class VertexGeminiTranscriptionGenerationConfig(TypedDict):
    audioTranscriptionConfig: ReadOnly[VertexGeminiTranscriptionAudioConfig]


class VertexGeminiTranscriptionRequest(TypedDict):
    contents: ReadOnly[tuple[VertexGeminiTranscriptionContent, ...]]
    generationConfig: ReadOnly[VertexGeminiTranscriptionGenerationConfig]


class VertexGeminiTranscriptionResponsePart(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str | None = None


class VertexGeminiTranscriptionResponseContent(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore")

    parts: tuple[VertexGeminiTranscriptionResponsePart, ...] = ()


class VertexGeminiTranscriptionCandidate(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore")

    content: VertexGeminiTranscriptionResponseContent | None = None


class VertexGeminiTranscriptionModalityTokens(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore")

    modality: str | None = None
    tokenCount: int = 0


class VertexGeminiTranscriptionUsageMetadata(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore")

    promptTokenCount: int = 0
    candidatesTokenCount: int = 0
    totalTokenCount: int = 0
    promptTokensDetails: tuple[VertexGeminiTranscriptionModalityTokens, ...] = ()


class VertexGeminiTranscriptionResponse(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore")

    candidates: tuple[VertexGeminiTranscriptionCandidate, ...] = ()
    usageMetadata: VertexGeminiTranscriptionUsageMetadata | None = None
