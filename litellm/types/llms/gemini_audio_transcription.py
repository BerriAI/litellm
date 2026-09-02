from typing import Literal, Required

from pydantic import BaseModel, ConfigDict
from typing_extensions import ReadOnly, TypedDict


class GeminiTranscriptionAudioInput(TypedDict):
    type: ReadOnly[Literal["audio"]]
    data: ReadOnly[str]
    mime_type: ReadOnly[str]


class GeminiTranscriptionVerbatimMode(TypedDict, total=False):
    type: ReadOnly[Required[Literal["verbatim"]]]
    timestamp_granularities: ReadOnly[tuple[Literal["word"], ...]]
    diarization_mode: ReadOnly[Literal["speaker"]]


class GeminiTranscriptionConfig(TypedDict, total=False):
    language_codes: ReadOnly[tuple[str, ...]]
    mode: ReadOnly[GeminiTranscriptionVerbatimMode]


class GeminiTranscriptionGenerationConfig(TypedDict):
    transcription_config: ReadOnly[GeminiTranscriptionConfig]


class GeminiTranscriptionInteractionRequest(TypedDict, total=False):
    model: ReadOnly[Required[str]]
    input: ReadOnly[Required[tuple[GeminiTranscriptionAudioInput, ...]]]
    generation_config: ReadOnly[GeminiTranscriptionGenerationConfig]


class GeminiTranscriptionWordAnnotation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    text: str | None = None
    speaker: str | None = None
    start_offset: str | None = None
    end_offset: str | None = None


class GeminiTranscriptionContent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    text: str | None = None
    annotations: tuple[GeminiTranscriptionWordAnnotation, ...] = ()


class GeminiTranscriptionStep(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    content: tuple[GeminiTranscriptionContent, ...] = ()


class GeminiTranscriptionModalityTokens(BaseModel):
    model_config = ConfigDict(extra="ignore")

    modality: str | None = None
    tokens: int = 0


class GeminiTranscriptionUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    input_tokens_by_modality: tuple[GeminiTranscriptionModalityTokens, ...] = ()


class GeminiTranscriptionInteractionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    status: str | None = None
    usage: GeminiTranscriptionUsage | None = None
    steps: tuple[GeminiTranscriptionStep, ...] = ()
