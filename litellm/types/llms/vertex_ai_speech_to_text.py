from dataclasses import dataclass

from pydantic import BaseModel
from typing_extensions import TypedDict


class VertexSpeechToTextAutoDecodingConfig(TypedDict):
    pass


class VertexSpeechToTextRecognitionFeatures(TypedDict):
    enableAutomaticPunctuation: bool


class VertexSpeechToTextRecognitionConfig(TypedDict):
    model: str
    languageCodes: list[str]
    features: VertexSpeechToTextRecognitionFeatures
    autoDecodingConfig: VertexSpeechToTextAutoDecodingConfig


class VertexSpeechToTextRecognizeRequest(TypedDict):
    config: VertexSpeechToTextRecognitionConfig
    content: str


class VertexSpeechToTextAlternative(BaseModel):
    transcript: str | None = None


class VertexSpeechToTextResult(BaseModel):
    alternatives: list[VertexSpeechToTextAlternative] = []
    languageCode: str | None = None


class VertexSpeechToTextResponseMetadata(BaseModel):
    totalBilledDuration: str | None = None


class VertexSpeechToTextRecognizeResponse(BaseModel):
    results: list[VertexSpeechToTextResult] = []
    metadata: VertexSpeechToTextResponseMetadata | None = None


@dataclass(frozen=True, slots=True)
class ChirpGrpcRequestConfig:
    recognizer: str
    model_name: str
    language_codes: tuple[str, ...]
    enable_automatic_punctuation: bool
    chunk_bytes: int
