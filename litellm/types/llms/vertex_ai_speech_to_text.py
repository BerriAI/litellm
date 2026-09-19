from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
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


class VertexSpeechStreamingConfigure(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["configure"] = "configure"
    model: str
    language_codes: tuple[str, ...]
    sample_rate_hertz: int


class VertexSpeechStreamingFinishTurn(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["finish_turn"] = "finish_turn"


class VertexSpeechStreamingDiscardTurn(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["discard_turn"] = "discard_turn"


VertexSpeechStreamingCommandUnion = (
    VertexSpeechStreamingConfigure | VertexSpeechStreamingFinishTurn | VertexSpeechStreamingDiscardTurn
)
VertexSpeechStreamingCommand = Annotated[VertexSpeechStreamingCommandUnion, Field(discriminator="kind")]


class VertexSpeechStreamingResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    transcript: str
    is_final: bool


class VertexSpeechStreamingResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["response"] = "response"
    speech_event: Literal["none", "begin", "end"]
    results: tuple[VertexSpeechStreamingResult, ...]
    billed_seconds: float


class VertexSpeechStreamingConfigured(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["configured"] = "configured"


class VertexSpeechStreamingTurnFinished(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["turn_finished"] = "turn_finished"


class VertexSpeechStreamingTurnDiscarded(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["turn_discarded"] = "turn_discarded"
    billed_seconds: float


VertexSpeechStreamingEventUnion = (
    VertexSpeechStreamingResponse
    | VertexSpeechStreamingConfigured
    | VertexSpeechStreamingTurnFinished
    | VertexSpeechStreamingTurnDiscarded
)
VertexSpeechStreamingEvent = Annotated[VertexSpeechStreamingEventUnion, Field(discriminator="kind")]
