from typing import Annotated, Literal

from pydantic import ConfigDict, Field
from typing_extensions import TypedDict

from litellm.types.llms.base import LiteLLMBaseModel


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


class VertexSpeechToTextAlternative(LiteLLMBaseModel):
    transcript: str | None = None


class VertexSpeechToTextResult(LiteLLMBaseModel):
    alternatives: list[VertexSpeechToTextAlternative] = Field(default=[])
    languageCode: str | None = None


class VertexSpeechToTextResponseMetadata(LiteLLMBaseModel):
    totalBilledDuration: str | None = None


class VertexSpeechToTextRecognizeResponse(LiteLLMBaseModel):
    results: list[VertexSpeechToTextResult] = Field(default=[])
    metadata: VertexSpeechToTextResponseMetadata | None = None


class VertexSpeechStreamingConfigure(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["configure"] = "configure"
    model: str
    language_codes: tuple[str, ...]
    sample_rate_hertz: int


class VertexSpeechStreamingFinishTurn(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["finish_turn"] = "finish_turn"


class VertexSpeechStreamingDiscardTurn(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["discard_turn"] = "discard_turn"


VertexSpeechStreamingCommandUnion = (
    VertexSpeechStreamingConfigure | VertexSpeechStreamingFinishTurn | VertexSpeechStreamingDiscardTurn
)
VertexSpeechStreamingCommand = Annotated[VertexSpeechStreamingCommandUnion, Field(discriminator="kind")]


class VertexSpeechStreamingResult(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    transcript: str
    is_final: bool


class VertexSpeechStreamingResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["response"] = "response"
    speech_event: Literal["none", "begin", "end"]
    results: tuple[VertexSpeechStreamingResult, ...]
    billed_seconds: float


class VertexSpeechStreamingConfigured(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["configured"] = "configured"


class VertexSpeechStreamingTurnFinished(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["turn_finished"] = "turn_finished"


class VertexSpeechStreamingTurnDiscarded(LiteLLMBaseModel):
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
