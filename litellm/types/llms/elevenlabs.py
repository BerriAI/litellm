from typing import Annotated, Literal, TypeAlias

from pydantic import ConfigDict, Field, StrictBool, StrictStr
from typing_extensions import ReadOnly, TypedDict

from litellm.types.llms.base_model import LiteLLMBaseModel

ElevenLabsCommitStrategy: TypeAlias = Literal["vad", "manual"]
ElevenLabsErrorType: TypeAlias = Literal[
    "error",
    "auth_error",
    "quota_exceeded",
    "commit_throttled",
    "unaccepted_terms",
    "rate_limited",
    "queue_overflow",
    "resource_exhausted",
    "session_time_limit_exceeded",
    "input_error",
    "invalid_request",
    "chunk_size_exceeded",
    "insufficient_audio_activity",
    "transcriber_error",
]


class ElevenLabsRealtimeOptions(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    keyterms: tuple[StrictStr, ...] | None = None
    secondary_languages: tuple[StrictStr, ...] | None = None
    no_verbatim: StrictBool | None = None
    filter_background_audio: StrictBool | None = None
    vad_threshold: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    vad_silence_threshold_secs: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    min_speech_duration_ms: int | None = Field(default=None, ge=0)
    min_silence_duration_ms: int | None = Field(default=None, ge=0)
    enable_logging: StrictBool | None = None


class ElevenLabsRealtimeConfigure(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["configure"] = "configure"
    query: str


class ElevenLabsInputAudioChunk(TypedDict):
    message_type: ReadOnly[Literal["input_audio_chunk"]]
    audio_base_64: ReadOnly[str]
    commit: ReadOnly[bool]
    sample_rate: ReadOnly[int]


class ElevenLabsSessionStarted(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    message_type: Literal["session_started"]
    session_id: str


class ElevenLabsPartialTranscript(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    message_type: Literal["partial_transcript"]
    text: str


class ElevenLabsCommittedTranscript(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    message_type: Literal["committed_transcript"]
    text: str


class ElevenLabsWarning(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    message_type: Literal["warning"]
    warning: str


class ElevenLabsErrorMessage(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    message_type: ElevenLabsErrorType
    error: str


ElevenLabsServerMessageUnion: TypeAlias = (
    ElevenLabsSessionStarted
    | ElevenLabsPartialTranscript
    | ElevenLabsCommittedTranscript
    | ElevenLabsWarning
    | ElevenLabsErrorMessage
)
ElevenLabsServerMessage: TypeAlias = Annotated[ElevenLabsServerMessageUnion, Field(discriminator="message_type")]
