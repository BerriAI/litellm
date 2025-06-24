from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from typing_extensions import Required, TypedDict

from .vertex_ai import (
    GenerationConfig,
    HttpxBlobType,
    HttpxContentType,
    Tools,
    UsageMetadata,
)


class GeminiFilesState(Enum):
    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


class GeminiFilesSource(Enum):
    SOURCE_UNSPECIFIED = "SOURCE_UNSPECIFIED"
    UPLOADED = "UPLOADED"
    GENERATED = "GENERATED"


class GeminiCreateFilesResponseObject(TypedDict):
    name: str
    displayName: str
    mimeType: str
    sizeBytes: str
    createTime: str
    updateTime: str
    expirationTime: str
    sha256Hash: str
    uri: str
    state: GeminiFilesState
    source: GeminiFilesSource
    error: dict
    metadata: dict


class BidiGenerateContentTranscription(TypedDict):
    text: str
    """Output only. The transcription of the audio."""


class BidiGenerateContentServerContent(TypedDict, total=False):
    generationComplete: bool
    """Output only. If true, indicates that the model is done generating."""

    turnComplete: bool
    """Output only. If true, indicates that the model has completed its turn. Generation will only start in response to additional client messages."""

    interrupted: bool
    """Output only. If true, indicates that a client message has interrupted current model generation. If the client is playing out the content in real time, this is a good signal to stop and empty the current playback queue."""

    groundingMetadata: dict
    """Output only. Grounding metadata for the generated content."""

    inputTranscription: BidiGenerateContentTranscription
    """Output only. Input audio transcription. The transcription is sent independently of the other server messages and there is no guaranteed ordering."""

    outputTranscription: BidiGenerateContentTranscription
    """Output only. Output audio transcription. The transcription is sent independently of the other server messages and there is no guaranteed ordering, in particular not between serverContent and this outputTranscription."""

    modelTurn: HttpxContentType
    """Output only. The content that the model is currently generating."""


class BidiGenerateContentServerMessage(TypedDict, total=False):
    usageMetadata: UsageMetadata
    """Output only. Usage metadata for the generated content."""

    serverContent: BidiGenerateContentServerContent
    """Output only. The content that the model is currently generating."""

    setupComplete: dict
    """Output only. The setup complete message."""


class BidiGenerateContentRealtimeInput(TypedDict, total=False):
    text: str
    """The text to be sent to the model."""

    audio: HttpxBlobType
    """The audio to be sent to the model."""

    video: HttpxBlobType
    """The video to be sent to the model."""

    audioStreamEnd: bool
    """Output only. If true, indicates that the audio stream has ended."""

    activityStart: bool
    """Output only. If true, indicates that the activity has started."""

    activityEnd: bool
    """Output only. If true, indicates that the activity has ended."""


StartOfSpeechSensitivityEnum = Literal[
    "START_SENSITIVITY_UNSPECIFIED", "START_SENSITIVITY_HIGH", "START_SENSITIVITY_LOW"
]
EndOfSpeechSensitivityEnum = Literal[
    "END_SENSITIVITY_UNSPECIFIED", "END_SENSITIVITY_HIGH", "END_SENSITIVITY_LOW"
]


class AutomaticActivityDetection(TypedDict, total=False):
    disabled: bool
    startOfSpeechSensitivity: StartOfSpeechSensitivityEnum
    prefixPaddingMs: int
    endOfSpeechSensitivity: EndOfSpeechSensitivityEnum
    silenceDurationMs: int


class BidiGenerateContentRealtimeInputConfig(TypedDict, total=False):
    automaticActivityDetection: AutomaticActivityDetection


class BidiGenerateContentSetup(TypedDict, total=False):
    model: str
    """The model to be used for the realtime session."""

    generationConfig: GenerationConfig
    """The generation config to be used for the realtime session."""

    systemInstruction: HttpxContentType
    """The system instruction to be used for the realtime session."""

    tools: List[Tools]
    """The tools to be used for the realtime session."""

    realtimeInputConfig: dict
    """The realtime config to be used for the realtime session."""

    sessionResumption: dict
    """The session resumption to be used for the realtime session."""

    sessionResumptionConfig: dict
    """The session resumption config to be used for the realtime session."""

    contextWindowCompression: dict
    """The context window compression to be used for the realtime session."""

    inputAudioTranscription: dict
    """The input audio transcription to be used for the realtime session."""

    outputAudioTranscription: dict
    """The output audio transcription to be used for the realtime session."""
