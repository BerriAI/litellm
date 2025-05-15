from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from typing_extensions import Required, TypedDict

from .vertex_ai import HttpxContentType, UsageMetadata


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
