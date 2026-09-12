from typing import Literal, TypeAlias

from typing_extensions import NotRequired, ReadOnly, TypedDict

MuseMode: TypeAlias = Literal["PUSH_TO_TALK", "ENDPOINTING"]
MuseAudioEncoding: TypeAlias = Literal["PCM_16KHZ", "PCM_24KHZ"]
MuseSampleRate: TypeAlias = Literal[16000, 24000]


class MuseAuthorization(TypedDict):
    accessToken: ReadOnly[str]


class MuseHandshake(TypedDict):
    authorization: ReadOnly[MuseAuthorization]
    audioEncoding: ReadOnly[MuseAudioEncoding]
    model: ReadOnly[str]
    mode: ReadOnly[MuseMode]
    partialMode: ReadOnly[Literal["CUMULATIVE"]]
    emitAudioProgress: ReadOnly[bool]
    languageBias: NotRequired[ReadOnly[tuple[str, ...]]]


class MuseTranscriptionAudioFormat(TypedDict):
    type: ReadOnly[Literal["audio/pcm"]]
    rate: ReadOnly[MuseSampleRate]


class MuseTranscriptionSettings(TypedDict):
    model: ReadOnly[str]
    language: NotRequired[ReadOnly[str]]


class MuseTurnDetection(TypedDict):
    type: ReadOnly[Literal["server_vad"]]


class MuseTranscriptionAudioInput(TypedDict):
    format: ReadOnly[MuseTranscriptionAudioFormat]
    transcription: ReadOnly[MuseTranscriptionSettings]
    turn_detection: ReadOnly[MuseTurnDetection | None]


class MuseTranscriptionAudio(TypedDict):
    input: ReadOnly[MuseTranscriptionAudioInput]


class MuseTranscriptionSession(TypedDict):
    id: ReadOnly[str]
    object: ReadOnly[Literal["realtime.transcription_session"]]
    type: ReadOnly[Literal["transcription"]]
    audio: ReadOnly[MuseTranscriptionAudio]


class MuseSessionCreatedEvent(TypedDict):
    type: ReadOnly[Literal["session.created"]]
    event_id: ReadOnly[str]
    session: ReadOnly[MuseTranscriptionSession]
