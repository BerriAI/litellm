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
