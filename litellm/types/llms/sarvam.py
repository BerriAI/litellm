from typing import Literal, TypeAlias

from typing_extensions import NotRequired, ReadOnly, TypedDict

SarvamEndpointing: TypeAlias = Literal["vad", "manual"]
SarvamSampleRate: TypeAlias = Literal[8000, 16000]
SarvamControlEventType: TypeAlias = Literal["speech_start", "speech_end", "flush", "end"]


class SarvamAudioInput(TypedDict):
    event: ReadOnly[Literal["audio_input"]]
    audio: ReadOnly[str]


class SarvamControlEvent(TypedDict):
    event: ReadOnly[SarvamControlEventType]


class SarvamConfigUpdate(TypedDict):
    event: ReadOnly[Literal["config.update"]]
    language_code: NotRequired[ReadOnly[str]]
    endpointing: NotRequired[ReadOnly[SarvamEndpointing]]
