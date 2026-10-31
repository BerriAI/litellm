from typing import Literal

from pydantic import BaseModel, ConfigDict
from typing_extensions import ReadOnly, TypedDict


class ElevenLabsSTTWord(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    type: Literal["word", "spacing", "audio_event"]
    start: float | None = None
    end: float | None = None
    speaker_id: str | None = None
    channel_index: int | None = None


class ElevenLabsSTTChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = ""
    language_code: str | None = None
    language_probability: float | None = None
    words: tuple[ElevenLabsSTTWord, ...] = ()
    channel_index: int | None = None
    audio_duration_secs: float | None = None


class ElevenLabsSTTMultichannelResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    transcripts: tuple[ElevenLabsSTTChunk, ...]
    audio_duration_secs: float | None = None


class OpenAIDiarizedSegment(TypedDict):
    id: ReadOnly[str]
    start: ReadOnly[float]
    end: ReadOnly[float]
    speaker: ReadOnly[str]
    text: ReadOnly[str]
    type: ReadOnly[Literal["transcript.text.segment"]]
