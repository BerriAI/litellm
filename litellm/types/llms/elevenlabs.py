from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ElevenLabsSTTWord(BaseModel):
    text: str
    type: Literal["word", "spacing", "audio_event"]
    start: Optional[float] = None
    end: Optional[float] = None
    speaker_id: Optional[str] = None
    channel_index: Optional[int] = None


class ElevenLabsSTTChunk(BaseModel):
    text: str = ""
    language_code: Optional[str] = None
    language_probability: Optional[float] = None
    words: List[ElevenLabsSTTWord] = Field(default_factory=list)
    channel_index: Optional[int] = None
    audio_duration_secs: Optional[float] = None


class ElevenLabsSTTMultichannelResponse(BaseModel):
    transcripts: List[ElevenLabsSTTChunk]
    audio_duration_secs: Optional[float] = None


class OpenAITranscriptionWord(TypedDict):
    word: str
    start: float
    end: float


class OpenAIDiarizedSegment(TypedDict):
    id: str
    start: float
    end: float
    speaker: str
    text: str
    type: Literal["transcript.text.segment"]
