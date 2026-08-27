"""Provider-agnostic SRT/WebVTT subtitle synthesis from timestamped transcription tokens."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, chain
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

CUE_MAX_TOKENS: Final = 15
CUE_MAX_DURATION_MS: Final = 5000

SRT_RESPONSE_FORMAT: Final = "srt"
VTT_RESPONSE_FORMAT: Final = "vtt"
SUBTITLE_RESPONSE_FORMATS: Final = frozenset((SRT_RESPONSE_FORMAT, VTT_RESPONSE_FORMAT))


@dataclass(frozen=True, slots=True)
class SubtitleToken:
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    speaker: str | int | None = None


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True, slots=True)
class _CueAccumulator:
    texts: tuple[str, ...] = ()
    start_ms: int | None = None
    end_ms: int | None = None
    speaker: str | int | None = None


def _completed_cue(accumulator: _CueAccumulator) -> tuple[SubtitleCue, ...]:
    if not accumulator.texts or accumulator.start_ms is None:
        return ()
    text: Final = "".join(accumulator.texts).strip()
    if not text:
        return ()
    end_ms: Final = accumulator.end_ms if accumulator.end_ms is not None else accumulator.start_ms
    return (SubtitleCue(start_ms=accumulator.start_ms, end_ms=end_ms, text=text),)


def _cue_break_reached(accumulator: _CueAccumulator, token: SubtitleToken) -> bool:
    if len(accumulator.texts) >= CUE_MAX_TOKENS:
        return True
    return (
        accumulator.start_ms is not None
        and token.start_ms is not None
        and token.start_ms - accumulator.start_ms >= CUE_MAX_DURATION_MS
    )


_AbsorbStep = tuple[tuple[SubtitleCue, ...], _CueAccumulator]


def _absorb_token(accumulator: _CueAccumulator, token: SubtitleToken) -> _AbsorbStep:
    if token.start_ms is None and accumulator.start_ms is None:
        return (), accumulator
    if token.speaker is not None and token.speaker != accumulator.speaker:
        return _completed_cue(accumulator), _CueAccumulator(
            texts=(token.text,),
            start_ms=token.start_ms,
            end_ms=token.end_ms,
            speaker=token.speaker,
        )
    if _cue_break_reached(accumulator, token):
        return _completed_cue(accumulator), _CueAccumulator(
            texts=(token.text,),
            start_ms=token.start_ms,
            end_ms=token.end_ms,
            speaker=accumulator.speaker,
        )
    return (), _CueAccumulator(
        texts=(*accumulator.texts, token.text),
        start_ms=accumulator.start_ms if accumulator.start_ms is not None else token.start_ms,
        end_ms=token.end_ms if token.end_ms is not None else accumulator.end_ms,
        speaker=accumulator.speaker,
    )


def _absorb_step(carry: _AbsorbStep, token: SubtitleToken) -> _AbsorbStep:
    return _absorb_token(carry[1], token)


def group_subtitle_tokens_into_cues(tokens: Sequence[SubtitleToken]) -> tuple[SubtitleCue, ...]:
    steps: Final = tuple(accumulate(tokens, _absorb_step, initial=((), _CueAccumulator())))
    completed: Final = chain.from_iterable(emitted for emitted, _ in steps)
    return (*completed, *_completed_cue(steps[-1][1]))


def _format_timestamp(total_ms: int, millis_separator: str) -> str:
    clamped: Final = max(total_ms, 0)
    hours, hour_remainder = divmod(clamped, 3_600_000)
    minutes, minute_remainder = divmod(hour_remainder, 60_000)
    seconds, millis = divmod(minute_remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{millis_separator}{millis:03d}"


def _render_srt(cues: Sequence[SubtitleCue]) -> str:
    lines: Final = tuple(
        line
        for index, cue in enumerate(cues, start=1)
        for line in (
            str(index),
            f"{_format_timestamp(cue.start_ms, ',')} --> {_format_timestamp(cue.end_ms, ',')}",
            cue.text,
            "",
        )
    )
    return "\n".join(lines)


def _render_vtt(cues: Sequence[SubtitleCue]) -> str:
    cue_lines: Final = tuple(
        line
        for cue in cues
        for line in (
            f"{_format_timestamp(cue.start_ms, '.')} --> {_format_timestamp(cue.end_ms, '.')}",
            cue.text,
            "",
        )
    )
    return "\n".join(("WEBVTT", "", *cue_lines))


def render_subtitle_tokens_as_srt(tokens: Sequence[SubtitleToken]) -> str:
    """Render tokens as an SRT document; empty string when no token has timestamp data."""
    cues: Final = group_subtitle_tokens_into_cues(tokens)
    if not cues:
        return ""
    return _render_srt(cues)


def render_subtitle_tokens_as_vtt(tokens: Sequence[SubtitleToken]) -> str:
    """Render tokens as a WebVTT document; the WEBVTT header is emitted even without cues."""
    return _render_vtt(group_subtitle_tokens_into_cues(tokens))


class TranscriptionWordTiming(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    word: str = ""
    start: float | None = None
    end: float | None = None
    speaker: str | None = None


_WORD_TIMINGS_ADAPTER: Final = TypeAdapter(tuple[TranscriptionWordTiming, ...])


def _seconds_to_ms(seconds: float | None) -> int | None:
    if seconds is None:
        return None
    return round(seconds * 1000)


def _word_to_subtitle_token(word: TranscriptionWordTiming) -> SubtitleToken:
    return SubtitleToken(
        text=f"{word.word} ",
        start_ms=_seconds_to_ms(word.start),
        end_ms=_seconds_to_ms(word.end),
        speaker=word.speaker,
    )


def _parse_word_timings(words: object) -> tuple[TranscriptionWordTiming, ...]:
    try:
        return _WORD_TIMINGS_ADAPTER.validate_python(words)
    except ValidationError:
        return ()


def synthesize_subtitle_document(words: object, response_format: str) -> str | None:
    """
    Build an SRT/VTT document from OpenAI verbose_json-style word dicts
    (word/start/end in float seconds, optional speaker). Returns None when the
    format is not a subtitle format or the words carry no usable timestamps.
    """
    if response_format not in SUBTITLE_RESPONSE_FORMATS:
        return None
    tokens: Final = tuple(_word_to_subtitle_token(word) for word in _parse_word_timings(words))
    cues: Final = group_subtitle_tokens_into_cues(tokens)
    if not cues:
        return None
    return _render_srt(cues) if response_format == SRT_RESPONSE_FORMAT else _render_vtt(cues)
