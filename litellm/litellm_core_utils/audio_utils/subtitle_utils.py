"""Provider-agnostic SRT/WebVTT subtitle synthesis from timestamped transcription tokens."""

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, groupby
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

CUE_MAX_CHARS: Final = 84
CUE_MAX_DURATION_MS: Final = 7000
CUE_GAP_MS: Final = 700

SRT_RESPONSE_FORMAT: Final = "srt"
VTT_RESPONSE_FORMAT: Final = "vtt"
SUBTITLE_RESPONSE_FORMATS: Final = frozenset((SRT_RESPONSE_FORMAT, VTT_RESPONSE_FORMAT))

_SENTENCE_END_CHARS: Final = (".", "!", "?", "。", "！", "？", "؟", "۔", "।", "॥", "։", "።")

_CJK_RANGES: Final = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0x31F0, 0x31FF),
)

_CJK_NO_BREAK_BEFORE: Final = "、。，．！？：；・ー…」』）〉》】〕"

_CJK_NO_BREAK_AFTER: Final = "「『（〈《【〔"


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
class _Word:
    text: str
    start_ms: int | None
    end_ms: int | None
    speaker: str | int | None


def _is_cjk(ch: str) -> bool:
    cp: Final = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _is_cjk_word_boundary(prev_ch: str, next_ch: str) -> bool:
    if not (_is_cjk(prev_ch) or _is_cjk(next_ch)):
        return False
    return next_ch not in _CJK_NO_BREAK_BEFORE and prev_ch not in _CJK_NO_BREAK_AFTER


def _text_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _starts_new_word(prev: SubtitleToken, token: SubtitleToken) -> bool:
    prev_last: Final = prev.text[-1:]
    first: Final = token.text[0]
    return (
        first.isspace()
        or prev_last.isspace()
        or token.speaker != prev.speaker
        or _is_cjk_word_boundary(prev_last, first)
    )


def _build_word(group: Sequence[SubtitleToken]) -> _Word:
    return _Word(
        text="".join(t.text for t in group),
        start_ms=next((t.start_ms for t in group if t.start_ms is not None), None),
        end_ms=next((t.end_ms for t in reversed(group) if t.end_ms is not None), None),
        speaker=group[0].speaker,
    )


def _merge_tokens_into_words(tokens: Sequence[SubtitleToken]) -> tuple[_Word, ...]:
    """
    Merge subword tokens (e.g. ``"Hel"``, ``"lo"``) into whole words.

    A token starts a new word when its text begins with whitespace, when the
    previous token's text ends with whitespace, when the speaker changes, or
    at a CJK character boundary (CJK scripts carry no spaces, so without this
    an entire utterance would fuse into a single unbreakable "word"; CJK
    punctuation stays attached to the preceding character per kinsoku rules).
    Each word carries the first/last available timestamps of its tokens.
    """
    kept: Final = tuple(t for t in tokens if t.text != "")
    starts: Final = tuple(i for i, t in enumerate(kept) if i == 0 or _starts_new_word(kept[i - 1], t))
    return tuple(_build_word(kept[begin:end]) for begin, end in zip(starts, (*starts[1:], len(kept))))


def _cue_start(ws: Sequence[_Word]) -> int | None:
    return next((w.start_ms for w in ws if w.start_ms is not None), None)


def _cue_end(ws: Sequence[_Word]) -> int | None:
    return next((w.end_ms for w in reversed(ws) if w.end_ms is not None), _cue_start(ws))


def _cue_text(ws: Sequence[_Word]) -> str:
    return "".join(w.text for w in ws).strip()


def _should_break(cue: Sequence[_Word], word: _Word) -> bool:
    speaker_changed: Final = word.speaker is not None and any(
        w.speaker is not None and w.speaker != word.speaker for w in cue
    )
    cue_start: Final = _cue_start(cue)
    cue_end: Final = _cue_end(cue)
    gap_exceeded: Final = word.start_ms is not None and cue_end is not None and (word.start_ms - cue_end) >= CUE_GAP_MS
    chars_exceeded: Final = _text_width(_cue_text(cue)) + _text_width(word.text) > CUE_MAX_CHARS
    word_end: Final = word.end_ms if word.end_ms is not None else word.start_ms
    duration_exceeded: Final = (
        word_end is not None and cue_start is not None and (word_end - cue_start) > CUE_MAX_DURATION_MS
    )
    return speaker_changed or gap_exceeded or chars_exceeded or duration_exceeded


def _cue_start_indices(words: Sequence[_Word]) -> tuple[int, ...]:
    def next_start(start: int, index: int) -> int:
        if words[index - 1].text.rstrip().endswith(_SENTENCE_END_CHARS):
            return index
        if _should_break(words[start:index], words[index]):
            return index
        return start

    if not words:
        return ()
    return tuple(start for start, _ in groupby(accumulate(range(1, len(words)), next_start, initial=0)))


def _build_cue(ws: Sequence[_Word]) -> SubtitleCue | None:
    text: Final = _cue_text(ws)
    start: Final = _cue_start(ws)
    if not text or start is None:
        return None
    end: Final = _cue_end(ws)
    return SubtitleCue(start_ms=start, end_ms=end if end is not None else start, text=text)


def group_subtitle_tokens_into_cues(tokens: Sequence[SubtitleToken]) -> tuple[SubtitleCue, ...]:
    """
    Group transcription tokens into subtitle cues aligned to the actual speech.

    Cues only ever break at word boundaries (tokens may be subwords, so they
    are first merged into words). A new cue starts when:
      - the speaker changes (if diarization is on),
      - a silence gap of at least CUE_GAP_MS separates two words, so
        subtitles never bridge pauses in speech,
      - adding the next word would exceed CUE_MAX_CHARS of display width
        (~two subtitle lines; East-Asian wide characters count double), or
      - adding the next word would make the cue span more than
        CUE_MAX_DURATION_MS.
    A cue also ends after sentence-final punctuation, which keeps cue breaks
    at natural seams. Cue timestamps come straight from token timestamps;
    words without timestamps stay attached to the surrounding cue, and a cue
    whose words carry no timestamps at all is dropped.
    """
    words: Final = _merge_tokens_into_words(tokens)
    starts: Final = _cue_start_indices(words)
    return tuple(
        cue
        for begin, end in zip(starts, (*starts[1:], len(words)))
        if (cue := _build_cue(words[begin:end])) is not None
    )


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
