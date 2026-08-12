"""
Shared utilities for the Soniox provider (https://soniox.com).
"""

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any, Final

from litellm.llms.base_llm.chat.transformation import BaseLLMException

# Soniox API base URL.
SONIOX_API_BASE: Final[str] = "https://api.soniox.com"

# Default polling interval in seconds when waiting for an async transcription
# to finish. Mirrors the Soniox SDK default.
SONIOX_DEFAULT_POLL_INTERVAL: Final[float] = 1.0

# Minimum polling interval (in seconds) the server will accept from caller-
# supplied `soniox_polling_interval` kwargs. Prevents an authenticated caller
# from forcing a worker into a tight poll loop with a zero/near-zero interval.
SONIOX_MIN_POLL_INTERVAL: Final[float] = 0.5

# Maximum polling interval (in seconds). Prevents a caller from setting an
# excessively large or non-finite interval that would keep a worker sleeping
# far longer than necessary between status checks.
SONIOX_MAX_POLL_INTERVAL: Final[float] = 60.0

# Default maximum number of polling attempts (1800 attempts * 1s ~= 30 minutes).
SONIOX_DEFAULT_MAX_POLL_ATTEMPTS: Final[int] = 1800

# Hard upper bound on polling attempts. Combined with `SONIOX_MIN_POLL_INTERVAL`
# this caps total polling time per request at ~3000s (50 minutes), preventing a
# caller from pinning a worker indefinitely via a huge attempt count.
SONIOX_MAX_POLL_ATTEMPTS: Final[int] = 6000

# Default cleanup behaviour: delete both the uploaded file (if any) and the
# transcription record after the transcript has been fetched.
SONIOX_DEFAULT_CLEANUP: Final[list[str]] = ["file", "transcription"]

# Body fields that may carry secrets and must be redacted before being
# forwarded to logging callbacks. Soniox accepts a webhook auth header value
# alongside the create-transcription request; that value lets the recipient
# authenticate webhook callbacks and must not leak into observability sinks.
SONIOX_SECRET_FIELDS: Final[list[str]] = ["webhook_auth_header_value"]


class SonioxException(BaseLLMException):
    """Provider-specific exception class for Soniox."""


def get_soniox_api_key(api_key: str | None = None) -> str | None:
    """Resolve the Soniox API key from arg or env var."""
    # Local import to avoid a circular import: litellm.secret_managers.main
    # imports from litellm at top-level.
    from litellm.secret_managers.main import get_secret_str

    return api_key or get_secret_str("SONIOX_API_KEY")


def get_soniox_api_base(api_base: str | None = None) -> str:
    """Resolve the Soniox API base URL from arg or env var (defaults to public API)."""
    from litellm.secret_managers.main import get_secret_str

    base: Final = api_base or get_secret_str("SONIOX_API_BASE") or SONIOX_API_BASE
    return base.rstrip("/")


def render_soniox_tokens(tokens: list[dict[str, Any]]) -> str:
    """
    Render a list of Soniox tokens to a readable transcript string.

    Mirrors the behaviour of the official Soniox SDK's `renderTokens` helper:
    - When the speaker changes, a `Speaker N:` tag is inserted.
    - When the language changes, a `[lang]` (or `[Translation][lang]`) tag is
      inserted.

    If neither speaker nor language information is present on any token (i.e.
    diarization and language identification are disabled), the function simply
    concatenates the token texts.
    """
    if not tokens:
        return ""

    text_parts: Final[list[str]] = []
    current_speaker: Any | None = None
    current_language: Any | None = None

    for token in tokens:
        text = token.get("text", "")
        speaker = token.get("speaker")
        language = token.get("language")
        is_translation = token.get("translation_status") == "translation"

        # Speaker changed -> emit a speaker tag.
        if speaker is not None and speaker != current_speaker:
            if current_speaker is not None:
                text_parts.append("\n\n")
            current_speaker = speaker
            current_language = None  # reset language whenever speaker changes
            text_parts.append(f"Speaker {current_speaker}:")

        # Language changed -> emit a language (or translation) tag.
        if language is not None and language != current_language:
            current_language = language
            prefix = "[Translation] " if is_translation else ""
            text_parts.append(f"\n{prefix}[{current_language}] ")
            text = text.lstrip() if isinstance(text, str) else text

        text_parts.append(text)

    return "".join(text_parts)


# ---------------------------------------------------------------------------
# SRT / VTT subtitle rendering
# ---------------------------------------------------------------------------

_CUE_MAX_CHARS: Final[int] = 84

_CUE_MAX_DURATION_MS: Final[int] = 7000

_CUE_GAP_MS: Final[int] = 700

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


def _is_cjk(ch: str) -> bool:
    cp: Final = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _is_cjk_word_boundary(prev_ch: str, next_ch: str) -> bool:
    if not (_is_cjk(prev_ch) or _is_cjk(next_ch)):
        return False
    return next_ch not in _CJK_NO_BREAK_BEFORE and prev_ch not in _CJK_NO_BREAK_AFTER


def _text_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _format_timestamp_srt(ms: int) -> str:
    """Format milliseconds as SRT timestamp: HH:MM:SS,mmm"""
    ms = max(ms, 0)
    hours: Final = ms // 3_600_000
    ms %= 3_600_000
    minutes: Final = ms // 60_000
    ms %= 60_000
    seconds: Final = ms // 1_000
    millis: Final = ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _format_timestamp_vtt(ms: int) -> str:
    """Format milliseconds as VTT timestamp: HH:MM:SS.mmm"""
    ms = max(ms, 0)
    hours: Final = ms // 3_600_000
    ms %= 3_600_000
    minutes: Final = ms // 60_000
    ms %= 60_000
    seconds: Final = ms // 1_000
    millis: Final = ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


@dataclass(frozen=True, slots=True)
class _Word:
    text: str
    start_ms: int | None
    end_ms: int | None
    speaker: str | int | None


@dataclass(frozen=True, slots=True)
class _Cue:
    start_ms: int
    end_ms: int
    text: str


def _keeps_token(token: Mapping[str, Any]) -> bool:
    text: Final = token.get("text", "")
    return isinstance(text, str) and text != "" and token.get("translation_status") != "translation"


def _starts_new_word(prev: Mapping[str, Any], token: Mapping[str, Any]) -> bool:
    prev_last: Final = prev["text"][-1:]
    first: Final = token["text"][0]
    return (
        first.isspace()
        or prev_last.isspace()
        or token.get("speaker") != prev.get("speaker")
        or _is_cjk_word_boundary(prev_last, first)
    )


def _build_word(group: Sequence[Mapping[str, Any]]) -> _Word:
    return _Word(
        text="".join(t["text"] for t in group),
        start_ms=next((t.get("start_ms") for t in group if t.get("start_ms") is not None), None),
        end_ms=next((t.get("end_ms") for t in reversed(group) if t.get("end_ms") is not None), None),
        speaker=group[0].get("speaker"),
    )


def _merge_tokens_into_words(tokens: Sequence[Mapping[str, Any]]) -> tuple[_Word, ...]:
    """
    Merge Soniox subword tokens (e.g. ``"Hel"``, ``"lo"``) into whole words.

    A token starts a new word when its text begins with whitespace, when the
    previous token's text ends with whitespace, when the speaker changes, or
    at a CJK character boundary (CJK scripts carry no spaces, so without this
    an entire utterance would fuse into a single unbreakable "word"; CJK
    punctuation stays attached to the preceding character per kinsoku rules).
    Each word carries the first/last available timestamps of its tokens.

    Translation tokens (``translation_status == "translation"``) are excluded:
    Soniox does not timestamp them, so they cannot be aligned to the audio and
    would otherwise mix translated text into original-language cues.
    """
    kept: Final = tuple(t for t in tokens if _keeps_token(t))
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
    gap_exceeded: Final = word.start_ms is not None and cue_end is not None and (word.start_ms - cue_end) >= _CUE_GAP_MS
    chars_exceeded: Final = _text_width(_cue_text(cue)) + _text_width(word.text) > _CUE_MAX_CHARS
    word_end: Final = word.end_ms if word.end_ms is not None else word.start_ms
    duration_exceeded: Final = (
        word_end is not None and cue_start is not None and (word_end - cue_start) > _CUE_MAX_DURATION_MS
    )
    return speaker_changed or gap_exceeded or chars_exceeded or duration_exceeded


def _cue_start_indices(words: Sequence[_Word]) -> tuple[int, ...]:
    def step(starts: tuple[int, ...], index: int) -> tuple[int, ...]:
        if words[index - 1].text.rstrip().endswith(_SENTENCE_END_CHARS):
            return (*starts, index)
        if _should_break(words[starts[-1] : index], words[index]):
            return (*starts, index)
        return starts

    return reduce(step, range(1, len(words)), (0,)) if words else ()


def _build_cue(ws: Sequence[_Word]) -> _Cue | None:
    text: Final = _cue_text(ws)
    start: Final = _cue_start(ws)
    if not text or start is None:
        return None
    end: Final = _cue_end(ws)
    return _Cue(start_ms=start, end_ms=end if end is not None else start, text=text)


def _group_tokens_into_cues(tokens: Sequence[Mapping[str, Any]]) -> tuple[_Cue, ...]:
    """
    Group Soniox tokens into subtitle cues aligned to the actual speech.

    Cues only ever break at word boundaries (Soniox tokens are subwords, so
    tokens are first merged into words). A new cue starts when:
      - the speaker changes (if diarization is on),
      - a silence gap of at least _CUE_GAP_MS separates two words, so
        subtitles never bridge pauses in speech,
      - adding the next word would exceed _CUE_MAX_CHARS of display width
        (~two subtitle lines; East-Asian wide characters count double), or
      - adding the next word would make the cue span more than
        _CUE_MAX_DURATION_MS.
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


def render_soniox_tokens_as_srt(tokens: list[dict[str, Any]]) -> str:
    """
    Render Soniox tokens as SRT (SubRip) subtitle format.

    Returns an empty string if no tokens have timestamp data.
    """
    cues: Final = _group_tokens_into_cues(tokens)
    if not cues:
        return ""

    return "\n".join(
        line
        for idx, cue in enumerate(cues, start=1)
        for line in (
            str(idx),
            f"{_format_timestamp_srt(cue.start_ms)} --> {_format_timestamp_srt(cue.end_ms)}",
            cue.text,
            "",
        )
    )


def render_soniox_tokens_as_vtt(tokens: list[dict[str, Any]]) -> str:
    """
    Render Soniox tokens as WebVTT subtitle format.

    Returns the VTT header even if no cues are present.
    """
    cues: Final = _group_tokens_into_cues(tokens)

    lines: Final = (
        "WEBVTT",
        "",
        *(
            line
            for cue in cues
            for line in (
                f"{_format_timestamp_vtt(cue.start_ms)} --> {_format_timestamp_vtt(cue.end_ms)}",
                cue.text,
                "",
            )
        ),
    )

    return "\n".join(lines)
