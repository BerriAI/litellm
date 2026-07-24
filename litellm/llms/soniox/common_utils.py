"""
Shared utilities for the Soniox provider (https://soniox.com).
"""

from typing import Any, Dict, List, Optional

from litellm.llms.base_llm.chat.transformation import BaseLLMException

# Soniox API base URL.
SONIOX_API_BASE: str = "https://api.soniox.com"

# Default polling interval in seconds when waiting for an async transcription
# to finish. Mirrors the Soniox SDK default.
SONIOX_DEFAULT_POLL_INTERVAL: float = 1.0

# Minimum polling interval (in seconds) the server will accept from caller-
# supplied `soniox_polling_interval` kwargs. Prevents an authenticated caller
# from forcing a worker into a tight poll loop with a zero/near-zero interval.
SONIOX_MIN_POLL_INTERVAL: float = 0.5

# Maximum polling interval (in seconds). Prevents a caller from setting an
# excessively large or non-finite interval that would keep a worker sleeping
# far longer than necessary between status checks.
SONIOX_MAX_POLL_INTERVAL: float = 60.0

# Default maximum number of polling attempts (1800 attempts * 1s ~= 30 minutes).
SONIOX_DEFAULT_MAX_POLL_ATTEMPTS: int = 1800

# Hard upper bound on polling attempts. Combined with `SONIOX_MIN_POLL_INTERVAL`
# this caps total polling time per request at ~3000s (50 minutes), preventing a
# caller from pinning a worker indefinitely via a huge attempt count.
SONIOX_MAX_POLL_ATTEMPTS: int = 6000

# Default cleanup behaviour: delete both the uploaded file (if any) and the
# transcription record after the transcript has been fetched.
SONIOX_DEFAULT_CLEANUP: List[str] = ["file", "transcription"]

# Body fields that may carry secrets and must be redacted before being
# forwarded to logging callbacks. Soniox accepts a webhook auth header value
# alongside the create-transcription request; that value lets the recipient
# authenticate webhook callbacks and must not leak into observability sinks.
SONIOX_SECRET_FIELDS: List[str] = ["webhook_auth_header_value"]


class SonioxException(BaseLLMException):
    """Provider-specific exception class for Soniox."""

    pass


def get_soniox_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Resolve the Soniox API key from arg or env var."""
    # Local import to avoid a circular import: litellm.secret_managers.main
    # imports from litellm at top-level.
    from litellm.secret_managers.main import get_secret_str

    return api_key or get_secret_str("SONIOX_API_KEY")


def get_soniox_api_base(api_base: Optional[str] = None) -> str:
    """Resolve the Soniox API base URL from arg or env var (defaults to public API)."""
    from litellm.secret_managers.main import get_secret_str

    base = api_base or get_secret_str("SONIOX_API_BASE") or SONIOX_API_BASE
    return base.rstrip("/")


def render_soniox_tokens(tokens: List[Dict[str, Any]]) -> str:
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

    text_parts: List[str] = []
    current_speaker: Optional[Any] = None
    current_language: Optional[Any] = None

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

_CUE_MAX_CHARS: int = 84

_CUE_MAX_DURATION_MS: int = 7000

_CUE_GAP_MS: int = 700

_SENTENCE_END_CHARS = (".", "!", "?", "。", "！", "？")


def _format_timestamp_srt(ms: int) -> str:
    """Format milliseconds as SRT timestamp: HH:MM:SS,mmm"""
    if ms < 0:
        ms = 0
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1_000
    millis = ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _format_timestamp_vtt(ms: int) -> str:
    """Format milliseconds as VTT timestamp: HH:MM:SS.mmm"""
    if ms < 0:
        ms = 0
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1_000
    millis = ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _merge_tokens_into_words(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge Soniox subword tokens (e.g. ``"Hel"``, ``"lo"``) into whole words.

    A token starts a new word when its text begins with whitespace, when the
    previous token's text ends with whitespace, or when the speaker changes.
    Each word carries the first/last available timestamps of its tokens.

    Translation tokens (``translation_status == "translation"``) are excluded:
    Soniox does not timestamp them, so they cannot be aligned to the audio and
    would otherwise mix translated text into original-language cues.
    """
    words: list[dict[str, Any]] = []
    for token in tokens:
        text = token.get("text", "")
        if not isinstance(text, str) or text == "":
            continue
        if token.get("translation_status") == "translation":
            continue
        is_continuation = (
            bool(words)
            and not text[0].isspace()
            and not words[-1]["text"][-1:].isspace()
            and token.get("speaker") == words[-1]["speaker"]
        )
        if is_continuation:
            last = words[-1]
            last["text"] += text
            if last["start_ms"] is None:
                last["start_ms"] = token.get("start_ms")
            if token.get("end_ms") is not None:
                last["end_ms"] = token.get("end_ms")
        else:
            words.append(
                {
                    "text": text,
                    "start_ms": token.get("start_ms"),
                    "end_ms": token.get("end_ms"),
                    "speaker": token.get("speaker"),
                }
            )
    return words


def _group_tokens_into_cues(
    tokens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Group Soniox tokens into subtitle cues aligned to the actual speech.

    Each cue has:
      - start_ms: int
      - end_ms: int
      - text: str

    Cues only ever break at word boundaries (Soniox tokens are subwords, so
    tokens are first merged into words). A new cue starts when:
      - the speaker changes (if diarization is on),
      - a silence gap of at least _CUE_GAP_MS separates two words, so
        subtitles never bridge pauses in speech,
      - adding the next word would exceed _CUE_MAX_CHARS (~two subtitle
        lines), or
      - the cue would span more than _CUE_MAX_DURATION_MS.
    A cue also ends after sentence-final punctuation, which keeps cue breaks
    at natural seams. Cue timestamps come straight from token timestamps;
    words without timestamps stay attached to the surrounding cue.
    """
    words = _merge_tokens_into_words(tokens)
    cues: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def _cue_start(ws: list[dict[str, Any]]) -> Optional[int]:
        return next((w["start_ms"] for w in ws if w["start_ms"] is not None), None)

    def _cue_end(ws: list[dict[str, Any]]) -> Optional[int]:
        return next((w["end_ms"] for w in reversed(ws) if w["end_ms"] is not None), _cue_start(ws))

    def _cue_text(ws: list[dict[str, Any]]) -> str:
        return "".join(w["text"] for w in ws).strip()

    def _flush() -> None:
        text = _cue_text(current)
        start = _cue_start(current)
        if text and start is not None:
            cues.append({"start_ms": start, "end_ms": _cue_end(current), "text": text})
        current.clear()

    for word in words:
        if current:
            start_ms = word["start_ms"]
            cue_start = _cue_start(current)
            cue_end = _cue_end(current)
            speaker_changed = word["speaker"] is not None and any(
                w["speaker"] is not None and w["speaker"] != word["speaker"] for w in current
            )
            gap_exceeded = start_ms is not None and cue_end is not None and (start_ms - cue_end) >= _CUE_GAP_MS
            chars_exceeded = len(_cue_text(current)) + len(word["text"]) > _CUE_MAX_CHARS
            duration_exceeded = (
                start_ms is not None and cue_start is not None and (start_ms - cue_start) >= _CUE_MAX_DURATION_MS
            )
            if speaker_changed or gap_exceeded or chars_exceeded or duration_exceeded:
                _flush()
        current.append(word)
        if word["text"].rstrip().endswith(_SENTENCE_END_CHARS):
            _flush()

    _flush()
    return cues


def render_soniox_tokens_as_srt(tokens: list[dict[str, Any]]) -> str:
    """
    Render Soniox tokens as SRT (SubRip) subtitle format.

    Returns an empty string if no tokens have timestamp data.
    """
    cues = _group_tokens_into_cues(tokens)
    if not cues:
        return ""

    lines: List[str] = []
    for idx, cue in enumerate(cues, start=1):
        start = _format_timestamp_srt(cue["start_ms"])
        end = _format_timestamp_srt(cue["end_ms"])
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(cue["text"])
        lines.append("")  # blank line between cues

    return "\n".join(lines)


def render_soniox_tokens_as_vtt(tokens: list[dict[str, Any]]) -> str:
    """
    Render Soniox tokens as WebVTT subtitle format.

    Returns the VTT header even if no cues are present.
    """
    cues = _group_tokens_into_cues(tokens)

    lines: List[str] = ["WEBVTT", ""]
    for cue in cues:
        start = _format_timestamp_vtt(cue["start_ms"])
        end = _format_timestamp_vtt(cue["end_ms"])
        lines.append(f"{start} --> {end}")
        lines.append(cue["text"])
        lines.append("")  # blank line between cues

    return "\n".join(lines)
