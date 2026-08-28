"""
Shared utilities for the Soniox provider (https://soniox.com).
"""

from typing import Any, Final

from litellm.litellm_core_utils.audio_utils.subtitle_utils import (
    SubtitleToken,
    render_subtitle_tokens_as_srt,
    render_subtitle_tokens_as_vtt,
)
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


def _soniox_token_to_subtitle_token(token: dict[str, Any]) -> SubtitleToken:
    return SubtitleToken(
        text=token.get("text", ""),
        start_ms=token.get("start_ms"),
        end_ms=token.get("end_ms"),
        speaker=token.get("speaker"),
    )


def render_soniox_tokens_as_srt(tokens: list[dict[str, Any]]) -> str:
    """
    Render Soniox tokens as SRT (SubRip) subtitle format.

    Returns an empty string if no tokens have timestamp data.
    """
    return render_subtitle_tokens_as_srt(tuple(_soniox_token_to_subtitle_token(token) for token in tokens))


def render_soniox_tokens_as_vtt(tokens: list[dict[str, Any]]) -> str:
    """
    Render Soniox tokens as WebVTT subtitle format.

    Returns the VTT header even if no cues are present.
    """
    return render_subtitle_tokens_as_vtt(tuple(_soniox_token_to_subtitle_token(token) for token in tokens))
