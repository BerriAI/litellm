"""
Shared base-URL handling for Melious.

Melious serves the OpenAI-compatible surface under ``/v1`` and the
Anthropic-compatible Messages API under the same root, so both configs resolve
``MELIOUS_API_BASE`` and normalize whatever shape the caller supplied.
"""

from typing import Final

MELIOUS_API_BASE: Final = "https://api.melious.ai"
MELIOUS_OPENAI_API_BASE: Final = f"{MELIOUS_API_BASE}/v1"

_CHAT_COMPLETIONS_SUFFIX: Final = "/chat/completions"
_MESSAGES_SUFFIX: Final = "/v1/messages"
_VERSION_SUFFIX: Final = "/v1"


def openai_api_base(api_base: str) -> str:
    """``{root}/v1``, whether the caller passed the root, ``/v1``, or the full chat URL."""
    trimmed: Final = api_base.rstrip("/").removesuffix(_CHAT_COMPLETIONS_SUFFIX)
    return trimmed if trimmed.endswith(_VERSION_SUFFIX) else f"{trimmed}{_VERSION_SUFFIX}"


def anthropic_messages_url(api_base: str) -> str:
    """``{root}/v1/messages``, whether the caller passed the root, ``/v1``, or the full URL."""
    trimmed: Final = api_base.rstrip("/")
    if trimmed.endswith(_MESSAGES_SUFFIX):
        return trimmed
    root: Final = trimmed.removesuffix(_CHAT_COMPLETIONS_SUFFIX).removesuffix(_VERSION_SUFFIX)
    return f"{root}{_MESSAGES_SUFFIX}"
