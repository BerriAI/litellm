"""
Shared helpers for OpenCode's managed inference APIs (Zen and Go).
"""

from collections.abc import Iterable, Mapping
from typing import Final

OPENCODE_SESSION_HEADER: Final = "x-opencode-session"


def resolve_opencode_session_id(litellm_params: Mapping[str, object]) -> str | None:
    """
    OpenCode requires a stable per-conversation id on every inference request so it can pin a
    conversation to one upstream and keep that upstream's prompt cache warm. LiteLLM keeps no
    conversation state of its own, so a caller-supplied session id is the only value that is
    genuinely stable across turns; the per-request ids are a last resort that keeps the request
    accepted even though it forfeits cache affinity.
    """
    metadata: Final = litellm_params.get("metadata")
    candidates: Final = (
        litellm_params.get("litellm_session_id"),
        metadata.get("session_id") if isinstance(metadata, Mapping) else None,
        litellm_params.get("litellm_trace_id"),
        litellm_params.get("litellm_call_id"),
    )
    return next((candidate for candidate in candidates if isinstance(candidate, str) and candidate), None)


def has_opencode_session_header(header_names: Iterable[str]) -> bool:
    return any(name.lower() == OPENCODE_SESSION_HEADER for name in header_names)
