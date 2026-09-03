"""
Shared helpers for OpenCode's managed inference APIs (Zen and Go).
"""

from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from typing import Final

from litellm.secret_managers.main import get_secret_str

OPENCODE_SESSION_HEADER: Final = "x-opencode-session"
OPENCODE_API_KEY_ENV_VAR: Final = "OPENCODE_API_KEY"


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


def with_opencode_session_header(
    headers: Mapping[str, object], litellm_params: Mapping[str, object]
) -> dict[str, object]:  # mutable-ok: every caller hands this straight back to validate_environment, which returns dict
    """Stamp the session header onto already-authenticated headers, leaving a caller's own value alone."""
    session_id: Final = None if has_opencode_session_header(headers) else resolve_opencode_session_id(litellm_params)
    stamped: Final = () if session_id is None else ((OPENCODE_SESSION_HEADER, session_id),)
    return dict(chain(headers.items(), stamped))


def opencode_endpoint_for_model(provider: str, model: str) -> str:
    """
    OpenCode splits its catalogue across three wire formats and the split is per model, not
    per family: on Go, qwen and minimax are served by `/messages` while glm, kimi and deepseek
    are served by `/chat/completions`. The cost map records each model's endpoint, so read it
    from there and fall back to the OpenAI-compatible path for anything unlisted.
    """
    import litellm

    entry: Final = litellm.model_cost.get(f"{provider}/{model}") or litellm.model_cost.get(model)
    endpoints: Final = entry.get("supported_endpoints") if isinstance(entry, Mapping) else None
    if isinstance(endpoints, Sequence) and not isinstance(endpoints, str) and endpoints:
        first: Final = endpoints[0]
        if isinstance(first, str):
            return first
    return "/v1/chat/completions"


def resolve_opencode_api_key(api_key: str | None) -> str | None:
    """One OpenCode account key authenticates both Zen and Go, so both surfaces read one variable."""
    return api_key or get_secret_str(OPENCODE_API_KEY_ENV_VAR)
