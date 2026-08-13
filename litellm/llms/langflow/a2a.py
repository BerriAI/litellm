import hashlib
from typing import Any, Final


def get_session_id_from_a2a_params(params: dict[str, Any]) -> str | None:
    message: Final = params.get("message", {})
    if isinstance(message, dict):
        return message.get("contextId")
    return getattr(message, "contextId", None)


def scope_session_to_principal(session_id: str, principal: str | None) -> str:
    """
    Bind a client-supplied A2A contextId to the authenticated principal.

    Without this, two distinct keys authorized for the same LangFlow agent could
    set the same contextId and read/append to each other's LangFlow memory. The
    principal is hashed (it is already a hashed token) so the raw value is never
    sent to the LangFlow backend, while the original contextId is kept as a
    suffix for operator-side correlation.
    """
    if not principal:
        return session_id
    principal_prefix: Final = hashlib.sha256(principal.encode("utf-8")).hexdigest()[:16]
    return f"{principal_prefix}-{session_id}"


def merge_a2a_session_into_litellm_params(
    litellm_params: dict[str, Any],
    params: dict[str, Any],
    principal: str | None = None,
) -> dict[str, Any]:
    merged: Final = dict(litellm_params)
    session_id: Final = get_session_id_from_a2a_params(params)
    if session_id and "session_id" not in merged:
        merged["session_id"] = scope_session_to_principal(session_id, principal)
    return merged
