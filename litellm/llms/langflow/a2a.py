from typing import Any, Final

from litellm.a2a_protocol.utils import (
    get_session_id_from_a2a_params,
    scope_session_to_principal,
)


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
