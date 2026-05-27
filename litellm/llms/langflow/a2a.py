from typing import Any, Dict, Optional


def get_session_id_from_a2a_params(params: Dict[str, Any]) -> Optional[str]:
    message = params.get("message", {})
    if isinstance(message, dict):
        return message.get("contextId")
    return getattr(message, "contextId", None)


def merge_a2a_session_into_litellm_params(
    litellm_params: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(litellm_params)
    session_id = get_session_id_from_a2a_params(params)
    if session_id and "session_id" not in merged:
        merged["session_id"] = session_id
    return merged
