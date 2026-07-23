"""Helpers for stateless provider-aware native Skills IDs."""

import base64
import json
from typing import Any

_SKILL_ID_PREFIX = "litellm_skill_"


def encode_skill_id(skill_id: str, model: str) -> str:
    payload = json.dumps({"id": skill_id, "model": model}, separators=(",", ":")).encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"{_SKILL_ID_PREFIX}{encoded}"


def decode_skill_id(skill_id: str) -> dict[str, str] | None:
    if not isinstance(skill_id, str) or not skill_id.startswith(_SKILL_ID_PREFIX):
        return None
    encoded = skill_id[len(_SKILL_ID_PREFIX) :]
    try:
        payload = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        decoded = json.loads(payload)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(decoded, dict)
        or not isinstance(decoded.get("id"), str)
        or not isinstance(decoded.get("model"), str)
    ):
        return None
    return {"id": decoded["id"], "model": decoded["model"]}


def get_original_skill_id(skill_id: str) -> str:
    decoded = decode_skill_id(skill_id)
    return decoded["id"] if decoded is not None else skill_id


def get_model_from_skill_id(skill_id: str) -> str | None:
    decoded = decode_skill_id(skill_id)
    return decoded["model"] if decoded is not None else None


def rewrite_skill_references(value: Any) -> Any:
    if isinstance(value, dict):
        rewritten = {key: rewrite_skill_references(item) for key, item in value.items()}
        if isinstance(rewritten.get("skill_id"), str):
            rewritten["skill_id"] = get_original_skill_id(rewritten["skill_id"])
        return rewritten
    if isinstance(value, list):
        return [rewrite_skill_references(item) for item in value]
    return value
