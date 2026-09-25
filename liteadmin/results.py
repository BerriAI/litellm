import json
import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue

POLICY_TEXT: Final = ("budget_id", "budget_duration", "budget_reset_at")
POLICY_NUMBERS: Final = ("spend", "max_budget", "soft_budget", "rpm_limit", "tpm_limit")
PAGINATION: Final = ("page", "current_page", "page_size", "total", "total_count", "total_pages")
COST_NUMBERS: Final = (
    "spend",
    "total_spend",
    "total_cost",
    "total_tokens",
    "total_input_tokens",
    "total_output_tokens",
)
TEXT_FIELDS: Final = MappingProxyType(
    {
        "key": (*POLICY_TEXT, "key_alias", "key_name", "team_id", "user_id", "organization_id", "expires"),
        "team": (*POLICY_TEXT, "team_id", "team_alias", "organization_id"),
        "user": (*POLICY_TEXT, "user_id", "user_alias", "user_email", "user_role"),
        "budget": (*POLICY_TEXT, "created_at", "updated_at"),
        "member": ("user_id", "user_email", "role"),
        "cost": ("model",),
        "spend": ("model", "team_id", "team_alias", "team_name", "customer", "group_by_day", "date"),
        "log": (
            "request_id",
            "model",
            "model_id",
            "team_id",
            "user",
            "call_type",
            "status",
            "startTime",
            "endTime",
            "cache_hit",
        ),
    }
)
NUMBER_FIELDS: Final = MappingProxyType(
    {
        "key": (*POLICY_NUMBERS, *PAGINATION),
        "team": (*POLICY_NUMBERS, *PAGINATION),
        "user": (*POLICY_NUMBERS, *PAGINATION),
        "budget": POLICY_NUMBERS,
        "member": ("spend", "max_budget_in_team"),
        "pagination": PAGINATION,
        "cost": COST_NUMBERS,
        "spend": COST_NUMBERS,
        "log": (
            *PAGINATION,
            "spend",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "request_duration_ms",
            "ttft_ms",
        ),
    }
)
NESTED: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "key": MappingProxyType({"keys": "key", "info": "key"}),
        "team": MappingProxyType(
            {"teams": "team", "team_info": "team", "keys": "key", "members": "member", "members_with_roles": "member"}
        ),
        "user": MappingProxyType({"users": "user", "user_info": "user"}),
        "budget": MappingProxyType({"data": "budget", "meta": "pagination"}),
        "spend": MappingProxyType(
            {"teams": "spend", "customers": "spend", "metadata": "cost", "model_details": "cost"}
        ),
        "log": MappingProxyType({"data": "log"}),
    }
)


def safe_text(value: str, secrets: tuple[str, ...]) -> str:
    pattern: Final = "|".join(re.escape(secret) for secret in secrets if secret)
    cleaned: Final = re.sub(pattern, "[redacted]", value) if pattern else value
    return re.sub(r"\bBearer\s+\S+|\bsk-[\w-]+", "[redacted]", cleaned, flags=re.IGNORECASE)[:400]


def safe_hash(value: JsonValue, secrets: tuple[str, ...]) -> str | None:
    return (
        value if isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value) and value not in secrets else None
    )


def project(kind: str, value: JsonValue, secrets: tuple[str, ...], depth: int = 0) -> JsonValue:
    if depth > 5:
        return None
    if isinstance(value, list):
        return [
            project(kind, item, secrets, depth + 1)
            for item in value[: (10 if kind in ("spend", "cost") and depth else 50)]
        ]
    if kind == "key" and isinstance(value, str):
        return safe_hash(value, secrets)
    if not isinstance(value, dict):
        return None
    text: Final = {
        key: safe_text(item, secrets)
        for key, item in value.items()
        if key in TEXT_FIELDS.get(kind, ()) and isinstance(item, str)
    }
    numbers: Final = {
        key: item
        for key, item in value.items()
        if key in NUMBER_FIELDS.get(kind, ())
        and isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(item)
    }
    hashes: Final = {
        key: safe_hash(item, secrets)
        for key, item in value.items()
        if (kind == "key" and key in ("token", "key_hash")) or (kind in ("spend", "cost") and key == "api_key")
    }
    booleans: Final = {
        key: item
        for key, item in value.items()
        if isinstance(item, bool)
        and ((key == "blocked" and kind in ("key", "team", "user", "budget")) or (key == "cache_hit" and kind == "log"))
    }
    lists: Final = {
        key: _text_list(item, secrets)
        for key, item in value.items()
        if (key == "models" and kind in ("key", "team", "user", "budget")) or (key == "teams" and kind == "user")
    }
    nested: Final = {
        key: project(child, value[key], secrets, depth + 1)
        for key, child in NESTED.get(kind, {}).items()
        if key in value
    }
    return {**text, **numbers, **hashes, **booleans, **lists, **nested}


def _text_list(value: JsonValue, secrets: tuple[str, ...]) -> JsonValue:
    return (
        [safe_text(item, secrets) for item in value[:50] if isinstance(item, str)] if isinstance(value, list) else None
    )


def project_result(kind: str, value: JsonValue, secrets: tuple[str, ...]) -> JsonValue:
    valid_shape: Final = (
        isinstance(value, list)
        if kind == "spend"
        else isinstance(value, (list, dict))
        if kind == "budget"
        else isinstance(value, dict)
    )
    projected: Final = project(kind, value, secrets) if valid_shape else None
    if projected is None:
        return {"notice": "The gateway returned an unsupported result shape. Open the resource to inspect it."}
    if len(json.dumps(projected)) > 24000:
        return {"notice": "The result is too large to summarize safely. Use a smaller page or narrower filters."}
    return projected
