"""Thin Python wrapper for the native Rust collector bridge."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, cast


class RustCollectorSpendLogs(Protocol):
    def __call__(
        self,
        logs: list[dict[str, Any]],
        auth_context: dict[str, Any],
        now: str,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


def load_rust_collector_spend_logs() -> RustCollectorSpendLogs | None:
    from litellm.rust_bridge import get_native_bridge

    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None
    return cast(
        RustCollectorSpendLogs,
        getattr(native_bridge, "normalize_collector_spend_logs", None),
    )


def normalize_collector_spend_logs(
    *,
    logs: list[dict[str, Any]],
    auth_context: dict[str, Any],
    now: str,
) -> list[dict[str, Any]]:
    rust_normalizer = load_rust_collector_spend_logs()
    if rust_normalizer is not None:
        return rust_normalizer(logs=logs, auth_context=auth_context, now=now)
    return _normalize_collector_spend_logs_python(
        logs=logs,
        auth_context=auth_context,
        now=now,
    )


_COLLECTOR_CALL_TYPE = "litellm-relay"
_DEFAULT_MODEL = "local-ai-traffic"
_DEFAULT_API_KEY = "litellm-relay"


def _normalize_collector_spend_logs_python(
    *,
    logs: list[dict[str, Any]],
    auth_context: dict[str, Any],
    now: str,
) -> list[dict[str, Any]]:
    return [
        _normalize_collector_spend_log_python(
            log=log,
            auth_context=auth_context,
            now=now,
        )
        for log in logs
    ]


def _normalize_collector_spend_log_python(
    *,
    log: dict[str, Any],
    auth_context: dict[str, Any],
    now: str,
) -> dict[str, Any]:
    collector_request_id = log.get("request_id")
    if not isinstance(collector_request_id, str) or not collector_request_id.strip():
        raise ValueError("missing required field: request_id")

    row: dict[str, Any] = {
        "request_id": _collector_request_id_for(auth_context, collector_request_id),
        "call_type": _COLLECTOR_CALL_TYPE,
        "api_key": auth_context.get("key_hash") or _DEFAULT_API_KEY,
        "spend": 0.0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "startTime": now,
        "endTime": now,
        "model": _DEFAULT_MODEL,
        "api_base": "",
        "custom_llm_provider": "",
        "user": auth_context.get("user_id"),
        "metadata": {},
        "cache_hit": "False",
        "cache_key": "",
        "request_tags": json.dumps([_DEFAULT_API_KEY]),
        "messages": {},
        "response": {},
        "proxy_server_request": {},
        "status": "success",
        "team_id": auth_context.get("team_id"),
        "organization_id": auth_context.get("organization_id"),
    }

    for key in _PASSTHROUGH_FIELDS:
        if key in log:
            row[key] = log[key]

    row["request_id"] = _collector_request_id_for(auth_context, collector_request_id)
    row["call_type"] = _COLLECTOR_CALL_TYPE
    row["api_key"] = auth_context.get("key_hash") or _DEFAULT_API_KEY
    row["spend"] = 0.0
    row["custom_llm_provider"] = ""
    row["team_id"] = auth_context.get("team_id")
    row["organization_id"] = auth_context.get("organization_id")
    row["user"] = auth_context.get("user_id")
    row["metadata"] = _normalize_metadata(
        log.get("metadata"), auth_context, collector_request_id
    )
    row["request_tags"] = _normalize_request_tags(log.get("request_tags"))
    return cast(dict[str, Any], _sanitize_json_value(row))


_PASSTHROUGH_FIELDS = {
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "startTime",
    "endTime",
    "completionStartTime",
    "model",
    "model_id",
    "model_group",
    "mcp_namespaced_tool_name",
    "agent_id",
    "api_base",
    "cache_hit",
    "cache_key",
    "end_user",
    "requester_ip_address",
    "messages",
    "response",
    "proxy_server_request",
    "session_id",
    "request_duration_ms",
    "status",
}


def _collector_request_id_for(
    auth_context: dict[str, Any], collector_request_id: str
) -> str:
    key_hash = auth_context.get("key_hash") or _DEFAULT_API_KEY
    digest = hashlib.sha256(f"{key_hash}:{collector_request_id}".encode()).hexdigest()
    return f"collector-{digest[:32]}"


def _normalize_metadata(
    metadata: Any,
    auth_context: dict[str, Any],
    collector_request_id: str,
) -> dict[str, Any]:
    if isinstance(metadata, dict):
        normalized = dict(metadata)
    elif metadata is None:
        normalized = {}
    else:
        normalized = {"relay_raw_metadata": metadata}

    normalized.update(
        {
            "source": _DEFAULT_API_KEY,
            "collector_request_id": collector_request_id,
            "relay_request_id": collector_request_id,
            "user_api_key": auth_context.get("key_hash"),
            "user_api_key_alias": auth_context.get("key_alias"),
            "user_api_key_user_id": auth_context.get("user_id"),
            "user_api_key_team_id": auth_context.get("team_id"),
            "user_api_key_team_alias": auth_context.get("team_alias"),
            "user_api_key_org_id": auth_context.get("organization_id"),
        }
    )
    return normalized


def _normalize_request_tags(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [value] if value.strip() else []
    elif isinstance(value, list):
        parsed = list(value)
    elif value is None:
        parsed = []
    else:
        parsed = [value]

    if _DEFAULT_API_KEY not in parsed:
        parsed.append(_DEFAULT_API_KEY)
    return json.dumps(parsed, separators=(",", ":"))


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key).replace("\x00", ""): _sanitize_json_value(item)
            for key, item in value.items()
        }
    return value
