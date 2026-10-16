"""
Shape of one Delta table row per LiteLLM request.

Zerobus validates every record against the target table and rejects unknown columns, so
the row is a fixed set of scalar columns for filtering plus JSON-encoded ``VARIANT``
columns for anything nested. ``create_table_sql`` renders the matching DDL.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

TRACE_TABLE_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "id": "STRING",
        "trace_id": "STRING",
        "session_id": "STRING",
        "litellm_call_id": "STRING",
        "call_type": "STRING",
        "status": "STRING",
        "model": "STRING",
        "model_group": "STRING",
        "model_id": "STRING",
        "custom_llm_provider": "STRING",
        "api_base": "STRING",
        "stream": "BOOLEAN",
        "cache_hit": "BOOLEAN",
        "start_time": "TIMESTAMP",
        "end_time": "TIMESTAMP",
        "completion_start_time": "TIMESTAMP",
        "response_time": "DOUBLE",
        "prompt_tokens": "LONG",
        "completion_tokens": "LONG",
        "total_tokens": "LONG",
        "response_cost": "DOUBLE",
        "saved_cache_cost": "DOUBLE",
        "api_key_hash": "STRING",
        "api_key_alias": "STRING",
        "team_id": "STRING",
        "team_alias": "STRING",
        "user_id": "STRING",
        "org_id": "STRING",
        "end_user": "STRING",
        "requester_ip_address": "STRING",
        "user_agent": "STRING",
        "request_tags": "VARIANT",
        "messages": "VARIANT",
        "response": "VARIANT",
        "error_str": "STRING",
        "error_information": "VARIANT",
        "metadata": "VARIANT",
        "model_parameters": "VARIANT",
        "hidden_params": "VARIANT",
        "guardrail_information": "VARIANT",
        "cost_breakdown": "VARIANT",
    }
)

_MICROSECONDS: Final = 1_000_000


def create_table_sql(table_name: str) -> str:
    columns: Final = ",\n".join(f"  {name} {delta_type}" for name, delta_type in TRACE_TABLE_COLUMNS.items())
    return f"CREATE TABLE {table_name} (\n{columns}\n);"


def _text(payload: Mapping[str, object], key: str) -> str | None:
    value: Final = payload.get(key)
    return value if isinstance(value, str) else None


def _flag(payload: Mapping[str, object], key: str) -> bool | None:
    value: Final = payload.get(key)
    return value if isinstance(value, bool) else None


def _number(payload: Mapping[str, object], key: str) -> float | None:
    value: Final = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _count(payload: Mapping[str, object], key: str) -> int | None:
    value: Final = _number(payload, key)
    return None if value is None else int(value)


def _timestamp_micros(payload: Mapping[str, object], key: str) -> int | None:
    """Delta ``TIMESTAMP`` over Zerobus is epoch microseconds; LiteLLM keeps epoch seconds."""
    seconds: Final = _number(payload, key)
    if seconds is None or seconds <= 0:
        return None
    return int(seconds * _MICROSECONDS)


def _json(payload: Mapping[str, object], key: str) -> str | None:
    value: Final = payload.get(key)
    return None if value is None else safe_dumps(value)


def _metadata(payload: Mapping[str, object]) -> Mapping[str, object]:
    value: Final = payload.get("metadata")
    return value if isinstance(value, Mapping) else MappingProxyType({})


def trace_row(payload: Mapping[str, object]) -> Mapping[str, object]:
    """One ``TRACE_TABLE_COLUMNS`` row for a ``StandardLoggingPayload``."""
    metadata: Final = _metadata(payload)
    return MappingProxyType(
        {
            "id": _text(payload, "id"),
            "trace_id": _text(payload, "trace_id"),
            "session_id": _text(payload, "session_id"),
            "litellm_call_id": _text(payload, "litellm_call_id"),
            "call_type": _text(payload, "call_type"),
            "status": _text(payload, "status"),
            "model": _text(payload, "model"),
            "model_group": _text(payload, "model_group"),
            "model_id": _text(payload, "model_id"),
            "custom_llm_provider": _text(payload, "custom_llm_provider"),
            "api_base": _text(payload, "api_base"),
            "stream": _flag(payload, "stream"),
            "cache_hit": _flag(payload, "cache_hit"),
            "start_time": _timestamp_micros(payload, "startTime"),
            "end_time": _timestamp_micros(payload, "endTime"),
            "completion_start_time": _timestamp_micros(payload, "completionStartTime"),
            "response_time": _number(payload, "response_time"),
            "prompt_tokens": _count(payload, "prompt_tokens"),
            "completion_tokens": _count(payload, "completion_tokens"),
            "total_tokens": _count(payload, "total_tokens"),
            "response_cost": _number(payload, "response_cost"),
            "saved_cache_cost": _number(payload, "saved_cache_cost"),
            "api_key_hash": _text(metadata, "user_api_key_hash"),
            "api_key_alias": _text(metadata, "user_api_key_alias"),
            "team_id": _text(metadata, "user_api_key_team_id"),
            "team_alias": _text(metadata, "user_api_key_team_alias"),
            "user_id": _text(metadata, "user_api_key_user_id"),
            "org_id": _text(metadata, "user_api_key_org_id"),
            "end_user": _text(payload, "end_user"),
            "requester_ip_address": _text(payload, "requester_ip_address"),
            "user_agent": _text(payload, "user_agent"),
            "request_tags": _json(payload, "request_tags"),
            "messages": _json(payload, "messages"),
            "response": _json(payload, "response"),
            "error_str": _text(payload, "error_str"),
            "error_information": _json(payload, "error_information"),
            "metadata": _json(payload, "metadata"),
            "model_parameters": _json(payload, "model_parameters"),
            "hidden_params": _json(payload, "hidden_params"),
            "guardrail_information": _json(payload, "guardrail_information"),
            "cost_breakdown": _json(payload, "cost_breakdown"),
        }
    )
