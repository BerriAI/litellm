from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Final

from pydantic import JsonValue, TypeAdapter
from typing_extensions import LiteralString

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)

_COLUMNS: Final[tuple[LiteralString, ...]] = (
    "request_id",
    "call_type",
    "api_key",
    "spend",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "startTime",
    "endTime",
    "request_duration_ms",
    "completionStartTime",
    "model",
    "model_id",
    "model_group",
    "custom_llm_provider",
    "api_base",
    "user",
    "metadata",
    "cache_hit",
    "cache_key",
    "request_tags",
    "team_id",
    "organization_id",
    "end_user",
    "requester_ip_address",
    "messages",
    "response",
    "session_id",
    "status",
    "mcp_namespaced_tool_name",
    "billing_agent_id",
    "agent_id",
    "proxy_server_request",
    "litellm_call_id",
    "created_at",
    "updated_at",
    "accounting_path",
)
_JSON_COLUMNS: Final = frozenset({"metadata", "request_tags", "messages", "response", "proxy_server_request"})


def _json_value(column: str, value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if column in _JSON_COLUMNS and isinstance(value, str):
        return _JSON.validate_json(value)
    return value


def spend_log_insert_statement(rows: Sequence[Mapping[str, object]]) -> tuple[LiteralString, str]:
    if not rows:
        raise ValueError("Spend log insert requires at least one row")
    fields: Final = frozenset(rows[0]) | {"created_at", "updated_at"}
    if not fields.issubset(_COLUMNS) or any(frozenset(row) | {"created_at", "updated_at"} != fields for row in rows):
        raise ValueError("Spend log insert requires matching known columns")
    columns: Final = ", ".join('"' + column + '"' for column in _COLUMNS if column in fields)
    now: Final = datetime.now(timezone.utc).isoformat()
    values: Final = tuple(
        {"created_at": now, "updated_at": now, **{column: _json_value(column, value) for column, value in row.items()}}
        for row in rows
    )
    sql: Final = (
        'INSERT INTO "LiteLLM_SpendLogs" ('
        + columns
        + ") SELECT "
        + columns
        + ' FROM jsonb_populate_recordset(NULL::"LiteLLM_SpendLogs", $1::jsonb) '
        "ON CONFLICT DO NOTHING RETURNING request_id"
    )
    return sql, safe_dumps(values, non_finite_as_null=True)


def spend_log_takeover_statement(row: Mapping[str, object]) -> tuple[LiteralString, str]:
    if not frozenset(row).issubset(_COLUMNS):
        raise ValueError("Batch takeover requires known SpendLogs columns")
    columns: Final[tuple[LiteralString, ...]] = tuple(
        column
        for column in _COLUMNS
        if column in row and column not in {"request_id", "accounting_path", "created_at", "updated_at"}
    )
    assignments: Final[LiteralString] = ", ".join(
        '"' + column + '" = replacement."' + column + '"' for column in columns
    )
    sql: Final = (
        'UPDATE "LiteLLM_SpendLogs" AS stored SET '
        + assignments
        + ', updated_at = CURRENT_TIMESTAMP FROM jsonb_populate_record(NULL::"LiteLLM_SpendLogs", $1::jsonb) replacement '
        "WHERE stored.request_id = replacement.request_id AND stored.call_type = 'aretrieve_batch' "
        "AND stored.status = 'success' AND stored.spend = 0 RETURNING stored.request_id"
    )
    return sql, safe_dumps(
        {column: _json_value(column, value) for column, value in row.items()}, non_finite_as_null=True
    )
