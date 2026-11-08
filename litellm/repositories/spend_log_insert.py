from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError
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
)
_JSON_COLUMNS: Final = frozenset({"metadata", "request_tags", "messages", "response", "proxy_server_request"})


class RejectedSpendLogRow(ValueError):
    """A row the raw insert cannot encode, so it is isolated like a row Postgres refuses."""


def _json_value(column: str, value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if column not in _JSON_COLUMNS or not isinstance(value, str):
        return value
    try:
        return _JSON.validate_json(value)
    except ValidationError as error:
        raise RejectedSpendLogRow(f"Spend log column {column} is not valid JSON") from error


def spend_log_insert_statement(rows: Sequence[Mapping[str, object]]) -> tuple[LiteralString, str]:
    if not rows:
        raise ValueError("Spend log insert requires at least one row")
    fields: Final = frozenset(rows[0]) | {"created_at", "updated_at"}
    if not fields.issubset(_COLUMNS) or any(frozenset(row) | {"created_at", "updated_at"} != fields for row in rows):
        raise RejectedSpendLogRow("Spend log insert requires matching known columns")
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
