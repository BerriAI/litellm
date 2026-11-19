from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Final, cast

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import LiteralString

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps, safe_json_structure

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
_INT_COLUMNS: Final = frozenset({"total_tokens", "prompt_tokens", "completion_tokens", "request_duration_ms"})
_TIMESTAMP_COLUMNS: Final = frozenset({"startTime", "endTime", "completionStartTime", "created_at", "updated_at"})


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


def spend_log_bound_insert_statement(
    rows: Sequence[Mapping[str, object]],
) -> tuple[LiteralString, *tuple[str | None, ...]]:
    if not rows:
        raise ValueError("Spend log insert requires at least one row")
    fields: Final = frozenset(rows[0]) | {"created_at", "updated_at"}
    if not fields.issubset(_COLUMNS) or any(frozenset(row) | {"created_at", "updated_at"} != fields for row in rows):
        raise RejectedSpendLogRow("Spend log insert requires matching known columns")
    ordered: Final = tuple(column for column in _COLUMNS if column in fields)
    if len(rows) * len(ordered) > 32_767 or DEFAULT_MAX_RECURSE_DEPTH < 2:
        return spend_log_insert_statement(rows)
    columns: Final = ", ".join(f'"{column}"' for column in ordered)
    now: Final = datetime.now(timezone.utc).isoformat()
    values: Final = tuple(
        {"created_at": now, "updated_at": now, **row} for row in rows
    )
    if any(not _can_bind(column, row[column]) for row in values for column in ordered):
        return spend_log_insert_statement(rows)
    tuples: Final = ", ".join(
        "(" + ", ".join(
            f"${row * len(ordered) + offset + 1}::{_column_type(column)}"
            for offset, column in enumerate(ordered)
        ) + ")" for row in range(len(rows))
    )
    query: Final = cast(
        LiteralString,
        f'INSERT INTO "LiteLLM_SpendLogs" ({columns}) VALUES {tuples} '
        "ON CONFLICT DO NOTHING RETURNING request_id",
    )
    return query, *(_binding(column, row[column]) for row in values for column in ordered)


def _column_type(column: str) -> LiteralString:
    if column in _JSON_COLUMNS:
        return "jsonb"
    if column in _INT_COLUMNS:
        return "int4"
    if column in _TIMESTAMP_COLUMNS:
        return "timestamp"
    return "float8" if column == "spend" else "text"


def _can_bind(column: str, value: object) -> bool:
    if value is None or column in _JSON_COLUMNS:
        return True
    if column in _INT_COLUMNS:
        return type(value) is int or isinstance(value, str)
    if column == "spend":
        return type(value) in (int, float) or isinstance(value, str)
    return isinstance(value, str) or (column in _TIMESTAMP_COLUMNS and isinstance(value, datetime))


def _binding(column: str, value: object) -> str | None:
    native: Final = _json_value(column, value)
    depth: Final = DEFAULT_MAX_RECURSE_DEPTH - 2
    if column in _JSON_COLUMNS:
        payload: Final = safe_dumps(native, max_depth=depth, non_finite_as_null=True)
        return None if payload == "null" else payload
    normalized: Final = safe_json_structure(native, max_depth=depth, non_finite_as_null=True)
    if normalized is None:
        return None
    return "0" if type(normalized) is float and normalized == 0 else str(normalized)
