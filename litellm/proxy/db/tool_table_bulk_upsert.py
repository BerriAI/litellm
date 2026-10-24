"""One multi-row ``INSERT ... ON CONFLICT ("tool_name") DO UPDATE`` per batch of discovered tools.

LiteLLM_ToolTable holds one row per tool name the proxy has seen. A batch arrives already folded to one
entry per tool name with a summed call_count, and this statement inserts the names the table has not seen
with the untrusted policies the columns default to, and for the names it has adds the batch's call_count
while leaving the policies and the discovering caller's fields alone. Same shape as daily_spend_bulk_upsert.
"""

import uuid
from collections.abc import Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import Final, TypeAlias

from litellm.proxy._types import ToolDiscoveryQueueItem

SqlValue: TypeAlias = str | int | None

_TOOL_ROW_COLUMNS: Final[tuple[str, ...]] = (
    "tool_id",
    "tool_name",
    "origin",
    "call_count",
    "created_by",
    "updated_by",
    "key_hash",
    "team_id",
    "key_alias",
    "user_agent",
)
_CASTS: Final[Mapping[str, str]] = MappingProxyType({"call_count": "integer"})


def _row_params(item: ToolDiscoveryQueueItem) -> tuple[SqlValue, ...]:
    created_by: Final = item.get("created_by") or "system"
    return (
        str(uuid.uuid4()),
        item.get("tool_name", ""),
        item.get("origin") or "user_defined",
        item.get("call_count", 1),
        created_by,
        created_by,
        item.get("key_hash"),
        item.get("team_id"),
        item.get("key_alias"),
        item.get("user_agent"),
    )


def build_tool_bulk_upsert(items: Sequence[ToolDiscoveryQueueItem]) -> tuple[str, tuple[SqlValue, ...]]:
    """The single statement writing one batch of distinct tool_names, plus its positional arguments."""
    width: Final = len(_TOOL_ROW_COLUMNS)
    rows: Final = ", ".join(
        "("
        + ", ".join(
            f"${row_index * width + offset + 1}::{_CASTS.get(column, 'text')}"
            for offset, column in enumerate(_TOOL_ROW_COLUMNS)
        )
        + ", (NOW() AT TIME ZONE 'UTC'), (NOW() AT TIME ZONE 'UTC'))"
        for row_index in range(len(items))
    )
    columns: Final = ", ".join(f'"{column}"' for column in _TOOL_ROW_COLUMNS)
    sql: Final = (
        f'INSERT INTO "LiteLLM_ToolTable" ({columns}, "last_used_at", "updated_at")\n'
        f"VALUES {rows}\n"
        'ON CONFLICT ("tool_name") DO UPDATE SET\n'
        '  "call_count" = "LiteLLM_ToolTable"."call_count" + EXCLUDED."call_count",\n'
        '  "last_used_at" = EXCLUDED."last_used_at",\n'
        '  "updated_at" = EXCLUDED."updated_at"'
    )
    return sql, tuple(chain.from_iterable(map(_row_params, items)))
