"""One multi-row ``INSERT ... ON CONFLICT DO UPDATE`` per batch of daily spend rows.

Emitting a statement per aggregated key put every replica's flush on the database as
hundreds of separate statements against the same handful of hot rows, each holding its
row locks for the rest of the enclosing batch transaction. Folding a batch into a single
statement keeps the aggregation identical while collapsing both the statement count and
the window in which those locks are held.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from types import MappingProxyType
from typing import Final, Literal

DailySpendEntity = Literal["user", "team", "org", "tag", "end_user", "agent"]

SqlValue = str | int | float | None

# A queued daily spend transaction, read by column name because the columns are data
# here rather than literals. The concrete TypedDicts in _types.py all satisfy this.
SpendRow = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DailySpendTable:
    """The physical table behind one entity's daily rollup."""

    name: str
    entity_id_column: str
    carries_request_id: bool = False


DAILY_SPEND_TABLES: Final[Mapping[DailySpendEntity, DailySpendTable]] = MappingProxyType(
    {
        "user": DailySpendTable(name="LiteLLM_DailyUserSpend", entity_id_column="user_id"),
        "team": DailySpendTable(name="LiteLLM_DailyTeamSpend", entity_id_column="team_id"),
        "org": DailySpendTable(name="LiteLLM_DailyOrganizationSpend", entity_id_column="organization_id"),
        "end_user": DailySpendTable(name="LiteLLM_DailyEndUserSpend", entity_id_column="end_user_id"),
        "agent": DailySpendTable(name="LiteLLM_DailyAgentSpend", entity_id_column="agent_id"),
        "tag": DailySpendTable(name="LiteLLM_DailyTagSpend", entity_id_column="tag", carries_request_id=True),
    }
)

# The unique constraint's columns after the entity id, in constraint order. A NULL can
# never match itself in a unique index, so every one of these is normalized to '': the
# conflict target has to be NULL-free or the row is re-inserted on every single flush.
_KEY_COLUMNS: Final = ("date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint")

_COUNTER_COLUMNS: Final = (
    "prompt_tokens",
    "completion_tokens",
    "api_requests",
    "successful_requests",
    "failed_requests",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "compression_saved_tokens",
)
_SPEND_COLUMNS: Final = (
    "spend",
    "compression_savings_spend",
    "prompt_caching_savings_spend",
    "autorouter_savings_spend",
)

_CASTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        **{column: "bigint" for column in _COUNTER_COLUMNS},
        **{column: "double precision" for column in _SPEND_COLUMNS},
    }
)


def _quoted(columns: Sequence[str]) -> str:
    return ", ".join(f'"{column}"' for column in columns)


def _as_text(value: object) -> str:
    return "" if value is None else str(value)


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def conflict_key(table: DailySpendTable, transaction: SpendRow) -> tuple[str, ...]:
    """The tuple the database arbitrates the upsert on, normalized free of NULLs."""
    return tuple(_as_text(transaction.get(column)) for column in (table.entity_id_column, *_KEY_COLUMNS))


def _merge(group: Sequence[SpendRow]) -> SpendRow:
    if len(group) == 1:
        return group[0]
    return {
        **group[0],
        **{column: sum(_as_int(row.get(column)) for row in group) for column in _COUNTER_COLUMNS},
        **{column: sum(_as_float(row.get(column)) for row in group) for column in _SPEND_COLUMNS},
    }


def merge_by_conflict_key(
    table: DailySpendTable,
    transactions: Sequence[SpendRow],
) -> tuple[tuple[tuple[str, ...], SpendRow], ...]:
    """Batch entries keyed by the conflict tuple, in a deterministic order.

    The queue keys transactions by their raw field values, so two entries differing only
    in a NULL versus an empty member reach the writer separately while arbitrating to the
    same row. Postgres rejects a statement whose values touch one row twice, so they are
    summed here into the single row they were always destined to become. Ordering by the
    key keeps concurrent writers taking row locks in the same sequence.
    """
    ordered: Final = sorted(transactions, key=lambda transaction: conflict_key(table, transaction))
    return tuple((key, _merge(tuple(group))) for key, group in groupby(ordered, key=lambda t: conflict_key(table, t)))


def _row_params(
    table: DailySpendTable,
    key: tuple[str, ...],
    transaction: SpendRow,
) -> tuple[SqlValue, ...]:
    request_id: Final = transaction.get("request_id")
    return (
        str(uuid.uuid4()),
        *key,
        None if transaction.get("model_group") is None else _as_text(transaction.get("model_group")),
        *(_as_int(transaction.get(column)) for column in _COUNTER_COLUMNS),
        *(_as_float(transaction.get(column)) for column in _SPEND_COLUMNS),
        *((None if request_id is None else _as_text(request_id),) if table.carries_request_id else ()),
    )


def _insert_columns(table: DailySpendTable) -> tuple[str, ...]:
    return (
        "id",
        table.entity_id_column,
        *_KEY_COLUMNS,
        "model_group",
        *_COUNTER_COLUMNS,
        *_SPEND_COLUMNS,
        *(("request_id",) if table.carries_request_id else ()),
    )


def build_bulk_upsert(
    table: DailySpendTable,
    batch: Sequence[tuple[tuple[str, ...], SpendRow]],
) -> tuple[str, tuple[SqlValue, ...]]:
    """The single statement writing one merged batch, plus its positional arguments."""
    columns: Final = _insert_columns(table)
    quoted_table: Final = f'"{table.name}"'
    rows: Final = ", ".join(
        "("
        + ", ".join(
            f"${row_index * len(columns) + offset + 1}::{_CASTS.get(column, 'text')}"
            for offset, column in enumerate(columns)
        )
        + ", (NOW() AT TIME ZONE 'UTC'))"
        for row_index in range(len(batch))
    )
    increments: Final = ", ".join(
        f'"{column}" = {quoted_table}."{column}" + EXCLUDED."{column}"'
        for column in (*_COUNTER_COLUMNS, *_SPEND_COLUMNS)
    )
    # request_id names one arbitrary contributing request, so an entry carrying none must
    # not blank out the one already recorded.
    request_id_update: Final = (
        f', "request_id" = COALESCE(EXCLUDED."request_id", {quoted_table}."request_id")'
        if table.carries_request_id
        else ""
    )
    sql: Final = (
        f'INSERT INTO {quoted_table} ({_quoted(columns)}, "updated_at")\n'
        f"VALUES {rows}\n"
        f"ON CONFLICT ({_quoted((table.entity_id_column, *_KEY_COLUMNS))}) DO UPDATE SET\n"
        f"  {increments}{request_id_update},\n"
        f"  \"updated_at\" = (NOW() AT TIME ZONE 'UTC')"
    )
    return sql, tuple(value for key, transaction in batch for value in _row_params(table, key, transaction))
