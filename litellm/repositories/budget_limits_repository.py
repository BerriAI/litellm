import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm.models.team import BudgetLimitEntry
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

BudgetLimitsTable: TypeAlias = Literal[
    "LiteLLM_VerificationToken", "LiteLLM_TeamTable", "LiteLLM_UserTable", "LiteLLM_BudgetTable"
]

_ID_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "LiteLLM_VerificationToken": "token",
        "LiteLLM_TeamTable": "team_id",
        "LiteLLM_UserTable": "user_id",
        "LiteLLM_BudgetTable": "budget_id",
    }
)
_WINDOWS: Final[TypeAdapter[tuple[Mapping[str, object], ...] | None]] = TypeAdapter(
    tuple[Mapping[str, object], ...] | None
)


class BudgetLimitsTransaction(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...

    async def execute_raw(self, query: str, *args: object) -> int: ...


async def compare_and_swap_budget_limits(
    transaction: BudgetLimitsTransaction,
    table: str,
    row_id: str,
    expected_json: str,
    updated_json: str,
) -> bool:
    query: Final = (
        f"UPDATE \"{table}\" SET budget_limits = $3::text::jsonb, updated_at = (NOW() AT TIME ZONE 'UTC')"
        f' WHERE "{_ID_COLUMNS[table]}" = $1::text'
        " AND COALESCE(budget_limits::jsonb, 'null'::jsonb) = $2::text::jsonb"
    )
    return await transaction.execute_raw(query, row_id, expected_json, updated_json) == 1


async def lock_budget_limits(
    transaction: BudgetLimitsTransaction, table: BudgetLimitsTable, row_id: str
) -> tuple[Mapping[str, object], ...] | None:
    rows: Final = await transaction.query_raw(
        f'SELECT budget_limits FROM "{table}" WHERE "{_ID_COLUMNS[table]}" = $1::text FOR UPDATE', row_id
    )
    if not rows:
        return None
    raw: Final = rows[0]["budget_limits"]
    return (_WINDOWS.validate_json(raw) if isinstance(raw, str) else _WINDOWS.validate_python(raw)) or ()


def _rebase_window(
    window: Mapping[str, object],
    original: Mapping[str, BudgetLimitEntry | Mapping[str, object]],
    current: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    duration: Final = str(window["budget_duration"])
    original_window: Final = original.get(duration)
    current_window: Final = current.get(duration)
    if original_window is None or current_window is None:
        return window
    try:
        before: Final = BudgetLimitEntry.model_validate(original_window)
        after: Final = BudgetLimitEntry.model_validate(current_window)
    except ValidationError:
        return window
    if after.reset_at is None or (before.reset_at == after.reset_at and before.window_start == after.window_start):
        return window
    if before.model_dump(exclude={"reset_at", "window_start"}) != after.model_dump(
        exclude={"reset_at", "window_start"}
    ):
        return window
    return {
        **{key: value for key, value in window.items() if key not in ("reset_at", "window_start")},
        "reset_at": after.reset_at.isoformat(),
        **({"window_start": after.window_start.isoformat()} if after.window_start is not None else {}),
    }


async def rebase_budget_limits_update(
    transaction: BudgetLimitsTransaction,
    table: BudgetLimitsTable,
    row_id: str,
    original: Sequence[BudgetLimitEntry | Mapping[str, object]] | None,
    data: Mapping[str, object],
) -> Mapping[str, object]:
    """Lock and rebase the payload before the caller writes it in this same transaction."""
    scheduled: Final = await _rebase_budget_duration_update(transaction, table, row_id, data)
    if "budget_limits" not in scheduled:
        return scheduled
    current: Final = await lock_budget_limits(transaction, table, row_id)
    raw: Final = scheduled["budget_limits"]
    requested: Final = _WINDOWS.validate_json(raw) if isinstance(raw, str) else _WINDOWS.validate_python(raw)
    if not requested or current is None:
        return scheduled
    before: Final = MappingProxyType(
        {
            original_duration: window
            for window in original or ()
            if isinstance(
                original_duration := window.budget_duration
                if isinstance(window, BudgetLimitEntry)
                else window.get("budget_duration"),
                str,
            )
        }
    )
    after: Final = MappingProxyType(
        {duration: window for window in current if isinstance(duration := window.get("budget_duration"), str)}
    )
    rebased: Final = tuple(_rebase_window(window, before, after) for window in requested)
    if rebased == requested:
        return scheduled
    return {**scheduled, "budget_limits": json.dumps(rebased)}


async def _rebase_budget_duration_update(
    transaction: BudgetLimitsTransaction,
    table: BudgetLimitsTable,
    row_id: str,
    data: Mapping[str, object],
) -> Mapping[str, object]:
    if "budget_duration" not in data:
        return data
    rows: Final = await transaction.query_raw(
        f'SELECT budget_duration, budget_reset_at FROM "{table}" WHERE "{_ID_COLUMNS[table]}" = $1::text FOR UPDATE',
        row_id,
    )
    current: Final[Mapping[str, object]] = rows[0] if rows else {}
    duration: Final = data["budget_duration"]
    if (
        isinstance(duration, str)
        and duration == current.get("budget_duration")
        and current.get("budget_reset_at") is not None
    ):
        return {key: value for key, value in data.items() if key != "budget_reset_at"}
    return {
        **data,
        "budget_reset_at": get_budget_reset_time(duration) if isinstance(duration, str) else None,
    }
