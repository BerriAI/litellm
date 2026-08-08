"""Shared resolution of ``budget_limits`` windows for keys and teams.

Each ``budget_limits`` entry carries a server-managed ``reset_at`` that anchors
its spend window (``window_start = reset_at - budget_duration``; see
``budget_reservation.get_budget_window_start``). Spend for a window is never
persisted as a running total; it is re-derived from ``LiteLLM_SpendLogs`` since
``window_start``, so moving ``reset_at`` moves the window and, on a cold spend
counter, the spend it enforces.

On update we therefore preserve ``reset_at`` for any window whose
``budget_duration`` already exists on the entity, so editing ``max_budget``
alone does not restart the window. Only genuinely new durations get a fresh
reset boundary. On create there is nothing to preserve, so every window is
initialized fresh.
"""

import json
from datetime import datetime
from typing import Mapping, Sequence

from pydantic import TypeAdapter

from litellm.models.team import BudgetLimitEntry
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

BudgetLimitWindow = BudgetLimitEntry | Mapping[str, object]
# Existing windows can arrive parsed (key rows, validated team models) or as the
# raw JSON string a prisma ``Json?`` column returns from ``find_unique`` (which
# is ``null`` when the field was cleared).
ExistingBudgetLimits = str | Sequence[BudgetLimitWindow] | None

_EXISTING_WINDOWS_ADAPTER: TypeAdapter[list[BudgetLimitEntry] | None] = TypeAdapter(list[BudgetLimitEntry] | None)


def _to_entry(window: BudgetLimitWindow) -> BudgetLimitEntry:
    if isinstance(window, BudgetLimitEntry):
        return window
    return BudgetLimitEntry.model_validate(window)


def _iter_existing(existing: ExistingBudgetLimits) -> Sequence[BudgetLimitEntry]:
    if existing is None:
        return ()
    if isinstance(existing, str):
        return _EXISTING_WINDOWS_ADAPTER.validate_json(existing) or ()
    return [_to_entry(window) for window in existing]


def _existing_reset_at_by_duration(existing: ExistingBudgetLimits) -> Mapping[str, datetime]:
    return {entry.budget_duration: entry.reset_at for entry in _iter_existing(existing) if entry.reset_at is not None}


def _resolve_window(window: BudgetLimitWindow, preserved: Mapping[str, datetime]) -> BudgetLimitEntry:
    entry = _to_entry(window)
    reset_at = preserved.get(entry.budget_duration) or get_budget_reset_time(budget_duration=entry.budget_duration)
    return entry.model_copy(update={"reset_at": reset_at})


def resolve_budget_limit_windows(
    incoming: Sequence[BudgetLimitWindow],
    existing: ExistingBudgetLimits = None,
) -> tuple[BudgetLimitEntry, ...]:
    """Return the windows to persist, each with ``reset_at`` resolved.

    A window whose ``budget_duration`` matches one in ``existing`` keeps that
    window's ``reset_at``; every other window is initialized to a fresh reset
    boundary. Any client-supplied ``reset_at`` on ``incoming`` is ignored, since
    the field is server-managed.
    """
    preserved = _existing_reset_at_by_duration(existing)
    return tuple(_resolve_window(window, preserved) for window in incoming)


def _to_stored_dict(window: BudgetLimitEntry) -> dict[str, object]:
    return {
        "budget_duration": window.budget_duration,
        "max_budget": window.max_budget,
        "reset_at": window.reset_at.isoformat() if window.reset_at is not None else None,
    }


def serialize_budget_limit_windows(windows: Sequence[BudgetLimitEntry]) -> str:
    """Serialize resolved windows to the JSON string stored in the DB column.

    ``reset_at`` is emitted with ``datetime.isoformat`` (``+00:00`` offset) to
    match the format written by the create paths and existing rows.
    """
    return json.dumps([_to_stored_dict(window) for window in windows])
