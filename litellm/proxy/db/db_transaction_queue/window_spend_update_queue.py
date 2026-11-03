"""Budget-window accounting values and time normalization."""

from datetime import datetime, timezone
from typing import TypedDict

from typing_extensions import ReadOnly


class WindowSpendTransaction(TypedDict):
    """A stored request's contribution to a budget window."""

    entity_type: ReadOnly[str]
    entity_id: ReadOnly[str]
    window_duration: ReadOnly[str]
    window_start: ReadOnly[str]
    spend: ReadOnly[float]
    started_at: ReadOnly[str | None]


def to_naive_utc(value: datetime) -> datetime:
    """LiteLLM_BudgetWindowSpend.window_start is TIMESTAMP(3), which holds naive UTC."""
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def window_spend_group_key(transaction: WindowSpendTransaction) -> tuple[str, str, str, str]:
    """Identity of a window increment: the row's primary key plus the window it
    belongs to. Two increments only aggregate when all four match."""
    return (
        transaction["entity_type"],
        transaction["entity_id"],
        transaction["window_duration"],
        transaction["window_start"],
    )


def build_window_spend_transaction(
    entity_type: str,
    entity_id: str,
    window_duration: str,
    window_start: datetime,
    spend: float,
    started_at: datetime | None = None,
) -> WindowSpendTransaction:
    return WindowSpendTransaction(
        entity_type=entity_type,
        entity_id=entity_id,
        window_duration=window_duration,
        window_start=to_naive_utc(window_start).isoformat(timespec="microseconds"),
        spend=spend,
        started_at=None
        if started_at is None
        else to_naive_utc(started_at.astimezone(timezone.utc)).isoformat(timespec="microseconds"),
    )
