from collections.abc import Mapping
from datetime import datetime, time, timedelta, timezone
from types import MappingProxyType
from typing import Final

from pydantic import ConfigDict

import litellm
from litellm.litellm_core_utils.duration_parser import (
    get_budget_window_start,
    get_next_standardized_reset_time,
    normalize_budget_window_start,
)
from litellm.models.budget_schedule import BudgetSchedule, is_scheduled_duration
from litellm.types.llms.base import LiteLLMBaseModel


class BudgetResetSettings(LiteLLMBaseModel):
    """Immutable, validated settings that govern when budgets reset.

    Parsed once from `litellm_settings` and injected into consumers (the reset
    job, management endpoints) so reset times never depend on reaching into
    module-level globals at call time.
    """

    model_config = ConfigDict(frozen=True)

    timezone: str = "UTC"
    reset_time_of_day: time = time(0, 0)


def parse_budget_reset_time(raw: object) -> time:
    """Parse a `budget_reset_time` config value (e.g. "12:00") into a `time`.

    Falls back to midnight when unset; raises a clear error on a malformed value
    so a bad config fails loudly at startup instead of silently resetting at midnight.
    """
    if raw is None or raw == "":
        return time(0, 0)
    if not isinstance(raw, str):
        raise ValueError(f"Invalid budget_reset_time {raw!r}; must be a quoted 24-hour 'HH:MM' string, e.g. \"12:00\"")
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return time(hour=parsed.hour, minute=parsed.minute, second=parsed.second)
        except ValueError:
            continue
    raise ValueError(
        f"Invalid budget_reset_time {raw!r}; expected a 24-hour 'HH:MM' or 'HH:MM:SS' string, e.g. \"12:00\""
    )


def get_budget_reset_timezone() -> str:
    """
    Get the budget reset timezone from litellm_settings.
    Falls back to UTC if not specified.

    litellm_settings values are set as attributes on the litellm module
    by proxy_server.py at startup (via setattr(litellm, key, value)).
    """
    return getattr(litellm, "timezone", None) or "UTC"


def get_budget_reset_settings() -> BudgetResetSettings:
    """Build validated reset settings from litellm_settings. Raises on a malformed
    `budget_reset_time`, which lets the proxy fail fast at startup."""
    return BudgetResetSettings(
        timezone=get_budget_reset_timezone(),
        reset_time_of_day=parse_budget_reset_time(getattr(litellm, "budget_reset_time", None)),
    )


def compute_budget_reset_at(
    budget_duration: str,
    settings: BudgetResetSettings,
    now: datetime | None = None,
    *,
    previous_reset: datetime | None = None,
) -> datetime:
    """Compute the next reset time for a budget duration using injected settings."""
    return get_next_standardized_reset_time(
        duration=budget_duration,
        current_time=now if now is not None else datetime.now(timezone.utc),
        timezone_str=settings.timezone,
        reset_time_of_day=settings.reset_time_of_day,
        elapsed_anchor=previous_reset,
    )


def get_budget_reset_time(budget_duration: str) -> datetime:
    """Get the budget reset time using the globally-configured timezone and reset time.

    Thin wrapper over `compute_budget_reset_at` for callers that don't yet receive
    `BudgetResetSettings` by injection (creation/update endpoints, startup backfill).
    """
    return compute_budget_reset_at(budget_duration, get_budget_reset_settings())


def get_budget_window_reset_fields(
    budget_duration: str,
    *,
    reset_at: datetime | None = None,
    settings: BudgetResetSettings | None = None,
    now: datetime | None = None,
) -> Mapping[str, str]:
    current: Final = now if now is not None else datetime.now(timezone.utc)
    schedule: Final = BudgetSchedule.from_label(budget_duration)
    if schedule is not None:
        return _scheduled_window_reset_fields(schedule, current)
    configured: Final = settings or get_budget_reset_settings()
    end: Final = reset_at if reset_at is not None else compute_budget_reset_at(budget_duration, configured, current)
    scheduled: Final = get_budget_window_start(budget_duration, end, configured.timezone, configured.reset_time_of_day)
    start: Final = normalize_budget_window_start(scheduled if reset_at is not None else min(scheduled, current))
    return MappingProxyType(
        {
            "reset_at": end.astimezone(timezone.utc).isoformat(),
            "window_start": start.astimezone(timezone.utc).isoformat(),
        }
    )


def _scheduled_window_reset_fields(schedule: BudgetSchedule, now: datetime) -> Mapping[str, str]:
    period: Final = schedule.period_at(now)
    start, end = period if period is not None else (schedule.next_start(now),) * 2
    return MappingProxyType({"reset_at": end.isoformat(), "window_start": start.isoformat()})


def keeps_budget_reset(requested_duration: str, stored_duration: str | None, stored_reset_at: datetime | None) -> bool:
    return stored_reset_at is not None and requested_duration == stored_duration


def _is_persistable_budget_duration(budget_duration: str) -> bool:
    from litellm.litellm_core_utils.duration_parser import duration_in_seconds

    try:
        seconds: Final = duration_in_seconds(budget_duration)
        if seconds <= 0:
            return False
        reset_at: Final = get_budget_reset_time(budget_duration=budget_duration)
        return reset_at - timedelta(seconds=seconds) < reset_at
    except (ValueError, OverflowError):
        return False


def budget_duration_error(budget_duration: str | None) -> str | None:
    """Why `budget_duration` cannot be persisted, or None when it is usable.

    A non-positive duration resolves to a reset time of "now", which leaves the row
    permanently due: the reset job re-reads it every tick and, once enough of them
    exist, they fill each batch and starve every other tenant's reset.
    """
    if budget_duration is None or _is_persistable_budget_duration(budget_duration):
        return None
    if is_scheduled_duration(budget_duration):
        return (
            None
            if BudgetSchedule.from_label(budget_duration) is not None
            else f"Invalid budget schedule '{budget_duration}'."
        )
    return f"Invalid budget_duration '{budget_duration}'. Use a format like '1h', '24h', '7d', or '30d'."
