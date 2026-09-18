from datetime import datetime, time, timezone

from pydantic import BaseModel, ConfigDict

import litellm
from litellm.litellm_core_utils.duration_parser import get_next_standardized_reset_time


class BudgetResetSettings(BaseModel):
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


def compute_budget_reset_at(budget_duration: str, settings: BudgetResetSettings) -> datetime:
    """Compute the next reset time for a budget duration using injected settings."""
    return get_next_standardized_reset_time(
        duration=budget_duration,
        current_time=datetime.now(timezone.utc),
        timezone_str=settings.timezone,
        reset_time_of_day=settings.reset_time_of_day,
    )


def get_budget_reset_time(budget_duration: str) -> datetime:
    """Get the budget reset time using the globally-configured timezone and reset time.

    Thin wrapper over `compute_budget_reset_at` for callers that don't yet receive
    `BudgetResetSettings` by injection (creation/update endpoints, startup backfill).
    """
    return compute_budget_reset_at(budget_duration, get_budget_reset_settings())


def _is_persistable_budget_duration(budget_duration: str) -> bool:
    from litellm.litellm_core_utils.duration_parser import duration_in_seconds

    try:
        if duration_in_seconds(budget_duration) <= 0:
            return False
        get_budget_reset_time(budget_duration=budget_duration)
    except (ValueError, OverflowError):
        return False
    return True


def budget_duration_error(budget_duration: str | None) -> str | None:
    """Why `budget_duration` cannot be persisted, or None when it is usable.

    A non-positive duration resolves to a reset time of "now", which leaves the row
    permanently due: the reset job re-reads it every tick and, once enough of them
    exist, they fill each batch and starve every other tenant's reset.
    """
    if budget_duration is None or _is_persistable_budget_duration(budget_duration):
        return None
    return f"Invalid budget_duration '{budget_duration}'. Use a format like '1h', '24h', '7d', or '30d'."
