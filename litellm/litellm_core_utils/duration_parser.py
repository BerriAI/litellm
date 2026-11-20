"""
Helper utilities for parsing durations - 1s, 1d, 10d, 30d, 1mo, 2mo

duration_in_seconds is used in diff parts of the code base, example
- Router - Provider budget routing
- Proxy - Key, Team Generation
"""

import re
import time as time_module
from datetime import datetime, time, timedelta, timezone, tzinfo
from typing import Final
from zoneinfo import ZoneInfo

from litellm._logging import verbose_logger

_BUDGET_DURATION_WORD_ALIASES: Final[dict[str, str]] = {
    "hourly": "1h",
    "daily": "24h",
    "weekly": "7d",
    "monthly": "30d",
}


def normalize_duration(duration: str) -> str:
    return _BUDGET_DURATION_WORD_ALIASES.get(duration.strip().lower(), duration)


def _extract_from_regex(duration: str) -> tuple[int, str]:
    match: Final = re.match(r"(\d+)(mo|[smhdw]?)", duration)

    if not match:
        raise ValueError("Invalid duration format")

    value, unit = match.groups()
    value = int(value)

    return value, unit


def get_last_day_of_month(year: int, month: int) -> int:
    # Handle December case
    if month == 12:
        return 31
    # Next month is January, so subtract a day from March 1st
    next_month: Final = datetime(year=year, month=month + 1, day=1)
    last_day_of_month: Final = (next_month - timedelta(days=1)).day
    return last_day_of_month


def duration_in_seconds(duration: str) -> int:
    """
    Parameters:
    - duration:
        - "<number>s" - seconds
        - "<number>m" - minutes
        - "<number>h" - hours
        - "<number>d" - days
        - "<number>w" - weeks
        - "<number>mo" - months

    Returns time in seconds till when budget needs to be reset
    """
    value, unit = _extract_from_regex(duration=normalize_duration(duration))

    if unit == "s":
        return value
    elif unit == "m":
        return value * 60
    elif unit == "h":
        return value * 3600
    elif unit == "d":
        return value * 86400
    elif unit == "w":
        return value * 604800
    elif unit == "mo":
        return _months_in_seconds(
            start=datetime.fromtimestamp(time_module.time(), tz=timezone.utc).astimezone(), months=value
        )

    else:
        raise ValueError(f"Unsupported duration unit, passed duration: {duration}")


def duration_in_seconds_from(start: datetime, duration: str) -> int:
    """How long a period of `duration` lasts when it begins at `start`.

    `duration_in_seconds` measures `Nmo` from now, so a monthly window that opened in a
    month of a different length needs its own start to know when it ends. Every other
    unit is a fixed length and ignores `start`.
    """
    value, unit = _extract_from_regex(duration=normalize_duration(duration))
    if unit == "mo":
        return _months_in_seconds(start=start, months=value)
    return duration_in_seconds(duration)


def _months_in_seconds(start: datetime, months: int) -> int:
    return int((_shift_months(start, months) - start).total_seconds())


def get_next_standardized_reset_time(
    duration: str,
    current_time: datetime,
    timezone_str: str = "UTC",
    reset_time_of_day: time = time(0, 0),
    *,
    elapsed_anchor: datetime | None = None,
) -> datetime:
    """
    Get the next standardized reset time based on the duration.

    All durations will reset at predictable intervals, aligned from the current time:
    - Nd: If N=1, reset at the next `reset_time_of_day`; if N>1, reset every N days from now
    - Nh/Nm/Ns: Fixed elapsed intervals, initially aligned to local day/hour/minute boundaries

    Parameters:
    - duration: Duration string (e.g. "30s", "30m", "30h", "30d")
    - current_time: Current datetime
    - timezone_str: Timezone string (e.g. "UTC", "US/Eastern", "Asia/Kolkata")
    - reset_time_of_day: Wall-clock time the reset lands on for day/week/month
      durations (defaults to midnight). Ignored for sub-day durations, where a
      time-of-day is meaningless.
    - elapsed_anchor: Saved reset instant for an existing elapsed-time schedule.
      Without one, retain the local alignment used when creating a budget.

    Returns:
    - Next reset time at a standardized interval in the specified timezone
    """
    # Set up timezone and normalize current time
    current_time, _ = _setup_timezone(current_time, timezone_str)

    # Parse duration
    value, unit = _parse_duration(normalize_duration(duration))
    if value is None:
        verbose_logger.warning(
            "Unrecognized budget_duration %r; falling back to a next-midnight reset. "
            "Use the <int><unit> format (e.g. '1h', '7d', '30d', '1mo').",
            duration,
        )
        return current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    # Midnight of the current day in the specified timezone
    base_midnight: Final = current_time.replace(hour=0, minute=0, second=0, microsecond=0)

    # Handle different time units
    if unit == "d":
        return _handle_day_reset(current_time, base_midnight, value, reset_time_of_day)
    elif unit == "w":
        return _handle_day_reset(current_time, base_midnight, value * 7, reset_time_of_day)
    elif unit == "h":
        return _next_elapsed_reset(current_time, timedelta(hours=value), elapsed_anchor or base_midnight)
    elif unit == "m":
        hour_start: Final = current_time.replace(minute=0, second=0, microsecond=0)
        return _next_elapsed_reset(current_time, timedelta(minutes=value), elapsed_anchor or hour_start)
    elif unit == "s":
        minute_start: Final = current_time.replace(second=0, microsecond=0)
        return _next_elapsed_reset(current_time, timedelta(seconds=value), elapsed_anchor or minute_start)
    elif unit == "mo":
        return _handle_month_reset(current_time, base_midnight, value, reset_time_of_day)
    else:
        # Unrecognized unit, default to next midnight
        return base_midnight + timedelta(days=1)


def _shift_months(moment: datetime, months: int) -> datetime:
    """`moment` moved by `months` calendar months, back when negative, clamped to the target month's last day."""
    total_months: Final = moment.year * 12 + moment.month - 1 + months
    year, month_index = divmod(total_months, 12)
    month: Final = month_index + 1
    return moment.replace(year=year, month=month, day=min(moment.day, get_last_day_of_month(year, month)))


def is_annual_budget_duration(duration: str) -> bool:
    return _parse_duration(normalize_duration(duration)) == (12, "mo")


def normalize_budget_window_start(moment: datetime) -> datetime:
    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


def get_annual_budget_window_start(reset_at: datetime, timezone_str: str) -> datetime:
    boundary: Final = _setup_timezone(normalize_budget_window_start(reset_at), timezone_str)[0]
    return _shift_months(boundary, -12).astimezone(timezone.utc)


def get_budget_window_start(duration: str, reset_at: datetime) -> datetime:
    """Start of the budget period that ends at `reset_at`, under the same rules
    `get_next_standardized_reset_time` used to pick it: `30d` and `Nmo` span calendar
    months and a duration it does not recognize resets at the next midnight."""
    return _shift_budget_window(duration, reset_at, -1)


def get_budget_window_end(
    duration: str, start: datetime, timezone_str: str = "UTC", reset_at: datetime | None = None
) -> datetime:
    boundary: Final = _setup_timezone(start, timezone_str)[0]
    configured: Final = _setup_timezone(reset_at, timezone_str)[0] if reset_at is not None else None
    if configured is not None and get_budget_window_start(duration, configured) == boundary:
        return configured.astimezone(timezone.utc)
    return _shift_budget_window(duration, boundary, 1).astimezone(timezone.utc)


def _shift_budget_window(duration: str, moment: datetime, direction: int) -> datetime:
    value, unit = _parse_duration(normalize_duration(duration))
    boundary: Final = normalize_budget_window_start(
        moment.astimezone(timezone.utc) if moment.tzinfo is not None and unit in ("h", "m", "s") else moment
    )
    if value is None:
        return boundary + timedelta(days=direction)
    match unit:
        case "mo":
            return _shift_months(boundary, direction * value)
        case "d":
            return _shift_months(boundary, direction) if value == 30 else boundary + timedelta(days=direction * value)
        case "w":
            return boundary + timedelta(weeks=direction * value)
        case "h":
            return boundary + timedelta(hours=direction * value)
        case "m":
            return boundary + timedelta(minutes=direction * value)
        case "s":
            return boundary + timedelta(seconds=direction * value)
        case _:
            return boundary + timedelta(days=direction)


def _setup_timezone(current_time: datetime, timezone_str: str = "UTC") -> tuple[datetime, tzinfo]:
    """Set up timezone and normalize current time to that timezone."""
    try:
        if timezone_str is None:
            tz: tzinfo = timezone.utc
        else:
            tz = ZoneInfo(timezone_str)
    except Exception:
        # If timezone is invalid, fall back to UTC
        tz = timezone.utc

    # Convert current_time to the target timezone
    if current_time.tzinfo is None:
        # Naive datetime - assume it's UTC
        utc_time: Final = current_time.replace(tzinfo=timezone.utc)
        current_time = utc_time.astimezone(tz)
    else:
        # Already has timezone - convert to target timezone
        current_time = current_time.astimezone(tz)

    return current_time, tz


def _parse_duration(duration: str) -> tuple[int | None, str | None]:
    """Parse the duration string into value and unit."""
    match: Final = re.match(r"(\d+)([a-z]+)", duration)
    if not match:
        return None, None

    value: Final = int(match.group(1))
    spelled_unit: Final = match.group(2)
    unit: Final = _extract_from_regex(duration)[1] or spelled_unit
    if unit == "mo" and spelled_unit != "mo" and value not in (1, 12):
        return value, spelled_unit
    return value, unit


def _apply_time_of_day(dt: datetime, reset_time_of_day: time) -> datetime:
    """Set the wall-clock time of `dt` to `reset_time_of_day`, keeping its date and tzinfo."""
    return dt.replace(
        hour=reset_time_of_day.hour,
        minute=reset_time_of_day.minute,
        second=reset_time_of_day.second,
        microsecond=reset_time_of_day.microsecond,
    )


def _next_occurrence(
    boundary_midnight: datetime,
    reset_time_of_day: time,
    current_time: datetime,
    period: timedelta,
) -> datetime:
    """Place the reset at `reset_time_of_day` on the boundary day, rolling forward one
    `period` if that instant has already passed (or is exactly now)."""
    candidate: Final = _apply_time_of_day(boundary_midnight, reset_time_of_day)
    if candidate <= current_time:
        return candidate + period
    return candidate


def _first_of_next_month(first_of_month: datetime) -> datetime:
    """Given the 1st of some month, return the 1st of the following month."""
    if first_of_month.month == 12:
        return first_of_month.replace(year=first_of_month.year + 1, month=1)
    return first_of_month.replace(month=first_of_month.month + 1)


def _handle_day_reset(
    current_time: datetime,
    base_midnight: datetime,
    value: int,
    reset_time_of_day: time,
) -> datetime:
    """Handle day-based reset times."""
    # Handle zero value - immediate expiration
    if value == 0:
        return current_time

    if value == 1:  # Daily reset at the configured time of day
        return _next_occurrence(base_midnight, reset_time_of_day, current_time, timedelta(days=1))
    elif value == 7:  # Weekly reset on Monday at the configured time of day
        days_until_monday: Final = (7 - current_time.weekday()) % 7
        upcoming_monday: Final = base_midnight + timedelta(days=days_until_monday)
        return _next_occurrence(upcoming_monday, reset_time_of_day, current_time, timedelta(days=7))
    elif value == 30:  # Monthly reset on 1st at the configured time of day
        return _handle_month_reset(current_time, base_midnight, 1, reset_time_of_day)
    else:  # Custom day value - next interval is value days from the start of today
        return _apply_time_of_day(base_midnight + timedelta(days=value), reset_time_of_day)


def _next_elapsed_reset(current_time: datetime, interval: timedelta, anchor: datetime) -> datetime:
    """Keep one UTC schedule across resets, including local hour/day changes."""
    if interval == timedelta(0):
        return current_time
    utc_anchor: Final = _setup_timezone(anchor, "UTC")[0]
    periods: Final = (current_time.astimezone(timezone.utc) - utc_anchor) // interval + 1
    return (utc_anchor + periods * interval).astimezone(current_time.tzinfo)


def _handle_month_reset(
    current_time: datetime,
    base_midnight: datetime,
    value: int,
    reset_time_of_day: time,
) -> datetime:
    """
    Handle monthly reset times. Resets land on the 1st at `reset_time_of_day`; if the
    1st of the current month at that time has already passed, roll to the 1st of next month.

    Args:
        current_time: Current datetime
        base_midnight: Midnight of current day
        value: Number of months, 1 for monthly or 12 for annual resets
        reset_time_of_day: Wall-clock time the reset lands on

    Returns:
        datetime: First day of the next reset month at `reset_time_of_day`
    """
    if value == 12:
        first_of_year: Final = base_midnight.replace(month=1, day=1)
        annual_candidate: Final = _apply_time_of_day(first_of_year, reset_time_of_day)
        return (
            annual_candidate.replace(year=annual_candidate.year + 1)
            if annual_candidate <= current_time
            else annual_candidate
        )
    if value != 1:
        raise ValueError("Calendar resets support 1 month or 12 month intervals")

    first_of_this_month: Final = base_midnight.replace(day=1)
    candidate: Final = _apply_time_of_day(first_of_this_month, reset_time_of_day)
    if candidate <= current_time:
        return _apply_time_of_day(_first_of_next_month(first_of_this_month), reset_time_of_day)
    return candidate
