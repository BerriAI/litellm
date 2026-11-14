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


def get_last_day_of_month(year, month):
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
) -> datetime:
    """
    Get the next standardized reset time based on the duration.

    All durations will reset at predictable intervals, aligned from the current time:
    - Nd: If N=1, reset at the next `reset_time_of_day`; if N>1, reset every N days from now
    - Nh: Every N hours, aligned to hour boundaries (e.g., 1:00, 2:00)
    - Nm: Every N minutes, aligned to minute boundaries (e.g., 1:05, 1:10)
    - Ns: Every N seconds, aligned to second boundaries

    Parameters:
    - duration: Duration string (e.g. "30s", "30m", "30h", "30d")
    - current_time: Current datetime
    - timezone_str: Timezone string (e.g. "UTC", "US/Eastern", "Asia/Kolkata")
    - reset_time_of_day: Wall-clock time the reset lands on for day/week/month
      durations (defaults to midnight). Ignored for sub-day durations, where a
      time-of-day is meaningless.

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
        return _handle_hour_reset(current_time, base_midnight, value)
    elif unit == "m":
        return _handle_minute_reset(current_time, base_midnight, value)
    elif unit == "s":
        return _handle_second_reset(current_time, base_midnight, value)
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


def get_annual_budget_window_start(reset_at: datetime, timezone_str: str) -> datetime:
    boundary: Final = _setup_timezone(reset_at, timezone_str)[0]
    return _shift_months(boundary, -12).astimezone(timezone.utc)


def budget_duration_in_months(duration: str) -> int | None:
    """Calendar months `duration` spans: `Nmo`, plus `30d`, which `_handle_day_reset`
    resets on the 1st and `get_budget_window_start` already counts as one month."""
    value, unit = _parse_duration(normalize_duration(duration))
    if not value:
        return None
    if unit == "mo":
        return value
    if unit == "d" and value == 30:
        return 1
    return None


def get_budget_window_start_from_reset(reset_at: datetime, duration: str, timezone_str: str = "UTC") -> datetime:
    """Start of the `duration` window that ends at `reset_at`.

    `Nmo` intervals — and `30d`, which the scheduler resets on the 1st — are counted on
    the calendar in `timezone_str` from the stored boundary, the same rule
    `get_next_standardized_reset_time` used to pick it, so a `2mo` window ending
    1 September opens on 1 July no matter when it is read. Measuring months with
    `duration_in_seconds` counts them from *now*, which puts that start on 2 July and
    leaves 1 July's spend outside the window it belongs to; for `30d` it lands on
    whatever day the closed month's length leaves, opening a window ending 1 March on
    31 January. Fixed-length units are subtracted as seconds either way."""
    months: Final = budget_duration_in_months(duration)
    if months is None:
        return reset_at - timedelta(seconds=duration_in_seconds(duration))
    boundary: Final = _setup_timezone(reset_at, timezone_str)[0]
    return _shift_months(boundary, -months).astimezone(timezone.utc)


def get_budget_window_start(duration: str, reset_at: datetime) -> datetime:
    """Start of the budget period that ends at `reset_at`, under the same rules
    `get_next_standardized_reset_time` used to pick it: `30d` and `Nmo` span calendar
    months and a duration it does not recognize resets at the next midnight."""
    value, unit = _parse_duration(normalize_duration(duration))
    if value is None:
        return reset_at - timedelta(days=1)
    match unit:
        case "mo":
            return _shift_months(reset_at, -value)
        case "d":
            return _shift_months(reset_at, -1) if value == 30 else reset_at - timedelta(days=value)
        case "w":
            return reset_at - timedelta(weeks=value)
        case "h":
            return reset_at - timedelta(hours=value)
        case "m":
            return reset_at - timedelta(minutes=value)
        case "s":
            return reset_at - timedelta(seconds=value)
        case _:
            return reset_at - timedelta(days=1)


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


def _handle_hour_reset(current_time: datetime, base_midnight: datetime, value: int) -> datetime:
    """Handle hour-based reset times."""
    # Handle zero value - immediate expiration
    if value == 0:
        return current_time

    current_hour: Final = current_time.hour
    current_minute: Final = current_time.minute
    current_second: Final = current_time.second
    current_microsecond: Final = current_time.microsecond

    # Calculate next hour aligned with the value
    if current_minute == 0 and current_second == 0 and current_microsecond == 0:
        next_hour = current_hour + value - (current_hour % value) if current_hour % value != 0 else current_hour + value
    else:
        next_hour = current_hour + value - (current_hour % value) if current_hour % value != 0 else current_hour + value

    # Handle overnight case
    if next_hour >= 24:
        next_hour = next_hour % 24
        next_day: Final = base_midnight + timedelta(days=1)
        return next_day.replace(hour=next_hour)

    return current_time.replace(hour=next_hour, minute=0, second=0, microsecond=0)


def _handle_minute_reset(current_time: datetime, base_midnight: datetime, value: int) -> datetime:
    """Handle minute-based reset times."""
    # Handle zero value - immediate expiration
    if value == 0:
        return current_time

    current_hour: Final = current_time.hour
    current_minute: Final = current_time.minute
    current_second: Final = current_time.second
    current_microsecond: Final = current_time.microsecond

    # Calculate next minute aligned with the value
    if current_second == 0 and current_microsecond == 0:
        next_minute = (
            current_minute + value - (current_minute % value) if current_minute % value != 0 else current_minute + value
        )
    else:
        next_minute = (
            current_minute + value - (current_minute % value) if current_minute % value != 0 else current_minute + value
        )

    # Handle hour rollover
    next_hour = current_hour + (next_minute // 60)
    next_minute = next_minute % 60

    # Handle overnight case
    if next_hour >= 24:
        next_hour = next_hour % 24
        next_day: Final = base_midnight + timedelta(days=1)
        return next_day.replace(hour=next_hour, minute=next_minute, second=0, microsecond=0)

    return current_time.replace(hour=next_hour, minute=next_minute, second=0, microsecond=0)


def _handle_second_reset(current_time: datetime, base_midnight: datetime, value: int) -> datetime:
    """Handle second-based reset times."""
    # Handle zero value - immediate expiration
    if value == 0:
        return current_time

    current_hour: Final = current_time.hour
    current_minute: Final = current_time.minute
    current_second: Final = current_time.second
    current_microsecond: Final = current_time.microsecond

    # Calculate next second aligned with the value
    if current_microsecond == 0:
        next_second = (
            current_second + value - (current_second % value) if current_second % value != 0 else current_second + value
        )
    else:
        next_second = (
            current_second + value - (current_second % value) if current_second % value != 0 else current_second + value
        )

    # Handle minute rollover
    additional_minutes: Final = next_second // 60
    next_second = next_second % 60
    next_minute = current_minute + additional_minutes

    # Handle hour rollover
    next_hour = current_hour + (next_minute // 60)
    next_minute = next_minute % 60

    # Handle overnight case
    if next_hour >= 24:
        next_hour = next_hour % 24
        next_day: Final = base_midnight + timedelta(days=1)
        return next_day.replace(hour=next_hour, minute=next_minute, second=next_second, microsecond=0)

    return current_time.replace(hour=next_hour, minute=next_minute, second=next_second, microsecond=0)


def _handle_month_reset(
    current_time: datetime,
    base_midnight: datetime,
    value: int,
    reset_time_of_day: time,
) -> datetime:
    """
    Handle monthly reset times. Resets land on the 1st at `reset_time_of_day`; if the
    1st of the current month at that time has already passed, roll `value` months ahead
    so `Nmo` windows stay aligned with the calendar the way `_shift_months` and
    `get_budget_window_start` already treat them.

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
    if value < 1:
        raise ValueError("Calendar resets require a positive number of months")

    first_of_this_month: Final = base_midnight.replace(day=1)
    candidate: Final = _apply_time_of_day(first_of_this_month, reset_time_of_day)
    if candidate <= current_time:
        return _apply_time_of_day(_shift_months(first_of_this_month, value), reset_time_of_day)
    return candidate
