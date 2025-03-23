"""
Helper utilities for parsing durations - 1s, 1d, 10d, 30d, 1mo, 2mo

duration_in_seconds is used in diff parts of the code base, example 
- Router - Provider budget routing
- Proxy - Key, Team Generation
"""

import re
import time
from datetime import datetime, timedelta
from typing import Tuple


def _extract_from_regex(duration: str) -> Tuple[int, str]:
    match = re.match(r"(\d+)(mo|[smhdw]?)", duration)

    if not match:
        raise ValueError("Invalid duration format")

    value, unit = match.groups()
    value = int(value)

    return value, unit


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

    Returns time in seconds till when budget needs to be reset. 
    Budgets reset at the top of the hour, day, week, etc. depending on the duration.
    For example:
        - if the duration is "1m", the budget will reset at the top of the next minute.
        - if the duration is "1h", the budget will reset at the top of the next hour.
        - if the duration is "1d", the budget will reset at the top of the next day (00:00:00 UTC).
        - if the duration is "1w", the budget will reset at the top of the next Monday (00:00:00 UTC).
        - if the duration is "1mo", the budget will reset at the top of the next month (00:00:00 UTC). 
    """
    value, unit = _extract_from_regex(duration=duration)
    current_time = datetime.now()

    if unit == "s":
        return value
    elif unit == "m":
        target = current_time.replace(second=0, microsecond=0) + timedelta(minutes=value)
        duration_until_target = target - current_time
        return int(duration_until_target.total_seconds())
    elif unit == "h":
        target = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=value)
        duration_until_target = target - current_time
        return int(duration_until_target.total_seconds())
    elif unit == "d": # Calculate the time until the nth day at 00:00:00 UTC from now in seconds
        target = current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=value)
        duration_until_target = target - current_time
        return int(duration_until_target.total_seconds())
    elif unit == "w": # Calculate the time until the nth Monday at 00:00:00 UTC from now in seconds        
        time_until_midnight = timedelta(hours=23 - current_time.hour, minutes=59 - current_time.minute, seconds=60 - current_time.second)
        time_from_midnight_to_monday = timedelta(days=7 - current_time.weekday() - 1)
        time_from_monday_to_target = timedelta(weeks=value - 1)
        duration_until_target = time_until_midnight + time_from_midnight_to_monday + time_from_monday_to_target
        return int(duration_until_target.total_seconds())
                
    elif unit == "mo":
        # calculate the time until the first day of the nth month at 00:00:00 UTC from now in seconds
        target_year = current_time.year + (current_time.month + value) // 12
        target_month = (current_time.month + value - 1) % 12 + 1

        print(f"target_year: {target_year}, target_month: {target_month}")

        target = datetime(
            year=target_year,
            month=target_month,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        duration_until_month = target - current_time
        return int(duration_until_month.total_seconds())
        

    else:
        raise ValueError(f"Unsupported duration unit, passed duration: {duration}")
