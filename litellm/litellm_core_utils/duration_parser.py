"""
Helper utilities for parsing durations - 1s, 1d, 10d, 30d, 1mo, 2mo

duration_in_seconds is used in diff parts of the code base, example 
- Router - Provider budget routing
- Proxy - Key, Team Generation
"""

import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Tuple


def _extract_from_regex(duration: str) -> Tuple[int, str]:
    match = re.match(r"(\d+)(mo|[smhdw]?)", duration)

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
    next_month = datetime(year=year, month=month + 1, day=1)
    last_day_of_month = (next_month - timedelta(days=1)).day
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
    value, unit = _extract_from_regex(duration=duration)

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
        now = time.time()
        current_time = datetime.fromtimestamp(now)

        if current_time.month == 12:
            target_year = current_time.year + 1
            target_month = 1
        else:
            target_year = current_time.year
            target_month = current_time.month + value

        # Determine the day to set for next month
        target_day = current_time.day
        last_day_of_target_month = get_last_day_of_month(target_year, target_month)

        if target_day > last_day_of_target_month:
            target_day = last_day_of_target_month

        next_month = datetime(
            year=target_year,
            month=target_month,
            day=target_day,
            hour=current_time.hour,
            minute=current_time.minute,
            second=current_time.second,
            microsecond=current_time.microsecond,
        )

        # Calculate the duration until the first day of the next month
        duration_until_next_month = next_month - current_time
        return int(duration_until_next_month.total_seconds())

    else:
        raise ValueError(f"Unsupported duration unit, passed duration: {duration}")

def get_next_standardized_reset_time(
    duration: str, 
    current_time: datetime,
    timezone_str: str = "UTC"
) -> datetime:
    """
    Get the next standardized reset time based on the duration.
    
    All durations will reset at predictable intervals, aligned from the current time:
    - Nd: If N=1, reset at next midnight; if N>1, reset every N days from now
    - Nh: Every N hours, aligned to hour boundaries (e.g., 1:00, 2:00)
    - Nm: Every N minutes, aligned to minute boundaries (e.g., 1:05, 1:10)
    - Ns: Every N seconds, aligned to second boundaries
    
    Parameters:
    - duration: Duration string (e.g. "30s", "30m", "30h", "30d")
    - current_time: Current datetime
    - timezone_str: Timezone string (e.g. "UTC", "US/Eastern", "Asia/Kolkata")
    
    Returns:
    - Next reset time at a standardized interval in the specified timezone
    """
    # Set up the timezone
    try:
        if timezone_str is None:
            timezone = ZoneInfo("UTC")
        else:
            timezone = ZoneInfo(timezone_str)
    except (ImportError, KeyError):
        # If timezone is invalid, fall back to UTC
        timezone = ZoneInfo("UTC")
    
    # Convert current_time to the target timezone
    if current_time.tzinfo is None:
        # Naive datetime - assume it's UTC
        utc_time = current_time.replace(tzinfo=ZoneInfo("UTC"))
        current_time = utc_time.astimezone(timezone)
    else:
        # Already has timezone - convert to target timezone
        current_time = current_time.astimezone(timezone)
    
    match = re.match(r"(\d+)([a-z]+)", duration)
    if not match:
        # Fall back to default if format is invalid
        return current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    value, unit = match.groups()
    value = int(value)
    
    # Midnight of the current day in the specified timezone
    base_midnight = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Handle days
    if unit == "d":
        if value == 1:  # Daily reset at midnight
            return base_midnight + timedelta(days=1)
        elif value == 7:  # Weekly reset on Monday at midnight
            days_until_monday = (7 - current_time.weekday()) % 7
            if days_until_monday == 0:  # If today is Monday
                days_until_monday = 7
            return base_midnight + timedelta(days=days_until_monday)
        elif value == 30:  # Monthly reset on 1st at midnight
            # Get 1st of next month at midnight
            if current_time.month == 12:
                next_reset = datetime(
                    year=current_time.year + 1, 
                    month=1, 
                    day=1, 
                    hour=0, 
                    minute=0, 
                    second=0, 
                    microsecond=0,
                    tzinfo=timezone
                )
            else:
                next_reset = datetime(
                    year=current_time.year, 
                    month=current_time.month + 1, 
                    day=1, 
                    hour=0, 
                    minute=0, 
                    second=0, 
                    microsecond=0,
                    tzinfo=timezone
                )
            return next_reset
        else:  # Custom day value - next interval is value days from current
            return current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=value)
    
    # Handle hours - find next hour boundary aligned with the frequency
    elif unit == "h":
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_second = current_time.second
        current_microsecond = current_time.microsecond
        
        # If exactly on the hour and no minutes/seconds/microseconds, use the next interval
        if current_minute == 0 and current_second == 0 and current_microsecond == 0:
            # Calculate next hour aligned with the value
            next_hour = current_hour + value - (current_hour % value) if current_hour % value != 0 else current_hour + value
        else:
            # Not on the hour boundary, round up to next aligned hour
            next_hour = current_hour + value - (current_hour % value) if current_hour % value != 0 else current_hour + value
        
        # Handle overnight case
        if next_hour >= 24:
            next_hour = next_hour % 24
            next_day = base_midnight + timedelta(days=1)
            return next_day.replace(hour=next_hour)
        
        return current_time.replace(hour=next_hour, minute=0, second=0, microsecond=0)
    
    # Handle minutes - find next minute boundary aligned with the frequency
    elif unit == "m":
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_second = current_time.second
        current_microsecond = current_time.microsecond
        
        # If exactly on the minute and no seconds/microseconds, use the next interval
        if current_second == 0 and current_microsecond == 0:
            # Calculate next minute aligned with the value
            next_minute = current_minute + value - (current_minute % value) if current_minute % value != 0 else current_minute + value
        else:
            # Not on the minute boundary, round up to next aligned minute
            next_minute = current_minute + value - (current_minute % value) if current_minute % value != 0 else current_minute + value
        
        # Handle hour rollover
        next_hour = current_hour + (next_minute // 60)
        next_minute = next_minute % 60
        
        # Handle overnight case
        if next_hour >= 24:
            next_hour = next_hour % 24
            next_day = base_midnight + timedelta(days=1)
            return next_day.replace(hour=next_hour, minute=next_minute, second=0, microsecond=0)
        
        return current_time.replace(hour=next_hour, minute=next_minute, second=0, microsecond=0)
    
    # Handle seconds - find next second boundary aligned with the frequency
    elif unit == "s":
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_second = current_time.second
        current_microsecond = current_time.microsecond
        
        # If exactly on the second and no microseconds, use the next interval
        if current_microsecond == 0:
            # Calculate next second aligned with the value
            next_second = current_second + value - (current_second % value) if current_second % value != 0 else current_second + value
        else:
            # Not on the second boundary, round up to next aligned second
            next_second = current_second + value - (current_second % value) if current_second % value != 0 else current_second + value
        
        # Handle minute rollover
        additional_minutes = next_second // 60
        next_second = next_second % 60
        next_minute = current_minute + additional_minutes
        
        # Handle hour rollover
        next_hour = current_hour + (next_minute // 60)
        next_minute = next_minute % 60
        
        # Handle overnight case
        if next_hour >= 24:
            next_hour = next_hour % 24
            next_day = base_midnight + timedelta(days=1)
            return next_day.replace(hour=next_hour, minute=next_minute, second=next_second, microsecond=0)
        
        return current_time.replace(hour=next_hour, minute=next_minute, second=next_second, microsecond=0)
    
    else:
        # Unrecognized unit, default to next midnight
        return base_midnight + timedelta(days=1)