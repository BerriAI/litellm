"""
Timezone-safe datetime parsing.

ISO-8601 strings arriving from API input, DB metadata, or serialized state may carry a
timezone offset ("...Z" / "+00:00") or not. Comparing an aware datetime with a naive one
raises TypeError, so every parse site must normalize. This module is the single allowed
entrypoint; a semgrep rule bans raw datetime.fromisoformat under litellm/proxy/.
"""

from datetime import datetime, timezone


def parse_utc_datetime(value: str | datetime) -> datetime:
    """Parse an ISO-8601 string (or pass through a datetime) into a tz-aware datetime.

    Naive values are assumed to be UTC, matching the convention used across the proxy
    (key expiry checks, budget windows, spend reports). The "Z" suffix is handled
    explicitly because datetime.fromisoformat only accepts it from Python 3.11 and the
    project floor is 3.10.

    Raises ValueError for unparseable strings and TypeError for any other type,
    mirroring datetime.fromisoformat's own contract so existing
    ``except (ValueError, TypeError)`` handlers stay fail-closed.
    """
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError(f"parse_utc_datetime expects str or datetime, got {type(value).__name__}")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
