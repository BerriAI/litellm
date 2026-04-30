"""
Schedule parsing + next-run computation for LiteLLM_ScheduledTaskTable.

Three schedule kinds:
  interval: '30s', '5m', '2h', '1d'
  cron:     standard 5-field crontab; honours schedule_tz (IANA)
  once:     ISO-8601 absolute timestamp ('2026-04-30T00:00:00Z' or with
            offset/microseconds). Fires once at that instant; fire_once=True
            flips status='fired' after the first claim.

All returned datetimes are UTC, timezone-aware.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.triggers.cron import (  # type: ignore[import-not-found, import-untyped]
    CronTrigger,
)


_INTERVAL_RE = re.compile(r"^\s*(\d+)\s*([smhd])\s*$", re.IGNORECASE)
_INTERVAL_UNITS = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
}
_FAR_FUTURE = datetime(9999, 1, 1, tzinfo=timezone.utc)


def _parse_interval(spec: str) -> timedelta:
    m = _INTERVAL_RE.match(spec or "")
    if not m:
        raise ValueError(f"invalid interval schedule_spec: {spec!r}")
    n = int(m.group(1))
    unit = _INTERVAL_UNITS[m.group(2).lower()]
    return timedelta(**{unit: n})


def _parse_once(spec: str) -> datetime:
    """
    Parse an ISO-8601 timestamp for kind='once'. Accepts trailing 'Z'
    (Python <3.11 datetime.fromisoformat doesn't), explicit offsets, and
    naive timestamps (treated as UTC). Always returns a tz-aware UTC
    datetime.
    """
    if not spec:
        raise ValueError("schedule_spec for kind='once' must not be empty")
    normalized = spec.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as e:
        raise ValueError(
            f"schedule_spec for kind='once' must be ISO-8601, got {spec!r}"
        ) from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def validate_schedule(
    *,
    kind: str,
    spec: str,
    tz: str | None,
) -> None:
    """Raise ValueError if any of (kind, spec, tz) is malformed."""
    if kind == "interval":
        _parse_interval(spec)
    elif kind == "cron":
        try:
            CronTrigger.from_crontab(spec, timezone=tz or "UTC")
        except Exception as e:
            raise ValueError(f"invalid cron schedule_spec: {spec!r} ({e})") from e
        if tz is not None:
            try:
                ZoneInfo(tz)
            except ZoneInfoNotFoundError as e:
                raise ValueError(f"invalid schedule_tz: {tz!r}") from e
    elif kind == "once":
        _parse_once(spec)
    else:
        raise ValueError(f"unknown schedule_kind: {kind!r}")


def compute_next_run(
    *,
    kind: str,
    spec: str,
    tz: str | None,
    from_time: datetime,
) -> datetime:
    """
    Returns UTC datetime.

    `from_time` should be timezone-aware; if naive, treated as UTC.

    For 'once', schedule_spec is an absolute ISO-8601 timestamp and is
    returned verbatim (in UTC). This is the value the caller wants the
    task to fire at — fire_once=True flips it to status='fired' after
    the first claim, so we never re-consult it.
    """
    if from_time.tzinfo is None:
        from_time = from_time.replace(tzinfo=timezone.utc)
    if kind == "interval":
        return from_time + _parse_interval(spec)
    if kind == "cron":
        trigger = CronTrigger.from_crontab(spec, timezone=tz or "UTC")
        nxt = trigger.get_next_fire_time(None, from_time)
        if nxt is None:
            return _FAR_FUTURE
        return nxt.astimezone(timezone.utc)
    if kind == "once":
        return _parse_once(spec)
    raise ValueError(f"unknown schedule_kind: {kind!r}")
