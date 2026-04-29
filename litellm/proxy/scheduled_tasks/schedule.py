"""
Schedule parsing + next-run computation for LiteLLM_ScheduledTaskTable.

Ported from test_local_agent_2/task_runner.py:299-340. Three schedule kinds:
  interval: '30s', '5m', '2h', '1d'
  cron:     standard 5-field crontab; honours schedule_tz (IANA)
  once:     parked at year 9999; fire_once=True flips status='fired' after
            first fire so we never consult next_run_at again.

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
        return
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
    Returns UTC datetime. For 'once', returns far-future sentinel.

    `from_time` should be timezone-aware; if naive, treated as UTC.
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
        return _FAR_FUTURE
    raise ValueError(f"unknown schedule_kind: {kind!r}")
