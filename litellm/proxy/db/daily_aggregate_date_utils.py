"""
Helpers for converting between in-memory `date` strings (`YYYY-MM-DD`) used by
the daily-aggregate spend pipeline and the native `DateTime @db.Date` column
that Prisma now persists.

The in-memory / Redis representation is intentionally kept as a string so
existing transaction keys, JSON serialization in Redis, and existing typed
dict shapes (`BaseDailySpendTransaction.date: str`) continue to work without
changing the wire format. Conversion happens at the Prisma boundary only.
"""

from datetime import date, datetime, timezone
from typing import Optional, Union

DateLike = Union[str, date, datetime]


def to_db_date(value: Optional[DateLike]) -> Optional[datetime]:
    """Normalize a date-like value to a UTC-midnight ``datetime``.

    prisma-client-py serializes Python ``datetime`` instances when writing to
    a ``DateTime @db.Date`` column. Postgres then truncates to the date
    portion. We always emit a tz-aware UTC datetime so the truncation is
    deterministic regardless of the server timezone.

    Accepts:
      - ``None`` (passed through; the DB column is NOT NULL so this only
        matters for code paths that explicitly handle missing dates)
      - ``str`` in ``YYYY-MM-DD`` (or ``YYYY-MM-DDTHH:...``) format
      - ``datetime.date`` (not a ``datetime`` instance)
      - ``datetime.datetime`` (made tz-aware as UTC if naive)
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str):
        date_part = value.split("T", 1)[0]
        parsed = datetime.strptime(date_part, "%Y-%m-%d")
        return parsed.replace(tzinfo=timezone.utc)
    raise TypeError(
        f"Cannot convert value of type {type(value).__name__!r} to a DB date"
    )


def to_date_str(value: Optional[DateLike]) -> Optional[str]:
    """Normalize a date-like value to ``YYYY-MM-DD``.

    Used when grouping records returned from Prisma (where ``record.date`` is
    a ``datetime``) and when rendering responses or chart keys. Returns
    ``None`` if ``value`` is ``None`` so callers can pass through optional
    fields.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value.split("T", 1)[0]
    raise TypeError(
        f"Cannot convert value of type {type(value).__name__!r} to a date string"
    )
