"""Collapsing stored health check rows to the latest one per model.

The store keeps one row per ``(model_id, model_name)`` pair, so the same deployment shows up more
than once whenever it was checked under several names or whenever rows saved before deployment ids
were stored are still around. Readers want the most recent check, which is what these helpers pick.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Protocol, TypeVar


class StoredHealthRow(Protocol):
    """The fields a reader needs off a stored health check row."""

    @property
    def model_name(self) -> str: ...

    @property
    def model_id(self) -> str | None: ...

    @property
    def checked_at(self) -> datetime | None: ...


RowT = TypeVar("RowT", bound=StoredHealthRow)

_NEVER_CHECKED: Final = datetime.min.replace(tzinfo=timezone.utc)


def _checked_at(row: StoredHealthRow) -> datetime:
    checked_at: Final = row.checked_at
    if checked_at is None:
        return _NEVER_CHECKED
    return checked_at if checked_at.tzinfo is not None else checked_at.replace(tzinfo=timezone.utc)


def _newest_per_key(rows: Iterable[RowT], key: Callable[[RowT], str | None]) -> Mapping[str, RowT]:
    oldest_first: Final = sorted(rows, key=_checked_at)
    return MappingProxyType({row_key: row for row in oldest_first if (row_key := key(row))})


def latest_by_deployment(rows: Iterable[RowT]) -> Mapping[str, RowT]:
    """The newest row per deployment, keyed by deployment id where the row carries one, else by model name."""
    return _newest_per_key(rows, lambda row: row.model_id or row.model_name)


def latest_by_model_name(rows: Iterable[RowT]) -> Mapping[str, RowT]:
    """The newest row per model name, across rows saved with and without a deployment id."""
    return _newest_per_key(rows, lambda row: row.model_name)


class ReportedHealthRow(StoredHealthRow, Protocol):
    """A stored row plus the fields a model hub publishes off it."""

    @property
    def status(self) -> str | None: ...

    @property
    def response_time_ms(self) -> float | None: ...


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """The health fields a model hub row carries, as the latest health check recorded them."""

    status: str | None
    response_time_ms: float | None
    checked_at: str | None


def snapshot_of(row: ReportedHealthRow) -> HealthSnapshot:
    """The published view of one stored health check row."""
    return HealthSnapshot(
        status=row.status,
        response_time_ms=row.response_time_ms,
        checked_at=row.checked_at.isoformat() if row.checked_at else None,
    )
