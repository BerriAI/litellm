"""Collapsing stored health check rows to the latest one per model.

The store keeps one row per ``(model_id, model_name)`` pair, so the same deployment shows up more
than once whenever it was checked under several names or whenever rows saved before deployment ids
were stored are still around. Readers want the most recent check, which is what these helpers pick.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from operator import itemgetter
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


def _sole_deployment_id_per_name(rows: Sequence[StoredHealthRow]) -> Mapping[str, str]:
    """For each model name exactly one deployment was checked under, that deployment's id."""
    identified: Final = sorted((row.model_name, model_id) for row in rows if (model_id := row.model_id) is not None)
    per_name: Final = (
        (name, frozenset(model_id for _, model_id in group)) for name, group in groupby(identified, key=itemgetter(0))
    )
    return MappingProxyType({name: sole for name, ids in per_name if len(ids) == 1 for sole in ids})


def _newest_check_per_deployment(rows: Sequence[StoredHealthRow]) -> Mapping[str, datetime]:
    """The most recent check each deployment carries on a row that names its id."""
    return MappingProxyType(
        {model_id: _checked_at(row) for row in sorted(rows, key=_checked_at) if (model_id := row.model_id) is not None}
    )


def _deployment_key(
    row: StoredHealthRow, sole_id_by_name: Mapping[str, str], newest_by_deployment: Mapping[str, datetime]
) -> str:
    model_id: Final = row.model_id
    if model_id is not None:
        return model_id
    sole: Final = sole_id_by_name.get(row.model_name)
    if sole is None or _checked_at(row) >= newest_by_deployment[sole]:
        return row.model_name
    return sole


def latest_by_deployment(rows: Iterable[RowT]) -> Mapping[str, RowT]:
    """The newest row per deployment, keyed by deployment id.

    A row saved before deployment ids were stored folds into the one deployment its model name was
    checked under once that deployment's own check has overtaken it, so a model reports one status
    rather than a current one beside a legacy one. It only ever folds away: a row carrying no
    deployment id names a model rather than a deployment, so it never stands in for one, and a row
    no deployment's own newer check has overtaken stays keyed by its model name.

    The fold reads the rows it is handed, so a caller that may read only some of them filters first:
    collapsing the whole table would let a row that caller cannot read fold away one it can.
    """
    stored: Final = tuple(rows)
    sole_id_by_name: Final = _sole_deployment_id_per_name(stored)
    newest_by_deployment: Final = _newest_check_per_deployment(stored)
    return _newest_per_key(stored, lambda row: _deployment_key(row, sole_id_by_name, newest_by_deployment))


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
