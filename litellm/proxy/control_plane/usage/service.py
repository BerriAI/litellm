"""Fleet-wide usage: fetch each dataplane's daily activity, then pure reducers over SpendMetrics."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from functools import reduce
from itertools import groupby
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import TypeAdapter

from litellm.proxy.control_plane.common.fanout import Failure, parse_reply
from litellm.proxy.control_plane.dataplanes.client import DataplaneClient
from litellm.proxy.control_plane.dataplanes.credentials import Caller
from litellm.proxy.control_plane.dataplanes.registry import Dataplane
from litellm.types.proxy.control_plane.usage import UsageGroupBy
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailySpendData,
    SpendAnalyticsPaginatedResponse,
    SpendMetrics,
)

DAILY_ACTIVITY_PATH: Final = "/user/daily/activity"
PAGE_SIZE: Final = 1000
MAX_PAGES: Final = 10
NO_TEAM: Final = "(no team)"

_DAILY: Final = TypeAdapter(SpendAnalyticsPaginatedResponse)
_NUMBERS: Final = TypeAdapter(Mapping[str, float])

DatedGroup: TypeAlias = tuple[date, str]
Grouped: TypeAlias = tuple[tuple[DatedGroup, SpendMetrics], ...]


@dataclass(frozen=True, slots=True)
class UsageWindow:
    start: date
    end: date

    def params(self, page: int) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "start_date": self.start.isoformat(),
                "end_date": self.end.isoformat(),
                "page": str(page),
                "page_size": str(PAGE_SIZE),
            }
        )


@dataclass(frozen=True, slots=True)
class DataplaneDays:
    dataplane_id: str
    days: tuple[DailySpendData, ...]


async def fetch_daily_activity(
    client: DataplaneClient, dataplane: Dataplane, caller: Caller, window: UsageWindow
) -> tuple[DailySpendData, ...] | Failure:
    return await _collect(client, dataplane, caller, window, page=1, collected=())


async def _collect(
    client: DataplaneClient,
    dataplane: Dataplane,
    caller: Caller,
    window: UsageWindow,
    page: int,
    collected: tuple[DailySpendData, ...],
) -> tuple[DailySpendData, ...] | Failure:
    parsed: Final = parse_reply(
        await client.get(dataplane, DAILY_ACTIVITY_PATH, caller, params=window.params(page)), _DAILY
    )
    if isinstance(parsed, Failure):
        return parsed
    rows: Final = collected + tuple(parsed.results)
    if not parsed.metadata.has_more:
        return rows
    if page >= MAX_PAGES:
        return Failure(f"more than {MAX_PAGES} pages of daily activity, narrow the window")
    return await _collect(client, dataplane, caller, window, page + 1, rows)


def add_metrics(left: SpendMetrics, right: SpendMetrics) -> SpendMetrics:
    first: Final = _NUMBERS.validate_python(left.model_dump())
    second: Final = _NUMBERS.validate_python(right.model_dump())
    return SpendMetrics.model_validate({name: value + second[name] for name, value in first.items()})


def sum_metrics(metrics: Iterable[SpendMetrics]) -> SpendMetrics:
    return reduce(add_metrics, metrics, SpendMetrics())


def _key_label(api_key: str, alias: str | None) -> str:
    return alias if alias else api_key


def group_day(day: DailySpendData, group_by: UsageGroupBy, dataplane_id: str) -> Mapping[str, SpendMetrics]:
    breakdown: Final = day.breakdown
    match group_by:
        case "dataplane":
            return MappingProxyType({dataplane_id: day.metrics})
        case "model":
            return MappingProxyType({name: entry.metrics for name, entry in breakdown.models.items()})
        case "model_group":
            return MappingProxyType({name: entry.metrics for name, entry in breakdown.model_groups.items()})
        case "provider":
            return MappingProxyType({name: entry.metrics for name, entry in breakdown.providers.items()})
        case "endpoint":
            return MappingProxyType({name: entry.metrics for name, entry in breakdown.endpoints.items()})
        case "api_key":
            by_label: Final = tuple(
                (_key_label(key, entry.metadata.key_alias), entry.metrics) for key, entry in breakdown.api_keys.items()
            )
            return MappingProxyType({label: metrics for (_, label), metrics in _merge_by_key(by_label, day.date)})
        case "team":
            by_team: Final = tuple(
                (entry.metadata.team_id or NO_TEAM, entry.metrics) for entry in breakdown.api_keys.values()
            )
            return MappingProxyType({team: metrics for (_, team), metrics in _merge_by_key(by_team, day.date)})


def _merge_by_key(entries: Iterable[tuple[str, SpendMetrics]], day: date) -> Grouped:
    return merge_grouped(((day, group), metrics) for group, metrics in entries)


def merge_grouped(entries: Iterable[tuple[DatedGroup, SpendMetrics]]) -> Grouped:
    ordered: Final = sorted(entries, key=_dated_group)
    return tuple(
        (key, sum_metrics(metrics for _, metrics in bucket)) for key, bucket in groupby(ordered, key=_dated_group)
    )


def _dated_group(entry: tuple[DatedGroup, SpendMetrics]) -> DatedGroup:
    return entry[0]


def _dated_groups(days: DataplaneDays, group_by: UsageGroupBy) -> Iterable[tuple[DatedGroup, SpendMetrics]]:
    for day in days.days:
        for group, metrics in group_day(day, group_by, days.dataplane_id).items():
            yield (day.date, group), metrics


def daily_rows(fleet: Iterable[DataplaneDays], group_by: UsageGroupBy) -> Grouped:
    return merge_grouped(
        entry for days in fleet for entry in _dated_groups(days, group_by)
    )  # comprehension-ok: flatten


def ranked(fleet: Iterable[DataplaneDays], group_by: UsageGroupBy, limit: int) -> tuple[tuple[str, SpendMetrics], ...]:
    per_group: Final = merge_grouped(
        ((date.min, group), metrics) for (_, group), metrics in daily_rows(fleet, group_by)
    )
    by_spend: Final = sorted(per_group, key=lambda item: item[1].spend, reverse=True)
    return tuple((group, metrics) for (_, group), metrics in by_spend[:limit])


def totals(fleet: Iterable[DataplaneDays]) -> SpendMetrics:
    return sum_metrics(day.metrics for days in fleet for day in days.days)  # comprehension-ok: flatten


def dataplane_totals(days: DataplaneDays) -> SpendMetrics:
    return sum_metrics(day.metrics for day in days.days)
