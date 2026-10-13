from datetime import date

import pytest

from litellm.proxy.control_plane.usage.service import (
    NO_TEAM,
    DataplaneDays,
    add_metrics,
    daily_rows,
    group_day,
    ranked,
    totals,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    BreakdownMetrics,
    DailySpendData,
    KeyMetadata,
    KeyMetricWithMetadata,
    MetricWithMetadata,
    SpendMetrics,
)

D1 = date(2026, 10, 1)
D2 = date(2026, 10, 2)


def _m(spend: float, requests: int = 1) -> SpendMetrics:
    return SpendMetrics(spend=spend, api_requests=requests, successful_requests=requests, total_tokens=requests * 10)


def _day(
    day: date,
    spend: float,
    models: dict[str, float] | None = None,
    keys: dict[str, tuple[float, str | None, str | None]] | None = None,
) -> DailySpendData:
    return DailySpendData(
        date=day,
        metrics=_m(spend),
        breakdown=BreakdownMetrics(
            models={name: MetricWithMetadata(metrics=_m(cost)) for name, cost in (models or {}).items()},
            api_keys={
                key: KeyMetricWithMetadata(metrics=_m(cost), metadata=KeyMetadata(key_alias=alias, team_id=team))
                for key, (cost, alias, team) in (keys or {}).items()
            },
        ),
    )


def test_add_metrics_sums_every_numeric_field() -> None:
    left = SpendMetrics.model_validate({name: 1 for name in SpendMetrics.model_fields})
    right = SpendMetrics.model_validate({name: 2 for name in SpendMetrics.model_fields})
    total = add_metrics(left, right)
    assert total.model_dump() == {name: 3 for name in SpendMetrics.model_fields}


def test_totals_add_up_across_dataplanes_and_days() -> None:
    fleet = (
        DataplaneDays("eu", (_day(D1, 1.0), _day(D2, 2.0))),
        DataplaneDays("us", (_day(D1, 4.0),)),
    )
    assert totals(fleet) == _m(7.0, requests=3)


def test_daily_rows_merge_the_same_model_on_the_same_day_from_two_dataplanes() -> None:
    fleet = (
        DataplaneDays(
            "eu", (_day(D1, 3.0, models={"gpt-4o": 2.0, "claude": 1.0}), _day(D2, 1.0, models={"claude": 1.0}))
        ),
        DataplaneDays("us", (_day(D1, 5.0, models={"gpt-4o": 5.0}),)),
    )
    rows = daily_rows(fleet, "model")
    assert [(day, group, metrics.spend) for (day, group), metrics in rows] == [
        (D1, "claude", 1.0),
        (D1, "gpt-4o", 7.0),
        (D2, "claude", 1.0),
    ]
    assert daily_rows(fleet, "dataplane") == (
        ((D1, "eu"), _m(3.0)),
        ((D1, "us"), _m(5.0)),
        ((D2, "eu"), _m(1.0)),
    )


def test_team_grouping_folds_keys_into_their_team_and_labels_keys_by_alias() -> None:
    day = _day(
        D1,
        6.0,
        keys={"hash-a": (1.0, "alice-key", "t1"), "hash-b": (2.0, None, "t1"), "hash-c": (3.0, "svc", None)},
    )
    assert {team: metrics.spend for team, metrics in group_day(day, "team", "eu").items()} == {"t1": 3.0, NO_TEAM: 3.0}
    assert set(group_day(day, "api_key", "eu")) == {"alice-key", "hash-b", "svc"}


def test_ranked_orders_by_fleet_spend_and_honours_the_limit() -> None:
    fleet = (
        DataplaneDays("eu", (_day(D1, 0, models={"a": 1.0, "b": 5.0}), _day(D2, 0, models={"a": 1.0}))),
        DataplaneDays("us", (_day(D1, 0, models={"c": 3.0, "a": 2.0}),)),
    )
    assert [(group, metrics.spend) for group, metrics in ranked(fleet, "model", limit=2)] == [("b", 5.0), ("a", 4.0)]
    assert len(ranked(fleet, "model", limit=10)) == 3


@pytest.mark.parametrize("dimension", ["model_group", "provider", "endpoint"])
def test_empty_breakdowns_group_to_nothing(dimension) -> None:
    assert dict(group_day(_day(D1, 1.0), dimension, "eu")) == {}


def test_two_keys_sharing_an_alias_add_up_instead_of_overwriting() -> None:
    day = _day(D1, 8.0, keys={"hash-a": (3.0, "shared", None), "hash-b": (5.0, "shared", None)})
    assert {label: metrics.spend for label, metrics in group_day(day, "api_key", "eu").items()} == {"shared": 8.0}
