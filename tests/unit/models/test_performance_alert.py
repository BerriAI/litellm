from datetime import date, datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.models.alert import AlertDetection, MetricEvidence
from litellm.models.performance_alert import (
    SCOPES,
    PerformanceScope,
    ResourceWindow,
    SweepState,
    advance,
    detect,
    new_sweep,
    next_scope,
    target_day,
)

DAY: Final = date(2026, 1, 8)
STEADY: Final = {
    "resource_id": "team-a",
    "spend": 10.0,
    "base_spend": 70.0,
    "requests": 1000,
    "base_requests": 7000,
    "failed": 10,
    "base_failed": 70,
    "response_ms": 100_000,
    "timed": 1000,
    "base_response_ms": 700_000,
    "base_timed": 7000,
    "active_days": 7,
    "timed_days": 7,
}


def _window(**changes: object) -> ResourceWindow:
    return ResourceWindow.model_validate({**STEADY, **changes})


def _types(**changes: object) -> tuple[str, ...]:
    return tuple(alert.type for alert in detect("team", _window(**changes), DAY))


def test_steady_usage_raises_nothing() -> None:
    assert _types() == ()


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("requests", (0, 100))
def test_quiet_or_idle_resources_do_not_raise_traffic_alerts(scope: PerformanceScope, requests: int) -> None:
    window: Final = _window(
        resource_id="a" * 64 if scope == "key" else "resource-a",
        requests=requests,
        failed=0,
        response_ms=requests * 100,
        timed=requests,
    )
    assert detect(scope, window, DAY) == ()


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({"spend": 60.0}, ("spend",)),
        ({"spend": 59.9}, ()),
        ({"base_spend": 280.0, "spend": 120.0}, ("spend",)),
        ({"base_spend": 280.0, "spend": 119.9}, ()),
        ({"spend": 5000.0, "active_days": 3}, ()),
        ({"spend": 5000.0, "base_spend": 0.0}, ()),
        ({"requests": 1100, "failed": 0, "response_ms": 110_000, "timed": 1100, "base_requests": 700}, ("traffic",)),
        ({"requests": 1099, "failed": 0, "response_ms": 109_900, "timed": 1099, "base_requests": 700}, ()),
        ({"requests": 50_000, "failed": 0, "active_days": 3}, ()),
        ({"response_ms": 600_000}, ("latency",)),
        ({"response_ms": 599_000}, ()),
        ({"response_ms": 60_000_000, "timed": 99}, ()),
        ({"response_ms": 600_000, "timed_days": 3}, ()),
        ({"failed": 40}, ("reliability",)),
        ({"failed": 25}, ()),
        ({"requests": 499, "failed": 400, "timed": 499, "response_ms": 49_900}, ()),
        ({"base_failed": 0, "failed": 21}, ("reliability",)),
    ),
)
def test_each_rule_fires_only_past_both_its_multiplier_and_its_floor(
    changes: dict[str, object], expected: tuple[str, ...]
) -> None:
    assert _types(**changes) == expected


def test_alert_saves_the_comparison_that_triggered_it() -> None:
    alerts: Final = detect("team", _window(spend=60.0), DAY)
    assert alerts == (
        AlertDetection.model_validate(
            {
                "type": "spend",
                "resource_type": "team",
                "resource_id": "team-a",
                "window_start": "2026-01-08T00:00:00Z",
                "window_end": "2026-01-09T00:00:00Z",
                "evidence": {"value": 60.0, "baseline": 10.0, "unit": "usd", "sample_size": 1000},
            }
        ),
    )


def test_latency_and_failure_rate_are_weighted_by_requests_not_averaged_per_day() -> None:
    alerts: Final = detect("model", _window(response_ms=600_000, failed=40), DAY)
    assert {alert.type: alert.evidence for alert in alerts} == {
        "latency": MetricEvidence(value=600.0, baseline=100.0, unit="milliseconds", sample_size=1000),
        "reliability": MetricEvidence(value=0.04, baseline=0.01, unit="ratio", sample_size=1000),
    }


def test_gateway_alerts_use_the_gateway_resource_id() -> None:
    alerts: Final = detect("gateway", _window(resource_id="anything", spend=60.0), DAY)
    assert tuple((alert.resource_type, alert.resource_id) for alert in alerts) == (("gateway", "gateway"),)


@pytest.mark.parametrize(
    ("now", "expected"),
    (
        (datetime(2026, 1, 10, 0, 59, tzinfo=timezone.utc), date(2026, 1, 8)),
        (datetime(2026, 1, 10, 1, 0, tzinfo=timezone.utc), date(2026, 1, 9)),
        (datetime(2026, 1, 10, 9, 0, tzinfo=timezone(timedelta(hours=8))), date(2026, 1, 9)),
    ),
)
def test_the_evaluated_day_changes_at_one_am_utc(now: datetime, expected: date) -> None:
    assert target_day(now) == expected


def test_gateway_moves_to_models_starting_from_the_beginning() -> None:
    assert advance(new_sweep(DAY), ("gateway",), 25) == SweepState(day=DAY, scope="model", checked=1)


def test_a_full_page_continues_from_its_last_resource() -> None:
    state: Final = SweepState(day=DAY, scope="team")
    assert advance(state, ("t1", "t2"), 2) == state.model_copy(update={"cursor": "t2", "checked": 2})


def test_a_scope_without_a_carried_start_moves_on_after_one_pass() -> None:
    finished: Final = advance(SweepState(day=DAY, scope="team", cursor="t9"), ("t10",), 25)
    assert (finished.scope, finished.cursor, finished.wrapped) == ("user", "", False)


def test_an_unfinished_day_restarts_its_scope_where_it_stopped_then_wraps() -> None:
    previous: Final = SweepState(day=DAY - timedelta(days=1), scope="team", cursor="t5", checked=40)
    today: Final = new_sweep(DAY, previous)
    assert (today.scope, today.start_scope, today.start_after, today.checked) == ("gateway", "team", "t5", 0)
    at_team: Final = next_scope(next_scope(today))
    assert (at_team.scope, at_team.cursor, at_team.wrapped) == ("team", "t5", False)
    after_tail: Final = advance(at_team, ("t6", "t7"), 25)
    assert (after_tail.scope, after_tail.cursor, after_tail.wrapped) == ("team", "", True)
    after_head: Final = advance(after_tail, ("t1", "t5"), 25)
    assert (after_head.scope, after_head.cursor, after_head.wrapped) == ("user", "", False)


@pytest.mark.parametrize(
    "previous",
    (
        SweepState(day=DAY - timedelta(days=1), scope="team", cursor="t5", completed=True),
        SweepState(day=DAY - timedelta(days=1), scope="gateway"),
    ),
)
def test_a_finished_or_barely_started_day_carries_nothing(previous: SweepState) -> None:
    assert new_sweep(DAY, previous) == SweepState(day=DAY)


def test_the_sweep_completes_after_keys() -> None:
    finished: Final = next_scope(SweepState(day=DAY, scope="key", cursor="k", wrapped=True, checked=9))
    assert finished.completed and finished.checked == 9


def test_a_skipped_resource_counts_as_skipped_not_checked() -> None:
    state: Final = SweepState(day=DAY, scope="user", cursor="u1", checked=4)
    assert advance(state, ("u2",), 1, skipped=1) == state.model_copy(update={"cursor": "u2", "skipped": 1})
