from datetime import date, datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.models.alert import AlertDetection, MetricEvidence
from litellm.models.performance_alert import (
    ResourceWindow,
    SweepState,
    advance,
    detect,
    new_sweep,
    next_scope,
    rotation_start,
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
    "exists": True,
}


def _window(**changes: object) -> ResourceWindow:
    return ResourceWindow.model_validate({**STEADY, **changes})


def _types(**changes: object) -> tuple[str, ...]:
    return tuple(alert.type for alert in detect("team", _window(**changes), DAY))


def test_steady_usage_raises_nothing() -> None:
    assert _types() == ()


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
        ({"requests": 99, "failed": 0, "timed": 99, "response_ms": 9900, "base_requests": 3500}, ("traffic",)),
        ({"requests": 100, "failed": 0, "timed": 100, "response_ms": 10_000, "base_requests": 3500}, ()),
        ({"requests": 0, "failed": 0, "timed": 0, "response_ms": 0, "base_requests": 3500, "active_days": 6}, ()),
        ({"requests": 0, "failed": 0, "timed": 0, "response_ms": 0, "base_requests": 3500, "exists": False}, ()),
        ({"requests": 0, "failed": 0, "timed": 0, "response_ms": 0, "base_requests": 3493}, ()),
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


def test_gateway_moves_to_models_starting_from_the_days_rotation_point() -> None:
    assert advance(new_sweep(DAY), ("gateway",), 25) == SweepState(
        day=DAY, scope="model", cursor=rotation_start(DAY), checked=1
    )


def test_a_full_page_continues_from_its_last_resource() -> None:
    state: Final = SweepState(day=DAY, scope="team", cursor=rotation_start(DAY))
    assert advance(state, ("t1", "t2"), 2) == state.model_copy(update={"cursor": "t2", "checked": 2})


def test_a_short_page_wraps_to_the_ids_before_the_rotation_point_then_moves_on() -> None:
    first_half: Final = advance(SweepState(day=DAY, scope="team", cursor="t9"), ("t10",), 25)
    assert (first_half.scope, first_half.cursor, first_half.wrapped) == ("team", "", True)
    second_half: Final = advance(first_half, ("a1",), 25)
    assert (second_half.scope, second_half.cursor, second_half.wrapped) == ("user", rotation_start(DAY), False)


def test_the_sweep_completes_after_keys() -> None:
    finished: Final = next_scope(SweepState(day=DAY, scope="key", cursor="k", wrapped=True, checked=9))
    assert finished.completed and finished.checked == 9


def test_a_skipped_resource_counts_as_skipped_not_checked() -> None:
    state: Final = SweepState(day=DAY, scope="user", cursor="u1", checked=4)
    assert advance(state, ("u2",), 1, skipped=1) == state.model_copy(update={"cursor": "u2", "skipped": 1})


def test_rotation_point_differs_between_days() -> None:
    assert rotation_start(DAY) != rotation_start(DAY + timedelta(days=1))
