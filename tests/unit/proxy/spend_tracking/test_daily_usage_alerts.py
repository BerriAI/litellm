from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.spend_tracking.daily_usage_alerts import DailyComparison, compare_daily_usage
from litellm.proxy.spend_tracking.usage_alert_job import completed_day
from litellm.repositories.usage_alert_metrics import DailyAlertMetrics
from litellm.types.proxy.usage_alerts import UsageAlertRuleInput

_DAY: Final = date(2026, 10, 20)


def _row(day: date, requests: int = 200, spend: float = 20, ms: float = 100_000, timed: int = 200) -> DailyAlertMetrics:
    return DailyAlertMetrics(
        entity_id="key-a", day=day, spend=spend, requests=requests, errors=10, response_time_ms=ms, timed_requests=timed
    )


@pytest.mark.parametrize(
    "metric,observed,baseline", [("spend", 100, 20), ("requests", 1000, 200), ("latency", 2000, 500)]
)
def test_spikes_use_recorded_reference_days_and_weighted_latency(metric: str, observed: float, baseline: float) -> None:
    rule: Final = UsageAlertRuleInput.model_validate(
        {"name": "spike", "scope_type": "key", "metric": metric, "direction": "increase"}
    )
    history: Final = tuple(_row(_DAY - timedelta(days=days)) for days in (1, 3, 7))
    current: Final = _row(_DAY, requests=1000, spend=100, ms=400_000)
    assert compare_daily_usage(rule, (*history, current), _DAY) == DailyComparison(
        3, observed, baseline, baseline * 3, True
    )


def test_missing_target_is_zero_for_counts_but_never_zero_latency() -> None:
    history: Final = tuple(_row(_DAY - timedelta(days=days)) for days in (1, 2, 3))
    rule: Final = UsageAlertRuleInput(name="drop", scope_type="key", metric="requests", direction="decrease")
    assert compare_daily_usage(rule, history, _DAY) == DailyComparison(3, 0, 200, 200 / 3, True)
    assert compare_daily_usage(rule.model_copy(update={"metric": "latency"}), history, _DAY) is None


def test_low_volume_is_insufficient_and_recorded_zero_baseline_uses_absolute_floor() -> None:
    rule: Final = UsageAlertRuleInput(name="spike", scope_type="key", metric="spend", direction="increase")
    history: Final = tuple(_row(_DAY - timedelta(days=days), requests=10, spend=0) for days in (1, 2, 3))
    assert compare_daily_usage(rule, (*history, _row(_DAY, spend=100)), _DAY) is None
    enough_requests: Final = tuple(replace(row, requests=200) for row in history)
    assert compare_daily_usage(rule, (*enough_requests, _row(_DAY, spend=100)), _DAY) == DailyComparison(
        3, 100, 0, 0, True
    )
    floor: Final = rule.model_copy(update={"min_value": 100})
    assert compare_daily_usage(floor, (*enough_requests, _row(_DAY, spend=100)), _DAY) == DailyComparison(
        3, 100, 0, 100, True
    )


def test_absolute_floor_suppresses_small_changes() -> None:
    history: Final = tuple(_row(_DAY - timedelta(days=days)) for days in (1, 2, 3))
    rule: Final = UsageAlertRuleInput(
        name="spike", scope_type="key", metric="spend", direction="increase", min_value=100
    )
    assert compare_daily_usage(rule, (*history, _row(_DAY, spend=100)), _DAY) == DailyComparison(3, 100, 20, 120, False)
    assert compare_daily_usage(rule, (*history, _row(_DAY, spend=60)), _DAY) == DailyComparison(3, 60, 20, 120, False)


def test_day_is_eligible_only_after_one_hour_utc_grace() -> None:
    assert completed_day(datetime(2026, 10, 21, 0, 59, tzinfo=timezone.utc)) == date(2026, 10, 19)
    assert completed_day(datetime(2026, 10, 21, 1, 0, tzinfo=timezone.utc)) == _DAY


def test_latency_weights_request_samples_instead_of_averaging_daily_averages() -> None:
    rule: Final = UsageAlertRuleInput(
        name="latency", scope_type="key", metric="latency", direction="increase", min_history_days=2
    )
    rows: Final = (
        _row(_DAY - timedelta(days=1), requests=100, ms=100_000, timed=100),
        _row(_DAY - timedelta(days=2), requests=400, ms=1_600_000, timed=400),
        _row(_DAY, requests=100, ms=1_100_000, timed=100),
    )
    assert compare_daily_usage(rule, rows, _DAY) == DailyComparison(2, 11000, 3400, 10200, True)


def test_errors_are_failed_request_count_and_require_absolute_change_floor() -> None:
    rule: Final = UsageAlertRuleInput(
        name="errors", scope_type="key", metric="errors", direction="increase", min_value=10
    )
    history: Final = tuple(_row(_DAY - timedelta(days=days)) for days in (1, 2, 3))
    current: Final = replace(_row(_DAY), errors=40)
    assert compare_daily_usage(rule, (*history, current), _DAY) == DailyComparison(3, 40, 10, 30, True)


def test_a_value_at_the_configured_multiplier_alerts() -> None:
    history: Final = tuple(_row(_DAY - timedelta(days=days)) for days in (1, 2, 3))
    rule: Final = UsageAlertRuleInput(name="spike", scope_type="key", metric="spend", direction="increase")
    assert compare_daily_usage(rule, (*history, _row(_DAY, spend=60)), _DAY) == DailyComparison(3, 60, 20, 60, True)


def test_default_notifications_require_opt_in_and_do_not_duplicate_legacy_user_spend() -> None:
    from litellm.proxy.spend_tracking.usage_alert_job import default_usage_rules

    defaults: Final = default_usage_rules()
    assert len(defaults) == 40
    assert all(not rule.notification_channels for rule in defaults)
    opted_in: Final = default_usage_rules(("slack",), legacy_user_spend_alerts=True)
    quiet: Final = tuple(rule for rule in opted_in if not rule.notification_channels)
    assert tuple((rule.scope_type, rule.metric, rule.direction) for rule in quiet) == (("user", "spend", "increase"),)


@pytest.mark.parametrize(
    "scope,entity",
    [("key", "unattributed"), ("key", "sk-example"), ("user", "default_user_id"), ("team", "litellm-dashboard")],
)
def test_sentinel_entities_cannot_create_alerts_with_unusable_activity_links(scope: str, entity: str) -> None:
    from litellm.proxy.spend_tracking.usage_alert_job import _observations
    from litellm.types.proxy.usage_alerts import UsageAlertRule

    rule: Final = UsageAlertRule.model_validate(
        {
            "id": "rule",
            "name": "spike",
            "scope_type": scope,
            "metric": "spend",
            "direction": "increase",
            "created_at": "2026-10-01T00:00:00Z",
            "updated_at": "2026-10-01T00:00:00Z",
            "created_by": None,
        }
    )
    history: Final = tuple(replace(_row(_DAY - timedelta(days=days)), entity_id=entity) for days in (1, 2, 3))
    current: Final = replace(_row(_DAY, spend=100), entity_id=entity)
    assert _observations(rule, (*history, current), _DAY) == ()


def test_custom_entity_rule_replaces_only_the_matching_default() -> None:
    from uuid import NAMESPACE_URL, uuid5

    from litellm.proxy.spend_tracking.usage_alert_job import _suppressed_entities
    from litellm.types.proxy.usage_alerts import UsageAlertRule

    default: Final = UsageAlertRule(
        id=str(uuid5(NAMESPACE_URL, "litellm:usage-alert:key:spend:increase")),
        name="default",
        scope_type="key",
        metric="spend",
        direction="increase",
        created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        created_by=None,
    )
    custom: Final = default.model_copy(update={"id": "custom", "scope_id": "key-a"})
    other_metric: Final = custom.model_copy(update={"id": "latency", "metric": "latency", "scope_id": "key-b"})
    assert _suppressed_entities(default, (default, custom, other_metric)) == frozenset({"key-a"})
    assert _suppressed_entities(custom, (default, custom, other_metric)) == frozenset()
    wildcard: Final = custom.model_copy(update={"scope_id": None})
    assert _suppressed_entities(default, (default, wildcard)) == frozenset({None})


def test_database_metric_rows_preserve_numeric_and_date_validation() -> None:
    from pydantic import TypeAdapter, ValidationError

    adapter: Final = TypeAdapter(tuple[DailyAlertMetrics, ...])
    record: Final = {
        "entity_id": "key-a",
        "day": "2026-10-20",
        "spend": "20",
        "requests": "200",
        "errors": "10",
        "response_time_ms": "100000",
        "timed_requests": "200",
    }
    assert adapter.validate_python([record]) == (_row(_DAY),)
    with pytest.raises(ValidationError):
        adapter.validate_python([{**record, "requests": "invalid"}])


def test_reporting_identity_preserves_canonical_rows_and_distinct_raw_buckets() -> None:
    from litellm.repositories.usage_alert_metrics import _reporting_id

    canonical: Final = _row(_DAY)
    assert _reporting_id("key", canonical) is canonical
    assert _reporting_id("model", canonical) is canonical
    assert _reporting_id("key", replace(canonical, entity_id="")).entity_id == "unattributed"
    assert _reporting_id("key", replace(canonical, entity_id="unattributed")).entity_id == "raw:unattributed"
    assert _reporting_id("key", replace(canonical, entity_id="raw:key")).entity_id == "raw:raw:key"
