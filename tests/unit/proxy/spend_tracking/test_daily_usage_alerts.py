import asyncio
import heapq
from collections import deque
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.spend_tracking.daily_usage_alerts import DailyComparison, compare_daily_usage
from litellm.proxy.spend_tracking.usage_alert_job import UsageAlertJob, completed_day
from litellm.repositories.analytics_budget import AnalyticsTransaction, BoundedAnalyticsDatabase
from litellm.repositories.usage_alert_health import UsageAlertHealthRepository
from litellm.repositories.usage_alert_metrics import DailyAlertMetrics, UsageAlertMetricsRepository
from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.proxy.usage_alert_defaults import default_usage_rule_id, default_usage_rules
from litellm.types.proxy.usage_alerts import UsageAlert, UsageAlertChannel, UsageAlertRule, UsageAlertRuleInput

_DAY: Final = date(2026, 10, 20)


class _JobBudgetClock(asyncio.SelectorEventLoop):
    _ready: deque[asyncio.Handle]
    _scheduled: list[asyncio.TimerHandle]

    def __init__(self) -> None:
        super().__init__()
        self.now = 1000.0

    def time(self) -> float:
        return self.now

    def _run_once(self) -> None:
        while self._scheduled and self._scheduled[0].cancelled():
            heapq.heappop(self._scheduled)
        if not self._ready and self._scheduled:
            self.now = max(self.now, self._scheduled[0].when())
        super()._run_once()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]  # CPython loop hook lacks stubs


class _BudgetTransaction:
    def __init__(self, database: "_BudgetDatabase", index: int) -> None:
        self.database = database
        self.index = index
        self.configured = False

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if not self.configured:
            self.configured = True
            return ({"acquired": True},)
        if self.index == 0:
            return ({"count": 1},)
        try:
            await asyncio.Event().wait()
        finally:
            self.database.cancelled_queries += (self.index,)
        raise AssertionError("stalled query unexpectedly completed")


class _BudgetDatabase:
    def __init__(self) -> None:
        self.transactions = 0
        self.active = 0
        self.cancelled_queries: tuple[int, ...] = ()
        self.query_rollback_finished = False
        self.lease_rollback_cancelled = False

    @asynccontextmanager
    async def tx(self, *, max_wait: timedelta, timeout: timedelta) -> AsyncGenerator[AnalyticsTransaction, None]:
        transaction: Final = _BudgetTransaction(self, self.transactions)
        self.transactions += 1
        self.active += 1
        try:
            yield transaction
        finally:
            try:
                if transaction.index == 1:
                    await asyncio.sleep(0.5)
                    self.query_rollback_finished = True
                elif transaction.index == 2:
                    try:
                        await asyncio.Event().wait()
                    finally:
                        self.lease_rollback_cancelled = True
            finally:
                self.active -= 1


class _UnexpectedDelivery:
    async def send(self, alert: UsageAlert, rule: UsageAlertRule, channel: UsageAlertChannel) -> bool:
        raise AssertionError("stalled evaluation must not deliver alerts")


def test_job_budget_includes_query_cancellation_and_lease_cleanup() -> None:
    async def scenario() -> None:
        existing: Final = asyncio.all_tasks()
        loop: Final = asyncio.get_running_loop()
        driver: Final = _BudgetDatabase()
        database: Final = BoundedAnalyticsDatabase(driver, background=True)

        async def projection(now: datetime) -> None:
            await asyncio.sleep(57)
            await database.query_raw("SELECT 1")

        job: Final = UsageAlertJob(
            UsageAlertsRepository(database),
            UsageAlertMetricsRepository(database),
            UsageAlertHealthRepository(database),
            _UnexpectedDelivery(),
            defaults=(),
            deprecation_projection=projection,
            work_budget_seconds=60,
        )
        started: Final = loop.time()
        await asyncio.wait_for(job.run(datetime(2026, 10, 21, 2, tzinfo=timezone.utc)), timeout=65)
        elapsed: Final = loop.time() - started
        assert elapsed <= 60, f"Job exceeded its 60-second budget including cleanup: {elapsed:g}s"
        assert driver.cancelled_queries == (1, 2)
        assert driver.query_rollback_finished and driver.lease_rollback_cancelled
        assert driver.active == 0 and asyncio.all_tasks() == existing
        healthy: Final = BoundedAnalyticsDatabase(_BudgetDatabase(), background=True)
        assert await healthy.query_raw("SELECT 1") == ({"count": 1},)

    loop: Final = _JobBudgetClock()
    try:
        loop.run_until_complete(scenario())
    finally:
        loop.close()


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


def test_low_volume_and_recorded_zero_spend_do_not_establish_a_multiplier_baseline() -> None:
    rule: Final = UsageAlertRuleInput(name="spike", scope_type="key", metric="spend", direction="increase")
    history: Final = tuple(_row(_DAY - timedelta(days=days), requests=10, spend=0) for days in (1, 2, 3))
    assert compare_daily_usage(rule, (*history, _row(_DAY, spend=100)), _DAY) is None
    enough_requests: Final = tuple(replace(row, requests=200) for row in history)
    assert compare_daily_usage(rule, (*enough_requests, _row(_DAY, spend=100)), _DAY) is None
    floor: Final = rule.model_copy(update={"min_value": 100})
    assert compare_daily_usage(floor, (*enough_requests, _row(_DAY, spend=100)), _DAY) is None


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
    assert len(defaults) == 25
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
    observations: Final = _observations(rule, (*history, current), _DAY)
    assert not observations.evaluated
    assert observations.changed == ()


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


@pytest.mark.parametrize(
    "active,spend,history_days,evaluated,persisted",
    [(False, 20, 3, True, False), (True, 20, 3, True, True), (False, 100, 3, True, True), (True, 100, 1, False, False)],
)
def test_healthy_evaluation_completes_and_recovers_only_with_sufficient_evidence(
    active: bool, spend: float, history_days: int, evaluated: bool, persisted: bool
) -> None:
    from litellm.proxy.spend_tracking.usage_alert_job import _observations
    from litellm.types.proxy.usage_alerts import UsageAlertRule

    rule: Final = UsageAlertRule(
        id="rule",
        name="spike",
        scope_type="key",
        metric="spend",
        direction="increase",
        created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        created_by=None,
    )
    history: Final = tuple(_row(_DAY - timedelta(days=day)) for day in range(1, history_days + 1))
    observations: Final = _observations(
        rule,
        (*history, _row(_DAY, spend=spend)),
        _DAY,
        active=frozenset({"key-a"}) if active else frozenset(),
    )
    assert observations.evaluated is evaluated
    assert bool(observations.changed) is persisted
    if observations.changed:
        assert observations.changed[0].breached is (spend >= 60)


@pytest.mark.parametrize("multiple", [10, 50])
def test_expensive_low_request_key_triggers_the_default_spend_issue(multiple: int) -> None:
    rule: Final = next(rule for rule in default_usage_rules() if rule.scope_type == "key" and rule.metric == "spend")
    history: Final = tuple(_row(_DAY - timedelta(days=day), requests=3, spend=10, timed=3) for day in (1, 2, 3))
    current: Final = _row(_DAY, requests=3, spend=10 * multiple, timed=3)
    assert compare_daily_usage(rule, (*history, current), _DAY) == DailyComparison(3, 10 * multiple, 10, 30, True)


def test_proportional_failure_growth_does_not_become_a_reliability_issue() -> None:
    rule: Final = next(
        rule for rule in default_usage_rules() if rule.scope_type == "key" and rule.metric == "error_rate"
    )
    history: Final = tuple(_row(_DAY - timedelta(days=day), requests=500) for day in (1, 2, 3))
    current: Final = replace(_row(_DAY, requests=5000), errors=100)
    assert compare_daily_usage(rule, (*history, current), _DAY) == DailyComparison(3, 2, 2, 6, False)


@pytest.mark.parametrize("errors,breached", [(15, False), (25, True)])
def test_error_rate_default_requires_two_percentage_points_as_well_as_the_multiplier(
    errors: int, breached: bool
) -> None:
    rule: Final = next(
        rule for rule in default_usage_rules() if rule.scope_type == "key" and rule.metric == "error_rate"
    )
    history: Final = tuple(replace(_row(_DAY - timedelta(days=day), requests=1000), errors=5) for day in (1, 2, 3))
    current: Final = replace(_row(_DAY, requests=1000), errors=errors)
    assert compare_daily_usage(rule, (*history, current), _DAY) == DailyComparison(3, errors / 10, 0.5, 2.5, breached)


def test_no_requests_cannot_report_zero_error_rate_or_resolve_a_reliability_issue() -> None:
    rule: Final = UsageAlertRuleInput(name="rate", scope_type="key", metric="error_rate", direction="decrease")
    history: Final = tuple(_row(_DAY - timedelta(days=day), requests=500) for day in (1, 2, 3))
    assert compare_daily_usage(rule, history, _DAY) is None


def test_traffic_drop_needs_a_full_recorded_week_and_an_eighty_percent_decline() -> None:
    rule: Final = next(
        rule
        for rule in default_usage_rules()
        if (rule.scope_type, rule.metric, rule.direction) == ("key", "requests", "decrease")
    )
    history: Final = tuple(_row(_DAY - timedelta(days=day), requests=500) for day in range(1, 8))
    assert compare_daily_usage(rule, history[:-1], _DAY) is None
    assert compare_daily_usage(rule, (*history, _row(_DAY, requests=101)), _DAY) == DailyComparison(
        7, 101, 500, 100, False
    )
    assert compare_daily_usage(rule, (*history, _row(_DAY, requests=100)), _DAY) == DailyComparison(
        7, 100, 500, 100, True
    )


def test_latency_default_requires_a_material_change_and_enough_timed_requests() -> None:
    rule: Final = next(rule for rule in default_usage_rules() if rule.scope_type == "key" and rule.metric == "latency")
    history: Final = tuple(_row(_DAY - timedelta(days=day), requests=500, ms=10_000, timed=100) for day in (1, 2, 3))
    assert compare_daily_usage(rule, (*history, _row(_DAY, ms=40_000, timed=100)), _DAY) == DailyComparison(
        3, 400, 100, 600, False
    )
    assert compare_daily_usage(rule, (*history, _row(_DAY, ms=60_000, timed=100)), _DAY) == DailyComparison(
        3, 600, 100, 600, True
    )
    assert compare_daily_usage(rule, (*history, _row(_DAY, ms=60_000, timed=99)), _DAY) is None


@pytest.mark.parametrize("metric", ["spend", "requests", "latency", "errors"])
def test_multiplier_alerts_need_a_positive_established_baseline(metric: str) -> None:
    rule: Final = UsageAlertRuleInput.model_validate(
        {"name": "baseline", "scope_type": "key", "metric": metric, "direction": "increase", "min_requests": 0}
    )
    current: Final = replace(_row(_DAY, requests=2000, spend=500), errors=100)
    assert compare_daily_usage(rule, (current,), _DAY) is None
    empty: Final = tuple(
        replace(_row(_DAY - timedelta(days=day), requests=0, spend=0, ms=0, timed=0), errors=0) for day in (1, 2, 3)
    )
    assert compare_daily_usage(rule, (*empty, current), _DAY) is None


def test_established_healthy_traffic_can_alert_on_new_failures_but_a_new_key_cannot() -> None:
    rule: Final = next(
        rule for rule in default_usage_rules() if rule.scope_type == "key" and rule.metric == "error_rate"
    )
    history: Final = tuple(replace(_row(_DAY - timedelta(days=day), requests=500), errors=0) for day in (1, 2, 3))
    current: Final = replace(_row(_DAY, requests=500), errors=10)
    assert compare_daily_usage(rule, (current,), _DAY) is None
    assert compare_daily_usage(rule, (*history[:2], current), _DAY) is None
    assert compare_daily_usage(rule, (*history, current), _DAY) == DailyComparison(3, 2, 0, 2, True)
    relaxed: Final = rule.model_copy(update={"min_requests": 0, "min_history_days": 1})
    assert compare_daily_usage(relaxed, (*history[:2], current), _DAY) is None
    assert compare_daily_usage(relaxed, (*history, replace(current, requests=499)), _DAY) is None


def test_default_traffic_drop_does_not_treat_a_normal_quiet_weekend_as_an_outage() -> None:
    definition: Final = next(
        rule
        for rule in default_usage_rules()
        if (rule.scope_type, rule.metric, rule.direction) == ("key", "requests", "decrease")
    )
    rule: Final = UsageAlertRule.model_validate(
        {
            **definition.model_dump(),
            "id": default_usage_rule_id("key", "requests", "decrease"),
            "created_at": "2026-10-01T00:00:00Z",
            "updated_at": "2026-10-01T00:00:00Z",
            "created_by": None,
        }
    )
    busy: Final = tuple(_row(_DAY - timedelta(days=day), requests=1000) for day in range(1, 7))
    history: Final = (*busy, _row(_DAY - timedelta(days=7), requests=1))
    current: Final = _row(_DAY, requests=1)
    assert compare_daily_usage(rule, (*history, current), _DAY) is None
    custom: Final = rule.model_copy(update={"id": "custom-policy"})
    custom_result: Final = compare_daily_usage(custom, (*history, current), _DAY)
    assert custom_result is not None and custom_result.breached


def test_error_rate_drop_requires_enough_current_requests() -> None:
    rule: Final = UsageAlertRuleInput(
        name="Reliability",
        scope_type="key",
        metric="error_rate",
        direction="decrease",
        min_requests=500,
    )
    history: Final = tuple(replace(_row(_DAY - timedelta(days=day), requests=1000), errors=100) for day in range(1, 8))
    assert compare_daily_usage(rule, (*history, replace(_row(_DAY, requests=1), errors=0)), _DAY) is None
    result: Final = compare_daily_usage(rule, (*history, replace(_row(_DAY, requests=500), errors=0)), _DAY)
    assert result is not None and result.breached


def test_suppressed_default_can_recover_existing_episode_without_creating_a_new_one() -> None:
    from litellm.proxy.spend_tracking.usage_alert_job import _observations

    rule: Final = UsageAlertRule(
        id="rule",
        name="Default",
        scope_type="key",
        metric="spend",
        direction="increase",
        created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        created_by=None,
    )
    history: Final = tuple(_row(_DAY - timedelta(days=day)) for day in range(1, 4))
    suppressed: Final = frozenset({"key-a"})
    active: Final = frozenset({"key-a"})
    recovered: Final = _observations(rule, (*history, _row(_DAY)), _DAY, suppressed, active=active)
    assert len(recovered.changed) == 1 and not recovered.changed[0].breached
    assert not _observations(rule, (*history, _row(_DAY, spend=100)), _DAY, suppressed, active=active).changed
    assert not _observations(rule, (*history, _row(_DAY, spend=100)), _DAY, suppressed).changed
