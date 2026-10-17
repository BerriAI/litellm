import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import UserAPIKeyAuth

_HOURS_TO_RESET: Final = 10.0
_ONE_MINUTE_IN_HOURS: Final = 1 / 60

SetGauge = Callable[[PrometheusLogger, datetime], None]


def _unregister_all_collectors() -> None:
    for collector in tuple(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)


@pytest.fixture(autouse=True)
def fresh_registry() -> Iterator[None]:
    _unregister_all_collectors()
    yield
    _unregister_all_collectors()


@pytest.fixture
def local_clock_ahead_of_utc(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is POSIX only")
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.fixture
def prometheus_logger(monkeypatch: pytest.MonkeyPatch) -> PrometheusLogger:
    monkeypatch.setattr(litellm, "enable_end_user_cost_tracking_prometheus_only", True)
    return PrometheusLogger()


def _set_key(logger: PrometheusLogger, reset_at: datetime) -> None:
    logger._set_key_budget_metrics(UserAPIKeyAuth(token="hashed-key", max_budget=10.0, budget_reset_at=reset_at))


def _set_team(logger: PrometheusLogger, reset_at: datetime) -> None:
    logger._set_team_budget_metrics(LiteLLM_TeamTable(team_id="team", max_budget=10.0, budget_reset_at=reset_at))


def _set_user(logger: PrometheusLogger, reset_at: datetime) -> None:
    logger._set_user_budget_metrics(LiteLLM_UserTable(user_id="user", max_budget=10.0, budget_reset_at=reset_at))


def _set_org(logger: PrometheusLogger, reset_at: datetime) -> None:
    logger._set_org_budget_metrics(org_id="org", org_alias="", spend=0.0, max_budget=10.0, budget_reset_at=reset_at)


def _set_customer(logger: PrometheusLogger, reset_at: datetime) -> None:
    logger._set_customer_budget_metrics(end_user_id="customer", spend=0.0, max_budget=10.0, budget_reset_at=reset_at)


_REMAINING_HOURS_GAUGES: Final = (
    pytest.param(_set_key, "litellm_api_key_budget_remaining_hours_metric", id="key"),
    pytest.param(_set_team, "litellm_team_budget_remaining_hours_metric", id="team"),
    pytest.param(_set_user, "litellm_user_budget_remaining_hours_metric", id="user"),
    pytest.param(_set_org, "litellm_org_budget_remaining_hours_metric", id="org"),
    pytest.param(_set_customer, "litellm_customer_budget_remaining_hours_metric", id="customer"),
)


def _sample_values(name: str) -> Iterator[float]:
    for metric in REGISTRY.collect():
        yield from (sample.value for sample in metric.samples if sample.name == name)


def _only_sample(name: str) -> float:
    values: Final = tuple(_sample_values(name))
    assert len(values) == 1, values
    return values[0]


@pytest.mark.parametrize(("set_gauge", "metric_name"), _REMAINING_HOURS_GAUGES)
@pytest.mark.usefixtures("local_clock_ahead_of_utc")
def test_naive_budget_reset_at_is_read_as_utc(
    prometheus_logger: PrometheusLogger, set_gauge: SetGauge, metric_name: str
) -> None:
    reset_at_utc: Final = datetime.now(timezone.utc) + timedelta(hours=_HOURS_TO_RESET)

    set_gauge(prometheus_logger, reset_at_utc.replace(tzinfo=None))

    assert _only_sample(metric_name) == pytest.approx(_HOURS_TO_RESET, abs=_ONE_MINUTE_IN_HOURS)


@pytest.mark.parametrize(("set_gauge", "metric_name"), _REMAINING_HOURS_GAUGES)
@pytest.mark.usefixtures("local_clock_ahead_of_utc")
def test_aware_budget_reset_at_in_another_zone_keeps_its_offset(
    prometheus_logger: PrometheusLogger, set_gauge: SetGauge, metric_name: str
) -> None:
    pacific: Final = timezone(timedelta(hours=-7))
    reset_at: Final = (datetime.now(timezone.utc) + timedelta(hours=_HOURS_TO_RESET)).astimezone(pacific)

    set_gauge(prometheus_logger, reset_at)

    assert _only_sample(metric_name) == pytest.approx(_HOURS_TO_RESET, abs=_ONE_MINUTE_IN_HOURS)
