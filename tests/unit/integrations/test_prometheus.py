import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Final

import pytest
from prometheus_client import REGISTRY
from typing_extensions import Self

import litellm
import litellm.integrations.prometheus as prometheus_module
from litellm.integrations.prometheus import PrometheusLogger
from litellm.models.team import LiteLLM_TeamTable, LiteLLM_TeamTableCachedObj
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec

_HOURS_TO_RESET: Final = 10.0
_HOURS_SINCE_EXPIRY: Final = 3.0
# One fixed UTC instant for every test, so the remaining hours are exact rather than approximately right
_NOW_UTC: Final = datetime(2026, 10, 20, 12, 0, 0, 250000, tzinfo=timezone.utc)
_RESET_AT_UTC: Final = _NOW_UTC + timedelta(hours=_HOURS_TO_RESET)
# What Postgres to_jsonb renders for a timestamp(3) column: ISO 8601, no offset, trailing zeros dropped
_RESET_AT_TO_JSONB: Final = "2026-10-20T22:00:00.25"

_PLUS_0530: Final = timezone(timedelta(hours=5, minutes=30))
_MINUS_0700: Final = timezone(timedelta(hours=-7))

SetGauge = Callable[[PrometheusLogger, datetime], None]


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> Self:
        return cls.fromtimestamp(_NOW_UTC.timestamp(), tz)


def _unregister_all_collectors() -> None:
    for collector in tuple(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)


@pytest.fixture(autouse=True)
def fresh_registry() -> Iterator[None]:
    _unregister_all_collectors()
    yield
    _unregister_all_collectors()


@pytest.fixture(autouse=True)
def frozen_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prometheus_module, "datetime", _FrozenDatetime)


@pytest.fixture(params=("UTC", "Asia/Kolkata", "America/Los_Angeles"))
def host_timezone(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Run under a zero, a positive and a negative UTC offset; a naive reset time read as local shifts by it."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is POSIX only")
    zone: Final[str] = request.param
    monkeypatch.setenv("TZ", zone)
    time.tzset()
    yield zone
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


def _same_instant_in_each_shape(reset_at_utc: datetime) -> tuple[datetime, ...]:
    """The naive UTC wall time the prefetch cache holds, then the instant carrying UTC, +05:30 and -07:00 offsets."""
    return (
        reset_at_utc.replace(tzinfo=None),
        reset_at_utc,
        reset_at_utc.astimezone(_PLUS_0530),
        reset_at_utc.astimezone(_MINUS_0700),
    )


def _sample_values(name: str) -> Iterator[float]:
    for metric in REGISTRY.collect():
        yield from (sample.value for sample in metric.samples if sample.name == name)


def _only_sample(name: str) -> float:
    values: Final = tuple(_sample_values(name))
    assert len(values) == 1, values
    return values[0]


def _gauge_after(logger: PrometheusLogger, set_gauge: SetGauge, metric_name: str, reset_at: datetime) -> float:
    set_gauge(logger, reset_at)
    return _only_sample(metric_name)


@pytest.mark.parametrize(("set_gauge", "metric_name"), _REMAINING_HOURS_GAUGES)
@pytest.mark.usefixtures("host_timezone")
def test_naive_budget_reset_at_is_read_as_utc(
    prometheus_logger: PrometheusLogger, set_gauge: SetGauge, metric_name: str
) -> None:
    set_gauge(prometheus_logger, _RESET_AT_UTC.replace(tzinfo=None))

    assert _only_sample(metric_name) == _HOURS_TO_RESET


@pytest.mark.parametrize(("set_gauge", "metric_name"), _REMAINING_HOURS_GAUGES)
@pytest.mark.usefixtures("host_timezone")
def test_aware_budget_reset_at_in_another_zone_keeps_its_offset(
    prometheus_logger: PrometheusLogger, set_gauge: SetGauge, metric_name: str
) -> None:
    set_gauge(prometheus_logger, _RESET_AT_UTC.astimezone(_MINUS_0700))

    assert _only_sample(metric_name) == _HOURS_TO_RESET


@pytest.mark.parametrize(("set_gauge", "metric_name"), _REMAINING_HOURS_GAUGES)
@pytest.mark.usefixtures("host_timezone")
def test_naive_and_aware_shapes_of_one_instant_give_identical_hours(
    prometheus_logger: PrometheusLogger, set_gauge: SetGauge, metric_name: str
) -> None:
    hours: Final = tuple(
        _gauge_after(prometheus_logger, set_gauge, metric_name, reset_at)
        for reset_at in _same_instant_in_each_shape(_RESET_AT_UTC)
    )

    assert hours == (_HOURS_TO_RESET,) * len(hours)


@pytest.mark.parametrize(("set_gauge", "metric_name"), _REMAINING_HOURS_GAUGES)
@pytest.mark.usefixtures("host_timezone")
def test_expired_budget_reports_negative_hours_unclamped(
    prometheus_logger: PrometheusLogger, set_gauge: SetGauge, metric_name: str
) -> None:
    expired_at_utc: Final = _NOW_UTC - timedelta(hours=_HOURS_SINCE_EXPIRY)

    hours: Final = tuple(
        _gauge_after(prometheus_logger, set_gauge, metric_name, reset_at)
        for reset_at in _same_instant_in_each_shape(expired_at_utc)
    )

    assert hours == (-_HOURS_SINCE_EXPIRY,) * len(hours)


@pytest.mark.usefixtures("host_timezone")
def test_cold_prefetch_rows_and_warm_lookups_report_the_same_hours(prometheus_logger: PrometheusLogger) -> None:
    """A cold read caches the to_jsonb row, which parses naive; a warm Prisma lookup carries the offset."""
    cold_team: Final = CacheCodec.deserialize(
        CacheCodec.serialize(
            {"team_id": "team", "max_budget": 10.0, "budget_reset_at": _RESET_AT_TO_JSONB},
            model_type=LiteLLM_TeamTableCachedObj,
        ),
        LiteLLM_TeamTableCachedObj,
    )
    cold_user: Final = CacheCodec.deserialize(
        CacheCodec.serialize(
            {"user_id": "user", "max_budget": 10.0, "budget_reset_at": _RESET_AT_TO_JSONB},
            model_type=LiteLLM_UserTable,
        ),
        LiteLLM_UserTable,
    )
    assert cold_team is not None and cold_user is not None
    assert cold_team.budget_reset_at is not None and cold_team.budget_reset_at.tzinfo is None
    assert cold_user.budget_reset_at is not None and cold_user.budget_reset_at.tzinfo is None

    prometheus_logger._set_team_budget_metrics(cold_team)
    prometheus_logger._set_user_budget_metrics(cold_user)
    cold_hours: Final = (
        _only_sample("litellm_team_budget_remaining_hours_metric"),
        _only_sample("litellm_user_budget_remaining_hours_metric"),
    )
    _set_team(prometheus_logger, _RESET_AT_UTC)
    _set_user(prometheus_logger, _RESET_AT_UTC)
    warm_hours: Final = (
        _only_sample("litellm_team_budget_remaining_hours_metric"),
        _only_sample("litellm_user_budget_remaining_hours_metric"),
    )

    assert cold_hours == warm_hours == (_HOURS_TO_RESET, _HOURS_TO_RESET)
