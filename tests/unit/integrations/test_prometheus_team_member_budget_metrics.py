"""Team member budget gauges: one series per (team, member) with an effective budget."""

from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger

NOW: Final = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
MAX_METRIC: Final = "litellm_team_member_max_budget_metric"
REMAINING_METRIC: Final = "litellm_remaining_team_member_budget_metric"
HOURS_METRIC: Final = "litellm_team_member_budget_remaining_hours_metric"
ALICE_IN_SEARCH: Final = {
    "team": "t-search",
    "team_alias": "search",
    "user": "alice",
    "user_email": "alice@example.com",
}


def _unregister_everything() -> None:
    for collector in tuple(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "prometheus_metrics_config", None)
    _unregister_everything()
    yield
    _unregister_everything()


def _sample(metric: str, labels: Mapping[str, str]) -> float | None:
    return REGISTRY.get_sample_value(metric, dict(labels))


def _set_alice(logger: PrometheusLogger, budget_reset_at: datetime | None) -> None:
    logger._set_team_member_budget_metrics(
        team_id="t-search",
        team_alias="search",
        user_id="alice",
        user_email="alice@example.com",
        max_budget=10.0,
        spend=4.0,
        budget_reset_at=budget_reset_at,
        now=NOW,
    )


def test_setter_emits_three_series_with_team_and_user_labels() -> None:
    logger: Final = PrometheusLogger()

    _set_alice(logger, budget_reset_at=NOW + timedelta(minutes=150))

    assert {
        MAX_METRIC: _sample(MAX_METRIC, ALICE_IN_SEARCH),
        REMAINING_METRIC: _sample(REMAINING_METRIC, ALICE_IN_SEARCH),
        HOURS_METRIC: _sample(HOURS_METRIC, ALICE_IN_SEARCH),
    } == {MAX_METRIC: 10.0, REMAINING_METRIC: 6.0, HOURS_METRIC: pytest.approx(2.5)}


def test_member_without_reset_time_gets_no_hours_series() -> None:
    logger: Final = PrometheusLogger()

    _set_alice(logger, budget_reset_at=None)

    assert _sample(MAX_METRIC, ALICE_IN_SEARCH) == 10.0
    assert _sample(REMAINING_METRIC, ALICE_IN_SEARCH) == 6.0
    assert _sample(HOURS_METRIC, ALICE_IN_SEARCH) is None


def test_include_labels_trims_member_series_to_the_configured_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "prometheus_metrics_config",
        [
            {"group": "member-max", "metrics": [MAX_METRIC], "include_labels": ["team", "user"]},
            {"group": "member-remaining", "metrics": [REMAINING_METRIC]},
            {"group": "member-hours", "metrics": [HOURS_METRIC], "include_labels": ["user"]},
        ],
    )
    logger: Final = PrometheusLogger()

    _set_alice(logger, budget_reset_at=NOW + timedelta(hours=1))

    assert _sample(MAX_METRIC, {"team": "t-search", "user": "alice"}) == 10.0
    assert _sample(REMAINING_METRIC, ALICE_IN_SEARCH) == 6.0
    assert _sample(HOURS_METRIC, {"user": "alice"}) == pytest.approx(1.0)
