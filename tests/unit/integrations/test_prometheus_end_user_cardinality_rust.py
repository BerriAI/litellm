from datetime import datetime, timedelta, timezone
from time import monotonic, time

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.integrations.prometheus_helpers import native_store
from tests.unit.integrations.prometheus_native_support import (
    enable_native_prometheus,
    labelsets,
    native_success,
    standard_payload,
)


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry(monkeypatch):
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    for name in (
        "enable_end_user_cost_tracking_prometheus_only",
        "prometheus_metrics_config",
        "prometheus_end_user_metrics_max_series_per_metric",
        "prometheus_end_user_metrics_ttl_seconds",
        "prometheus_end_user_metrics_cleanup_interval_seconds",
    ):
        monkeypatch.setattr(litellm, name, getattr(litellm, name))

    yield

    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def manual_clock(monkeypatch):
    real_store = native_store._native.NativePrometheusStore

    def build(max_series=None, ttl_seconds=None, cleanup_interval_seconds=None):
        return real_store(max_series, ttl_seconds, cleanup_interval_seconds, manual_clock=True)

    monkeypatch.setattr(native_store._native, "NativePrometheusStore", build)


def _set_clock(logger: PrometheusLogger, monotonic_now: float) -> None:
    store = logger._native_store
    assert store is not None
    store.store.set_time(monotonic_now, time())


async def _spend_for(logger: PrometheusLogger, end_user: str, monkeypatch) -> None:
    now = datetime.now()
    kwargs = {
        "model": "gpt-4",
        "litellm_params": {"metadata": {"user_api_key_end_user_id": end_user}},
        "start_time": now,
        "end_time": now,
        "standard_logging_object": standard_payload(
            metadata={"user_api_key_end_user_id": end_user}, end_user=end_user, response_cost=0.01
        ),
    }
    await native_success(logger, kwargs, None, monkeypatch=monkeypatch)


def _end_users(metric: str) -> set[str]:
    return {labels["end_user"] for labels in labelsets(metric)}


def _end_user_config(monkeypatch) -> None:
    monkeypatch.setattr(litellm, "enable_end_user_cost_tracking_prometheus_only", True)
    monkeypatch.setattr(
        litellm,
        "prometheus_metrics_config",
        [
            {
                "group": "end-user-spend",
                "metrics": ["litellm_spend_metric"],
                "include_labels": ["end_user"],
            }
        ],
    )


@pytest.mark.asyncio
async def test_prometheus_end_user_series_are_capped_per_metric(monkeypatch):
    _end_user_config(monkeypatch)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_max_series_per_metric", 3)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_ttl_seconds", None)
    logger = PrometheusLogger()

    for index in range(6):
        await _spend_for(logger, f"end-user-{index}", monkeypatch)

    assert len(labelsets("litellm_spend_metric_total")) == 3
    assert labelsets("litellm_spend_metric_total") == [
        {"end_user": "end-user-3"},
        {"end_user": "end-user-4"},
        {"end_user": "end-user-5"},
    ]


@pytest.mark.asyncio
async def test_bounded_prometheus_series_tracker_treats_zero_max_as_unlimited(monkeypatch):
    _end_user_config(monkeypatch)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_max_series_per_metric", 0)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_ttl_seconds", None)
    logger = PrometheusLogger()

    for index in range(3):
        await _spend_for(logger, f"end-user-{index}", monkeypatch)

    assert _end_users("litellm_spend_metric_total") == {"end-user-0", "end-user-1", "end-user-2"}


@pytest.mark.asyncio
async def test_prometheus_end_user_series_expire_by_ttl(monkeypatch, manual_clock):
    _end_user_config(monkeypatch)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_max_series_per_metric", None)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_ttl_seconds", 10.0)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_cleanup_interval_seconds", 0.0)
    logger = PrometheusLogger()

    current_time = [monotonic()]
    _set_clock(logger, current_time[0])
    await _spend_for(logger, "stale-end-user", monkeypatch)

    current_time[0] += 11.0
    _set_clock(logger, current_time[0])
    await _spend_for(logger, "fresh-end-user", monkeypatch)

    assert _end_users("litellm_spend_metric_total") == {"fresh-end-user"}


@pytest.mark.asyncio
async def test_prometheus_end_user_not_tracked_by_default(monkeypatch):
    monkeypatch.setattr(litellm, "enable_end_user_cost_tracking_prometheus_only", None)
    logger = PrometheusLogger()
    labels = logger.get_labels_for_metric("litellm_spend_metric")
    assert "end_user" in labels

    await _spend_for(logger, "not-exported", monkeypatch)

    assert _end_users("litellm_spend_metric_total") == {"None"}


def test_prometheus_customer_budget_series_are_capped_per_metric(monkeypatch):
    monkeypatch.setattr(litellm, "enable_end_user_cost_tracking_prometheus_only", True)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_max_series_per_metric", 2)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_ttl_seconds", None)
    logger = PrometheusLogger()

    for index in range(5):
        logger._set_customer_budget_metrics(
            end_user_id=f"customer-{index}",
            spend=1.0,
            max_budget=10.0,
            budget_reset_at=None,
        )

    assert _end_users("litellm_remaining_customer_budget_metric") == {"customer-3", "customer-4"}
    assert _end_users("litellm_customer_max_budget_metric") == {"customer-3", "customer-4"}


def test_prometheus_customer_budget_series_expire_by_ttl(monkeypatch, manual_clock):
    monkeypatch.setattr(litellm, "enable_end_user_cost_tracking_prometheus_only", True)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_max_series_per_metric", None)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_ttl_seconds", 10.0)
    monkeypatch.setattr(litellm, "prometheus_end_user_metrics_cleanup_interval_seconds", 0.0)
    logger = PrometheusLogger()

    current_time = [monotonic()]
    _set_clock(logger, current_time[0])
    logger._set_customer_budget_metrics(
        end_user_id="customer-with-removed-budget",
        spend=1.0,
        max_budget=10.0,
        budget_reset_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    current_time[0] += 11.0
    _set_clock(logger, current_time[0])
    logger._set_customer_budget_metrics(
        end_user_id="customer-still-budgeted",
        spend=1.0,
        max_budget=10.0,
        budget_reset_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    assert _end_users("litellm_remaining_customer_budget_metric") == {"customer-still-budgeted"}
    assert _end_users("litellm_customer_max_budget_metric") == {"customer-still-budgeted"}
    assert _end_users("litellm_customer_budget_remaining_hours_metric") == {"customer-still-budgeted"}
