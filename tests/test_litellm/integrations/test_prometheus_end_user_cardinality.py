from time import monotonic

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.integrations.prometheus_helpers import bounded_prometheus_series_tracker
from litellm.integrations.prometheus_helpers.bounded_prometheus_series_tracker import (
    BoundedPrometheusSeriesTracker,
)
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    old_enable_end_user = litellm.enable_end_user_cost_tracking_prometheus_only
    old_metrics_config = litellm.prometheus_metrics_config
    old_max_series = litellm.prometheus_end_user_metrics_max_series_per_metric
    old_ttl_seconds = litellm.prometheus_end_user_metrics_ttl_seconds
    old_cleanup_interval_seconds = (
        litellm.prometheus_end_user_metrics_cleanup_interval_seconds
    )

    yield

    litellm.enable_end_user_cost_tracking_prometheus_only = old_enable_end_user
    litellm.prometheus_metrics_config = old_metrics_config
    litellm.prometheus_end_user_metrics_max_series_per_metric = old_max_series
    litellm.prometheus_end_user_metrics_ttl_seconds = old_ttl_seconds
    litellm.prometheus_end_user_metrics_cleanup_interval_seconds = (
        old_cleanup_interval_seconds
    )

    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def test_prometheus_end_user_series_are_capped_per_metric():
    litellm.enable_end_user_cost_tracking_prometheus_only = True
    litellm.prometheus_metrics_config = [
        {
            "group": "end-user-spend",
            "metrics": ["litellm_spend_metric"],
            "include_labels": ["end_user"],
        }
    ]
    litellm.prometheus_end_user_metrics_max_series_per_metric = 3
    litellm.prometheus_end_user_metrics_ttl_seconds = None
    logger = PrometheusLogger()

    for index in range(6):
        PrometheusLogger._inc_labeled_counter(
            logger,
            logger.litellm_spend_metric,
            "litellm_spend_metric",
            UserAPIKeyLabelValues(end_user=f"end-user-{index}"),
            amount=0.01,
        )

    assert len(logger.litellm_spend_metric._metrics) == 3
    assert set(logger.litellm_spend_metric._metrics) == {
        ("end-user-3",),
        ("end-user-4",),
        ("end-user-5",),
    }


def test_bounded_prometheus_series_tracker_is_label_agnostic():
    class FakeMetric:
        def __init__(self):
            self.removed_label_values = []

        def remove(self, *label_values):
            self.removed_label_values.append(label_values)

    metric = FakeMetric()
    tracker = BoundedPrometheusSeriesTracker()

    for index in range(4):
        tracker.track_series(
            metric=metric,
            metric_name="generic_metric",
            label_values=(f"route-{index}", "200"),
            max_series=2,
            ttl_seconds=None,
            cleanup_interval_seconds=60.0,
        )

    assert metric.removed_label_values == [
        ("route-0", "200"),
        ("route-1", "200"),
    ]


def test_bounded_prometheus_series_tracker_treats_zero_max_as_unlimited():
    # A misconfigured ``max_series=0`` must not silently evict every emission.
    class FakeMetric:
        def __init__(self):
            self.removed_label_values = []

        def remove(self, *label_values):
            self.removed_label_values.append(label_values)

    metric = FakeMetric()
    tracker = BoundedPrometheusSeriesTracker()

    for index in range(3):
        tracker.track_series(
            metric=metric,
            metric_name="generic_metric",
            label_values=(f"end-user-{index}",),
            max_series=0,
            ttl_seconds=None,
            cleanup_interval_seconds=60.0,
        )

    assert metric.removed_label_values == []


def test_prometheus_end_user_series_expire_by_ttl(monkeypatch):
    litellm.enable_end_user_cost_tracking_prometheus_only = True
    litellm.prometheus_metrics_config = [
        {
            "group": "end-user-spend",
            "metrics": ["litellm_spend_metric"],
            "include_labels": ["end_user"],
        }
    ]
    litellm.prometheus_end_user_metrics_max_series_per_metric = None
    litellm.prometheus_end_user_metrics_ttl_seconds = 10.0
    litellm.prometheus_end_user_metrics_cleanup_interval_seconds = 0.0
    logger = PrometheusLogger()

    current_time = [monotonic()]
    monkeypatch.setattr(
        bounded_prometheus_series_tracker.time,
        "monotonic",
        lambda: current_time[0],
    )
    PrometheusLogger._inc_labeled_counter(
        logger,
        logger.litellm_spend_metric,
        "litellm_spend_metric",
        UserAPIKeyLabelValues(end_user="stale-end-user"),
        amount=0.01,
    )

    current_time[0] += 11.0
    PrometheusLogger._inc_labeled_counter(
        logger,
        logger.litellm_spend_metric,
        "litellm_spend_metric",
        UserAPIKeyLabelValues(end_user="fresh-end-user"),
        amount=0.01,
    )

    assert set(logger.litellm_spend_metric._metrics) == {("fresh-end-user",)}


def test_prometheus_end_user_not_tracked_by_default():
    litellm.enable_end_user_cost_tracking_prometheus_only = None
    labels = PrometheusLogger().get_labels_for_metric("litellm_spend_metric")
    assert "end_user" in labels

    label_values = UserAPIKeyLabelValues(end_user="not-exported")
    from litellm.integrations.prometheus import prometheus_label_factory

    prometheus_labels = prometheus_label_factory(labels, label_values)
    assert prometheus_labels["end_user"] is None
