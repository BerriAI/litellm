"""
Tests that prometheus_latency_buckets, prometheus_exclude_metrics, and
prometheus_exclude_labels are respected end-to-end:
- histograms are registered with the custom boundaries
- observed values land in the expected bucket
- excluded metrics are replaced by NoOpMetric
- excluded labels are stripped from all metrics
"""

from datetime import datetime, timedelta

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import NoOpMetric, PrometheusLogger
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues


@pytest.fixture(autouse=True)
def reset_prometheus_registry():
    """Unregister only collectors added during the test, avoiding private-API churn."""
    before = set(REGISTRY._names_to_collectors.keys())
    yield
    after = set(REGISTRY._names_to_collectors.keys())
    for name in after - before:
        try:
            REGISTRY.unregister(REGISTRY._names_to_collectors[name])
        except Exception:
            pass


@pytest.fixture(autouse=True)
def reset_litellm_prometheus_settings():
    original_buckets = litellm.prometheus_latency_buckets
    original_exclude_metrics = litellm.prometheus_exclude_metrics
    original_exclude_labels = litellm.prometheus_exclude_labels
    yield
    litellm.prometheus_latency_buckets = original_buckets
    litellm.prometheus_exclude_metrics = original_exclude_metrics
    litellm.prometheus_exclude_labels = original_exclude_labels


def _make_enum_values() -> UserAPIKeyLabelValues:
    return UserAPIKeyLabelValues(
        end_user=None,
        hashed_api_key="test-key",
        api_key_alias="test-alias",
        requested_model="gpt-4",
        model_group="gpt-4",
        team=None,
        team_alias=None,
        user=None,
        user_email=None,
        status_code="200",
        model="gpt-4",
        litellm_model_name="gpt-4",
        tags=[],
        model_id="model-id-1",
        api_base="https://api.openai.com",
        api_provider="openai",
        exception_status=None,
        exception_class=None,
        custom_metadata_labels={},
        route=None,
    )


def _get_bucket_count(metric_name: str, le: str) -> float:
    """Return the bucket count for a specific upper bound from the live registry."""
    for metric_family in REGISTRY.collect():
        if metric_family.name == metric_name:
            for sample in metric_family.samples:
                if (
                    sample.name == f"{metric_name}_bucket"
                    and sample.labels.get("le") == le
                ):
                    return sample.value
    raise AssertionError(f"Bucket le={le} not found in metric {metric_name}")


def _get_registered_bucket_les(metric_name: str) -> set:
    """Return the set of 'le' label values registered for a histogram metric."""
    return {
        sample.labels["le"]
        for family in REGISTRY.collect()
        if family.name == metric_name
        for sample in family.samples
        if sample.name.endswith("_bucket")
    }


def _observe_once(logger: PrometheusLogger) -> None:
    """Make one zero-latency observation so that labeled histograms emit samples.

    prometheus_client only populates the registry for labeled metric instances
    that have received at least one ``observe()`` call.  Without this, REGISTRY.collect()
    returns an empty sample list for any labeled Histogram, making bucket-boundary
    inspection via the registry impossible.
    """
    now = datetime.now()
    logger._set_latency_metrics(
        kwargs={
            "start_time": now,
            "end_time": now,
            "api_call_start_time": now,
            "litellm_params": {"metadata": {}},
            "model": "gpt-4",
        },
        model="gpt-4",
        user_api_key="test-key",
        user_api_key_alias="test-alias",
        user_api_team=None,
        user_api_team_alias=None,
        enum_values=_make_enum_values(),
    )


def test_custom_latency_buckets_registered():
    """Histograms use prometheus_latency_buckets when set."""
    litellm.prometheus_latency_buckets = (1.0, 5.0, float("inf"))

    logger = PrometheusLogger()

    assert list(logger._get_latency_buckets()) == [1.0, 5.0, float("inf")]

    # prometheus_client only emits samples for labeled metrics that have been observed;
    # trigger a zero-latency observation so the registry is populated before we inspect it.
    _observe_once(logger)

    # Verify via the public registry text format, not private _upper_bounds
    les = _get_registered_bucket_les("litellm_request_total_latency_metric")
    assert les == {"1.0", "5.0", "+Inf"}

    les_api = _get_registered_bucket_les("litellm_llm_api_latency_metric")
    assert les_api == {"1.0", "5.0", "+Inf"}


def test_observation_lands_in_correct_custom_bucket():
    """
    With buckets [1.0, 5.0, inf], a 2-second observation must:
    - NOT be counted in the le=1.0 bucket
    - BE counted in the le=5.0 bucket
    - BE counted in the le=+Inf bucket
    """
    litellm.prometheus_latency_buckets = (1.0, 5.0, float("inf"))

    logger = PrometheusLogger()
    enum_values = _make_enum_values()

    now = datetime.now()
    kwargs = {
        "start_time": now - timedelta(seconds=2),
        "end_time": now,
        "api_call_start_time": now - timedelta(seconds=2),
        "litellm_params": {"metadata": {}},
        "model": "gpt-4",
    }

    logger._set_latency_metrics(
        kwargs=kwargs,
        model="gpt-4",
        user_api_key="test-key",
        user_api_key_alias="test-alias",
        user_api_team=None,
        user_api_team_alias=None,
        enum_values=enum_values,
    )

    # le=1.0 bucket must be 0 (2s > 1.0)
    assert _get_bucket_count("litellm_request_total_latency_metric", "1.0") == 0.0

    # le=5.0 bucket must be 1 (2s <= 5.0)
    assert _get_bucket_count("litellm_request_total_latency_metric", "5.0") == 1.0

    # le=+Inf must also be 1
    assert _get_bucket_count("litellm_request_total_latency_metric", "+Inf") == 1.0


def test_default_buckets_used_when_not_set():
    """When prometheus_latency_buckets is None, the module-level LATENCY_BUCKETS constant is used."""
    from litellm.types.integrations.prometheus import LATENCY_BUCKETS

    litellm.prometheus_latency_buckets = None

    logger = PrometheusLogger()

    assert list(logger._get_latency_buckets()) == list(LATENCY_BUCKETS)

    # Trigger an observation so the labeled histogram emits samples to the registry.
    _observe_once(logger)

    # Verify via the registry
    les = _get_registered_bucket_les("litellm_request_total_latency_metric")
    expected = {str(b) if b != float("inf") else "+Inf" for b in LATENCY_BUCKETS}
    assert les == expected


def test_exclude_metrics_replaces_with_noop():
    """Excluded metrics are replaced by NoOpMetric and never registered in Prometheus."""
    litellm.prometheus_exclude_metrics = ["litellm_overhead_latency_metric"]

    logger = PrometheusLogger()

    assert isinstance(logger.litellm_overhead_latency_metric, NoOpMetric)

    # The metric must not appear in the registry at all
    registered_names = {f.name for f in REGISTRY.collect()}
    assert "litellm_overhead_latency_metric" not in registered_names

    # Other metrics are real
    assert not isinstance(logger.litellm_request_total_latency_metric, NoOpMetric)


def test_exclude_labels_strips_label_from_metrics():
    """Excluded labels are absent from the label set of all affected metrics."""
    litellm.prometheus_exclude_labels = ["end_user"]

    logger = PrometheusLogger()

    # Verify via the registry: no sample for litellm_request_total_latency_metric
    # should carry the end_user label key
    for family in REGISTRY.collect():
        if family.name == "litellm_request_total_latency_metric":
            for sample in family.samples:
                assert (
                    "end_user" not in sample.labels
                ), f"end_user label found in sample {sample}"
