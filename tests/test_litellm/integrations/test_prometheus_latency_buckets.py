"""
Tests that prometheus_latency_buckets is respected end-to-end:
- histograms are registered with the custom boundaries
- observed values land in the expected bucket
"""

from datetime import datetime, timedelta

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues


@pytest.fixture(autouse=True)
def reset_prometheus_registry():
    collectors = list(REGISTRY._collector_to_names.keys())
    for c in collectors:
        REGISTRY.unregister(c)
    yield
    collectors = list(REGISTRY._collector_to_names.keys())
    for c in collectors:
        REGISTRY.unregister(c)


@pytest.fixture(autouse=True)
def reset_litellm_buckets():
    original = litellm.prometheus_latency_buckets
    yield
    litellm.prometheus_latency_buckets = original


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


def test_custom_latency_buckets_registered():
    """Histograms use prometheus_latency_buckets when set."""
    litellm.prometheus_latency_buckets = (1.0, 5.0, float("inf"))

    logger = PrometheusLogger()

    assert logger._get_latency_buckets() == (1.0, 5.0, float("inf"))
    assert logger.litellm_request_total_latency_metric._upper_bounds == [
        1.0,
        5.0,
        float("inf"),
    ]
    assert logger.litellm_llm_api_latency_metric._upper_bounds == [
        1.0,
        5.0,
        float("inf"),
    ]


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

    assert logger._get_latency_buckets() == LATENCY_BUCKETS
    assert logger.litellm_request_total_latency_metric._upper_bounds == list(
        LATENCY_BUCKETS
    )
