"""
Unit tests for litellm_total_overhead_latency_metric.

The metric reports total internal latency LiteLLM adds around the provider
call = SDK overhead (litellm_overhead_time_ms) + pre/post-call guardrail
durations. During-call (moderation) guardrails run concurrently with the LLM
call and are excluded so they don't inflate the overhead.
"""

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import StandardLoggingPayload


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    """Clean up prometheus registry before/after each test."""
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)
    yield
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)


def test_get_guardrail_overhead_seconds_sums_pre_post_excludes_during():
    """Helper sums pre/post durations, excludes during_call, tolerates missing values."""
    payload = StandardLoggingPayload(
        guardrail_information=[
            {"guardrail_mode": GuardrailEventHooks.pre_call, "duration": 0.1},
            {"guardrail_mode": GuardrailEventHooks.during_call, "duration": 0.5},
            {"guardrail_mode": GuardrailEventHooks.post_call, "duration": 0.25},
            {"guardrail_mode": GuardrailEventHooks.post_call},  # no duration -> 0
        ],
    )
    # 0.1 (pre) + 0.25 (post) = 0.35; during_call 0.5 excluded; missing duration -> 0
    assert abs(PrometheusLogger._get_guardrail_overhead_seconds(payload) - 0.35) < 1e-6


def test_get_guardrail_overhead_seconds_no_guardrails_is_zero():
    """No guardrail_information at all -> 0.0."""
    assert (
        PrometheusLogger._get_guardrail_overhead_seconds(
            StandardLoggingPayload(model="gpt-4o")
        )
        == 0.0
    )


def test_get_guardrail_overhead_seconds_accepts_plain_string_mode():
    """guardrail_mode may arrive as a plain string after serialization."""
    payload = StandardLoggingPayload(
        guardrail_information=[
            {"guardrail_mode": "pre_call", "duration": 0.2},
            {"guardrail_mode": "during_call", "duration": 0.9},
        ],
    )
    # only pre_call counts; during_call excluded
    assert abs(PrometheusLogger._get_guardrail_overhead_seconds(payload) - 0.2) < 1e-6


def test_total_overhead_metric_is_registered():
    """The total-overhead histogram is defined and registered on logger init."""
    logger = PrometheusLogger()
    assert logger.litellm_total_overhead_latency_metric is not None

    registered = [
        name
        for name in REGISTRY._names_to_collectors
        if name.startswith("litellm_total_overhead_latency_metric")
    ]
    assert registered, "litellm_total_overhead_latency_metric not registered"
