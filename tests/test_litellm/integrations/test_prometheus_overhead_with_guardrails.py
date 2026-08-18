"""
Unit tests for litellm_overhead_with_guardrails_latency_metric.

The metric reports total internal latency LiteLLM adds around the provider
call = SDK overhead (litellm_overhead_time_ms) + pre/post-call guardrail
durations. During-call (moderation) guardrails run concurrently with the LLM
call and are excluded so they don't inflate the overhead.
"""

from unittest.mock import MagicMock

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


def test_get_guardrail_overhead_seconds_excludes_list_mode_with_during_call():
    """A list-typed guardrail_mode containing during_call must be excluded.

    guardrail_mode is typed Optional[Union[GuardrailEventHooks,
    List[GuardrailEventHooks], GuardrailMode]]; a list mixing in during_call is
    not additive (concurrent) overhead and must not be counted.
    """
    payload = StandardLoggingPayload(
        guardrail_information=[
            {
                "guardrail_mode": [
                    GuardrailEventHooks.pre_call,
                    GuardrailEventHooks.during_call,
                ],
                "duration": 0.3,
            },
            {"guardrail_mode": GuardrailEventHooks.post_call, "duration": 0.05},
        ],
    )
    # the list entry mixes in during_call -> excluded; only the post_call counts
    assert abs(PrometheusLogger._get_guardrail_overhead_seconds(payload) - 0.05) < 1e-6


def test_get_guardrail_overhead_seconds_counts_pure_pre_post_list_mode():
    """A list-typed mode containing only additive (pre/post) phases is counted."""
    payload = StandardLoggingPayload(
        guardrail_information=[
            {"guardrail_mode": [GuardrailEventHooks.pre_call], "duration": 0.1},
            {"guardrail_mode": ["post_call"], "duration": 0.2},
        ],
    )
    assert abs(PrometheusLogger._get_guardrail_overhead_seconds(payload) - 0.3) < 1e-6


def test_get_guardrail_overhead_seconds_excludes_logging_only_and_mcp():
    """logging_only and MCP-specific modes do not block the response -> excluded."""
    payload = StandardLoggingPayload(
        guardrail_information=[
            {"guardrail_mode": GuardrailEventHooks.logging_only, "duration": 0.4},
            {"guardrail_mode": GuardrailEventHooks.pre_mcp_call, "duration": 0.3},
            {"guardrail_mode": GuardrailEventHooks.during_mcp_call, "duration": 0.2},
            {"guardrail_mode": GuardrailEventHooks.post_call, "duration": 0.05},
        ],
    )
    # only the post_call guardrail is additive, user-visible overhead
    assert abs(PrometheusLogger._get_guardrail_overhead_seconds(payload) - 0.05) < 1e-6


def test_get_guardrail_overhead_seconds_ignores_dict_mode_without_error():
    """guardrail_mode may be a GuardrailMode TypedDict (an unhashable dict at
    runtime, from the enterprise Mode-hook path). It must not raise TypeError and
    must not be counted (the phase can't be resolved to a blocking pre/post)."""
    payload = StandardLoggingPayload(
        guardrail_information=[
            # GuardrailMode TypedDict -> plain dict at runtime
            {"guardrail_mode": {"tags": {"default": ["pre_call"]}}, "duration": 0.3},
            {"guardrail_mode": GuardrailEventHooks.post_call, "duration": 0.05},
        ],
    )
    # dict-typed mode is ignored (no TypeError); only the post_call entry counts
    assert abs(PrometheusLogger._get_guardrail_overhead_seconds(payload) - 0.05) < 1e-6


def test_get_guardrail_overhead_seconds_ignores_dict_inside_list_mode():
    """A list-typed mode containing a dict must not raise and the dict is ignored."""
    payload = StandardLoggingPayload(
        guardrail_information=[
            {"guardrail_mode": [GuardrailEventHooks.pre_call, {"k": "v"}], "duration": 0.1},
        ],
    )
    # the dict is ignored; remaining mode is pre_call -> counted
    assert abs(PrometheusLogger._get_guardrail_overhead_seconds(payload) - 0.1) < 1e-6


def test_get_guardrail_overhead_seconds_handles_single_dict_payload():
    """guardrail_information may be a single dict (e.g. xecguard) rather than a
    list. Iterating it would yield string keys and crash the success-metrics
    block, so a lone dict must be evaluated as one entry, not raise."""
    payload = StandardLoggingPayload(
        guardrail_information={
            "guardrail_mode": "logging_only",
            "duration": 0.7,
            "guardrail_name": "xecguard",
        },
    )
    # the single dict is logging_only -> excluded, and must not raise
    assert PrometheusLogger._get_guardrail_overhead_seconds(payload) == 0.0


def test_get_guardrail_overhead_seconds_counts_single_pre_call_dict():
    """A single pre/post-call dict (not wrapped in a list) is still counted."""
    payload = StandardLoggingPayload(
        guardrail_information={"guardrail_mode": "pre_call", "duration": 0.3},
    )
    assert abs(PrometheusLogger._get_guardrail_overhead_seconds(payload) - 0.3) < 1e-6


def _patch_label_factory(monkeypatch):
    monkeypatch.setattr(
        "litellm.integrations.prometheus.prometheus_label_factory",
        lambda **kwargs: {},
    )


def test_overhead_with_guardrails_recorded_when_only_guardrails_no_sdk_overhead(monkeypatch):
    """Guardrail-only overhead is recorded even when SDK overhead is absent."""
    _patch_label_factory(monkeypatch)
    logger = PrometheusLogger()
    mock_metric = MagicMock()
    logger.litellm_overhead_with_guardrails_latency_metric = mock_metric

    payload = StandardLoggingPayload(
        hidden_params={},  # no litellm_overhead_time_ms
        guardrail_information=[
            {"guardrail_mode": GuardrailEventHooks.post_call, "duration": 0.2}
        ],
    )
    logger._set_overhead_with_guardrails_metric(
        payload, enum_values=MagicMock(), label_context=MagicMock()
    )

    mock_metric.labels.return_value.observe.assert_called_once()
    observed = mock_metric.labels.return_value.observe.call_args[0][0]
    assert abs(observed - 0.2) < 1e-6


def test_overhead_with_guardrails_recorded_when_sdk_overhead_is_zero(monkeypatch):
    """SDK overhead of exactly 0 (walrus-falsy) must not suppress the metric."""
    _patch_label_factory(monkeypatch)
    logger = PrometheusLogger()
    mock_metric = MagicMock()
    logger.litellm_overhead_with_guardrails_latency_metric = mock_metric

    payload = StandardLoggingPayload(
        hidden_params={"litellm_overhead_time_ms": 0.0},
        guardrail_information=[
            {"guardrail_mode": GuardrailEventHooks.pre_call, "duration": 0.1}
        ],
    )
    logger._set_overhead_with_guardrails_metric(
        payload, enum_values=MagicMock(), label_context=MagicMock()
    )

    observed = mock_metric.labels.return_value.observe.call_args[0][0]
    assert abs(observed - 0.1) < 1e-6


def test_overhead_with_guardrails_skipped_when_no_overhead_and_no_guardrails(monkeypatch):
    """Nothing to record -> the metric is not touched."""
    _patch_label_factory(monkeypatch)
    logger = PrometheusLogger()
    mock_metric = MagicMock()
    logger.litellm_overhead_with_guardrails_latency_metric = mock_metric

    payload = StandardLoggingPayload(hidden_params={})
    logger._set_overhead_with_guardrails_metric(
        payload, enum_values=MagicMock(), label_context=MagicMock()
    )

    mock_metric.labels.assert_not_called()


def test_overhead_with_guardrails_metric_is_registered():
    """The overhead-with-guardrails histogram is defined and registered on logger init."""
    logger = PrometheusLogger()
    assert logger.litellm_overhead_with_guardrails_latency_metric is not None

    registered = [
        name
        for name in REGISTRY._names_to_collectors
        if name.startswith("litellm_overhead_with_guardrails_latency_metric")
    ]
    assert registered, "litellm_overhead_with_guardrails_latency_metric not registered"
