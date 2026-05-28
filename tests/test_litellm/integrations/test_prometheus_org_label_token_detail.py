"""LIT-2825 — verify that the token-type detail metrics carry org_id / org_alias
labels alongside the primary input/output token metrics.

The bug as filed: spend / token / request metrics emit ``org_id`` but not
``org_alias``. The org-label set on the primary token metrics already
includes both via :pyattr:`PrometheusMetricLabels._org_label_metrics`, but
the per-detail metrics (cached / cache-creation / audio input + reasoning /
audio output) were missed. This test pins the regression in two ways:

1. Each detail metric appears in :pyattr:`_org_label_metrics`.
2. The Prometheus counter created from the resulting label list actually
   carries ``org_id`` and ``org_alias`` (so dashboards that group by org
   alias work end-to-end), and the values flow through a real
   ``_inc_labeled_counter`` call.
"""

from unittest.mock import patch

import pytest
from prometheus_client import REGISTRY, generate_latest

from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import (
    PrometheusMetricLabels,
    UserAPIKeyLabelNames,
    UserAPIKeyLabelValues,
)


# Token-type detail metrics added to the org-label set in LIT-2825.
TOKEN_DETAIL_METRICS = (
    "litellm_input_cached_tokens_metric",
    "litellm_input_cache_creation_tokens_metric",
    "litellm_input_audio_tokens_metric",
    "litellm_output_reasoning_tokens_metric",
    "litellm_output_audio_tokens_metric",
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Reset the Prometheus default registry around each test."""
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.mark.parametrize("metric_name", TOKEN_DETAIL_METRICS)
def test_metric_in_org_label_set(metric_name):
    """The five detail metrics are members of _org_label_metrics."""
    assert metric_name in PrometheusMetricLabels._org_label_metrics


@pytest.mark.parametrize("metric_name", TOKEN_DETAIL_METRICS)
def test_get_labels_includes_org_id_and_org_alias(metric_name):
    """get_labels() returns both org_id and org_alias for the detail metrics."""
    labels = PrometheusMetricLabels.get_labels(metric_name)
    assert UserAPIKeyLabelNames.ORG_ID.value in labels
    assert UserAPIKeyLabelNames.ORG_ALIAS.value in labels
    # Defensive: order doesn't matter and there are no duplicates.
    assert labels.count(UserAPIKeyLabelNames.ORG_ID.value) == 1
    assert labels.count(UserAPIKeyLabelNames.ORG_ALIAS.value) == 1


@pytest.mark.parametrize("metric_name", TOKEN_DETAIL_METRICS)
def test_counter_labelnames_include_org(metric_name):
    """The instantiated counter on ``PrometheusLogger`` carries the org labels."""
    logger = PrometheusLogger()
    counter = getattr(logger, metric_name)
    labelnames = set(counter._labelnames)
    assert UserAPIKeyLabelNames.ORG_ID.value in labelnames
    assert UserAPIKeyLabelNames.ORG_ALIAS.value in labelnames


def _build_payload_and_enum_values():
    payload = {
        "id": "req-abc",
        "call_type": "completion",
        "cache_hit": False,
        "saved_cache_cost": 0.0,
        "startTime": 0,
        "endTime": 1,
        "completionStartTime": 0,
        "model": "gpt-4o-mini",
        "model_id": "model-xyz",
        "model_group": "gpt-4o-mini",
        "api_base": "https://api.openai.com",
        "metadata": {
            "user_api_key_hash": "hash1",
            "user_api_key_alias": "alias1",
            "user_api_key_team_id": "team1",
            "user_api_key_team_alias": "team-alias-1",
            "user_api_key_user_id": "user1",
            "user_api_key_user_email": "u@example.com",
            "user_api_key_org_id": "org-1",
            "user_api_key_org_alias": "Org One",
            "user_api_key_request_route": "/chat/completions",
            "requester_ip_address": "10.0.0.1",
            "user_agent": "pytest/1",
            "usage_object": {
                "prompt_tokens": 120,
                "completion_tokens": 80,
                "total_tokens": 200,
                "prompt_tokens_details": {
                    "cached_tokens": 60,
                    "cache_creation_tokens": 20,
                    "audio_tokens": 10,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 25,
                    "audio_tokens": 5,
                },
            },
        },
        "stream": False,
        "completion_tokens": 80,
        "prompt_tokens": 120,
        "total_tokens": 200,
        "cache_key": None,
        "response_cost": 0.0123,
        "request_tags": [],
        "end_user": None,
        "requester_ip_address": "10.0.0.1",
        "status": "success",
        "custom_llm_provider": "openai",
        "hidden_params": {
            "additional_headers": {},
            "model_id": "model-xyz",
            "cache_key": None,
            "api_base": "https://api.openai.com",
            "response_cost": 0.0123,
        },
    }
    enum_values = UserAPIKeyLabelValues(
        end_user=None,
        hashed_api_key="hash1",
        api_key_alias="alias1",
        requested_model="gpt-4o-mini",
        model_group="gpt-4o-mini",
        team="team1",
        team_alias="team-alias-1",
        org_id="org-1",
        org_alias="Org One",
        user="user1",
        user_email="u@example.com",
        status_code="200",
        model="gpt-4o-mini",
        litellm_model_name="gpt-4o-mini",
        tags=[],
        model_id="model-xyz",
        api_base="https://api.openai.com",
        api_provider="openai",
        exception_status=None,
        exception_class=None,
        custom_metadata_labels={},
        route="/chat/completions",
        client_ip="10.0.0.1",
        user_agent="pytest/1",
        stream="false",
    )
    return payload, enum_values


_EXPECTED_VALUES = {
    "litellm_input_cached_tokens_metric": 60.0,
    "litellm_input_cache_creation_tokens_metric": 20.0,
    "litellm_input_audio_tokens_metric": 10.0,
    "litellm_output_reasoning_tokens_metric": 25.0,
    "litellm_output_audio_tokens_metric": 5.0,
}


def test_increment_token_detail_metrics_emits_org_labels():
    """End-to-end: the live increment path stamps org_id / org_alias on each series.

    Drives the same code path that
    ``PrometheusLogger.async_log_success_event`` invokes on every
    completion (see :pymeth:`PrometheusLogger._increment_token_detail_metrics`).
    Without the fix, the resulting series have no ``org_id`` / ``org_alias``
    labels at all — the prometheus_client text exposition format is
    sensitive to label-name changes, so this also doubles as a backstop
    against accidentally renaming the keys.
    """
    logger = PrometheusLogger()
    payload, enum_values = _build_payload_and_enum_values()
    logger._increment_token_detail_metrics(payload, enum_values)

    scrape = generate_latest().decode("utf-8", errors="replace")
    for metric_name, expected_value in _EXPECTED_VALUES.items():
        series_line = next(
            (
                line
                for line in scrape.splitlines()
                if line.startswith(f"{metric_name}_total{{")
            ),
            None,
        )
        assert series_line is not None, (
            f"{metric_name}_total series was not emitted; scrape was:\n{scrape}"
        )
        # Both org labels present with the configured values.
        assert 'org_id="org-1"' in series_line, series_line
        assert 'org_alias="Org One"' in series_line, series_line
        # The value is the per-token-type count, not a count of calls.
        assert series_line.endswith(f" {expected_value}"), series_line


def test_get_labels_excludes_org_when_metric_not_in_set():
    """Sanity: ``get_labels`` does not blanket-add org labels."""
    # A budget metric explicitly scoped to API keys should not get org_*.
    labels = PrometheusMetricLabels.get_labels(
        "litellm_remaining_api_key_budget_metric"
    )
    assert UserAPIKeyLabelNames.ORG_ID.value not in labels
    assert UserAPIKeyLabelNames.ORG_ALIAS.value not in labels
