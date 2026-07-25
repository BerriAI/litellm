import sys
from typing import get_args

import pytest

sys.path.insert(0, "../../../..")

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import (
    DEFINED_PROMETHEUS_METRICS,
    PROMETHEUS_METRICS_WILDCARD,
    PrometheusMetricsConfig,
)


def _bare_logger() -> PrometheusLogger:
    """A PrometheusLogger instance with __init__ skipped, for testing the
    config-parsing helpers in isolation without re-registering ~70 real
    metrics against prometheus_client's global registry on every test."""
    return PrometheusLogger.__new__(PrometheusLogger)


def test_wildcard_sets_default_labels_for_every_metric():
    logger = _bare_logger()
    configs = [
        PrometheusMetricsConfig(
            group="defaults",
            metrics=[PROMETHEUS_METRICS_WILDCARD],
            include_labels=["hashed_api_key", "team"],
        )
    ]

    label_filters = logger._build_label_filters(configs)

    assert set(label_filters.keys()) == set(get_args(DEFINED_PROMETHEUS_METRICS))
    for labels in label_filters.values():
        assert labels == ["hashed_api_key", "team"]


def test_named_group_overrides_wildcard_for_that_metric():
    logger = _bare_logger()
    configs = [
        PrometheusMetricsConfig(
            group="defaults",
            metrics=[PROMETHEUS_METRICS_WILDCARD],
            include_labels=["hashed_api_key", "team"],
        ),
        PrometheusMetricsConfig(
            group="spend_needs_more",
            metrics=["litellm_spend_metric"],
            include_labels=["hashed_api_key", "team", "end_user"],
        ),
    ]

    label_filters = logger._build_label_filters(configs)

    assert label_filters["litellm_spend_metric"] == [
        "hashed_api_key",
        "team",
        "end_user",
    ]
    assert label_filters["litellm_total_tokens_metric"] == ["hashed_api_key", "team"]


def test_wildcard_group_does_not_disable_other_metrics():
    """A wildcard-only config should leave every metric enabled - it's a
    label filter, not an allowlist."""
    logger = _bare_logger()
    litellm.prometheus_metrics_config = [
        {
            "group": "defaults",
            "metrics": [PROMETHEUS_METRICS_WILDCARD],
            "include_labels": ["team"],
        }
    ]

    logger._parse_prometheus_config()

    assert logger.enabled_metrics == set()
    assert logger._is_metric_enabled("litellm_spend_metric") is True
    assert logger._is_metric_enabled("litellm_mcp_tool_calls_total") is True


def test_wildcard_combined_with_named_enable_list():
    """Wildcard for labels + a separate group naming which metrics are
    enabled - the two concerns are independent."""
    logger = _bare_logger()
    litellm.prometheus_metrics_config = [
        {
            "group": "enabled",
            "metrics": ["litellm_spend_metric", "litellm_total_tokens_metric"],
        },
        {
            "group": "defaults",
            "metrics": [PROMETHEUS_METRICS_WILDCARD],
            "include_labels": ["team"],
        },
    ]

    label_filters = logger._parse_prometheus_config()

    assert logger.enabled_metrics == {
        "litellm_spend_metric",
        "litellm_total_tokens_metric",
    }
    assert logger._is_metric_enabled("litellm_spend_metric") is True
    assert logger._is_metric_enabled("litellm_cache_hits_metric") is False
    assert label_filters["litellm_spend_metric"] == ["team"]


def test_wildcard_mixed_with_named_metric_in_same_group_raises():
    logger = _bare_logger()
    configs = [
        PrometheusMetricsConfig(
            group="broken",
            metrics=[PROMETHEUS_METRICS_WILDCARD, "litellm_spend_metric"],
            include_labels=["team"],
        )
    ]

    with pytest.raises(ValueError, match="mixes the wildcard"):
        logger._validate_all_configurations(configs)


def test_wildcard_with_unknown_label_is_reported():
    logger = _bare_logger()
    configs = [
        PrometheusMetricsConfig(
            group="typo",
            metrics=[PROMETHEUS_METRICS_WILDCARD],
            include_labels=["taem"],  # typo, should be "team"
        )
    ]

    results = logger._validate_all_configurations(configs)

    assert results.has_errors
    assert any("taem" in err.message for err in results.label_errors)


def test_wildcard_with_no_include_labels_is_a_noop():
    logger = _bare_logger()
    configs = [
        PrometheusMetricsConfig(
            group="pointless",
            metrics=[PROMETHEUS_METRICS_WILDCARD],
            include_labels=None,
        )
    ]

    label_filters = logger._build_label_filters(configs)

    assert label_filters == {}