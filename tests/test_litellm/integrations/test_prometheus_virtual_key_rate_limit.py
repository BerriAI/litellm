"""
Unit tests for _set_virtual_key_rate_limit_metrics using prometheus_label_factory.

When custom_prometheus_metadata_labels is configured (e.g., ["onyx_feature"]),
PrometheusMetricLabels.get_labels() appends the custom labels to every metric's
label set. The Gauge is created with these extra labels, so callers must provide
values for them. Using hardcoded positional args to .labels() fails with
    ValueError: Incorrect label count
because the custom labels are never passed.

The fix uses prometheus_label_factory (same pattern as all other metrics in the
callback) so that custom metadata labels are resolved from enum_values
automatically.

Fixes https://github.com/BerriAI/litellm/issues/24760
"""

import os
import sys

import pytest
from prometheus_client import REGISTRY

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues


@pytest.fixture(scope="function")
def prometheus_logger():
    """Create a PrometheusLogger instance for testing."""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


@pytest.fixture(scope="function")
def prometheus_logger_with_custom_labels():
    """Create a PrometheusLogger with custom_prometheus_metadata_labels configured."""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    original = litellm.custom_prometheus_metadata_labels
    litellm.custom_prometheus_metadata_labels = ["onyx_feature"]
    try:
        logger = PrometheusLogger()
        yield logger
    finally:
        litellm.custom_prometheus_metadata_labels = original


class TestVirtualKeyRateLimitMetrics:
    """
    Test that _set_virtual_key_rate_limit_metrics works correctly with
    prometheus_label_factory, including when custom_prometheus_metadata_labels
    is configured.
    """

    def test_set_virtual_key_rate_limit_metrics_basic(self, prometheus_logger):
        """
        _set_virtual_key_rate_limit_metrics should not raise with default labels.
        """
        enum_values = UserAPIKeyLabelValues(
            hashed_api_key="test-key-hash",
            api_key_alias="test-alias",
            model="gpt-4o",
            model_id="model-123",
        )

        kwargs = {
            "litellm_params": {
                "metadata": {
                    "model_group": "gpt-4o",
                },
            },
        }
        metadata = {}

        # Should not raise
        prometheus_logger._set_virtual_key_rate_limit_metrics(
            kwargs=kwargs,
            metadata=metadata,
            enum_values=enum_values,
        )

    def test_set_virtual_key_rate_limit_metrics_with_custom_labels(
        self, prometheus_logger_with_custom_labels
    ):
        """
        _set_virtual_key_rate_limit_metrics should not raise ValueError when
        custom_prometheus_metadata_labels adds extra labels to the Gauge.

        Before the fix, this crashed with:
            ValueError: Incorrect label count
        because the Gauge had 5 labels (4 default + 1 custom) but only 4
        positional args were passed.
        """
        enum_values = UserAPIKeyLabelValues(
            hashed_api_key="test-key-hash",
            api_key_alias="test-alias",
            model="gpt-4o",
            model_id="model-123",
            custom_metadata_labels={"onyx_feature": "test-feature"},
        )

        kwargs = {
            "litellm_params": {
                "metadata": {
                    "model_group": "gpt-4o",
                },
            },
        }
        metadata = {}

        # Should not raise ValueError: Incorrect label count
        prometheus_logger_with_custom_labels._set_virtual_key_rate_limit_metrics(
            kwargs=kwargs,
            metadata=metadata,
            enum_values=enum_values,
        )

    def test_set_virtual_key_rate_limit_metrics_with_none_values(
        self, prometheus_logger
    ):
        """
        _set_virtual_key_rate_limit_metrics should handle None values gracefully.
        """
        enum_values = UserAPIKeyLabelValues(
            hashed_api_key=None,
            api_key_alias=None,
            model=None,
            model_id=None,
        )

        kwargs = {
            "litellm_params": {
                "metadata": {},
            },
        }
        metadata = {}

        # Should not raise
        prometheus_logger._set_virtual_key_rate_limit_metrics(
            kwargs=kwargs,
            metadata=metadata,
            enum_values=enum_values,
        )

    def test_set_virtual_key_rate_limit_metrics_sets_remaining_values(
        self, prometheus_logger
    ):
        """
        _set_virtual_key_rate_limit_metrics should correctly set the remaining
        request/token counts from metadata.
        """
        enum_values = UserAPIKeyLabelValues(
            hashed_api_key="test-key-hash",
            api_key_alias="test-alias",
            model="gpt-4o",
            model_id="model-123",
        )

        kwargs = {
            "litellm_params": {
                "metadata": {
                    "model_group": "gpt-4o",
                },
            },
        }
        metadata = {
            "litellm-key-remaining-requests-gpt-4o": 100,
            "litellm-key-remaining-tokens-gpt-4o": 50000,
        }

        # Should not raise
        prometheus_logger._set_virtual_key_rate_limit_metrics(
            kwargs=kwargs,
            metadata=metadata,
            enum_values=enum_values,
        )
