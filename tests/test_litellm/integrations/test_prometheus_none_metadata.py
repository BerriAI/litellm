"""
Unit tests for Prometheus handling of None metadata in litellm_params.

When the Responses API sends streaming requests, litellm_params.metadata
can be None, causing AttributeError: 'NoneType' object has no attribute 'get'
in set_llm_deployment_success_metrics.
"""

import os
import sys
from datetime import datetime

import pytest
from prometheus_client import REGISTRY

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues


@pytest.fixture(scope="function")
def prometheus_logger():
    """Create a PrometheusLogger instance for testing."""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


class TestNoneMetadataHandling:
    """
    Test that Prometheus metrics don't crash when metadata is None.

    This targets the bug where Responses API streaming sets
    litellm_params["metadata"] = None, causing:
        _metadata.get("model_info") -> AttributeError
    """

    def test_set_llm_deployment_success_metrics_with_none_metadata(
        self, prometheus_logger
    ):
        """
        set_llm_deployment_success_metrics should not raise when
        litellm_params.metadata is None.
        """
        request_kwargs = {
            "litellm_params": {
                "metadata": None,  # Bug trigger
                "custom_llm_provider": "openai",
            },
            "model": "gpt-4o",
            "standard_logging_object": {
                "api_base": "https://api.openai.com",
                "hidden_params": {
                    "additional_headers": None,
                    "litellm_overhead_time_ms": None,
                },
                "metadata": {
                    "user_api_key_hash": "test-key",
                    "user_api_key_alias": None,
                    "user_api_key_team_id": None,
                    "user_api_key_team_alias": None,
                },
                "model": "gpt-4o",
                "response_cost": 0.001,
            },
        }
        enum_values = UserAPIKeyLabelValues(
            end_user=None,
            hashed_api_key="test-key",
            api_key_alias=None,
            team=None,
            team_alias=None,
            requested_model="gpt-4o",
        )

        # Should not raise AttributeError
        prometheus_logger.set_llm_deployment_success_metrics(
            request_kwargs=request_kwargs,
            start_time=datetime.now(),
            end_time=datetime.now(),
            enum_values=enum_values,
            output_tokens=10.0,
        )

    def test_set_llm_deployment_success_metrics_with_missing_litellm_params(
        self, prometheus_logger
    ):
        """
        set_llm_deployment_success_metrics should not raise when
        litellm_params is missing entirely.
        """
        request_kwargs = {
            "model": "gpt-4o",
            "standard_logging_object": {
                "api_base": "https://api.openai.com",
                "hidden_params": {
                    "additional_headers": None,
                    "litellm_overhead_time_ms": None,
                },
                "metadata": {
                    "user_api_key_hash": "test-key",
                    "user_api_key_alias": None,
                    "user_api_key_team_id": None,
                    "user_api_key_team_alias": None,
                },
                "model": "gpt-4o",
                "response_cost": 0.001,
            },
        }
        enum_values = UserAPIKeyLabelValues(
            end_user=None,
            hashed_api_key="test-key",
            api_key_alias=None,
            team=None,
            team_alias=None,
            requested_model="gpt-4o",
        )

        # Should not raise
        prometheus_logger.set_llm_deployment_success_metrics(
            request_kwargs=request_kwargs,
            start_time=datetime.now(),
            end_time=datetime.now(),
            enum_values=enum_values,
            output_tokens=10.0,
        )

    def test_set_llm_deployment_success_metrics_with_litellm_metadata_key(
        self, prometheus_logger
    ):
        """
        set_llm_deployment_success_metrics should pick up litellm_metadata
        when metadata is None, using get_litellm_metadata_from_kwargs.
        """
        request_kwargs = {
            "litellm_params": {
                "metadata": None,
                "litellm_metadata": {"model_info": {"id": "test-model-id"}},
                "custom_llm_provider": "openai",
            },
            "model": "gpt-4o",
            "standard_logging_object": {
                "api_base": "https://api.openai.com",
                "hidden_params": {
                    "additional_headers": None,
                    "litellm_overhead_time_ms": None,
                },
                "metadata": {
                    "user_api_key_hash": "test-key",
                    "user_api_key_alias": None,
                    "user_api_key_team_id": None,
                    "user_api_key_team_alias": None,
                },
                "model": "gpt-4o",
                "response_cost": 0.001,
            },
        }
        enum_values = UserAPIKeyLabelValues(
            end_user=None,
            hashed_api_key="test-key",
            api_key_alias=None,
            team=None,
            team_alias=None,
            requested_model="gpt-4o",
        )

        # Should not raise, and should pick up litellm_metadata
        prometheus_logger.set_llm_deployment_success_metrics(
            request_kwargs=request_kwargs,
            start_time=datetime.now(),
            end_time=datetime.now(),
            enum_values=enum_values,
            output_tokens=10.0,
        )
