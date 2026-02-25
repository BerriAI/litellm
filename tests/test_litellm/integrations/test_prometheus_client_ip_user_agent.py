from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.integrations.prometheus import (
    UserAPIKeyLabelValues,
)


@pytest.mark.asyncio
async def test_async_post_call_failure_hook_includes_client_ip_user_agent():
    """
    Test that async_post_call_failure_hook includes client_ip and user_agent in UserAPIKeyLabelValues
    """
    # Mocking
    # Mocking
    with patch(
        "litellm.integrations.prometheus.PrometheusLogger.__init__", return_value=None
    ):
        logger = PrometheusLogger()
        # Initialize attributes manually as __init__ is mocked
        logger.litellm_proxy_failed_requests_metric = MagicMock()
        logger.litellm_proxy_total_requests_metric = MagicMock()
        logger.get_labels_for_metric = MagicMock(
            return_value=["client_ip", "user_agent"]
        )

    request_data = {
        "model": "gpt-4",
        "metadata": {
            "requester_ip_address": "127.0.0.1",
            "user_agent": "test-agent",
        },
    }
    user_api_key_dict = UserAPIKeyAuth(token="test_token")
    original_exception = Exception("Test exception")

    # Mock prometheus_label_factory to inspect arguments
    with patch(
        "litellm.integrations.prometheus.prometheus_label_factory"
    ) as mock_label_factory:
        mock_label_factory.return_value = {}

        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
        )

        # Verification
        assert mock_label_factory.call_count >= 1

        # Check calls
        calls = mock_label_factory.call_args_list
        found = False
        for call in calls:
            kwargs = call.kwargs
            enum_values = kwargs.get("enum_values")
            if isinstance(enum_values, UserAPIKeyLabelValues):
                if (
                    enum_values.client_ip == "127.0.0.1"
                    and enum_values.user_agent == "test-agent"
                ):
                    found = True
                    break

        assert (
            found
        ), "UserAPIKeyLabelValues should contain client_ip='127.0.0.1' and user_agent='test-agent'"


@pytest.mark.asyncio
async def test_async_post_call_success_hook_includes_client_ip_user_agent():
    """
    Test that async_log_success_event includes client_ip and user_agent in UserAPIKeyLabelValues.
    
    Note: After PR #21159, the metric increment was moved from async_post_call_success_hook 
    to async_log_success_event to prevent double-counting.
    """
    # Mocking
    with patch(
        "litellm.integrations.prometheus.PrometheusLogger.__init__", return_value=None
    ):
        logger = PrometheusLogger()
        logger.litellm_proxy_total_requests_metric = MagicMock()
        logger.get_labels_for_metric = MagicMock(
            return_value=["client_ip", "user_agent"]
        )
        logger._should_skip_metrics_for_invalid_key = MagicMock(return_value=False)
        logger._increment_top_level_request_and_spend_metrics = MagicMock()
        logger._increment_token_metrics = MagicMock()
        logger._increment_remaining_budget_metrics = AsyncMock()
        logger._set_virtual_key_rate_limit_metrics = MagicMock()
        logger._set_latency_metrics = MagicMock()
        logger.set_llm_deployment_success_metrics = MagicMock()
        logger._increment_cache_metrics = MagicMock()

    kwargs = {
        "model": "gpt-4",
        "litellm_params": {
            "metadata": {}
        },
        "start_time": None,
        "standard_logging_object": {
            "model_group": "gpt-4",
            "model_id": "model_1",
            "api_base": "http://api.base",
            "custom_llm_provider": "openai",
            "completion_tokens": 10,
            "total_tokens": 20,
            "response_cost": 0.01,
            "request_tags": [],
            "metadata": {
                "user_api_key_user_id": "user_1",
                "user_api_key_hash": "hash_1",
                "user_api_key_alias": "alias_1",
                "user_api_key_team_id": "team_1",
                "user_api_key_team_alias": "team_alias_1",
                "user_api_key_user_email": "test@example.com",
                "user_api_key_request_route": "/chat/completions",
                "requester_ip_address": "192.168.1.1",
                "user_agent": "success-agent",
            },
        },
    }

    # Mock prometheus_label_factory to inspect arguments
    with patch(
        "litellm.integrations.prometheus.prometheus_label_factory"
    ) as mock_label_factory:
        mock_label_factory.return_value = {}

        await logger.async_log_success_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        # Verification
        assert mock_label_factory.call_count >= 1

        # Check calls
        calls = mock_label_factory.call_args_list
        found = False
        for call in calls:
            kwargs_args = call.kwargs
            enum_values = kwargs_args.get("enum_values")
            if isinstance(enum_values, UserAPIKeyLabelValues):
                if (
                    enum_values.client_ip == "192.168.1.1"
                    and enum_values.user_agent == "success-agent"
                ):
                    found = True
                    break

        assert (
            found
        ), "UserAPIKeyLabelValues should contain client_ip='192.168.1.1' and user_agent='success-agent'"


def test_set_llm_deployment_failure_metrics_includes_client_ip_user_agent():
    """
    Test that set_llm_deployment_failure_metrics includes client_ip and user_agent in UserAPIKeyLabelValues
    """
    # Mocking
    # Mocking
    with patch(
        "litellm.integrations.prometheus.PrometheusLogger.__init__", return_value=None
    ):
        logger = PrometheusLogger()
        logger.litellm_deployment_failure_responses = MagicMock()
        logger.litellm_deployment_total_requests = MagicMock()
        logger.get_labels_for_metric = MagicMock(
            return_value=["client_ip", "user_agent"]
        )
        logger.set_deployment_partial_outage = MagicMock()

    request_kwargs = {
        "model": "gpt-4",
        "standard_logging_object": {
            "metadata": {
                "requester_ip_address": "10.0.0.1",
                "user_agent": "failure-deployment",
                "user_api_key_team_id": "team_1",
                "user_api_key_team_alias": "team_alias_1",
                "user_api_key_alias": "key_alias_1",
            },
            "model_group": "group_1",
            "api_base": "http://api.base",
            "model_id": "model_1",
        },
        "litellm_params": {},
        "exception": Exception("Deployment failure"),
    }

    # Mock prometheus_label_factory to inspect arguments
    with patch(
        "litellm.integrations.prometheus.prometheus_label_factory"
    ) as mock_label_factory:
        mock_label_factory.return_value = {}

        logger.set_llm_deployment_failure_metrics(request_kwargs=request_kwargs)

        # Verification
        assert mock_label_factory.call_count >= 1

        # Check calls
        calls = mock_label_factory.call_args_list
        found = False
        for call in calls:
            kwargs = call.kwargs
            enum_values = kwargs.get("enum_values")
            if isinstance(enum_values, UserAPIKeyLabelValues):
                if (
                    enum_values.client_ip == "10.0.0.1"
                    and enum_values.user_agent == "failure-deployment"
                ):
                    found = True
                    break

        assert (
            found
        ), "UserAPIKeyLabelValues should contain client_ip='10.0.0.1' and user_agent='failure-deployment'"


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_async_post_call_failure_hook_includes_client_ip_user_agent())
    asyncio.run(test_async_post_call_success_hook_includes_client_ip_user_agent())
    test_set_llm_deployment_failure_metrics_includes_client_ip_user_agent()
    print("✅ All client_ip and user_agent tests passed!")
