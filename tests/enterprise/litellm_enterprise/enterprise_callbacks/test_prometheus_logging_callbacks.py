import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

import pytest
from prometheus_client import REGISTRY, CollectorRegistry

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import (
    StandardLoggingHiddenParams,
    StandardLoggingMetadata,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
)

try:
    from litellm_enterprise.integrations.prometheus import (
        PrometheusLogger,
        UserAPIKeyLabelValues,
        get_custom_labels_from_metadata,
    )
except Exception:
    PrometheusLogger = None
from litellm.proxy._types import UserAPIKeyAuth

verbose_logger.setLevel(logging.DEBUG)

litellm.set_verbose = True
import time


@pytest.fixture
def prometheus_logger() -> PrometheusLogger:
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def create_standard_logging_payload() -> StandardLoggingPayload:
    return StandardLoggingPayload(
        id="test_id",
        call_type="completion",
        stream=False,
        response_cost=0.1,
        response_cost_failure_debug_info=None,
        status="success",
        total_tokens=30,
        prompt_tokens=20,
        completion_tokens=10,
        startTime=1234567890.0,
        endTime=1234567891.0,
        completionStartTime=1234567890.5,
        model_map_information=StandardLoggingModelInformation(
            model_map_key="gpt-3.5-turbo", model_map_value=None
        ),
        model="gpt-3.5-turbo",
        model_id="model-123",
        model_group="openai-gpt",
        custom_llm_provider="openai",
        api_base="https://api.openai.com",
        metadata=StandardLoggingMetadata(
            user_api_key_hash="test_hash",
            user_api_key_alias="test_alias",
            user_api_key_team_id="test_team",
            user_api_key_user_id="test_user",
            user_api_key_user_email="test@example.com",
            user_api_key_team_alias="test_team_alias",
            user_api_key_org_id=None,
            spend_logs_metadata=None,
            requester_ip_address="127.0.0.1",
            requester_metadata=None,
            user_api_key_end_user_id="test_end_user",
        ),
        cache_hit=False,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        messages=[{"role": "user", "content": "Hello, world!"}],
        response={"choices": [{"message": {"content": "Hi there!"}}]},
        error_str=None,
        model_parameters={"stream": True},
        hidden_params=StandardLoggingHiddenParams(
            model_id="model-123",
            cache_key=None,
            api_base="https://api.openai.com",
            response_cost="0.1",
            additional_headers=None,
        ),
    )


def test_safe_get_remaining_budget(prometheus_logger):
    assert prometheus_logger._safe_get_remaining_budget(100, 30) == 70
    assert prometheus_logger._safe_get_remaining_budget(100, None) == 100
    assert prometheus_logger._safe_get_remaining_budget(None, 30) == float("inf")
    assert prometheus_logger._safe_get_remaining_budget(None, None) == float("inf")


@pytest.mark.asyncio
async def test_async_log_success_event(prometheus_logger):
    standard_logging_object = create_standard_logging_payload()
    kwargs = {
        "model": "gpt-3.5-turbo",
        "stream": True,
        "litellm_params": {
            "metadata": {
                "user_api_key": "test_key",
                "user_api_key_user_id": "test_user",
                "user_api_key_team_id": "test_team",
                "user_api_key_end_user_id": "test_end_user",
            }
        },
        "start_time": datetime.now(),
        "completion_start_time": datetime.now(),
        "api_call_start_time": datetime.now(),
        "end_time": datetime.now() + timedelta(seconds=1),
        "standard_logging_object": standard_logging_object,
    }
    response_obj = MagicMock()

    # Mock the prometheus client methods

    # High Level Metrics - request/spend
    prometheus_logger.litellm_requests_metric = MagicMock()
    prometheus_logger.litellm_spend_metric = MagicMock()

    # Token Metrics
    prometheus_logger.litellm_tokens_metric = MagicMock()
    prometheus_logger.litellm_input_tokens_metric = MagicMock()
    prometheus_logger.litellm_output_tokens_metric = MagicMock()

    # Remaining Budget Metrics
    prometheus_logger.litellm_remaining_team_budget_metric = MagicMock()
    prometheus_logger.litellm_remaining_api_key_budget_metric = MagicMock()

    # Virtual Key Rate limit Metrics
    prometheus_logger.litellm_remaining_api_key_requests_for_model = MagicMock()
    prometheus_logger.litellm_remaining_api_key_tokens_for_model = MagicMock()

    # Latency Metrics
    prometheus_logger.litellm_llm_api_time_to_first_token_metric = MagicMock()
    prometheus_logger.litellm_llm_api_latency_metric = MagicMock()
    prometheus_logger.litellm_request_total_latency_metric = MagicMock()

    await prometheus_logger.async_log_success_event(
        kwargs, response_obj, kwargs["start_time"], kwargs["end_time"]
    )

    # Assert that the metrics were incremented
    prometheus_logger.litellm_requests_metric.labels.assert_called()
    prometheus_logger.litellm_spend_metric.labels.assert_called()

    # Token Metrics
    prometheus_logger.litellm_tokens_metric.labels.assert_called()
    prometheus_logger.litellm_input_tokens_metric.labels.assert_called()
    prometheus_logger.litellm_output_tokens_metric.labels.assert_called()

    # Remaining Budget Metrics
    prometheus_logger.litellm_remaining_team_budget_metric.labels.assert_called()
    prometheus_logger.litellm_remaining_api_key_budget_metric.labels.assert_called()

    # Virtual Key Rate limit Metrics
    prometheus_logger.litellm_remaining_api_key_requests_for_model.labels.assert_called()
    prometheus_logger.litellm_remaining_api_key_tokens_for_model.labels.assert_called()

    # Latency Metrics
    prometheus_logger.litellm_llm_api_time_to_first_token_metric.labels.assert_called()
    prometheus_logger.litellm_llm_api_latency_metric.labels.assert_called()
    prometheus_logger.litellm_request_total_latency_metric.labels.assert_called()


def test_increment_token_metrics(prometheus_logger):
    """
    Test the increment_token_metrics method

    input, output, and total tokens metrics are incremented by the values in the standard logging payload
    """
    prometheus_logger.litellm_tokens_metric = MagicMock()
    prometheus_logger.litellm_input_tokens_metric = MagicMock()
    prometheus_logger.litellm_output_tokens_metric = MagicMock()

    standard_logging_payload = create_standard_logging_payload()
    standard_logging_payload["total_tokens"] = 100
    standard_logging_payload["prompt_tokens"] = 50
    standard_logging_payload["completion_tokens"] = 50

    enum_values = UserAPIKeyLabelValues(
        litellm_model_name=standard_logging_payload["model"],
        api_provider=standard_logging_payload["custom_llm_provider"],
        hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
        api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
        team=standard_logging_payload["metadata"]["user_api_key_team_id"],
        team_alias=standard_logging_payload["metadata"]["user_api_key_team_alias"],
        **standard_logging_payload,
    )

    prometheus_logger._increment_token_metrics(
        standard_logging_payload,
        end_user_id="user1",
        user_api_key="key1",
        user_api_key_alias="alias1",
        model="gpt-3.5-turbo",
        user_api_team="team1",
        user_api_team_alias="team_alias1",
        user_id="user1",
        enum_values=enum_values,
    )

    prometheus_logger.litellm_tokens_metric.labels.assert_called_once_with(
        end_user=None,
        user=None,
        user_email=None,
        hashed_api_key="test_hash",
        api_key_alias="test_alias",
        team="test_team",
        team_alias="test_team_alias",
        requested_model=None,
        model="gpt-3.5-turbo",
    )
    prometheus_logger.litellm_tokens_metric.labels().inc.assert_called_once_with(100)

    prometheus_logger.litellm_input_tokens_metric.labels.assert_called_once_with(
        end_user=None,
        user=None,
        user_email=None,
        hashed_api_key="test_hash",
        api_key_alias="test_alias",
        team="test_team",
        team_alias="test_team_alias",
        requested_model=None,
        model="gpt-3.5-turbo",
    )
    prometheus_logger.litellm_input_tokens_metric.labels().inc.assert_called_once_with(
        50
    )

    prometheus_logger.litellm_output_tokens_metric.labels.assert_called_once_with(
        end_user=None,
        user=None,
        user_email=None,
        hashed_api_key="test_hash",
        api_key_alias="test_alias",
        team="test_team",
        team_alias="test_team_alias",
        requested_model=None,
        model="gpt-3.5-turbo",
    )
    prometheus_logger.litellm_output_tokens_metric.labels().inc.assert_called_once_with(
        50
    )


@pytest.mark.asyncio
async def test_increment_remaining_budget_metrics(prometheus_logger):
    """
    Test the increment_remaining_budget_metrics method

    - team and api key remaining budget metrics are set to the difference between max budget and spend
    - team and api key max budget metrics are set to their respective max budgets
    - team and api key remaining hours metrics are set based on budget reset timestamps
    """
    # Mock all budget-related metrics
    prometheus_logger.litellm_remaining_team_budget_metric = MagicMock()
    prometheus_logger.litellm_remaining_api_key_budget_metric = MagicMock()
    prometheus_logger.litellm_team_max_budget_metric = MagicMock()
    prometheus_logger.litellm_api_key_max_budget_metric = MagicMock()
    prometheus_logger.litellm_team_budget_remaining_hours_metric = MagicMock()
    prometheus_logger.litellm_api_key_budget_remaining_hours_metric = MagicMock()

    # Create a future budget reset time for testing
    future_reset_time_team = datetime.now() + timedelta(hours=10)
    future_reset_time_key = datetime.now() + timedelta(hours=12)
    # Mock the get_team_object and get_key_object functions to return objects with budget reset times
    with patch(
        "litellm.proxy.auth.auth_checks.get_team_object"
    ) as mock_get_team, patch(
        "litellm.proxy.auth.auth_checks.get_key_object"
    ) as mock_get_key:

        mock_get_team.return_value = MagicMock(budget_reset_at=future_reset_time_team)
        mock_get_key.return_value = MagicMock(budget_reset_at=future_reset_time_key)

        litellm_params = {
            "metadata": {
                "user_api_key_team_spend": 50,
                "user_api_key_team_max_budget": 100,
                "user_api_key_spend": 25,
                "user_api_key_max_budget": 75,
            }
        }

        await prometheus_logger._increment_remaining_budget_metrics(
            user_api_team="team1",
            user_api_team_alias="team_alias1",
            user_api_key="key1",
            user_api_key_alias="alias1",
            litellm_params=litellm_params,
            response_cost=10,
        )

        # Test remaining budget metrics
        prometheus_logger.litellm_remaining_team_budget_metric.labels.assert_called_once_with(
            team="team1", team_alias="team_alias1"
        )
        prometheus_logger.litellm_remaining_team_budget_metric.labels().set.assert_called_once_with(
            40  # 100 - (50 + 10)
        )

        prometheus_logger.litellm_remaining_api_key_budget_metric.labels.assert_called_once_with(
            hashed_api_key="key1", api_key_alias="alias1"
        )
        prometheus_logger.litellm_remaining_api_key_budget_metric.labels().set.assert_called_once_with(
            40  # 75 - (25 + 10)
        )

        # Test max budget metrics
        prometheus_logger.litellm_team_max_budget_metric.labels.assert_called_once_with(
            team="team1", team_alias="team_alias1"
        )
        prometheus_logger.litellm_team_max_budget_metric.labels().set.assert_called_once_with(
            100
        )

        prometheus_logger.litellm_api_key_max_budget_metric.labels.assert_called_once_with(
            hashed_api_key="key1", api_key_alias="alias1"
        )
        prometheus_logger.litellm_api_key_max_budget_metric.labels().set.assert_called_once_with(
            75
        )

        # Test remaining hours metrics
        prometheus_logger.litellm_team_budget_remaining_hours_metric.labels.assert_called_once_with(
            team="team1", team_alias="team_alias1"
        )
        # The remaining hours should be approximately 10 (with some small difference due to test execution time)
        remaining_hours_call = prometheus_logger.litellm_team_budget_remaining_hours_metric.labels().set.call_args[
            0
        ][
            0
        ]
        assert 9.9 <= remaining_hours_call <= 10.0

        prometheus_logger.litellm_api_key_budget_remaining_hours_metric.labels.assert_called_once_with(
            hashed_api_key="key1", api_key_alias="alias1"
        )
        # The remaining hours should be approximately 10 (with some small difference due to test execution time)
        remaining_hours_call = prometheus_logger.litellm_api_key_budget_remaining_hours_metric.labels().set.call_args[
            0
        ][
            0
        ]
        assert 11.9 <= remaining_hours_call <= 12.0


def test_set_latency_metrics(prometheus_logger):
    """
    Test the set_latency_metrics method

    time to first token, llm api latency, and request total latency metrics are set to the values in the standard logging payload
    """
    standard_logging_payload = create_standard_logging_payload()
    prometheus_logger.litellm_llm_api_time_to_first_token_metric = MagicMock()
    prometheus_logger.litellm_llm_api_latency_metric = MagicMock()
    prometheus_logger.litellm_request_total_latency_metric = MagicMock()

    enum_values = UserAPIKeyLabelValues(
        litellm_model_name=standard_logging_payload["model"],
        api_provider=standard_logging_payload["custom_llm_provider"],
        hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
        api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
        team=standard_logging_payload["metadata"]["user_api_key_team_id"],
        team_alias=standard_logging_payload["metadata"]["user_api_key_team_alias"],
        requested_model=standard_logging_payload["model_group"],
        user=standard_logging_payload["metadata"]["user_api_key_user_id"],
        **standard_logging_payload,
    )

    now = datetime.now()
    kwargs = {
        "end_time": now,  # when the request ends
        "start_time": now - timedelta(seconds=2),  # when the request starts
        "api_call_start_time": now - timedelta(seconds=1.5),  # when the api call starts
        "completion_start_time": now
        - timedelta(seconds=1),  # when the completion starts
        "stream": True,
    }

    prometheus_logger._set_latency_metrics(
        kwargs=kwargs,
        model="gpt-3.5-turbo",
        user_api_key="key1",
        user_api_key_alias="alias1",
        user_api_team="team1",
        user_api_team_alias="team_alias1",
        enum_values=enum_values,
    )

    # completion_start_time - api_call_start_time
    prometheus_logger.litellm_llm_api_time_to_first_token_metric.labels.assert_called_once_with(
        "gpt-3.5-turbo", "key1", "alias1", "team1", "team_alias1"
    )
    prometheus_logger.litellm_llm_api_time_to_first_token_metric.labels().observe.assert_called_once_with(
        0.5
    )

    # end_time - api_call_start_time
    prometheus_logger.litellm_llm_api_latency_metric.labels.assert_called_once_with(
        end_user=None,
        user="test_user",
        hashed_api_key="test_hash",
        api_key_alias="test_alias",
        team="test_team",
        team_alias="test_team_alias",
        requested_model="openai-gpt",
        model="gpt-3.5-turbo",
    )
    prometheus_logger.litellm_llm_api_latency_metric.labels().observe.assert_called_once_with(
        1.5
    )

    # total latency for the request
    prometheus_logger.litellm_request_total_latency_metric.labels.assert_called_once_with(
        end_user=None,
        user="test_user",
        hashed_api_key="test_hash",
        api_key_alias="test_alias",
        team="test_team",
        team_alias="test_team_alias",
        requested_model="openai-gpt",
        model="gpt-3.5-turbo",
    )
    prometheus_logger.litellm_request_total_latency_metric.labels().observe.assert_called_once_with(
        2.0
    )


def test_set_latency_metrics_missing_timestamps(prometheus_logger):
    """
    Test that _set_latency_metrics handles missing timestamp values gracefully
    """
    # Mock all metrics used in the method
    prometheus_logger.litellm_llm_api_time_to_first_token_metric = MagicMock()
    prometheus_logger.litellm_llm_api_latency_metric = MagicMock()
    prometheus_logger.litellm_request_total_latency_metric = MagicMock()

    standard_logging_payload = create_standard_logging_payload()
    enum_values = UserAPIKeyLabelValues(
        litellm_model_name=standard_logging_payload["model"],
        api_provider=standard_logging_payload["custom_llm_provider"],
        hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
        api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
        team=standard_logging_payload["metadata"]["user_api_key_team_id"],
        team_alias=standard_logging_payload["metadata"]["user_api_key_team_alias"],
    )

    # Test case where completion_start_time is None
    kwargs = {
        "end_time": datetime.now(),
        "start_time": datetime.now() - timedelta(seconds=2),
        "api_call_start_time": datetime.now() - timedelta(seconds=1.5),
        "completion_start_time": None,  # Missing completion start time
        "stream": True,
    }

    # This should not raise an exception
    prometheus_logger._set_latency_metrics(
        kwargs=kwargs,
        model="gpt-3.5-turbo",
        user_api_key="key1",
        user_api_key_alias="alias1",
        user_api_team="team1",
        user_api_team_alias="team_alias1",
        enum_values=enum_values,
    )

    # Verify time to first token metric was not called due to missing completion_start_time
    prometheus_logger.litellm_llm_api_time_to_first_token_metric.labels.assert_not_called()

    # Other metrics should still be called
    prometheus_logger.litellm_llm_api_latency_metric.labels.assert_called_once()
    prometheus_logger.litellm_request_total_latency_metric.labels.assert_called_once()


def test_set_latency_metrics_missing_api_call_start(prometheus_logger):
    """
    Test that _set_latency_metrics handles missing api_call_start_time gracefully
    """
    # Mock all metrics used in the method
    prometheus_logger.litellm_llm_api_time_to_first_token_metric = MagicMock()
    prometheus_logger.litellm_llm_api_latency_metric = MagicMock()
    prometheus_logger.litellm_request_total_latency_metric = MagicMock()

    standard_logging_payload = create_standard_logging_payload()
    enum_values = UserAPIKeyLabelValues(
        litellm_model_name=standard_logging_payload["model"],
        api_provider=standard_logging_payload["custom_llm_provider"],
        hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
        api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
        team=standard_logging_payload["metadata"]["user_api_key_team_id"],
        team_alias=standard_logging_payload["metadata"]["user_api_key_team_alias"],
    )

    # Test case where api_call_start_time is None
    kwargs = {
        "end_time": datetime.now(),
        "start_time": datetime.now() - timedelta(seconds=2),
        "api_call_start_time": None,  # Missing API call start time
        "completion_start_time": datetime.now() - timedelta(seconds=1),
        "stream": True,
    }

    # This should not raise an exception
    prometheus_logger._set_latency_metrics(
        kwargs=kwargs,
        model="gpt-3.5-turbo",
        user_api_key="key1",
        user_api_key_alias="alias1",
        user_api_team="team1",
        user_api_team_alias="team_alias1",
        enum_values=enum_values,
    )

    # Verify API latency metrics were not called due to missing api_call_start_time
    prometheus_logger.litellm_llm_api_time_to_first_token_metric.labels.assert_not_called()
    prometheus_logger.litellm_llm_api_latency_metric.labels.assert_not_called()

    # Total request latency should still be called
    prometheus_logger.litellm_request_total_latency_metric.labels.assert_called_once()


def test_increment_top_level_request_and_spend_metrics(prometheus_logger):
    """
    Test the increment_top_level_request_and_spend_metrics method

    - litellm_requests_metric is incremented by 1
    - litellm_spend_metric is incremented by the response cost in the standard logging payload
    """
    standard_logging_payload = create_standard_logging_payload()
    enum_values = UserAPIKeyLabelValues(
        litellm_model_name=standard_logging_payload["model"],
        api_provider=standard_logging_payload["custom_llm_provider"],
        hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
        api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
        team=standard_logging_payload["metadata"]["user_api_key_team_id"],
        team_alias=standard_logging_payload["metadata"]["user_api_key_team_alias"],
        **standard_logging_payload,
    )
    prometheus_logger.litellm_requests_metric = MagicMock()
    prometheus_logger.litellm_spend_metric = MagicMock()

    prometheus_logger._increment_top_level_request_and_spend_metrics(
        end_user_id="user1",
        user_api_key="key1",
        user_api_key_alias="alias1",
        model="gpt-3.5-turbo",
        user_api_team="team1",
        user_api_team_alias="team_alias1",
        user_id="user1",
        response_cost=0.1,
        enum_values=enum_values,
    )

    prometheus_logger.litellm_requests_metric.labels.assert_called_once_with(
        end_user=None,
        user=None,
        hashed_api_key="test_hash",
        api_key_alias="test_alias",
        team="test_team",
        team_alias="test_team_alias",
        model="gpt-3.5-turbo",
        user_email=None,
    )
    prometheus_logger.litellm_requests_metric.labels().inc.assert_called_once()

    # The spend metric uses keyword arguments (same as requests metric)
    prometheus_logger.litellm_spend_metric.labels.assert_called_once_with(
        end_user=None,
        user=None,
        hashed_api_key="test_hash",
        api_key_alias="test_alias",
        team="test_team",
        team_alias="test_team_alias",
        model="gpt-3.5-turbo",
        user_email=None,
    )
    prometheus_logger.litellm_spend_metric.labels().inc.assert_called_once_with(0.1)


@pytest.mark.asyncio
async def test_async_log_failure_event(prometheus_logger):
    # NOTE: almost all params for this metric are read from standard logging payload
    standard_logging_object = create_standard_logging_payload()
    kwargs = {
        "model": "gpt-3.5-turbo",
        "litellm_params": {
            "custom_llm_provider": "openai",
        },
        "start_time": datetime.now(),
        "completion_start_time": datetime.now(),
        "api_call_start_time": datetime.now(),
        "end_time": datetime.now() + timedelta(seconds=1),
        "standard_logging_object": standard_logging_object,
        "exception": Exception("Test error"),
    }
    response_obj = MagicMock()

    # Mock the metrics
    prometheus_logger.litellm_llm_api_failed_requests_metric = MagicMock()
    prometheus_logger.litellm_deployment_failure_responses = MagicMock()
    prometheus_logger.litellm_deployment_total_requests = MagicMock()
    prometheus_logger.set_deployment_partial_outage = MagicMock()

    await prometheus_logger.async_log_failure_event(
        kwargs, response_obj, kwargs["start_time"], kwargs["end_time"]
    )

    # litellm_llm_api_failed_requests_metric incremented
    """
    Expected metrics
    end_user_id,
    user_api_key,
    user_api_key_alias,
    model,
    user_api_team,
    user_api_team_alias,
    user_id,
    """
    prometheus_logger.litellm_llm_api_failed_requests_metric.labels.assert_called_once_with(
        None,
        "test_hash",
        "test_alias",
        "gpt-3.5-turbo",
        "test_team",
        "test_team_alias",
        "test_user",
    )
    prometheus_logger.litellm_llm_api_failed_requests_metric.labels().inc.assert_called_once()

    # deployment should be marked in partial outage
    prometheus_logger.set_deployment_partial_outage.assert_called_once_with(
        litellm_model_name="gpt-3.5-turbo",
        model_id="model-123",
        api_base="https://api.openai.com",
        api_provider="openai",
    )

    # deployment failure responses incremented
    prometheus_logger.litellm_deployment_failure_responses.labels.assert_called_once_with(
        litellm_model_name="gpt-3.5-turbo",
        model_id="model-123",
        api_base="https://api.openai.com",
        api_provider="openai",
        exception_status="None",
        exception_class="Exception",
        requested_model="openai-gpt",  # passed in standard logging payload
        hashed_api_key="test_hash",
        api_key_alias="test_alias",
        team="test_team",
        team_alias="test_team_alias",
    )
    prometheus_logger.litellm_deployment_failure_responses.labels().inc.assert_called_once()

    # deployment total requests incremented
    prometheus_logger.litellm_deployment_total_requests.labels.assert_called_once_with(
        litellm_model_name="gpt-3.5-turbo",
        model_id="model-123",
        api_base="https://api.openai.com",
        api_provider="openai",
        requested_model="openai-gpt",  # passed in standard logging payload
        hashed_api_key="test_hash",
        api_key_alias="test_alias",
        team="test_team",
        team_alias="test_team_alias",
    )
    prometheus_logger.litellm_deployment_total_requests.labels().inc.assert_called_once()


@pytest.mark.asyncio
async def test_async_post_call_failure_hook(prometheus_logger):
    """
    Test for the async_post_call_failure_hook method

    it should increment the litellm_proxy_failed_requests_metric and litellm_proxy_total_requests_metric
    """
    # Mock the prometheus metrics
    prometheus_logger.litellm_proxy_failed_requests_metric = MagicMock()
    prometheus_logger.litellm_proxy_total_requests_metric = MagicMock()

    # Create test data
    request_data = {"model": "gpt-3.5-turbo"}

    original_exception = litellm.RateLimitError(
        message="Test error", llm_provider="openai", model="gpt-3.5-turbo"
    )

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_key",
        key_alias="test_alias",
        team_id="test_team",
        team_alias="test_team_alias",
        user_id="test_user",
        end_user_id="test_end_user",
        request_route="/chat/completions",
    )

    # Call the function
    await prometheus_logger.async_post_call_failure_hook(
        request_data=request_data,
        original_exception=original_exception,
        user_api_key_dict=user_api_key_dict,
    )

    # Assert failed requests metric was incremented with correct labels
    prometheus_logger.litellm_proxy_failed_requests_metric.labels.assert_called_once_with(
        end_user=None,
        user="test_user",
        user_email=None,
        hashed_api_key="test_key",
        api_key_alias="test_alias",
        team="test_team",
        team_alias="test_team_alias",
        requested_model="gpt-3.5-turbo",
        exception_status="429",
        exception_class="Openai.RateLimitError",
        route=user_api_key_dict.request_route,
    )
    prometheus_logger.litellm_proxy_failed_requests_metric.labels().inc.assert_called_once()

    # Assert total requests metric was incremented with correct labels
    prometheus_logger.litellm_proxy_total_requests_metric.labels.assert_called_once_with(
        end_user=None,
        hashed_api_key="test_key",
        api_key_alias="test_alias",
        requested_model="gpt-3.5-turbo",
        team="test_team",
        team_alias="test_team_alias",
        user="test_user",
        status_code="429",
        user_email=None,
        route=user_api_key_dict.request_route,
    )
    prometheus_logger.litellm_proxy_total_requests_metric.labels().inc.assert_called_once()


@pytest.mark.asyncio
async def test_async_post_call_success_hook(prometheus_logger):
    """
    Test for the async_post_call_success_hook method

    it should increment the litellm_proxy_total_requests_metric
    """
    # Mock the prometheus metric
    prometheus_logger.litellm_proxy_total_requests_metric = MagicMock()

    # Create test data
    data = {"model": "gpt-3.5-turbo"}

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_key",
        key_alias="test_alias",
        team_id="test_team",
        team_alias="test_team_alias",
        user_id="test_user",
        end_user_id="test_end_user",
        request_route="/chat/completions",
    )

    response = {"choices": [{"message": {"content": "test response"}}]}

    # Call the function
    await prometheus_logger.async_post_call_success_hook(
        data=data, user_api_key_dict=user_api_key_dict, response=response
    )

    # Assert total requests metric was incremented with correct labels
    prometheus_logger.litellm_proxy_total_requests_metric.labels.assert_called_once_with(
        end_user=None,
        hashed_api_key="test_key",
        api_key_alias="test_alias",
        requested_model="gpt-3.5-turbo",
        team="test_team",
        team_alias="test_team_alias",
        user="test_user",
        status_code="200",
        user_email=None,
        route=user_api_key_dict.request_route,
    )
    prometheus_logger.litellm_proxy_total_requests_metric.labels().inc.assert_called_once()


def test_set_llm_deployment_success_metrics(prometheus_logger):
    # Mock all the metrics used in the method
    prometheus_logger.litellm_remaining_requests_metric = MagicMock()
    prometheus_logger.litellm_remaining_tokens_metric = MagicMock()
    prometheus_logger.litellm_deployment_success_responses = MagicMock()
    prometheus_logger.litellm_deployment_total_requests = MagicMock()
    prometheus_logger.litellm_deployment_latency_per_output_token = MagicMock()
    prometheus_logger.set_deployment_healthy = MagicMock()
    prometheus_logger.litellm_overhead_latency_metric = MagicMock()

    standard_logging_payload = create_standard_logging_payload()

    standard_logging_payload["hidden_params"]["additional_headers"] = {
        "x_ratelimit_remaining_requests": 123,
        "x_ratelimit_remaining_tokens": 4321,
    }
    standard_logging_payload["model_group"] = "my_custom_model_group"
    standard_logging_payload["hidden_params"]["litellm_overhead_time_ms"] = 100

    # Create test data
    request_kwargs = {
        "model": "gpt-3.5-turbo",
        "litellm_params": {
            "custom_llm_provider": "openai",
            "metadata": {"model_info": {"id": "model-123"}},
        },
        "standard_logging_object": standard_logging_payload,
    }

    enum_values = UserAPIKeyLabelValues(
        requested_model=standard_logging_payload["model_group"],
        litellm_model_name=standard_logging_payload["model"],
        api_provider=standard_logging_payload["custom_llm_provider"],
        hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
        api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
        team=standard_logging_payload["metadata"]["user_api_key_team_id"],
        team_alias=standard_logging_payload["metadata"]["user_api_key_team_alias"],
        **standard_logging_payload,
    )

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)
    output_tokens = 10

    # Call the function
    prometheus_logger.set_llm_deployment_success_metrics(
        request_kwargs=request_kwargs,
        start_time=start_time,
        end_time=end_time,
        output_tokens=output_tokens,
        enum_values=enum_values,
    )

    # Verify remaining requests metric
    prometheus_logger.litellm_remaining_requests_metric.labels.assert_called_once_with(
        model_group="my_custom_model_group",  # model_group / requested model from create_standard_logging_payload()
        api_provider="openai",  # llm provider
        api_base="https://api.openai.com",  # api base
        litellm_model_name="gpt-3.5-turbo",  # actual model used - litellm model name
        hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
        api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
    )

    prometheus_logger.litellm_remaining_requests_metric.labels().set.assert_called_once_with(
        123
    )

    # Verify remaining tokens metric
    prometheus_logger.litellm_remaining_tokens_metric.labels.assert_called_once_with(
        api_base="https://api.openai.com",
        api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
        api_provider="openai",
        hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
        litellm_model_name="gpt-3.5-turbo",
        model_group="my_custom_model_group",
    )

    prometheus_logger.litellm_remaining_tokens_metric.labels().set.assert_called_once_with(
        4321
    )

    # Verify deployment healthy state
    prometheus_logger.set_deployment_healthy.assert_called_once_with(
        litellm_model_name="gpt-3.5-turbo",
        model_id="model-123",
        api_base="https://api.openai.com",
        api_provider="openai",
    )

    # Verify success responses metric
    prometheus_logger.litellm_deployment_success_responses.labels.assert_called_once_with(
        litellm_model_name="gpt-3.5-turbo",
        model_id="model-123",
        api_base="https://api.openai.com",
        api_provider="openai",
        requested_model="my_custom_model_group",
        hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
        api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
        team=standard_logging_payload["metadata"]["user_api_key_team_id"],
        team_alias=standard_logging_payload["metadata"]["user_api_key_team_alias"],
    )
    prometheus_logger.litellm_deployment_success_responses.labels().inc.assert_called_once()

    # Verify total requests metric
    prometheus_logger.litellm_deployment_total_requests.labels.assert_called_once_with(
        litellm_model_name="gpt-3.5-turbo",
        model_id="model-123",
        api_base="https://api.openai.com",
        api_provider="openai",
        requested_model="my_custom_model_group",
        hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
        api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
        team=standard_logging_payload["metadata"]["user_api_key_team_id"],
        team_alias=standard_logging_payload["metadata"]["user_api_key_team_alias"],
    )
    prometheus_logger.litellm_deployment_total_requests.labels().inc.assert_called_once()

    # Verify latency per output token metric
    prometheus_logger.litellm_deployment_latency_per_output_token.labels.assert_called_once_with(
        litellm_model_name="gpt-3.5-turbo",
        model_id="model-123",
        api_base="https://api.openai.com",
        api_provider="openai",
        hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
        api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
        team=standard_logging_payload["metadata"]["user_api_key_team_id"],
        team_alias=standard_logging_payload["metadata"]["user_api_key_team_alias"],
    )
    prometheus_logger.litellm_overhead_latency_metric.labels.assert_called_once_with(
        api_base="https://api.openai.com",
        api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
        api_provider="openai",
        hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
        litellm_model_name="gpt-3.5-turbo",
        model_group="my_custom_model_group",
    )

    # Calculate expected latency per token (1 second / 10 tokens = 0.1 seconds per token)
    expected_latency_per_token = 0.1
    prometheus_logger.litellm_deployment_latency_per_output_token.labels().observe.assert_called_once_with(
        expected_latency_per_token
    )


@pytest.mark.asyncio
async def test_log_success_fallback_event(prometheus_logger):
    prometheus_logger.litellm_deployment_successful_fallbacks = MagicMock()

    original_model_group = "gpt-3.5-turbo"
    kwargs = {
        "model": "gpt-4",
        "metadata": {
            "user_api_key_hash": "test_hash",
            "user_api_key_alias": "test_alias",
            "user_api_key_team_id": "test_team",
            "user_api_key_team_alias": "test_team_alias",
        },
    }
    original_exception = litellm.RateLimitError(
        message="Test error", llm_provider="openai", model="gpt-3.5-turbo"
    )

    await prometheus_logger.log_success_fallback_event(
        original_model_group=original_model_group,
        kwargs=kwargs,
        original_exception=original_exception,
    )

    prometheus_logger.litellm_deployment_successful_fallbacks.labels.assert_called_once_with(
        requested_model=original_model_group,
        fallback_model="gpt-4",
        hashed_api_key="test_hash",
        api_key_alias="test_alias",
        team="test_team",
        team_alias="test_team_alias",
        exception_status="429",
        exception_class="Openai.RateLimitError",
    )
    prometheus_logger.litellm_deployment_successful_fallbacks.labels().inc.assert_called_once()


@pytest.mark.asyncio
async def test_log_failure_fallback_event(prometheus_logger):
    prometheus_logger.litellm_deployment_failed_fallbacks = MagicMock()

    original_model_group = "gpt-3.5-turbo"
    kwargs = {
        "model": "gpt-4",
        "metadata": {
            "user_api_key_hash": "test_hash",
            "user_api_key_alias": "test_alias",
            "user_api_key_team_id": "test_team",
            "user_api_key_team_alias": "test_team_alias",
        },
    }
    original_exception = litellm.RateLimitError(
        message="Test error", llm_provider="openai", model="gpt-3.5-turbo"
    )

    await prometheus_logger.log_failure_fallback_event(
        original_model_group=original_model_group,
        kwargs=kwargs,
        original_exception=original_exception,
    )

    prometheus_logger.litellm_deployment_failed_fallbacks.labels.assert_called_once_with(
        requested_model=original_model_group,
        fallback_model="gpt-4",
        hashed_api_key="test_hash",
        api_key_alias="test_alias",
        team="test_team",
        team_alias="test_team_alias",
        exception_status="429",
        exception_class="Openai.RateLimitError",
    )
    prometheus_logger.litellm_deployment_failed_fallbacks.labels().inc.assert_called_once()


def test_deployment_state_management(prometheus_logger):
    prometheus_logger.litellm_deployment_state = MagicMock()

    test_params = {
        "litellm_model_name": "gpt-3.5-turbo",
        "model_id": "model-123",
        "api_base": "https://api.openai.com",
        "api_provider": "openai",
    }

    # Test set_deployment_healthy (state=0)
    prometheus_logger.set_deployment_healthy(**test_params)
    prometheus_logger.litellm_deployment_state.labels.assert_called_with(
        litellm_model_name=test_params["litellm_model_name"],
        model_id=test_params["model_id"],
        api_base=test_params["api_base"],
        api_provider=test_params["api_provider"],
    )
    prometheus_logger.litellm_deployment_state.labels().set.assert_called_with(0)

    # Test set_deployment_partial_outage (state=1)
    prometheus_logger.set_deployment_partial_outage(**test_params)
    prometheus_logger.litellm_deployment_state.labels().set.assert_called_with(1)

    # Test set_deployment_complete_outage (state=2)
    prometheus_logger.set_deployment_complete_outage(**test_params)
    prometheus_logger.litellm_deployment_state.labels().set.assert_called_with(2)


def test_increment_deployment_cooled_down(prometheus_logger):

    prometheus_logger.litellm_deployment_cooled_down = MagicMock()

    prometheus_logger.increment_deployment_cooled_down(
        litellm_model_name="gpt-3.5-turbo",
        model_id="model-123",
        api_base="https://api.openai.com",
        api_provider="openai",
        exception_status="429",
    )

    prometheus_logger.litellm_deployment_cooled_down.labels.assert_called_once_with(
        "gpt-3.5-turbo", "model-123", "https://api.openai.com", "openai", "429"
    )
    prometheus_logger.litellm_deployment_cooled_down.labels().inc.assert_called_once()


@pytest.mark.parametrize("enable_end_user_cost_tracking_prometheus_only", [True, False])
def test_prometheus_factory(monkeypatch, enable_end_user_cost_tracking_prometheus_only):
    from litellm_enterprise.integrations.prometheus import prometheus_label_factory

    from litellm.types.integrations.prometheus import UserAPIKeyLabelValues

    monkeypatch.setattr(
        "litellm.enable_end_user_cost_tracking_prometheus_only",
        enable_end_user_cost_tracking_prometheus_only,
    )

    enum_values = UserAPIKeyLabelValues(
        end_user="test_end_user",
        api_key_hash="test_hash",
        api_key_alias="test_alias",
    )
    supported_labels = ["end_user", "api_key_hash", "api_key_alias"]
    returned_dict = prometheus_label_factory(
        supported_enum_labels=supported_labels, enum_values=enum_values
    )

    if enable_end_user_cost_tracking_prometheus_only is True:
        assert returned_dict["end_user"] == "test_end_user"
    else:
        assert returned_dict["end_user"] == None


def test_get_custom_labels_from_metadata(monkeypatch):
    monkeypatch.setattr(
        "litellm.custom_prometheus_metadata_labels", ["metadata.foo", "metadata.bar"]
    )
    metadata = {"foo": "bar", "bar": "baz", "taz": "qux"}
    assert get_custom_labels_from_metadata(metadata) == {
        "metadata_foo": "bar",
        "metadata_bar": "baz",
    }


def test_get_custom_labels_from_metadata_tags(monkeypatch):
    monkeypatch.setattr("litellm.custom_prometheus_metadata_labels", [])
    metadata = {"foo": "bar", "bar": "baz", "taz": "qux"}
    assert get_custom_labels_from_metadata(metadata) == {}


def test_get_custom_labels_from_tags(monkeypatch):
    from litellm_enterprise.integrations.prometheus import get_custom_labels_from_tags

    monkeypatch.setattr(
        "litellm.custom_prometheus_tags", ["prod", "test-env", "batch.job"]
    )
    tags = ["prod", "debug", "batch.job"]
    result = get_custom_labels_from_tags(tags)
    assert result == {
        "tag_prod": "true",
        "tag_test_env": "false",  # not in request tags
        "tag_batch_job": "true",  # dot replaced with underscore
    }


def test_get_custom_labels_from_tags_empty_config(monkeypatch):
    from litellm_enterprise.integrations.prometheus import get_custom_labels_from_tags

    monkeypatch.setattr("litellm.custom_prometheus_tags", [])
    tags = ["prod", "debug"]
    result = get_custom_labels_from_tags(tags)
    assert result == {}


def test_get_custom_labels_from_tags_no_tags(monkeypatch):
    from litellm_enterprise.integrations.prometheus import get_custom_labels_from_tags

    monkeypatch.setattr("litellm.custom_prometheus_tags", ["prod", "test"])
    tags = []
    result = get_custom_labels_from_tags(tags)
    assert result == {
        "tag_prod": "false",
        "tag_test": "false",
    }


def test_get_custom_labels_from_tags_wildcard_patterns(monkeypatch):
    """Test wildcard pattern matching for custom labels from tags."""
    from litellm_enterprise.integrations.prometheus import get_custom_labels_from_tags

    # Configure tags with wildcard patterns
    monkeypatch.setattr(
        "litellm.custom_prometheus_tags",
        [
            "User-Agent: curl/*",
            "User-Agent: python-requests/*",
            "Environment: prod*",
            "Service: api-gateway*",
            "exact-match",
        ],
    )

    # Test tags that should match the wildcard patterns
    tags = [
        "User-Agent: curl/7.68.0",
        "User-Agent: python-requests/2.28.1",
        "Environment: production",
        "Service: api-gateway-v2",
        "exact-match",
        "other-tag",
    ]

    result = get_custom_labels_from_tags(tags)

    expected = {
        "tag_User_Agent__curl__": "true",  # matches "User-Agent: curl/*"
        "tag_User_Agent__python_requests__": "true",  # matches "User-Agent: python-requests/*"
        "tag_Environment__prod_": "true",  # matches "Environment: prod*"
        "tag_Service__api_gateway_": "true",  # matches "Service: api-gateway*"
        "tag_exact_match": "true",  # exact match
    }

    assert result == expected


def test_get_custom_labels_from_tags_wildcard_no_matches(monkeypatch):
    """Test wildcard patterns that don't match any tags."""
    from litellm_enterprise.integrations.prometheus import get_custom_labels_from_tags

    # Configure tags with wildcard patterns
    monkeypatch.setattr(
        "litellm.custom_prometheus_tags",
        ["User-Agent: firefox/*", "Environment: dev*", "Service: web-app*"],
    )

    # Test tags that should NOT match the wildcard patterns
    tags = [
        "User-Agent: curl/7.68.0",  # doesn't match "User-Agent: firefox/*"
        "Environment: production",  # doesn't match "Environment: dev*"
        "Service: api-gateway-v2",  # doesn't match "Service: web-app*"
        "other-tag",
    ]

    result = get_custom_labels_from_tags(tags)

    expected = {
        "tag_User_Agent__firefox__": "false",  # no match for "User-Agent: firefox/*"
        "tag_Environment__dev_": "false",  # no match for "Environment: dev*"
        "tag_Service__web_app_": "false",  # no match for "Service: web-app*"
    }

    assert result == expected


def test_tag_matches_wildcard_configured_pattern():
    """Test the helper function for wildcard pattern matching."""
    from litellm_enterprise.integrations.prometheus import (
        _tag_matches_wildcard_configured_pattern,
    )

    # Test cases that should match
    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["User-Agent: curl/7.68.0", "prod", "other"],
            configured_tag="User-Agent: curl/*",
        )
        is True
    )

    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["User-Agent: python-requests/2.28.1", "test"],
            configured_tag="User-Agent: python-requests/*",
        )
        is True
    )

    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["Environment: production", "debug"],
            configured_tag="Environment: prod*",
        )
        is True
    )

    # Test exact match (no wildcard)
    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["prod", "test"], configured_tag="prod"
        )
        is True
    )

    # Test cases that should NOT match
    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["User-Agent: firefox/98.0", "prod"],
            configured_tag="User-Agent: curl/*",
        )
        is False
    )

    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["Environment: development", "test"],
            configured_tag="Environment: prod*",
        )
        is False
    )

    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=["staging", "test"], configured_tag="prod"
        )
        is False
    )

    # Test with empty tags
    assert (
        _tag_matches_wildcard_configured_pattern(
            tags=[], configured_tag="User-Agent: curl/*"
        )
        is False
    )


@pytest.mark.asyncio(scope="session")
async def test_initialize_remaining_budget_metrics(prometheus_logger):
    """
    Test that _initialize_remaining_budget_metrics correctly sets budget metrics for all teams
    """
    litellm.prometheus_initialize_budget_metrics = True
    # Mock the prisma client and get_paginated_teams function
    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_paginated_teams"
    ) as mock_get_teams:

        # Create mock team data with proper datetime objects for budget_reset_at
        future_reset = datetime.now() + timedelta(hours=24)  # Reset 24 hours from now
        mock_teams = [
            MagicMock(
                team_id="team1",
                team_alias="alias1",
                max_budget=100,
                spend=30,
                budget_reset_at=future_reset,
            ),
            MagicMock(
                team_id="team2",
                team_alias="alias2",
                max_budget=200,
                spend=50,
                budget_reset_at=future_reset,
            ),
            MagicMock(
                team_id="team3",
                team_alias=None,
                max_budget=300,
                spend=100,
                budget_reset_at=future_reset,
            ),
        ]

        # Mock get_paginated_teams to return our test data
        mock_get_teams.return_value = (mock_teams, len(mock_teams))

        # Mock the Prometheus metrics
        prometheus_logger.litellm_remaining_team_budget_metric = MagicMock()
        prometheus_logger.litellm_team_budget_remaining_hours_metric = MagicMock()

        # Call the function
        await prometheus_logger._initialize_remaining_budget_metrics()

        # Verify the remaining budget metric was set correctly for each team
        expected_budget_calls = [
            call.labels("team1", "alias1").set(70),  # 100 - 30
            call.labels("team2", "alias2").set(150),  # 200 - 50
            call.labels("team3", "").set(200),  # 300 - 100
        ]

        prometheus_logger.litellm_remaining_team_budget_metric.assert_has_calls(
            expected_budget_calls, any_order=True
        )

        # Get all the calls made to the hours metric
        hours_calls = (
            prometheus_logger.litellm_team_budget_remaining_hours_metric.mock_calls
        )

        # Verify the structure and approximate values of the hours calls
        assert len(hours_calls) == 6  # 3 teams * 2 calls each (labels + set)

        # Helper function to extract hours value from call
        def get_hours_from_call(call_obj):
            if "set" in str(call_obj):
                return call_obj[1][0]  # Extract the hours value
            return None

        # Verify each team's hours are approximately 24 (within reasonable bounds)
        hours_values = [
            get_hours_from_call(call)
            for call in hours_calls
            if get_hours_from_call(call) is not None
        ]
        for hours in hours_values:
            assert (
                23.9 <= hours <= 24.0
            ), f"Hours value {hours} not within expected range"

        # Verify the labels were called with correct team information
        label_calls = [
            call.labels(team="team1", team_alias="alias1"),
            call.labels(team="team2", team_alias="alias2"),
            call.labels(team="team3", team_alias=""),
        ]
        prometheus_logger.litellm_team_budget_remaining_hours_metric.assert_has_calls(
            label_calls, any_order=True
        )


@pytest.mark.asyncio
async def test_initialize_remaining_budget_metrics_exception_handling(
    prometheus_logger,
):
    """
    Test that _initialize_remaining_budget_metrics properly handles exceptions
    """
    litellm.prometheus_initialize_budget_metrics = True
    # Mock the prisma client and get_paginated_teams function to raise an exception
    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_paginated_teams"
    ) as mock_get_teams, patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._list_key_helper"
    ) as mock_list_keys:

        # Make get_paginated_teams raise an exception
        mock_get_teams.side_effect = Exception("Database error")
        mock_list_keys.side_effect = Exception("Key listing error")

        # Mock the Prometheus metrics
        prometheus_logger.litellm_remaining_team_budget_metric = MagicMock()
        prometheus_logger.litellm_remaining_api_key_budget_metric = MagicMock()

        # Mock the logger to capture the error
        with patch("litellm._logging.verbose_logger.exception") as mock_logger:
            # Call the function
            await prometheus_logger._initialize_remaining_budget_metrics()

            # Verify both errors were logged
            assert mock_logger.call_count == 2
            assert (
                "Error initializing teams budget metrics"
                in mock_logger.call_args_list[0][0][0]
            )
            assert (
                "Error initializing keys budget metrics"
                in mock_logger.call_args_list[1][0][0]
            )

        # Verify the metrics were never called
        prometheus_logger.litellm_remaining_team_budget_metric.assert_not_called()
        prometheus_logger.litellm_remaining_api_key_budget_metric.assert_not_called()


@pytest.mark.asyncio(scope="session")
async def test_initialize_api_key_budget_metrics(prometheus_logger):
    """
    Test that _initialize_api_key_budget_metrics correctly sets budget metrics for all API keys
    """
    litellm.prometheus_initialize_budget_metrics = True
    # Mock the prisma client and _list_key_helper function
    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._list_key_helper"
    ) as mock_list_keys:

        # Create mock key data with proper datetime objects for budget_reset_at
        future_reset = datetime.now() + timedelta(hours=24)  # Reset 24 hours from now
        key1 = UserAPIKeyAuth(
            api_key="key1_hash",
            key_alias="alias1",
            team_id="team1",
            max_budget=100,
            spend=30,
            budget_reset_at=future_reset,
        )
        key1.token = "key1_hash"
        key2 = UserAPIKeyAuth(
            api_key="key2_hash",
            key_alias="alias2",
            team_id="team2",
            max_budget=200,
            spend=50,
            budget_reset_at=future_reset,
        )
        key2.token = "key2_hash"

        key3 = UserAPIKeyAuth(
            api_key="key3_hash",
            key_alias=None,
            team_id="team3",
            max_budget=300,
            spend=100,
            budget_reset_at=future_reset,
        )
        key3.token = "key3_hash"

        mock_keys = [
            key1,
            key2,
            key3,
        ]

        # Mock _list_key_helper to return our test data
        mock_list_keys.return_value = {"keys": mock_keys, "total_count": len(mock_keys)}

        # Mock the Prometheus metrics
        prometheus_logger.litellm_remaining_api_key_budget_metric = MagicMock()
        prometheus_logger.litellm_api_key_budget_remaining_hours_metric = MagicMock()
        prometheus_logger.litellm_api_key_max_budget_metric = MagicMock()

        # Call the function
        await prometheus_logger._initialize_api_key_budget_metrics()

        # Verify the remaining budget metric was set correctly for each key
        expected_budget_calls = [
            call.labels("key1_hash", "alias1").set(70),  # 100 - 30
            call.labels("key2_hash", "alias2").set(150),  # 200 - 50
            call.labels("key3_hash", "").set(200),  # 300 - 100
        ]

        prometheus_logger.litellm_remaining_api_key_budget_metric.assert_has_calls(
            expected_budget_calls, any_order=True
        )

        # Get all the calls made to the hours metric
        hours_calls = (
            prometheus_logger.litellm_api_key_budget_remaining_hours_metric.mock_calls
        )

        # Verify the structure and approximate values of the hours calls
        assert len(hours_calls) == 6  # 3 keys * 2 calls each (labels + set)

        # Helper function to extract hours value from call
        def get_hours_from_call(call_obj):
            if "set" in str(call_obj):
                return call_obj[1][0]  # Extract the hours value
            return None

        # Verify each key's hours are approximately 24 (within reasonable bounds)
        hours_values = [
            get_hours_from_call(call)
            for call in hours_calls
            if get_hours_from_call(call) is not None
        ]
        for hours in hours_values:
            assert (
                23.9 <= hours <= 24.0
            ), f"Hours value {hours} not within expected range"

        # Verify max budget metric was set correctly for each key
        expected_max_budget_calls = [
            call.labels("key1_hash", "alias1").set(100),
            call.labels("key2_hash", "alias2").set(200),
            call.labels("key3_hash", "").set(300),
        ]
        prometheus_logger.litellm_api_key_max_budget_metric.assert_has_calls(
            expected_max_budget_calls, any_order=True
        )


def test_set_team_budget_metrics_multiple_teams(prometheus_logger):
    """
    Test that _set_team_budget_metrics correctly handles multiple teams with different budgets and reset times
    """
    # Create test teams with different budgets and reset times
    teams = [
        MagicMock(
            team_id="team1",
            team_alias="alias1",
            spend=50.0,
            max_budget=100.0,
            budget_reset_at=datetime(2024, 12, 31, tzinfo=timezone.utc),
        ),
        MagicMock(
            team_id="team2",
            team_alias="alias2",
            spend=75.0,
            max_budget=150.0,
            budget_reset_at=datetime(2024, 6, 30, tzinfo=timezone.utc),
        ),
        MagicMock(
            team_id="team3",
            team_alias="alias3",
            spend=25.0,
            max_budget=200.0,
            budget_reset_at=datetime(2024, 3, 31, tzinfo=timezone.utc),
        ),
    ]

    # Mock the metrics
    prometheus_logger.litellm_remaining_team_budget_metric = MagicMock()
    prometheus_logger.litellm_team_max_budget_metric = MagicMock()
    prometheus_logger.litellm_team_budget_remaining_hours_metric = MagicMock()

    # Set metrics for each team
    for team in teams:
        prometheus_logger._set_team_budget_metrics(team)

    # Verify remaining budget metric calls
    expected_remaining_budget_calls = [
        call.labels(team="team1", team_alias="alias1").set(50.0),  # 100 - 50
        call.labels(team="team2", team_alias="alias2").set(75.0),  # 150 - 75
        call.labels(team="team3", team_alias="alias3").set(175.0),  # 200 - 25
    ]
    prometheus_logger.litellm_remaining_team_budget_metric.assert_has_calls(
        expected_remaining_budget_calls, any_order=True
    )

    # Verify max budget metric calls
    expected_max_budget_calls = [
        call.labels("team1", "alias1").set(100.0),
        call.labels("team2", "alias2").set(150.0),
        call.labels("team3", "alias3").set(200.0),
    ]
    prometheus_logger.litellm_team_max_budget_metric.assert_has_calls(
        expected_max_budget_calls, any_order=True
    )

    # Verify budget reset metric calls
    # Note: The exact hours will depend on the current time, so we'll just verify the structure
    assert (
        prometheus_logger.litellm_team_budget_remaining_hours_metric.labels.call_count
        == 3
    )
    assert (
        prometheus_logger.litellm_team_budget_remaining_hours_metric.labels().set.call_count
        == 3
    )


def test_set_team_budget_metrics_null_values(prometheus_logger):
    """
    Test that _set_team_budget_metrics correctly handles null/None values
    """
    # Create test team with null values
    team = MagicMock(
        team_id="team_null",
        team_alias=None,  # Test null alias
        spend=None,  # Test null spend
        max_budget=None,  # Test null max_budget
        budget_reset_at=None,  # Test null reset time
    )

    # Mock the metrics
    prometheus_logger.litellm_remaining_team_budget_metric = MagicMock()
    prometheus_logger.litellm_team_max_budget_metric = MagicMock()
    prometheus_logger.litellm_team_budget_remaining_hours_metric = MagicMock()

    # Set metrics for the team
    prometheus_logger._set_team_budget_metrics(team)

    # Verify remaining budget metric is set to infinity when max_budget is None
    prometheus_logger.litellm_remaining_team_budget_metric.labels.assert_called_once_with(
        team="team_null", team_alias=""
    )
    prometheus_logger.litellm_remaining_team_budget_metric.labels().set.assert_called_once_with(
        float("inf")
    )

    # Verify max budget metric is not set when max_budget is None
    prometheus_logger.litellm_team_max_budget_metric.assert_not_called()

    # Verify reset metric is not set when budget_reset_at is None
    prometheus_logger.litellm_team_budget_remaining_hours_metric.assert_not_called()


def test_set_team_budget_metrics_with_custom_labels(prometheus_logger, monkeypatch):
    """
    Test that _set_team_budget_metrics correctly handles custom prometheus labels
    """
    # Set custom prometheus labels
    custom_labels = ["metadata.organization", "metadata.environment"]
    monkeypatch.setattr("litellm.custom_prometheus_metadata_labels", custom_labels)

    # Create test team with custom metadata
    team = MagicMock(
        team_id="team1",
        team_alias="alias1",
        spend=50.0,
        max_budget=100.0,
        budget_reset_at=datetime(2024, 12, 31, tzinfo=timezone.utc),
    )

    # Mock the metrics
    prometheus_logger.litellm_remaining_team_budget_metric = MagicMock()
    prometheus_logger.litellm_team_max_budget_metric = MagicMock()
    prometheus_logger.litellm_team_budget_remaining_hours_metric = MagicMock()

    # Set metrics for the team
    prometheus_logger._set_team_budget_metrics(team)

    # Verify remaining budget metric includes custom labels
    prometheus_logger.litellm_remaining_team_budget_metric.labels.assert_called_once_with(
        team="team1",
        team_alias="alias1",
        metadata_organization=None,
        metadata_environment=None,
    )
    prometheus_logger.litellm_remaining_team_budget_metric.labels().set.assert_called_once_with(
        50.0
    )  # 100 - 50

    # Verify max budget metric includes custom labels
    prometheus_logger.litellm_team_max_budget_metric.labels.assert_called_once_with(
        team="team1",
        team_alias="alias1",
        metadata_organization=None,
        metadata_environment=None,
    )
    prometheus_logger.litellm_team_max_budget_metric.labels().set.assert_called_once_with(
        100.0
    )


def test_prometheus_label_factory_with_custom_tags(monkeypatch):
    """
    Test that prometheus_label_factory correctly handles custom tags
    """
    from litellm_enterprise.integrations.prometheus import (
        get_custom_labels_from_tags,
        prometheus_label_factory,
    )

    from litellm.types.integrations.prometheus import UserAPIKeyLabelValues

    # Set custom tags configuration
    monkeypatch.setattr("litellm.custom_prometheus_tags", ["prod", "test-env"])

    # Create enum_values with tags
    enum_values = UserAPIKeyLabelValues(
        hashed_api_key="key123",
        team="team1",
        tags=["prod", "debug"],  # Only "prod" is in our custom_prometheus_tags
    )

    # Test with supported labels including custom tags
    supported_labels = ["hashed_api_key", "team", "tag_prod", "tag_test_env"]

    result = prometheus_label_factory(
        supported_enum_labels=supported_labels,
        enum_values=enum_values,
    )

    expected = {
        "hashed_api_key": "key123",
        "team": "team1",
        "tag_prod": "true",  # present in tags
        "tag_test_env": "false",  # not present in tags
    }

    assert result == expected


def test_prometheus_label_factory_with_no_custom_tags(monkeypatch):
    """
    Test that prometheus_label_factory works when no custom tags are configured
    """
    from litellm_enterprise.integrations.prometheus import (
        get_custom_labels_from_tags,
        prometheus_label_factory,
    )

    from litellm.types.integrations.prometheus import UserAPIKeyLabelValues

    # Set empty custom tags configuration
    monkeypatch.setattr("litellm.custom_prometheus_tags", [])

    # Create enum_values with tags
    enum_values = UserAPIKeyLabelValues(
        hashed_api_key="key123",
        team="team1",
        tags=["prod", "debug"],
    )

    # Test with basic supported labels (no custom tags)
    supported_labels = ["hashed_api_key", "team"]

    result = prometheus_label_factory(
        supported_enum_labels=supported_labels,
        enum_values=enum_values,
    )

    expected = {
        "hashed_api_key": "key123",
        "team": "team1",
    }

    assert result == expected


def test_get_exception_class_name(prometheus_logger):
    """
    Test that _get_exception_class_name correctly formats the exception class name
    """
    # Test case 1: Exception with llm_provider
    rate_limit_error = litellm.RateLimitError(
        message="Rate limit exceeded", llm_provider="openai", model="gpt-3.5-turbo"
    )
    assert (
        prometheus_logger._get_exception_class_name(rate_limit_error)
        == "Openai.RateLimitError"
    )

    # Test case 2: Exception with empty llm_provider
    auth_error = litellm.AuthenticationError(
        message="Invalid API key", llm_provider="", model="gpt-4"
    )
    assert (
        prometheus_logger._get_exception_class_name(auth_error) == "AuthenticationError"
    )

    # Test case 3: Exception with None llm_provider
    context_window_error = litellm.ContextWindowExceededError(
        message="Context length exceeded", llm_provider=None, model="gpt-4"
    )
    assert (
        prometheus_logger._get_exception_class_name(context_window_error)
        == "ContextWindowExceededError"
    )


def test_set_llm_deployment_success_metrics_with_label_filtering():
    """
    Test that set_llm_deployment_success_metrics correctly uses prometheus_label_factory
    and respects label filtering configuration to prevent "Incorrect label names" errors.
    """
    from litellm.types.integrations.prometheus import PrometheusMetricsConfig

    # Create a prometheus logger with label filtering configuration
    config = [
        PrometheusMetricsConfig(
            group="test_group",
            metrics=[
                "litellm_overhead_latency_metric",
                "litellm_remaining_requests_metric",
                "litellm_remaining_tokens_metric",
                "litellm_deployment_success_responses",
                "litellm_deployment_total_requests",
            ],
            include_labels=[
                "litellm_model_name",
                "api_provider",
                "hashed_api_key",
            ],  # Limited labels
        )
    ]

    # Mock litellm.prometheus_metrics_config
    with patch("litellm.prometheus_metrics_config", config):
        # Clear registry before creating new logger
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            REGISTRY.unregister(collector)

        prometheus_logger = PrometheusLogger()

        # Mock all the metrics used in the method
        prometheus_logger.litellm_overhead_latency_metric = MagicMock()
        prometheus_logger.litellm_remaining_requests_metric = MagicMock()
        prometheus_logger.litellm_remaining_tokens_metric = MagicMock()
        prometheus_logger.litellm_deployment_success_responses = MagicMock()
        prometheus_logger.litellm_deployment_total_requests = MagicMock()
        prometheus_logger.set_deployment_healthy = MagicMock()

        # Create standard logging payload
        standard_logging_payload = create_standard_logging_payload()
        standard_logging_payload["hidden_params"]["additional_headers"] = {
            "x_ratelimit_remaining_requests": 123,
            "x_ratelimit_remaining_tokens": 4321,
        }
        standard_logging_payload["hidden_params"]["litellm_overhead_time_ms"] = 100

        # Create test data
        request_kwargs = {
            "model": "gpt-3.5-turbo",
            "litellm_params": {
                "custom_llm_provider": "openai",
                "metadata": {"model_info": {"id": "model-123"}},
            },
            "standard_logging_object": standard_logging_payload,
        }

        enum_values = UserAPIKeyLabelValues(
            litellm_model_name=standard_logging_payload["model"],
            api_provider=standard_logging_payload["custom_llm_provider"],
            hashed_api_key=standard_logging_payload["metadata"]["user_api_key_hash"],
            api_key_alias=standard_logging_payload["metadata"]["user_api_key_alias"],
            team=standard_logging_payload["metadata"]["user_api_key_team_id"],
            team_alias=standard_logging_payload["metadata"]["user_api_key_team_alias"],
            requested_model=standard_logging_payload["model_group"],
            model=standard_logging_payload["model"],
            model_id=standard_logging_payload["model_id"],
            api_base=standard_logging_payload["api_base"],
        )

        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=1)
        output_tokens = 10

        # Call the function - this should not raise "Incorrect label names" error
        prometheus_logger.set_llm_deployment_success_metrics(
            request_kwargs=request_kwargs,
            start_time=start_time,
            end_time=end_time,
            output_tokens=output_tokens,
            enum_values=enum_values,
        )

        # Verify that metrics were called with filtered labels (only the configured ones)
        # The exact labels depend on what get_labels_for_metric returns for each metric

        # Verify overhead latency metric was called with filtered labels
        prometheus_logger.litellm_overhead_latency_metric.labels.assert_called_once()
        overhead_labels = (
            prometheus_logger.litellm_overhead_latency_metric.labels.call_args[1]
        )

        # Should only contain the filtered labels that are supported for this metric
        expected_filtered_labels = {
            "litellm_model_name",
            "api_provider",
            "hashed_api_key",
        }
        actual_labels = set(k for k in overhead_labels.keys() if k is not None)

        # Verify that only expected labels are present (subset of configured labels)
        assert actual_labels <= expected_filtered_labels

        # Verify remaining requests metric was called with filtered labels
        prometheus_logger.litellm_remaining_requests_metric.labels.assert_called_once()
        requests_labels = (
            prometheus_logger.litellm_remaining_requests_metric.labels.call_args[1]
        )
        actual_labels = set(k for k in requests_labels.keys() if k is not None)
        assert actual_labels <= expected_filtered_labels

        # Verify remaining tokens metric was called with filtered labels
        prometheus_logger.litellm_remaining_tokens_metric.labels.assert_called_once()
        tokens_labels = (
            prometheus_logger.litellm_remaining_tokens_metric.labels.call_args[1]
        )
        actual_labels = set(k for k in tokens_labels.keys() if k is not None)
        assert actual_labels <= expected_filtered_labels

        # Verify deployment success responses metric was called with filtered labels
        prometheus_logger.litellm_deployment_success_responses.labels.assert_called_once()
        success_labels = (
            prometheus_logger.litellm_deployment_success_responses.labels.call_args[1]
        )
        actual_labels = set(k for k in success_labels.keys() if k is not None)
        assert actual_labels <= expected_filtered_labels

        # Verify deployment total requests metric was called with filtered labels
        prometheus_logger.litellm_deployment_total_requests.labels.assert_called_once()
        total_labels = (
            prometheus_logger.litellm_deployment_total_requests.labels.call_args[1]
        )
        actual_labels = set(total_labels.keys())
        assert actual_labels.issubset(expected_filtered_labels.union({None}))

        # Verify all metrics were actually called (no exceptions were raised)
        prometheus_logger.litellm_overhead_latency_metric.labels().observe.assert_called_once()
        prometheus_logger.litellm_remaining_requests_metric.labels().set.assert_called_once_with(
            123
        )
        prometheus_logger.litellm_remaining_tokens_metric.labels().set.assert_called_once_with(
            4321
        )
        prometheus_logger.litellm_deployment_success_responses.labels().inc.assert_called_once()
        prometheus_logger.litellm_deployment_total_requests.labels().inc.assert_called_once()


@pytest.mark.asyncio
async def test_prometheus_token_metrics_with_prometheus_config():
    """
    Test that validates the renamed token metrics are incremented correctly with a prometheus config.

    This test ensures that after the metric renaming (git diff):
    - litellm_total_tokens -> litellm_total_tokens_metric
    - litellm_input_tokens -> litellm_input_tokens_metric
    - litellm_output_tokens -> litellm_output_tokens_metric

    All three metrics should be properly incremented when making a successful completion request.
    """
    from prometheus_client import CollectorRegistry, Counter

    import litellm
    from litellm.types.integrations.prometheus import PrometheusMetricsConfig

    # Clear registry before test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)

    # Set up prometheus configuration that includes the token metrics
    config = [
        PrometheusMetricsConfig(
            group="token_metrics_test",
            metrics=[
                "litellm_total_tokens_metric",
                "litellm_input_tokens_metric",
                "litellm_output_tokens_metric",
                "litellm_requests_metric",
            ],
            include_labels=[
                "model",
                "hashed_api_key",
                "api_key_alias",
                "team",
                "team_alias",
            ],
        )
    ]

    # Mock litellm.prometheus_metrics_config
    with patch("litellm.prometheus_metrics_config", config):
        # Create PrometheusLogger with the configuration
        prometheus_logger = PrometheusLogger()

        # Test data with specific token counts
        standard_logging_payload = create_standard_logging_payload()
        standard_logging_payload["total_tokens"] = 1500
        standard_logging_payload["prompt_tokens"] = 900
        standard_logging_payload["completion_tokens"] = 600
        standard_logging_payload["response_cost"] = 0.075

        kwargs = {
            "model": "gpt-3.5-turbo",
            "stream": False,
            "litellm_params": {
                "metadata": {
                    "user_api_key": "test_key_hash",
                    "user_api_key_user_id": "test_user",
                    "user_api_key_team_id": "test_team",
                    "user_api_key_alias": "test_alias",
                    "user_api_key_team_alias": "test_team_alias",
                }
            },
            "start_time": datetime.now() - timedelta(seconds=2),
            "completion_start_time": datetime.now() - timedelta(seconds=1),
            "api_call_start_time": datetime.now() - timedelta(seconds=1.5),
            "end_time": datetime.now(),
            "standard_logging_object": standard_logging_payload,
        }
        response_obj = MagicMock()

        # Make the completion call through the logger
        await prometheus_logger.async_log_success_event(
            kwargs, response_obj, kwargs["start_time"], kwargs["end_time"]
        )

        await asyncio.sleep(2)

        print("final registry values", REGISTRY._collector_to_names)

        # Get metric collectors directly from registry
        metric_collectors = {}
        for collector, names in REGISTRY._collector_to_names.items():
            metric_name = names[0]  # First name is the base metric name
            metric_collectors[metric_name] = collector

        print("=== Final Metric Values (Direct Access) ===")

        # Expected values
        expected_values = {
            "litellm_total_tokens_metric": 1500.0,
            "litellm_input_tokens_metric": 900.0,
            "litellm_output_tokens_metric": 600.0,
            "litellm_requests_metric": 1.0,
        }

        expected_label_values = {
            "api_key_alias": "test_alias",
            "hashed_api_key": "test_hash",
            "model": "gpt-3.5-turbo",
            "team": "test_team",
            "team_alias": "test_team_alias",
        }

        # Validate each metric directly
        for metric_name, expected_value in expected_values.items():
            if metric_name in metric_collectors:
                collector = metric_collectors[metric_name]

                # Get all samples for this metric
                samples = list(collector.collect())[0].samples

                # Find the _total sample (the actual counter value)
                total_sample = None
                for sample in samples:
                    if sample.name.endswith("_total"):
                        total_sample = sample
                        break

                if total_sample:
                    actual_value = total_sample.value
                    actual_labels = total_sample.labels

                    print(
                        f"✓ {metric_name}: expected={expected_value}, actual={actual_value}"
                    )
                    print(f"  Labels: {actual_labels}")

                    # Validate the value
                    assert (
                        actual_value == expected_value
                    ), f"Expected {expected_value}, got {actual_value} for {metric_name}"

                    # Validate the labels
                    for (
                        label_key,
                        expected_label_value,
                    ) in expected_label_values.items():
                        actual_label_value = actual_labels.get(label_key)
                        assert (
                            actual_label_value == expected_label_value
                        ), f"Expected label {label_key}={expected_label_value}, got {actual_label_value}"

                    print(f"  ✓ {metric_name} VALIDATED")
                else:
                    raise AssertionError(f"No _total sample found for {metric_name}")
            else:
                raise AssertionError(f"Metric {metric_name} not found in registry")

        print("✓ All token metrics validated successfully!")

        # check final value of metrics in registry
