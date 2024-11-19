import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import logging
import uuid

import pytest
from prometheus_client import REGISTRY, CollectorRegistry

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.prometheus import PrometheusLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import (
    StandardLoggingPayload,
    StandardLoggingMetadata,
    StandardLoggingHiddenParams,
    StandardLoggingModelInformation,
)
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth

verbose_logger.setLevel(logging.DEBUG)

litellm.set_verbose = True
import time


@pytest.fixture
def prometheus_logger():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def create_standard_logging_payload() -> StandardLoggingPayload:
    return StandardLoggingPayload(
        id="test_id",
        call_type="completion",
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
        api_base="https://api.openai.com",
        metadata=StandardLoggingMetadata(
            user_api_key_hash="test_hash",
            user_api_key_alias="test_alias",
            user_api_key_team_id="test_team",
            user_api_key_user_id="test_user",
            user_api_key_team_alias="test_team_alias",
            user_api_key_org_id=None,
            spend_logs_metadata=None,
            requester_ip_address="127.0.0.1",
            requester_metadata=None,
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
        "litellm_params": {
            "metadata": {
                "user_api_key": "test_key",
                "user_api_key_user_id": "test_user",
                "user_api_key_team_id": "test_team",
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

    prometheus_logger._increment_token_metrics(
        standard_logging_payload,
        end_user_id="user1",
        user_api_key="key1",
        user_api_key_alias="alias1",
        model="gpt-3.5-turbo",
        user_api_team="team1",
        user_api_team_alias="team_alias1",
        user_id="user1",
    )

    prometheus_logger.litellm_tokens_metric.labels.assert_called_once_with(
        "user1", "key1", "alias1", "gpt-3.5-turbo", "team1", "team_alias1", "user1"
    )
    prometheus_logger.litellm_tokens_metric.labels().inc.assert_called_once_with(100)

    prometheus_logger.litellm_input_tokens_metric.labels.assert_called_once_with(
        "user1", "key1", "alias1", "gpt-3.5-turbo", "team1", "team_alias1", "user1"
    )
    prometheus_logger.litellm_input_tokens_metric.labels().inc.assert_called_once_with(
        50
    )

    prometheus_logger.litellm_output_tokens_metric.labels.assert_called_once_with(
        "user1", "key1", "alias1", "gpt-3.5-turbo", "team1", "team_alias1", "user1"
    )
    prometheus_logger.litellm_output_tokens_metric.labels().inc.assert_called_once_with(
        50
    )


def test_increment_remaining_budget_metrics(prometheus_logger):
    """
    Test the increment_remaining_budget_metrics method

    team and api key budget metrics are set to the difference between max budget and spend
    """
    prometheus_logger.litellm_remaining_team_budget_metric = MagicMock()
    prometheus_logger.litellm_remaining_api_key_budget_metric = MagicMock()

    litellm_params = {
        "metadata": {
            "user_api_key_team_spend": 50,
            "user_api_key_team_max_budget": 100,
            "user_api_key_spend": 25,
            "user_api_key_max_budget": 75,
        }
    }

    prometheus_logger._increment_remaining_budget_metrics(
        user_api_team="team1",
        user_api_team_alias="team_alias1",
        user_api_key="key1",
        user_api_key_alias="alias1",
        litellm_params=litellm_params,
    )

    prometheus_logger.litellm_remaining_team_budget_metric.labels.assert_called_once_with(
        "team1", "team_alias1"
    )
    prometheus_logger.litellm_remaining_team_budget_metric.labels().set.assert_called_once_with(
        50
    )

    prometheus_logger.litellm_remaining_api_key_budget_metric.labels.assert_called_once_with(
        "key1", "alias1"
    )
    prometheus_logger.litellm_remaining_api_key_budget_metric.labels().set.assert_called_once_with(
        50
    )


def test_set_latency_metrics(prometheus_logger):
    """
    Test the set_latency_metrics method

    time to first token, llm api latency, and request total latency metrics are set to the values in the standard logging payload
    """
    standard_logging_payload = create_standard_logging_payload()
    standard_logging_payload["model_parameters"] = {"stream": True}
    prometheus_logger.litellm_llm_api_time_to_first_token_metric = MagicMock()
    prometheus_logger.litellm_llm_api_latency_metric = MagicMock()
    prometheus_logger.litellm_request_total_latency_metric = MagicMock()

    now = datetime.now()
    kwargs = {
        "end_time": now,  # when the request ends
        "start_time": now - timedelta(seconds=2),  # when the request starts
        "api_call_start_time": now - timedelta(seconds=1.5),  # when the api call starts
        "completion_start_time": now
        - timedelta(seconds=1),  # when the completion starts
    }

    prometheus_logger._set_latency_metrics(
        kwargs=kwargs,
        model="gpt-3.5-turbo",
        user_api_key="key1",
        user_api_key_alias="alias1",
        user_api_team="team1",
        user_api_team_alias="team_alias1",
        standard_logging_payload=standard_logging_payload,
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
        "gpt-3.5-turbo", "key1", "alias1", "team1", "team_alias1"
    )
    prometheus_logger.litellm_llm_api_latency_metric.labels().observe.assert_called_once_with(
        1.5
    )

    # total latency for the request
    prometheus_logger.litellm_request_total_latency_metric.labels.assert_called_once_with(
        "gpt-3.5-turbo", "key1", "alias1", "team1", "team_alias1"
    )
    prometheus_logger.litellm_request_total_latency_metric.labels().observe.assert_called_once_with(
        2.0
    )


def test_increment_top_level_request_and_spend_metrics(prometheus_logger):
    """
    Test the increment_top_level_request_and_spend_metrics method

    - litellm_requests_metric is incremented by 1
    - litellm_spend_metric is incremented by the response cost in the standard logging payload
    """
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
    )

    prometheus_logger.litellm_requests_metric.labels.assert_called_once_with(
        "user1", "key1", "alias1", "gpt-3.5-turbo", "team1", "team_alias1", "user1"
    )
    prometheus_logger.litellm_requests_metric.labels().inc.assert_called_once()

    prometheus_logger.litellm_spend_metric.labels.assert_called_once_with(
        "user1", "key1", "alias1", "gpt-3.5-turbo", "team1", "team_alias1", "user1"
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
    )

    # Call the function
    await prometheus_logger.async_post_call_failure_hook(
        request_data=request_data,
        original_exception=original_exception,
        user_api_key_dict=user_api_key_dict,
    )

    # Assert failed requests metric was incremented with correct labels
    prometheus_logger.litellm_proxy_failed_requests_metric.labels.assert_called_once_with(
        end_user="test_end_user",
        hashed_api_key="test_key",
        api_key_alias="test_alias",
        requested_model="gpt-3.5-turbo",
        team="test_team",
        team_alias="test_team_alias",
        user="test_user",
        exception_status=429,
        exception_class="RateLimitError",
    )
    prometheus_logger.litellm_proxy_failed_requests_metric.labels().inc.assert_called_once()

    # Assert total requests metric was incremented with correct labels
    prometheus_logger.litellm_proxy_total_requests_metric.labels.assert_called_once_with(
        "test_end_user",
        "test_key",
        "test_alias",
        "gpt-3.5-turbo",
        "test_team",
        "test_team_alias",
        "test_user",
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
    )

    response = {"choices": [{"message": {"content": "test response"}}]}

    # Call the function
    await prometheus_logger.async_post_call_success_hook(
        data=data, user_api_key_dict=user_api_key_dict, response=response
    )

    # Assert total requests metric was incremented with correct labels
    prometheus_logger.litellm_proxy_total_requests_metric.labels.assert_called_once_with(
        "test_end_user",
        "test_key",
        "test_alias",
        "gpt-3.5-turbo",
        "test_team",
        "test_team_alias",
        "test_user",
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

    standard_logging_payload = create_standard_logging_payload()

    standard_logging_payload["hidden_params"]["additional_headers"] = {
        "x_ratelimit_remaining_requests": 123,
        "x_ratelimit_remaining_tokens": 4321,
    }

    # Create test data
    request_kwargs = {
        "model": "gpt-3.5-turbo",
        "litellm_params": {
            "custom_llm_provider": "openai",
            "metadata": {"model_info": {"id": "model-123"}},
        },
        "standard_logging_object": standard_logging_payload,
    }

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)
    output_tokens = 10

    # Call the function
    prometheus_logger.set_llm_deployment_success_metrics(
        request_kwargs=request_kwargs,
        start_time=start_time,
        end_time=end_time,
        output_tokens=output_tokens,
    )

    # Verify remaining requests metric
    prometheus_logger.litellm_remaining_requests_metric.labels.assert_called_once_with(
        "openai-gpt",  # model_group / requested model from create_standard_logging_payload()
        "openai",  # llm provider
        "https://api.openai.com",  # api base
        "gpt-3.5-turbo",  # actual model used - litellm model name
        standard_logging_payload["metadata"]["user_api_key_hash"],
        standard_logging_payload["metadata"]["user_api_key_alias"],
    )
    prometheus_logger.litellm_remaining_requests_metric.labels().set.assert_called_once_with(
        123
    )

    # Verify remaining tokens metric
    prometheus_logger.litellm_remaining_tokens_metric.labels.assert_called_once_with(
        "openai-gpt",  # model_group / requested model from create_standard_logging_payload()
        "openai",  # llm provider
        "https://api.openai.com",  # api base
        "gpt-3.5-turbo",  # actual model used - litellm model name
        standard_logging_payload["metadata"]["user_api_key_hash"],
        standard_logging_payload["metadata"]["user_api_key_alias"],
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
        requested_model="openai-gpt",  # requested model from create_standard_logging_payload()
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
        requested_model="openai-gpt",  # requested model from create_standard_logging_payload()
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
        exception_class="RateLimitError",
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
        exception_class="RateLimitError",
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
        test_params["litellm_model_name"],
        test_params["model_id"],
        test_params["api_base"],
        test_params["api_provider"],
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
