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

verbose_logger.setLevel(logging.DEBUG)

litellm.set_verbose = True
import time

prometheus_logger = PrometheusLogger()


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


def test_safe_get_remaining_budget():
    assert prometheus_logger._safe_get_remaining_budget(100, 30) == 70
    assert prometheus_logger._safe_get_remaining_budget(100, None) == 100
    assert prometheus_logger._safe_get_remaining_budget(None, 30) == float("inf")
    assert prometheus_logger._safe_get_remaining_budget(None, None) == float("inf")


@pytest.mark.asyncio
async def test_async_log_success_event():
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
