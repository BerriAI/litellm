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

verbose_logger.setLevel(logging.DEBUG)

litellm.set_verbose = True
import time


@pytest.mark.asyncio()
async def test_async_prometheus_success_logging():
    run_id = str(uuid.uuid4())
    litellm.set_verbose = True
    litellm.success_callback = ["prometheus"]
    litellm.failure_callback = ["prometheus"]

    response = await litellm.acompletion(
        model="claude-instant-1.2",
        messages=[{"role": "user", "content": "what llm are u"}],
        max_tokens=10,
        mock_response="hi",
        temperature=0.2,
        metadata={
            "id": run_id,
            "tags": ["tag1", "tag2"],
            "user_api_key": "6eb81e014497d89f3cc1aa9da7c2b37bda6b7fea68e4b710d33d94201e68970c",
            "user_api_key_alias": "ishaans-prometheus-key",
            "user_api_end_user_max_budget": None,
            "litellm_api_version": "1.40.19",
            "global_max_parallel_requests": None,
            "user_api_key_user_id": "admin",
            "user_api_key_org_id": None,
            "user_api_key_team_id": "dbe2f686-a686-4896-864a-4c3924458709",
            "user_api_key_team_alias": "testing-team",
        },
    )
    print(response)
    await asyncio.sleep(3)

    # get prometheus logger
    from litellm.litellm_core_utils.litellm_logging import _in_memory_loggers

    for callback in _in_memory_loggers:
        if isinstance(callback, PrometheusLogger):
            test_prometheus_logger = callback

    print("done with success request")

    print(
        "vars of test_prometheus_logger",
        vars(test_prometheus_logger.litellm_requests_metric),
    )

    # Get the metrics
    metrics = {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            metrics[sample.name] = sample.value

    print("metrics from prometheus", metrics)
    assert metrics["litellm_requests_metric_total"] == 1.0
    assert metrics["litellm_total_tokens_total"] == 30.0
    assert metrics["litellm_deployment_success_responses_total"] == 1.0
    assert metrics["litellm_deployment_total_requests_total"] == 1.0
    assert metrics["litellm_deployment_latency_per_output_token_bucket"] == 1.0


@pytest.mark.asyncio()
async def test_async_prometheus_success_logging_with_callbacks():
    run_id = str(uuid.uuid4())
    litellm.set_verbose = True

    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = ["prometheus"]

    # Get initial metric values
    initial_metrics = {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            initial_metrics[sample.name] = sample.value

    response = await litellm.acompletion(
        model="claude-instant-1.2",
        messages=[{"role": "user", "content": "what llm are u"}],
        max_tokens=10,
        mock_response="hi",
        temperature=0.2,
        metadata={
            "id": run_id,
            "tags": ["tag1", "tag2"],
            "user_api_key": "6eb81e014497d89f3cc1aa9da7c2b37bda6b7fea68e4b710d33d94201e68970c",
            "user_api_key_alias": "ishaans-prometheus-key",
            "user_api_end_user_max_budget": None,
            "litellm_api_version": "1.40.19",
            "global_max_parallel_requests": None,
            "user_api_key_user_id": "admin",
            "user_api_key_org_id": None,
            "user_api_key_team_id": "dbe2f686-a686-4896-864a-4c3924458709",
            "user_api_key_team_alias": "testing-team",
        },
    )
    print(response)
    await asyncio.sleep(3)

    # get prometheus logger
    from litellm.litellm_core_utils.litellm_logging import _in_memory_loggers

    for callback in _in_memory_loggers:
        if isinstance(callback, PrometheusLogger):
            test_prometheus_logger = callback

    print("done with success request")

    print(
        "vars of test_prometheus_logger",
        vars(test_prometheus_logger.litellm_requests_metric),
    )

    # Get the updated metrics
    updated_metrics = {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            updated_metrics[sample.name] = sample.value

    print("metrics from prometheus", updated_metrics)

    # Assert the delta for each metric
    assert (
        updated_metrics["litellm_requests_metric_total"]
        - initial_metrics.get("litellm_requests_metric_total", 0)
        == 1.0
    )
    assert (
        updated_metrics["litellm_total_tokens_total"]
        - initial_metrics.get("litellm_total_tokens_total", 0)
        == 30.0
    )
    assert (
        updated_metrics["litellm_deployment_success_responses_total"]
        - initial_metrics.get("litellm_deployment_success_responses_total", 0)
        == 1.0
    )
    assert (
        updated_metrics["litellm_deployment_total_requests_total"]
        - initial_metrics.get("litellm_deployment_total_requests_total", 0)
        == 1.0
    )
    assert (
        updated_metrics["litellm_deployment_latency_per_output_token_bucket"]
        - initial_metrics.get("litellm_deployment_latency_per_output_token_bucket", 0)
        == 1.0
    )
