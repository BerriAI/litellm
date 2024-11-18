"""
Unit tests for prometheus metrics
"""

import pytest
import aiohttp
import asyncio
import uuid
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


async def make_bad_chat_completion_request(session, key):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "fake-azure-endpoint",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        return status, response_text


async def make_good_chat_completion_request(session, key):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "fake-openai-endpoint",
        "messages": [{"role": "user", "content": f"Hello {uuid.uuid4()}"}],
        "tags": ["teamB"],
    }
    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        return status, response_text


async def make_chat_completion_request_with_fallback(session, key):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "fake-azure-endpoint",
        "messages": [{"role": "user", "content": "Hello"}],
        "fallbacks": ["fake-openai-endpoint"],
    }
    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

    # make a request with a failed fallback
    data = {
        "model": "fake-azure-endpoint",
        "messages": [{"role": "user", "content": "Hello"}],
        "fallbacks": ["unknown-model"],
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

    return


@pytest.mark.asyncio
async def test_proxy_failure_metrics():
    """
    - Make 1 bad chat completion call to "fake-azure-endpoint"
    - GET /metrics
    - assert the failure metric for the requested model is incremented by 1
    - Assert the Exception class and status code are correct
    """
    async with aiohttp.ClientSession() as session:
        # Make a bad chat completion call
        status, response_text = await make_bad_chat_completion_request(
            session, "sk-1234"
        )

        # Check if the request failed as expected
        assert status == 429, f"Expected status 429, but got {status}"

        # Get metrics
        async with session.get("http://0.0.0.0:4000/metrics") as response:
            metrics = await response.text()

        print("/metrics", metrics)

        # Check if the failure metric is present and correct
        expected_metric = 'litellm_proxy_failed_requests_metric_total{api_key_alias="None",end_user="None",exception_class="RateLimitError",exception_status="429",hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",requested_model="fake-azure-endpoint",team="None",team_alias="None",user="default_user_id"} 1.0'

        assert (
            expected_metric in metrics
        ), "Expected failure metric not found in /metrics"
        expected_llm_deployment_failure = 'litellm_deployment_failure_responses_total{api_base="https://exampleopenaiendpoint-production.up.railway.app",api_provider="openai",exception_class="RateLimitError",exception_status="429",litellm_model_name="429",model_id="7499d31f98cd518cf54486d5a00deda6894239ce16d13543398dc8abf870b15f",requested_model="fake-azure-endpoint"} 1.0'
        assert expected_llm_deployment_failure

        assert (
            'litellm_proxy_total_requests_metric_total{api_key_alias="None",end_user="None",hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",requested_model="fake-azure-endpoint",team="None",team_alias="None",user="default_user_id"} 1.0'
            in metrics
        )

        assert (
            'litellm_deployment_failure_responses_total{api_base="https://exampleopenaiendpoint-production.up.railway.app",api_key_alias="None",api_provider="openai",exception_class="RateLimitError",exception_status="429",hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",litellm_model_name="429",model_id="7499d31f98cd518cf54486d5a00deda6894239ce16d13543398dc8abf870b15f",requested_model="fake-azure-endpoint",team="None",team_alias="None"}'
            in metrics
        )


@pytest.mark.asyncio
async def test_proxy_success_metrics():
    """
    Make 1 good /chat/completions call to "openai/gpt-3.5-turbo"
    GET /metrics
    Assert the success metric is incremented by 1
    """

    async with aiohttp.ClientSession() as session:
        # Make a good chat completion call
        status, response_text = await make_good_chat_completion_request(
            session, "sk-1234"
        )

        # Check if the request succeeded as expected
        assert status == 200, f"Expected status 200, but got {status}"

        # Get metrics
        async with session.get("http://0.0.0.0:4000/metrics") as response:
            metrics = await response.text()

        print("/metrics", metrics)

        # Check if the success metric is present and correct
        assert (
            'litellm_request_total_latency_metric_bucket{api_key_alias="None",hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",le="0.005",model="fake",team="None",team_alias="None"}'
            in metrics
        )

        assert (
            'litellm_llm_api_latency_metric_bucket{api_key_alias="None",hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",le="0.005",model="fake",team="None",team_alias="None"}'
            in metrics
        )

        # assert (
        #     'litellm_deployment_latency_per_output_token_count{api_base="https://exampleopenaiendpoint-production.up.railway.app/",api_key_alias="None",api_provider="openai",hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",litellm_model_name="fake",model_id="team-b-model",team="None",team_alias="None"}'
        #     in metrics
        # )

        verify_latency_metrics(metrics)


def verify_latency_metrics(metrics: str):
    """
    Assert that LATENCY_BUCKETS distribution is used for
    - litellm_request_total_latency_metric_bucket
    - litellm_llm_api_latency_metric_bucket
    """
    from litellm.types.integrations.prometheus import LATENCY_BUCKETS
    import re

    metric_names = [
        "litellm_request_total_latency_metric_bucket",
        "litellm_llm_api_latency_metric_bucket",
    ]

    for metric_name in metric_names:
        # Extract all 'le' values for the current metric
        pattern = rf'{metric_name}{{.*?le="(.*?)".*?}}'
        le_values = re.findall(pattern, metrics)

        # Convert to set for easier comparison
        actual_buckets = set(le_values)

        print("actual_buckets", actual_buckets)
        expected_buckets = []
        for bucket in LATENCY_BUCKETS:
            expected_buckets.append(str(bucket))

        # replace inf with +Inf
        expected_buckets = [
            bucket.replace("inf", "+Inf") for bucket in expected_buckets
        ]

        print("expected_buckets", expected_buckets)
        expected_buckets = set(expected_buckets)
        # Verify all expected buckets are present
        assert (
            actual_buckets == expected_buckets
        ), f"Mismatch in {metric_name} buckets. Expected: {expected_buckets}, Got: {actual_buckets}"


@pytest.mark.asyncio
async def test_proxy_fallback_metrics():
    """
    Make 1 request with a client side fallback - check metrics
    """

    async with aiohttp.ClientSession() as session:
        # Make a good chat completion call
        await make_chat_completion_request_with_fallback(session, "sk-1234")

        # Get metrics
        async with session.get("http://0.0.0.0:4000/metrics") as response:
            metrics = await response.text()

        print("/metrics", metrics)

        # Check if successful fallback metric is incremented
        assert (
            'litellm_deployment_successful_fallbacks_total{api_key_alias="None",exception_class="RateLimitError",exception_status="429",fallback_model="fake-openai-endpoint",hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",requested_model="fake-azure-endpoint",team="None",team_alias="None"} 1.0'
            in metrics
        )

        # Check if failed fallback metric is incremented
        assert (
            'litellm_deployment_failed_fallbacks_total{api_key_alias="None",exception_class="RateLimitError",exception_status="429",fallback_model="unknown-model",hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",requested_model="fake-azure-endpoint",team="None",team_alias="None"} 1.0'
            in metrics
        )
