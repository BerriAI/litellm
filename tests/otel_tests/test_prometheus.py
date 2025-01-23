"""
Unit tests for prometheus metrics
"""

import pytest
import aiohttp
import asyncio
import uuid
import os
import sys
from openai import AsyncOpenAI
import time
from typing import Dict, Any

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

END_USER_ID = "my-test-user-34"


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
        "user": END_USER_ID,  # test if disable end user tracking for prometheus works
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
            'litellm_proxy_total_requests_metric_total{api_key_alias="None",end_user="None",hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",requested_model="fake-azure-endpoint",status_code="429",team="None",team_alias="None",user="default_user_id"} 1.0'
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

        assert END_USER_ID not in metrics

        # Check if the success metric is present and correct
        assert (
            'litellm_request_total_latency_metric_bucket{api_key_alias="None",end_user="None",hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",le="0.005",model="fake",requested_model="fake-openai-endpoint",team="None",team_alias="None",user="default_user_id"}'
            in metrics
        )

        assert (
            'litellm_llm_api_latency_metric_bucket{api_key_alias="None",end_user="None",hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",le="0.005",model="fake",requested_model="fake-openai-endpoint",team="None",team_alias="None",user="default_user_id"}'
            in metrics
        )

        verify_latency_metrics(metrics)


def verify_latency_metrics(metrics: str):
    """
    Assert that LATENCY_BUCKETS distribution is used for
    - litellm_request_total_latency_metric_bucket
    - litellm_llm_api_latency_metric_bucket

    Very important to verify that the overhead latency metric is present
    """
    from litellm.types.integrations.prometheus import LATENCY_BUCKETS
    import re
    import time

    time.sleep(2)

    metric_names = [
        "litellm_request_total_latency_metric_bucket",
        "litellm_llm_api_latency_metric_bucket",
        "litellm_overhead_latency_metric_bucket",
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


async def create_test_team(
    session: aiohttp.ClientSession, team_data: Dict[str, Any]
) -> str:
    """Create a new team and return the team_id"""
    url = "http://0.0.0.0:4000/team/new"
    headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json",
    }

    async with session.post(url, headers=headers, json=team_data) as response:
        assert (
            response.status == 200
        ), f"Failed to create team. Status: {response.status}"
        team_info = await response.json()
        return team_info["team_id"]


async def get_prometheus_metrics(session: aiohttp.ClientSession) -> str:
    """Fetch current prometheus metrics"""
    async with session.get("http://0.0.0.0:4000/metrics") as response:
        assert response.status == 200
        return await response.text()


def extract_budget_metrics(metrics_text: str, team_id: str) -> Dict[str, float]:
    """Extract budget-related metrics for a specific team"""
    import re

    metrics = {}

    # Get remaining budget
    remaining_pattern = f'litellm_remaining_team_budget_metric{{team="{team_id}",team_alias="[^"]*"}} ([0-9.]+)'
    remaining_match = re.search(remaining_pattern, metrics_text)
    metrics["remaining"] = float(remaining_match.group(1)) if remaining_match else None

    # Get total budget
    total_pattern = f'litellm_team_max_budget_metric{{team="{team_id}",team_alias="[^"]*"}} ([0-9.]+)'
    total_match = re.search(total_pattern, metrics_text)
    metrics["total"] = float(total_match.group(1)) if total_match else None

    return metrics


async def create_test_key(session: aiohttp.ClientSession, team_id: str) -> str:
    """Generate a new key for the team and return it"""
    url = "http://0.0.0.0:4000/key/generate"
    headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json",
    }
    data = {
        "team_id": team_id,
    }

    async with session.post(url, headers=headers, json=data) as response:
        assert (
            response.status == 200
        ), f"Failed to generate key. Status: {response.status}"
        key_info = await response.json()
        return key_info["key"]


async def get_team_info(session: aiohttp.ClientSession, team_id: str) -> Dict[str, Any]:
    """Fetch team info and return the response"""
    url = f"http://0.0.0.0:4000/team/info?team_id={team_id}"
    headers = {
        "Authorization": "Bearer sk-1234",
    }

    async with session.get(url, headers=headers) as response:
        assert (
            response.status == 200
        ), f"Failed to get team info. Status: {response.status}"
        return await response.json()


@pytest.mark.asyncio
async def test_team_budget_metrics():
    """
    Test team budget tracking metrics:
    1. Create a team with max_budget
    2. Generate a key for the team
    3. Make chat completion requests using OpenAI SDK with team's key
    4. Verify budget decreases over time
    5. Verify request costs are being tracked correctly
    6. Verify prometheus metrics match /team/info spend data
    """
    async with aiohttp.ClientSession() as session:
        # Setup test team
        team_data = {
            "team_alias": "budget_test_team",
            "max_budget": 10,
            "budget_duration": "7d",
        }
        team_id = await create_test_team(session, team_data)
        print("team_id", team_id)
        # Generate key for the team
        team_key = await create_test_key(session, team_id)

        # Initialize OpenAI client with team's key
        client = AsyncOpenAI(base_url="http://0.0.0.0:4000", api_key=team_key)

        # Make initial request and check budget
        await client.chat.completions.create(
            model="fake-openai-endpoint",
            messages=[{"role": "user", "content": f"Hello {uuid.uuid4()}"}],
        )

        await asyncio.sleep(11)  # Wait for metrics to update

        # Get metrics after request
        metrics_after_first = await get_prometheus_metrics(session)
        print("metrics_after_first", metrics_after_first)
        first_budget = extract_budget_metrics(metrics_after_first, team_id)

        print(f"Budget after 1 request: {first_budget}")
        assert (
            first_budget["remaining"] < 10.0
        ), "remaining budget should be less than 10.0 after first request"
        assert first_budget["total"] == 10.0, "Total budget metric is incorrect"

        # Get team info and verify spend matches prometheus metrics
        team_info = await get_team_info(session, team_id)
        print("team_info", team_info)
        _team_info_data = team_info["team_info"]

        # Calculate spend from prometheus (total - remaining)
        team_info_spend = float(_team_info_data["spend"])
        team_info_max_budget = float(_team_info_data["max_budget"])
        team_info_remaining_budget = team_info_max_budget - team_info_spend
        print("\n\n\n###### Final budget metrics ######\n\n\n")
        print("team_info_remaining_budget", team_info_remaining_budget)
        print("prometheus_remaining_budget", first_budget["remaining"])
        print(
            "diff between team_info_remaining_budget and prometheus_remaining_budget",
            team_info_remaining_budget - first_budget["remaining"],
        )

        # Verify spends match within a small delta (floating point comparison)
        assert (
            abs(team_info_remaining_budget - first_budget["remaining"]) <= 0.00000
        ), f"Spend mismatch: Prometheus={team_info_remaining_budget}, Team Info={first_budget['remaining']}"
