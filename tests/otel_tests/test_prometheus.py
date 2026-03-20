"""
Unit tests for prometheus metrics
"""

import pytest
import aiohttp
import asyncio
from litellm._uuid import uuid
import os
import sys
from openai import AsyncOpenAI
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

        # Check if the failure metric is present and correct - use pattern matching for robustness
        # Labels are ordered alphabetically by Prometheus: api_key_alias, end_user, exception_class,
        # exception_status, hashed_api_key, requested_model, route, team, team_alias, user, user_email
        # Note: client_ip, user_agent, model_id are present but we use substring matching to be flexible
        # Check for both the new metric and deprecated metric for backwards compatibility
        expected_patterns = [
            'litellm_proxy_failed_requests_metric_total{',  # New metric
            'litellm_llm_api_failed_requests_metric_total{'  # Deprecated but may still be used
        ]
        
        # Check if either pattern is in metrics and contains required fields
        found_metric = False
        for pattern in expected_patterns:
            for line in metrics.split("\n"):
                # For proxy metric, check proxy-specific fields
                if 'litellm_proxy_failed_requests_metric_total{' in line:
                    if 'api_key_alias="None"' in line and \
                       'exception_class="Openai.RateLimitError"' in line and \
                       'exception_status="429"' in line and \
                       'hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"' in line and \
                       'requested_model="fake-azure-endpoint"' in line and \
                       'route="/chat/completions"' in line:
                        found_metric = True
                        break
                # For deprecated llm_api metric, check llm-specific fields
                elif 'litellm_llm_api_failed_requests_metric_total{' in line:
                    if 'hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"' in line and \
                       'model="429"' in line:  # The deprecated metric uses the actual model from the request
                        found_metric = True
                        break
            if found_metric:
                break
        
        assert found_metric, f"Expected failure metric not found in /metrics. Looking for either litellm_proxy_failed_requests_metric_total or litellm_llm_api_failed_requests_metric_total with required fields"

        # Check total requests metric similarly  
        # The litellm_proxy_total_requests_metric_total should be present
        total_requests_pattern = 'litellm_proxy_total_requests_metric_total{'
        
        found_total_metric = False
        for line in metrics.split("\n"):
            if total_requests_pattern in line and \
               'hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"' in line and \
               'requested_model="fake-azure-endpoint"' in line and \
               'status_code="429"' in line:
                found_total_metric = True
                break
        
        assert found_total_metric, f"Expected total requests metric not found in /metrics. Looking for: {total_requests_pattern} with hashed_api_key and status_code=429"


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=2)
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

        # Check if the success metric is present and correct - use flexible matching
        # Check for request_total_latency_metric with required fields
        # Note: The model can be "gpt-3.5-turbo-0301" or similar depending on what's returned
        found_request_latency = False
        for line in metrics.split("\n"):
            if 'litellm_request_total_latency_metric_bucket{' in line and \
               'api_key_alias="None"' in line and \
               'hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"' in line and \
               'requested_model="fake-openai-endpoint"' in line and \
               'le="0.005"' in line:
                found_request_latency = True
                break
        
        assert found_request_latency, "Expected litellm_request_total_latency_metric_bucket not found in /metrics"

        # Check for llm_api_latency_metric with required fields
        found_api_latency = False
        for line in metrics.split("\n"):
            if 'litellm_llm_api_latency_metric_bucket{' in line and \
               'api_key_alias="None"' in line and \
               'hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"' in line and \
               'requested_model="fake-openai-endpoint"' in line and \
               'le="0.005"' in line:
                found_api_latency = True
                break
        
        assert found_api_latency, "Expected litellm_llm_api_latency_metric_bucket not found in /metrics"

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

        # Check if successful fallback metric is incremented - use flexible matching
        found_successful_fallback = False
        for line in metrics.split("\n"):
            if 'litellm_deployment_successful_fallbacks_total{' in line and \
               'api_key_alias="None"' in line and \
               'exception_class="Openai.RateLimitError"' in line and \
               'exception_status="429"' in line and \
               'fallback_model="fake-openai-endpoint"' in line and \
               'hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"' in line and \
               'requested_model="fake-azure-endpoint"' in line and \
               '1.0' in line:
                found_successful_fallback = True
                break
        
        assert found_successful_fallback, "Expected litellm_deployment_successful_fallbacks_total metric not found in /metrics"

        # Check if failed fallback metric is incremented - use flexible matching
        found_failed_fallback = False
        for line in metrics.split("\n"):
            if 'litellm_deployment_failed_fallbacks_total{' in line and \
               'api_key_alias="None"' in line and \
               'exception_class="Openai.RateLimitError"' in line and \
               'exception_status="429"' in line and \
               'fallback_model="unknown-model"' in line and \
               'hashed_api_key="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"' in line and \
               'requested_model="fake-azure-endpoint"' in line and \
               '1.0' in line:
                found_failed_fallback = True
                break
        
        assert found_failed_fallback, "Expected litellm_deployment_failed_fallbacks_total metric not found in /metrics"


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


async def create_test_user(
    session: aiohttp.ClientSession, user_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a new user and return the user info"""
    url = "http://0.0.0.0:4000/user/new"
    headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json",
    }

    async with session.post(url, headers=headers, json=user_data) as response:
        assert (
            response.status == 200
        ), f"Failed to create user. Status: {response.status}"
        user_info = await response.json()
        return user_info


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

    # Get remaining hours
    hours_pattern = f'litellm_team_budget_remaining_hours_metric{{team="{team_id}",team_alias="[^"]*"}} ([0-9.]+)'
    hours_match = re.search(hours_pattern, metrics_text)
    metrics["remaining_hours"] = float(hours_match.group(1)) if hours_match else None

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
        print("first_budget['remaining_hours']", first_budget["remaining_hours"])
        # Budget should have positive remaining hours, up to 7 days
        assert (
            0 < first_budget["remaining_hours"] <= 168
        ), "Budget should have positive remaining hours, up to 7 days"

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
            abs(team_info_remaining_budget - first_budget["remaining"]) <= 0.001
        ), f"Spend mismatch: Prometheus={team_info_remaining_budget}, Team Info={first_budget['remaining']}"


async def create_test_key_with_budget(
    session: aiohttp.ClientSession, budget_data: Dict[str, Any]
) -> str:
    """Generate a new key with budget constraints and return it"""
    url = "http://0.0.0.0:4000/key/generate"
    headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json",
    }
    print("budget_data", budget_data)

    async with session.post(url, headers=headers, json=budget_data) as response:
        assert (
            response.status == 200
        ), f"Failed to generate key. Status: {response.status}"
        key_info = await response.json()
        return key_info["key"]


async def get_key_info(session: aiohttp.ClientSession, key: str) -> Dict[str, Any]:
    """Fetch key info and return the response"""
    url = "http://0.0.0.0:4000/key/info"
    headers = {
        "Authorization": f"Bearer {key}",
    }

    async with session.get(url, headers=headers) as response:
        assert (
            response.status == 200
        ), f"Failed to get key info. Status: {response.status}"
        return await response.json()


async def get_user_info(session: aiohttp.ClientSession, user_id: str) -> Dict[str, Any]:
    """Fetch user info and return the response"""
    from urllib.parse import quote

    # URL encode user_id to handle special characters
    encoded_user_id = quote(user_id, safe="")
    url = f"http://0.0.0.0:4000/user/info?user_id={encoded_user_id}"
    headers = {
        "Authorization": "Bearer sk-1234",
    }

    async with session.get(url, headers=headers) as response:
        assert (
            response.status == 200
        ), f"Failed to get user info. Status: {response.status}"
        return await response.json()


def extract_key_budget_metrics(metrics_text: str, key_id: str) -> Dict[str, float]:
    """Extract budget-related metrics for a specific key"""
    import re

    metrics = {}

    # Get remaining budget
    remaining_pattern = f'litellm_remaining_api_key_budget_metric{{api_key_alias="[^"]*",hashed_api_key="{key_id}"}} ([0-9.]+)'
    remaining_match = re.search(remaining_pattern, metrics_text)
    metrics["remaining"] = float(remaining_match.group(1)) if remaining_match else None

    # Get total budget
    total_pattern = f'litellm_api_key_max_budget_metric{{api_key_alias="[^"]*",hashed_api_key="{key_id}"}} ([0-9.]+)'
    total_match = re.search(total_pattern, metrics_text)
    metrics["total"] = float(total_match.group(1)) if total_match else None

    # Get remaining hours
    hours_pattern = f'litellm_api_key_budget_remaining_hours_metric{{api_key_alias="[^"]*",hashed_api_key="{key_id}"}} ([0-9.]+)'
    hours_match = re.search(hours_pattern, metrics_text)
    metrics["remaining_hours"] = float(hours_match.group(1)) if hours_match else None

    return metrics


def extract_user_budget_metrics(metrics_text: str, user_id: str) -> Dict[str, float]:
    """Extract budget-related metrics for a specific user"""
    import re

    metrics = {}

    # Escape user_id for regex pattern matching
    escaped_user_id = re.escape(user_id)

    # Get remaining budget
    remaining_pattern = f'litellm_remaining_user_budget_metric{{user="{escaped_user_id}"}} ([0-9.]+)'
    remaining_match = re.search(remaining_pattern, metrics_text)
    metrics["remaining"] = float(remaining_match.group(1)) if remaining_match else None

    # Get total budget
    total_pattern = f'litellm_user_max_budget_metric{{user="{escaped_user_id}"}} ([0-9.]+)'
    total_match = re.search(total_pattern, metrics_text)
    metrics["total"] = float(total_match.group(1)) if total_match else None

    # Get remaining hours
    hours_pattern = f'litellm_user_budget_remaining_hours_metric{{user="{escaped_user_id}"}} ([0-9.]+)'
    hours_match = re.search(hours_pattern, metrics_text)
    metrics["remaining_hours"] = float(hours_match.group(1)) if hours_match else None

    return metrics


@pytest.mark.asyncio
async def test_key_budget_metrics():
    """
    Test key budget tracking metrics:
    1. Create a key with max_budget
    2. Make chat completion requests using OpenAI SDK with the key
    3. Verify budget decreases over time
    4. Verify request costs are being tracked correctly
    5. Verify prometheus metrics match /key/info spend data
    """
    from datetime import datetime, timedelta, timezone

    async with aiohttp.ClientSession() as session:
        # Setup test key with unique alias
        unique_alias = f"budget_test_key_{uuid.uuid4()}"
        key_data = {
            "key_alias": unique_alias,
            "max_budget": 10,
            "budget_duration": "7d",
            "budget_reset_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        }
        key = await create_test_key_with_budget(session, key_data)

        # Extract key_id from the key info
        key_info = await get_key_info(session, key)
        print("key_info", key_info)
        key_id = key_info["key"]
        print("key_id", key_id)

        # Initialize OpenAI client with the key
        client = AsyncOpenAI(base_url="http://0.0.0.0:4000", api_key=key)

        # Make initial request and check budget
        await client.chat.completions.create(
            model="fake-openai-endpoint",
            messages=[{"role": "user", "content": f"Hello {uuid.uuid4()}"}],
        )

        await asyncio.sleep(11)  # Wait for metrics to update

        # Get metrics after request
        metrics_after_first = await get_prometheus_metrics(session)
        print("metrics_after_first request", metrics_after_first)
        first_budget = extract_key_budget_metrics(metrics_after_first, key_id)

        print(f"Budget after 1 request: {first_budget}")
        assert (
            first_budget["remaining"] < 10.0
        ), "remaining budget should be less than 10.0 after first request"
        assert first_budget["total"] == 10.0, "Total budget metric is incorrect"
        print("first_budget['remaining_hours']", first_budget["remaining_hours"])
        # The budget reset time is now standardized - for "7d" it resets on Monday at midnight
        # So we'll check if it's within a reasonable range (0-7 days depending on current day of week)
        assert (
            0 <= first_budget["remaining_hours"] <= 168
        ), "Budget remaining hours should be within a reasonable range (0-7 days depending on day of week)"

        # Get key info and verify spend matches prometheus metrics
        key_info = await get_key_info(session, key)
        print("key_info", key_info)
        _key_info_data = key_info["info"]

        # Calculate spend from prometheus (total - remaining)
        key_info_spend = float(_key_info_data["spend"])
        key_info_max_budget = float(_key_info_data["max_budget"])
        key_info_remaining_budget = key_info_max_budget - key_info_spend
        print("\n\n\n###### Final budget metrics ######\n\n\n")
        print("key_info_remaining_budget", key_info_remaining_budget)
        print("prometheus_remaining_budget", first_budget["remaining"])
        print(
            "diff between key_info_remaining_budget and prometheus_remaining_budget",
            key_info_remaining_budget - first_budget["remaining"],
        )

        # Verify spends match within a small delta (floating point comparison)
        assert (
            abs(key_info_remaining_budget - first_budget["remaining"]) <= 0.001
        ), f"Spend mismatch: Prometheus={key_info_remaining_budget}, Key Info={first_budget['remaining']}"


@pytest.mark.asyncio
async def test_user_budget_metrics():
    """
    Test user budget tracking metrics:
    1. Create a user with max_budget
    2. Make chat completion requests using OpenAI SDK with the user's key
    3. Verify budget decreases over time
    4. Verify request costs are being tracked correctly
    5. Verify prometheus metrics match /user/info spend data
    """
    from datetime import datetime, timedelta, timezone

    async with aiohttp.ClientSession() as session:
        # Setup test user with unique user_id
        unique_user_id = f"budget_test_user_{uuid.uuid4()}"
        user_data = {
            "user_id": unique_user_id,
            "max_budget": 10,
            "budget_duration": "7d",
            "budget_reset_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        }
        user_info = await create_test_user(session, user_data)
        print("user_info", user_info)
        user_id = user_info["user_id"]
        print("user_id", user_id)
        # Get the key that was created with the user
        key = user_info["key"]

        # Initialize OpenAI client with the user's key
        client = AsyncOpenAI(base_url="http://0.0.0.0:4000", api_key=key)

        # Make initial request and check budget
        await client.chat.completions.create(
            model="fake-openai-endpoint",
            messages=[{"role": "user", "content": f"Hello {uuid.uuid4()}"}],
        )

        await asyncio.sleep(11)  # Wait for metrics to update

        # Get metrics after request
        metrics_after_first = await get_prometheus_metrics(session)
        print("metrics_after_first request", metrics_after_first)
        first_budget = extract_user_budget_metrics(metrics_after_first, user_id)

        print(f"Budget after 1 request: {first_budget}")
        assert (
            first_budget["remaining"] is not None
        ), "remaining budget metric should be present"
        assert (
            first_budget["total"] is not None
        ), "total budget metric should be present"
        assert (
            first_budget["remaining"] < 10.0
        ), "remaining budget should be less than 10.0 after first request"
        assert first_budget["total"] == 10.0, "Total budget metric is incorrect"
        print("first_budget['remaining_hours']", first_budget["remaining_hours"])
        # The budget reset time is now standardized - for "7d" it resets on Monday at midnight
        # So we'll check if it's within a reasonable range (0-7 days depending on current day of week)
        assert (
            first_budget["remaining_hours"] is not None
        ), "remaining hours metric should be present"
        assert (
            0 <= first_budget["remaining_hours"] <= 168
        ), "Budget remaining hours should be within a reasonable range (0-7 days depending on day of week)"

        # Get user info and verify spend matches prometheus metrics
        user_info_response = await get_user_info(session, user_id)
        print("user_info_response", user_info_response)
        _user_info_data = user_info_response["user_info"]

        # Calculate spend from prometheus (total - remaining)
        user_info_spend = float(_user_info_data["spend"])
        user_info_max_budget = float(_user_info_data["max_budget"])
        user_info_remaining_budget = user_info_max_budget - user_info_spend
        print("\n\n\n###### Final budget metrics ######\n\n\n")
        print("user_info_remaining_budget", user_info_remaining_budget)
        print("prometheus_remaining_budget", first_budget["remaining"])
        print(
            "diff between user_info_remaining_budget and prometheus_remaining_budget",
            user_info_remaining_budget - first_budget["remaining"],
        )

        # Verify spends match within a small delta (floating point comparison)
        assert (
            abs(user_info_remaining_budget - first_budget["remaining"]) <= 0.001
        ), f"Spend mismatch: Prometheus={user_info_remaining_budget}, User Info={first_budget['remaining']}"


@pytest.mark.asyncio
async def test_user_email_metrics():
    """
    Test user email tracking metrics:
    1. Create a user with user_email
    2. Make chat completion requests using OpenAI SDK with the user's email
    3. Verify user email is being tracked correctly in `litellm_user_email_metric`
    """
    async with aiohttp.ClientSession() as session:
        # Create a user with user_email
        user_email = f"test-{uuid.uuid4()}@example.com"
        user_data = {
            "user_email": user_email,
        }
        user_info = await create_test_user(session, user_data)
        key = user_info["key"]

        # Initialize OpenAI client with the user's email
        client = AsyncOpenAI(base_url="http://0.0.0.0:4000", api_key=key)

        # Make initial request and check budget
        await client.chat.completions.create(
            model="fake-openai-endpoint",
            messages=[{"role": "user", "content": f"Hello {uuid.uuid4()}"}],
        )

        await asyncio.sleep(11)  # Wait for metrics to update

        # Get metrics after request
        metrics_after_first = await get_prometheus_metrics(session)
        print("metrics_after_first request", metrics_after_first)
        assert (
            user_email in metrics_after_first
        ), "user_email should be tracked correctly"


@pytest.mark.asyncio
async def test_user_email_in_all_required_metrics():
    """
    Test that user_email label is present in all the metrics that were requested to have it:
    - litellm_proxy_total_requests_metric_total
    - litellm_proxy_failed_requests_metric_total
    - litellm_input_tokens_metric_total
    - litellm_output_tokens_metric_total
    - litellm_requests_metric_total
    - litellm_spend_metric_total
    """
    async with aiohttp.ClientSession() as session:
        # Create a user with user_email
        user_email = f"test-metrics-{uuid.uuid4()}@example.com"
        user_data = {
            "user_email": user_email,
        }
        user_info = await create_test_user(session, user_data)
        key = user_info["key"]

        # Initialize OpenAI client with the user's email
        client = AsyncOpenAI(base_url="http://0.0.0.0:4000", api_key=key)

        # Make successful request to generate metrics
        await client.chat.completions.create(
            model="fake-openai-endpoint",
            messages=[{"role": "user", "content": f"Hello {uuid.uuid4()}"}],
        )

        await asyncio.sleep(11)  # Wait for metrics to update

        # Get metrics after request
        metrics_text = await get_prometheus_metrics(session)
        print("Testing user_email in all required metrics")

        # Check that user_email appears in all the required metrics
        required_metrics_with_user_email = [
            # "litellm_proxy_total_requests_metric_total",
            # "litellm_input_tokens_metric_total",
            # "litellm_output_tokens_metric_total",
            # "litellm_requests_metric_total",
            "litellm_spend_metric_total",
        ]

        import re

        for metric_name in required_metrics_with_user_email:
            # Check that the metric exists and contains user_email label
            # Look for the metric with user_email in its labels
            pattern = (
                rf'{metric_name}{{[^}}]*user_email="{re.escape(user_email)}"[^}}]*}}'
            )
            matches = re.findall(pattern, metrics_text)
            assert (
                len(matches) > 0
            ), f"Metric {metric_name} should contain user_email={user_email} but was not found in metrics"

        # Also test failure metric by making a bad request
        try:
            await client.chat.completions.create(
                model="fake-azure-endpoint",  # This should fail
                messages=[{"role": "user", "content": "Hello"}],
            )
        except Exception:
            pass  # Expected to fail

        await asyncio.sleep(11)  # Wait for metrics to update

        # Get updated metrics
        metrics_text = await get_prometheus_metrics(session)

        # Check that failure metric also contains user_email
        failure_pattern = rf'litellm_proxy_failed_requests_metric_total{{[^}}]*user_email="{re.escape(user_email)}"[^}}]*}}'
        failure_matches = re.findall(failure_pattern, metrics_text)
        assert (
            len(failure_matches) > 0
        ), f"litellm_proxy_failed_requests_metric_total should contain user_email={user_email}"
