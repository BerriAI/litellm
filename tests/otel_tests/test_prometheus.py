"""
Unit tests for prometheus metrics
"""

import pytest
import aiohttp
import asyncio


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
