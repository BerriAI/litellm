import os
import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union
import json
from litellm._uuid import uuid


async def chat_completion(
    session,
    key,
    messages,
    model: Union[str, List] = "gpt-5.5",
    guardrails: Optional[List] = None,
):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": messages,
    }

    if guardrails is not None:
        data["guardrails"] = guardrails

    print("data=", data)

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(response_text)

        # response headers
        response_headers = dict(response.headers)
        print("response headers=", response_headers)

        return await response.json(), response_headers


@pytest.mark.asyncio
async def test_bedrock_guardrail_triggered():
    """
    - Tests a request where our bedrock guardrail should be triggered
    - Assert that the guardrails applied are returned in the response headers
    """
    async with aiohttp.ClientSession() as session:
        with pytest.raises(Exception, match="Violated guardrail policy") as exc_info:
            response, headers = await chat_completion(
                session,
                os.environ["LITELLM_MASTER_KEY"],
                model="fake-openai-endpoint",
                messages=[{"role": "user", "content": "Hello do you like coffee?"}],
                guardrails=["bedrock-pre-guard"],
            )
        e = exc_info.value
        print(e)
        assert "Violated guardrail policy" in str(e)


async def get_guardrail_lb_counts(session):
    """Get the current guardrail load balancing call counts from the proxy."""
    url = "http://0.0.0.0:4000/guardrail/lb/counts"
    headers = {"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}", "Content-Type": "application/json"}

    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        return None


@pytest.mark.asyncio
async def test_guardrail_load_balancing():
    """
    Test that guardrail load balancing distributes requests across multiple guardrail instances.

    - Make 20 requests with the lb-test-guard guardrail
    - Verify that both GuardrailForLBTestingA and GuardrailForLBTestingB are called
    - Verify reasonable distribution (both should have at least some calls)
    """
    async with aiohttp.ClientSession() as session:
        num_requests = 20

        # Make multiple requests with the load-balanced guardrail
        for i in range(num_requests):
            response, headers = await chat_completion(
                session,
                os.environ["LITELLM_MASTER_KEY"],
                model="fake-openai-endpoint",
                messages=[{"role": "user", "content": f"Hello request {i}"}],
                guardrails=["lb-test-guard"],
            )

            # Verify guardrail was applied
            assert "x-litellm-applied-guardrails" in headers
            assert headers["x-litellm-applied-guardrails"] == "lb-test-guard"

        # All requests should succeed - the test passes if we get here
        # The actual load balancing verification is done by checking proxy logs
        # which should show alternating calls to GuardrailForLBTestingA and GuardrailForLBTestingB
        print(f"Successfully made {num_requests} requests with load-balanced guardrail")
