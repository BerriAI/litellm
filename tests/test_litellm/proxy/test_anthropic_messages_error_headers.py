"""
Test that x-litellm-model-id header is returned on /v1/messages error responses.

This test verifies that the model_id header is propagated correctly when
requests fail after router selection (e.g., due to unsupported parameters).
"""

import pytest
import asyncio
import aiohttp

LITELLM_MASTER_KEY = "sk-1234"


async def anthropic_messages_with_headers(session, key, model="gpt-4", **extra_params):
    """
    Make a request to /v1/messages and return response headers.
    """
    url = "http://0.0.0.0:4000/v1/messages"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "Hello!"},
        ],
        **extra_params,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Status: {status}")
        print(f"Response: {response_text}")
        print()

        raw_headers = response.raw_headers
        raw_headers_json = {}

        for item in response.raw_headers:
            raw_headers_json[item[0].decode("utf-8")] = item[1].decode("utf-8")

        return {
            "status": status,
            "headers": raw_headers_json,
            "response_text": response_text,
        }


@pytest.mark.asyncio
async def test_anthropic_messages_error_with_model_id_header():
    """
    Test that x-litellm-model-id header is returned on error responses.

    This test:
    1. Makes a request to /v1/messages with an unsupported parameter (reasoning_effort)
    2. Verifies that the request fails with a 400 error
    3. Verifies that the x-litellm-model-id header is present in the error response

    The error occurs AFTER router selection, so model_id should be available
    and included in the error response headers.
    """
    async with aiohttp.ClientSession() as session:
        key = LITELLM_MASTER_KEY
        result = await anthropic_messages_with_headers(
            session=session,
            key=key,
            model="gpt-4",
            reasoning_effort="low",  # Unsupported param that triggers error
        )

        # Verify the request failed
        assert result["status"] == 400, f"Expected 400, got {result['status']}"

        # Verify model_id header is present
        assert "x-litellm-model-id" in result["headers"], (
            f"x-litellm-model-id header missing in error response. "
            f"Headers: {result['headers'].keys()}"
        )

        # Verify the header has a non-empty value
        model_id = result["headers"]["x-litellm-model-id"]
        assert model_id, "x-litellm-model-id header is empty"
        print(f"Successfully retrieved model_id on error response: {model_id}")


@pytest.mark.asyncio
async def test_anthropic_messages_success_with_model_id_header():
    """
    Test that x-litellm-model-id header is returned on successful responses.

    This is a baseline test to ensure the header is present on success too.
    """
    async with aiohttp.ClientSession() as session:
        key = LITELLM_MASTER_KEY
        result = await anthropic_messages_with_headers(
            session=session,
            key=key,
            model="gpt-4",
        )

        # Verify the request succeeded
        assert result["status"] == 200, f"Expected 200, got {result['status']}"

        # Verify model_id header is present
        assert "x-litellm-model-id" in result["headers"], (
            f"x-litellm-model-id header missing in success response. "
            f"Headers: {result['headers'].keys()}"
        )

        # Verify the header has a non-empty value
        model_id = result["headers"]["x-litellm-model-id"]
        assert model_id, "x-litellm-model-id header is empty"
        print(f"Successfully retrieved model_id on success response: {model_id}")
