import pytest
import asyncio
import aiohttp
import json
from httpx import AsyncClient
from typing import Any, Optional, List


async def generate_key(session, models: Optional[List[str]] = None):
    """Helper function to generate a key with specific model access"""
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {}
    if models is not None:
        data["models"] = models
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def chat_completion(session, key: str, model: str):
    """Make a chat completion request using OpenAI SDK"""
    from openai import AsyncOpenAI
    import uuid

    client = AsyncOpenAI(api_key=key, base_url="http://0.0.0.0:4000/v1")

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"Say hello! {uuid.uuid4()}"}],
        extra_body={
            "mock_response": "mock_response",
        },
    )
    return response


@pytest.mark.parametrize(
    "key_models, test_model, expect_success",
    [
        (["openai/*"], "anthropic/claude-2", False),  # Non-matching model
        (["gpt-4"], "gpt-4", True),  # Exact model match
        (["bedrock/*"], "bedrock/anthropic.claude-3", True),  # Bedrock wildcard
        (["bedrock/anthropic.*"], "bedrock/anthropic.claude-3", True),  # Pattern match
        (["bedrock/anthropic.*"], "bedrock/amazon.titan", False),  # Pattern non-match
        (None, "gpt-4", True),  # No model restrictions
        ([], "gpt-4", True),  # Empty model list
    ],
)
@pytest.mark.asyncio
async def test_model_access_patterns(key_models, test_model, expect_success):
    """
    Test model access patterns for API keys:
    1. Create key with specific model access pattern
    2. Attempt to make completion with test model
    3. Verify access is granted/denied as expected
    """
    async with aiohttp.ClientSession() as session:
        # Generate key with specified model access
        key_gen = await generate_key(session=session, models=key_models)
        key = key_gen["key"]

        try:
            response = await chat_completion(
                session=session,
                key=key,
                model=test_model,
            )
            if not expect_success:
                pytest.fail(f"Expected request to fail for model {test_model}")
            assert (
                response is not None
            ), "Should get valid response when access is allowed"
        except Exception as e:
            if expect_success:
                pytest.fail(f"Expected request to succeed but got error: {e}")
            _error_body = e.body

            # Assert error structure and values
            assert _error_body["type"] == "key_model_access_denied"
            assert _error_body["param"] == "model"
            assert _error_body["code"] == "401"
            assert "API Key not allowed to access model" in _error_body["message"]


@pytest.mark.asyncio
async def test_model_access_update():
    """
    Test updating model access for an existing key:
    1. Create key with restricted model access
    2. Verify access patterns
    3. Update key with new model access
    4. Verify new access patterns
    """
    client = AsyncClient(base_url="http://0.0.0.0:4000")
    headers = {"Authorization": "Bearer sk-1234"}

    # Create initial key with restricted access
    response = await client.post(
        "/key/generate", json={"models": ["openai/gpt-4"]}, headers=headers
    )
    assert response.status_code == 200
    key_data = response.json()
    key = key_data["key"]

    # Test initial access
    async with aiohttp.ClientSession() as session:
        # Should work with gpt-4
        await chat_completion(session=session, key=key, model="openai/gpt-4")

        # Should fail with gpt-3.5-turbo
        with pytest.raises(Exception) as exc_info:
            await chat_completion(
                session=session, key=key, model="openai/gpt-3.5-turbo"
            )
        assert (
            "Invalid model" in str(exc_info.value)
            or "permission denied" in str(exc_info.value).lower()
        )

    # Update key with new model access
    response = await client.post(
        "/key/update", json={"key": key, "models": ["openai/*"]}, headers=headers
    )
    assert response.status_code == 200

    # Test updated access
    async with aiohttp.ClientSession() as session:
        # Both models should now work
        await chat_completion(session=session, key=key, model="openai/gpt-4")
        await chat_completion(session=session, key=key, model="openai/gpt-3.5-turbo")

        # Non-OpenAI model should still fail
        with pytest.raises(Exception) as exc_info:
            await chat_completion(session=session, key=key, model="anthropic/claude-2")
        assert (
            "Invalid model" in str(exc_info.value)
            or "permission denied" in str(exc_info.value).lower()
        )
