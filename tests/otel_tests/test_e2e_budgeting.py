import pytest
import asyncio
import aiohttp
import json


async def make_calls_until_budget_exceeded(session, key: str, call_function, **kwargs):
    """Helper function to make API calls until budget is exceeded. Verify that the budget is exceeded error is returned."""
    MAX_CALLS = 50
    call_count = 0
    try:
        while call_count < MAX_CALLS:
            await call_function(session=session, key=key, **kwargs)
            call_count += 1
        pytest.fail(f"Budget was not exceeded after {MAX_CALLS} calls")
    except Exception as e:
        print("vars: ", vars(e))
        print("e.body: ", e.body)

        error_dict = e.body
        print("error_dict: ", error_dict)

        # Check error structure and values that should be consistent
        assert (
            error_dict["code"] == "400"
        ), f"Expected error code 400, got: {error_dict['code']}"
        assert (
            error_dict["type"] == "budget_exceeded"
        ), f"Expected error type budget_exceeded, got: {error_dict['type']}"

        # Check message contains required parts without checking specific values
        message = error_dict["message"]
        assert (
            "Budget has been exceeded!" in message
        ), f"Expected message to start with 'Budget has been exceeded!', got: {message}"
        assert (
            "Current cost:" in message
        ), f"Expected message to contain 'Current cost:', got: {message}"
        assert (
            "Max budget:" in message
        ), f"Expected message to contain 'Max budget:', got: {message}"

        return call_count


async def generate_key(
    session,
    max_budget=None,
):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "max_budget": max_budget,
    }
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def chat_completion(session, key: str, model: str):
    """Make a chat completion request using OpenAI SDK"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=key, base_url="http://0.0.0.0:4000/v1"  # Point to our local proxy
    )

    response = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Say hello!" * 100}]
    )
    return response


@pytest.mark.asyncio
async def test_chat_completion_low_budget():
    """
    Test budget enforcement for chat completions:
    1. Create key with $0.01 budget
    2. Make chat completion calls until budget exceeded
    3. Verify budget exceeded error
    """
    async with aiohttp.ClientSession() as session:
        # Create key with $0.01 budget
        key_gen = await generate_key(session=session, max_budget=0.0000000005)
        print("response from key generation: ", key_gen)
        key = key_gen["key"]

        # Make calls until budget exceeded
        calls_made = await make_calls_until_budget_exceeded(
            session=session,
            key=key,
            call_function=chat_completion,
            model="fake-openai-endpoint",
        )

        assert (
            calls_made > 0
        ), "Should make at least one successful call before budget exceeded"


@pytest.mark.asyncio
async def test_chat_completion_zero_budget():
    """
    Test budget enforcement for chat completions:
    1. Create key with $0.01 budget
    2. Make chat completion calls until budget exceeded
    3. Verify budget exceeded error
    """
    async with aiohttp.ClientSession() as session:
        # Create key with $0.01 budget
        key_gen = await generate_key(session=session, max_budget=0.000000000)
        print("response from key generation: ", key_gen)
        key = key_gen["key"]

        # Make calls until budget exceeded
        calls_made = await make_calls_until_budget_exceeded(
            session=session,
            key=key,
            call_function=chat_completion,
            model="fake-openai-endpoint",
        )

        assert calls_made == 0, "Should make no calls before budget exceeded"


@pytest.mark.asyncio
async def test_chat_completion_high_budget():
    """
    Test budget enforcement for chat completions:
    1. Create key with $0.01 budget
    2. Make chat completion calls until budget exceeded
    3. Verify budget exceeded error
    """
    async with aiohttp.ClientSession() as session:
        # Create key with $0.01 budget
        key_gen = await generate_key(session=session, max_budget=0.001)
        print("response from key generation: ", key_gen)
        key = key_gen["key"]

        # Make calls until budget exceeded
        calls_made = await make_calls_until_budget_exceeded(
            session=session,
            key=key,
            call_function=chat_completion,
            model="fake-openai-endpoint",
        )

        assert (
            calls_made > 0
        ), "Should make at least one successful call before budget exceeded"
