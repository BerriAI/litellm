# What is this?
## This tests if the proxy fallbacks work as expected
import pytest
import asyncio
import aiohttp
from large_text import text
import time
from typing import Optional


async def generate_key(
    session,
    i,
    models: list,
    calling_key="sk-1234",
):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {
        "Authorization": f"Bearer {calling_key}",
        "Content-Type": "application/json",
    }
    data = {
        "models": models,
    }

    print(f"data: {data}")

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response {i} (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")

        return await response.json()


async def chat_completion(
    session,
    key: str,
    model: str,
    messages: list,
    return_headers: bool = False,
    extra_headers: Optional[dict] = None,
    **kwargs,
):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if extra_headers is not None:
        headers.update(extra_headers)
    data = {"model": model, "messages": messages, **kwargs}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            if return_headers:
                return None, response.headers
            else:
                raise Exception(f"Request did not return a 200 status code: {status}")

        if return_headers:
            return await response.json(), response.headers
        else:
            return await response.json()


@pytest.mark.asyncio
async def test_chat_completion():
    """
    make chat completion call with prompt > context window. expect it to work with fallback
    """
    async with aiohttp.ClientSession() as session:
        model = "gpt-3.5-turbo"
        messages = [
            {"role": "system", "content": text},
            {"role": "user", "content": "Who was Alexander?"},
        ]
        await chat_completion(
            session=session, key="sk-1234", model=model, messages=messages
        )


@pytest.mark.parametrize("has_access", [True, False])
@pytest.mark.asyncio
async def test_chat_completion_client_fallbacks(has_access):
    """
    make chat completion call with prompt > context window. expect it to work with fallback
    """

    async with aiohttp.ClientSession() as session:
        models = ["gpt-3.5-turbo"]

        if has_access:
            models.append("gpt-instruct")

        ## CREATE KEY WITH MODELS
        generated_key = await generate_key(session=session, i=0, models=models)
        calling_key = generated_key["key"]
        model = "gpt-3.5-turbo"
        messages = [
            {"role": "user", "content": "Who was Alexander?"},
        ]

        ## CALL PROXY
        try:
            await chat_completion(
                session=session,
                key=calling_key,
                model=model,
                messages=messages,
                mock_testing_fallbacks=True,
                fallbacks=["gpt-instruct"],
            )
            if not has_access:
                pytest.fail(
                    "Expected this to fail, submitted fallback model that key did not have access to"
                )
        except Exception as e:
            if has_access:
                pytest.fail("Expected this to work: {}".format(str(e)))


@pytest.mark.asyncio
async def test_chat_completion_with_retries():
    """
    make chat completion call with prompt > context window. expect it to work with fallback
    """
    async with aiohttp.ClientSession() as session:
        model = "fake-openai-endpoint-4"
        messages = [
            {"role": "system", "content": text},
            {"role": "user", "content": "Who was Alexander?"},
        ]
        response, headers = await chat_completion(
            session=session,
            key="sk-1234",
            model=model,
            messages=messages,
            mock_testing_rate_limit_error=True,
            return_headers=True,
        )
        print(f"headers: {headers}")
        assert headers["x-litellm-attempted-retries"] == "1"
        assert headers["x-litellm-max-retries"] == "50"


@pytest.mark.asyncio
async def test_chat_completion_with_timeout():
    """
    make chat completion call with low timeout and `mock_timeout`: true. Expect it to fail and correct timeout to be set in headers.
    """
    async with aiohttp.ClientSession() as session:
        model = "fake-openai-endpoint-5"
        messages = [
            {"role": "system", "content": text},
            {"role": "user", "content": "Who was Alexander?"},
        ]
        start_time = time.time()
        response, headers = await chat_completion(
            session=session,
            key="sk-1234",
            model=model,
            messages=messages,
            num_retries=0,
            mock_timeout=True,
            return_headers=True,
        )
        end_time = time.time()
        print(f"headers: {headers}")
        assert (
            headers["x-litellm-timeout"] == "1.0"
        )  # assert model-specific timeout used


@pytest.mark.asyncio
async def test_chat_completion_with_timeout_from_request():
    """
    make chat completion call with low timeout and `mock_timeout`: true. Expect it to fail and correct timeout to be set in headers.
    """
    async with aiohttp.ClientSession() as session:
        model = "fake-openai-endpoint-5"
        messages = [
            {"role": "system", "content": text},
            {"role": "user", "content": "Who was Alexander?"},
        ]
        extra_headers = {
            "x-litellm-timeout": "0.001",
        }
        start_time = time.time()
        response, headers = await chat_completion(
            session=session,
            key="sk-1234",
            model=model,
            messages=messages,
            num_retries=0,
            mock_timeout=True,
            extra_headers=extra_headers,
            return_headers=True,
        )
        end_time = time.time()
        print(f"headers: {headers}")
        assert (
            headers["x-litellm-timeout"] == "0.001"
        )  # assert model-specific timeout used


@pytest.mark.parametrize("has_access", [True, False])
@pytest.mark.asyncio
async def test_chat_completion_client_fallbacks_with_custom_message(has_access):
    """
    make chat completion call with prompt > context window. expect it to work with fallback
    """

    async with aiohttp.ClientSession() as session:
        models = ["gpt-3.5-turbo"]

        if has_access:
            models.append("gpt-instruct")

        ## CREATE KEY WITH MODELS
        generated_key = await generate_key(session=session, i=0, models=models)
        calling_key = generated_key["key"]
        model = "gpt-3.5-turbo"
        messages = [
            {"role": "user", "content": "Who was Alexander?"},
        ]

        ## CALL PROXY
        try:
            await chat_completion(
                session=session,
                key=calling_key,
                model=model,
                messages=messages,
                mock_testing_fallbacks=True,
                fallbacks=[
                    {
                        "model": "gpt-instruct",
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "This is a custom message",
                            }
                        ],
                    }
                ],
            )
            if not has_access:
                pytest.fail(
                    "Expected this to fail, submitted fallback model that key did not have access to"
                )
        except Exception as e:
            if has_access:
                pytest.fail("Expected this to work: {}".format(str(e)))


import asyncio
from openai import AsyncOpenAI
from typing import List
import time


async def make_request(client: AsyncOpenAI, model: str) -> bool:
    try:
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Who was Alexander?"}],
        )
        return True
    except Exception as e:
        print(f"Error with {model}: {str(e)}")
        return False


async def run_good_model_test(client: AsyncOpenAI, num_requests: int) -> bool:
    tasks = [make_request(client, "good-model") for _ in range(num_requests)]
    good_results = await asyncio.gather(*tasks)
    return all(good_results)


@pytest.mark.asyncio
async def test_chat_completion_bad_and_good_model():
    """
    Prod test - ensure even if bad model is down, good model is still working.
    """
    client = AsyncOpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")
    num_requests = 100
    num_iterations = 3

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        start_time = time.time()

        # Fire and forget bad model requests
        for _ in range(num_requests):
            asyncio.create_task(make_request(client, "bad-model"))

        # Wait only for good model requests
        success = await run_good_model_test(client, num_requests)
        print(
            f"Iteration {iteration + 1}: {'✓' if success else '✗'} ({time.time() - start_time:.2f}s)"
        )
        assert success, "Not all good model requests succeeded"
