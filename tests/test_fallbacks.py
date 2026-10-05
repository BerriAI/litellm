import os
from typing import Final

# What is this?
## This tests if the proxy fallbacks work as expected
import pytest
import asyncio
import aiohttp
from tests.large_text import text
import time
from typing import Optional
from openai import AsyncOpenAI, PermissionDeniedError

PROXY_BASE_URL: Final = os.environ.get("LITELLM_PROXY_BASE_URL", "http://0.0.0.0:4000")


async def generate_key(
    session,
    i,
    models: list,
    calling_key="sk-1234",
):
    url: Final = f"{PROXY_BASE_URL}/key/generate"
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
    url: Final = f"{PROXY_BASE_URL}/chat/completions"
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


@pytest.mark.parametrize("has_access", [True, False])
@pytest.mark.asyncio
async def test_chat_completion_client_fallbacks(has_access: bool) -> None:
    models: Final = ["gpt-3.5-turbo", "gpt-6-luna"] if has_access else ["gpt-3.5-turbo"]
    async with aiohttp.ClientSession() as session:
        generated_key: Final = await generate_key(session=session, i=0, models=models)
    async with AsyncOpenAI(api_key=generated_key["key"], base_url=PROXY_BASE_URL, max_retries=0) as client:
        request: Final = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Who was Alexander?"}],
            "max_tokens": 32,
            "temperature": 0,
            "extra_body": {
                "mock_testing_fallbacks": True,
                "fallbacks": ["gpt-6-luna"],
            },
        }
        if not has_access:
            with pytest.raises(PermissionDeniedError) as denied:
                await client.chat.completions.create(**request)
            assert denied.value.status_code == 403
            assert "gpt-6-luna" in str(denied.value)
            return
        response: Final = await client.chat.completions.create(**request)
        assert response.model == "gpt-6-luna"
        assert response.choices[0].message.content


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
async def test_chat_completion_with_fallbacks():
    """
    make chat completion call with prompt > context window. expect it to work with fallback
    """
    async with aiohttp.ClientSession() as session:
        model = "badly-configured-openai-endpoint"
        messages = [
            {"role": "system", "content": text},
            {"role": "user", "content": "Who was Alexander?"},
        ]
        response, headers = await chat_completion(
            session=session,
            key="sk-1234",
            model=model,
            messages=messages,
            fallbacks=["fake-openai-endpoint-5"],
            return_headers=True,
        )
        print(f"headers: {headers}")
        assert headers["x-litellm-attempted-fallbacks"] == "1"


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
async def test_chat_completion_client_fallbacks_with_custom_message(has_access: bool) -> None:
    original_messages: Final = [{"role": "user", "content": "Who was Alexander?"}]
    custom_messages: Final = [
        {
            "role": "user",
            "content": (
                "Describe the weather in a coastal city during winter, including the usual temperature, rain, wind, "
                "and the clothing a visitor should bring."
            ),
        }
    ]
    models: Final = ["gpt-3.5-turbo", "gpt-6-luna"] if has_access else ["gpt-3.5-turbo"]
    async with aiohttp.ClientSession() as session:
        generated_key: Final = await generate_key(session=session, i=0, models=models)
    async with AsyncOpenAI(api_key=generated_key["key"], base_url=PROXY_BASE_URL, max_retries=0) as client:
        request: Final = {
            "model": "gpt-3.5-turbo",
            "messages": original_messages,
            "max_tokens": 32,
            "temperature": 0,
            "extra_body": {
                "mock_testing_fallbacks": True,
                "fallbacks": [
                    {
                        "model": "gpt-6-luna",
                        "messages": custom_messages,
                    }
                ],
            },
        }
        if not has_access:
            with pytest.raises(PermissionDeniedError) as denied:
                await client.chat.completions.create(**request)
            assert denied.value.status_code == 403
            assert "gpt-6-luna" in str(denied.value)
            return
        response: Final = await client.chat.completions.create(**request)
        assert response.model == "gpt-6-luna"
        assert response.choices[0].message.content
        custom_control: Final = await client.chat.completions.create(
            model="gpt-6-luna",
            messages=custom_messages,
            max_tokens=32,
            temperature=0,
        )
        original_control: Final = await client.chat.completions.create(
            model="gpt-6-luna",
            messages=original_messages,
            max_tokens=32,
            temperature=0,
        )
        assert response.usage is not None
        assert custom_control.usage is not None
        assert original_control.usage is not None
        assert custom_control.usage.completion_tokens > 0
        assert original_control.usage.completion_tokens > 0
        assert custom_control.usage.prompt_tokens != original_control.usage.prompt_tokens
        assert response.usage.prompt_tokens == custom_control.usage.prompt_tokens


from typing import List


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
