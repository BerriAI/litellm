import os
from typing import Final
# What this tests ?
## Tests /chat/completions by generating a key and then making a chat completions-request
import pytest
from openai import AsyncOpenAI

LITELLM_MASTER_KEY = os.environ["LITELLM_MASTER_KEY"]


def response_header_check(response):
    """
    - assert if response headers < 4kb (nginx limit).
    """
    headers_size = sum(len(k) + len(v) for k, v in response.raw_headers)
    assert headers_size < 4096, "Response headers exceed the 4kb limit"


async def moderation(session, key):
    url = "http://0.0.0.0:4000/moderations"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {"model": "text-moderation-stable", "input": "I want to kill the cat."}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return await response.json()


async def embeddings(session, key, model="text-embedding-ada-002"):
    url = "http://0.0.0.0:4000/embeddings"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "input": ["hello world"],
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        response_header_check(
            response
        )  # calling the function to check response headers


@pytest.mark.asyncio
async def test_chat_completion_streaming():
    """
    [PROD Test] Ensures logprobs are returned correctly
    """
    client = AsyncOpenAI(api_key=os.environ["LITELLM_MASTER_KEY"], base_url="http://0.0.0.0:4000")

    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-large",
        messages=[{"role": "user", "content": "Hello!"}],
        logprobs=True,
        top_logprobs=2,
        stream=True,
    )

    response_str = ""

    async for chunk in response:
        response_str += chunk.choices[0].delta.content or ""

    print(f"response_str: {response_str}")


@pytest.mark.asyncio
async def test_completion_streaming_usage_metrics():
    """
    [PROD Test] Ensures usage metrics are returned correctly when `include_usage` is set to `True`
    """
    client: Final = AsyncOpenAI(
        api_key=os.environ["LITELLM_MASTER_KEY"], base_url=os.environ.get("LITELLM_PROXY_BASE_URL", "http://0.0.0.0:4000")
    )

    response = await client.completions.create(
        model="gpt-6-luna",
        prompt="hey",
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=4,
        temperature=0.00000001,
    )

    last_chunk = None
    async for chunk in response:
        print("chunk", chunk)
        last_chunk = chunk

    assert last_chunk is not None, "No chunks were received"
    assert last_chunk.usage is not None, "Usage information was not received"
    assert last_chunk.usage.prompt_tokens > 0, "Prompt tokens should be greater than 0"
    assert last_chunk.usage.completion_tokens > 0, "Completion tokens should be greater than 0"
    assert last_chunk.usage.total_tokens > 0, "Total tokens should be greater than 0"


