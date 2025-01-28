import json
import os
import sys
from datetime import datetime
import pytest

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system-path

import litellm

from local_testing.test_streaming import streaming_format_tests


@pytest.mark.asyncio()
async def test_aiohttp_openai():
    litellm.set_verbose = True
    response = await litellm.acompletion(
        model="aiohttp_openai/fake-model",
        messages=[{"role": "user", "content": "Hello, world!"}],
        api_base="https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions",
        api_key="fake-key",
    )
    print(response)


@pytest.mark.asyncio()
async def test_aiohttp_openai_gpt_4o():
    litellm.set_verbose = True
    response = await litellm.acompletion(
        model="aiohttp_openai/gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    print(response)


@pytest.mark.asyncio()
async def test_completion_model_stream():
    litellm.set_verbose = True
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key is not None, "API key is not set in environment variables"

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        response = await litellm.acompletion(
            api_key=api_key, model="aiohttp_openai/gpt-4o", messages=messages, stream=True, max_tokens=50
        )

        complete_response = ""
        idx = 0  # Initialize index manually
        async for chunk in response:  # Use async for to handle async iterator
            chunk, finished = streaming_format_tests(idx, chunk)  # Await if streaming_format_tests is async
            print(f"outside chunk: {chunk}")
            if finished:
                break
            complete_response += chunk
            idx += 1  # Increment index manually

        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"complete response: {complete_response}")

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
