import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import json

import pytest

import litellm
from litellm import completion


@pytest.mark.asyncio
async def test_cohere_generate_api_completion():
    try:
        litellm.set_verbose = False
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="cohere/command-nightly",
            messages=messages,
            max_tokens=10,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_cohere_generate_api_stream():
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = await litellm.acompletion(
            model="cohere/command-nightly",
            messages=messages,
            max_tokens=10,
            stream=True,
        )
        print("async cohere stream response", response)
        async for chunk in response:
            print(chunk)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_cohere_stream_bad_key():
    try:
        api_key = "bad-key"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        completion(
            model="command-nightly",
            messages=messages,
            stream=True,
            max_tokens=50,
            api_key=api_key,
        )

    except litellm.AuthenticationError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
