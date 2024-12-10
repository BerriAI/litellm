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
