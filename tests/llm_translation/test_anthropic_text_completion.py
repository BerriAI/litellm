import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

import litellm.types
import litellm.types.utils
from litellm.llms.anthropic.chat import ModelResponseIterator

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["claude-2", "anthropic/claude-2"])
@pytest.mark.flaky(retries=6, delay=1)
async def test_acompletion_claude2(model):
    try:
        litellm.set_verbose = True
        messages = [
            {
                "role": "system",
                "content": "Your goal is generate a joke on the topic user gives.",
            },
            {"role": "user", "content": "Generate a 3 liner joke for me"},
        ]
        # test without max-tokens
        response = await litellm.acompletion(model=model, messages=messages)
        # Add any assertions here to check the response
        print(response)
        print(response.usage)
        print(response.usage.completion_tokens)
        print(response["usage"]["completion_tokens"])
        # print("new cost tracking")
    except litellm.InternalServerError:
        pytest.skip("model is overloaded.")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_acompletion_claude2_stream():
    try:
        litellm.set_verbose = False
        messages = [
            {
                "role": "system",
                "content": "Your goal is generate a joke on the topic user gives.",
            },
            {"role": "user", "content": "Generate a 3 liner joke for me"},
        ]
        # test without max-tokens
        response = await litellm.acompletion(
            model="anthropic_text/claude-2",
            messages=messages,
            stream=True,
            max_tokens=10,
        )
        async for chunk in response:
            print(chunk)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
