import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import time

import pytest

import litellm
from litellm import completion


# @pytest.mark.skip(reason="beta test - this is a new feature")
@pytest.mark.asyncio
async def test_datadog_logging():
    try:
        litellm.success_callback = ["datadog"]
        litellm.set_verbose = True
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)

        await asyncio.sleep(5)
    except Exception as e:
        print(e)
