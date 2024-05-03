import json
import sys
import os
import io, asyncio

import logging

logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))

from litellm import completion
import litellm

import time
import pytest


@pytest.mark.asyncio
async def test_async_log_langfuse():
    litellm.success_callback = ["async_langfuse"]
    litellm.set_verbose = True
    response = await litellm.acompletion(
        model="gpt-4",
        messages=[{"role": "user", "content": "This is a test"}],
        max_tokens=5,
        temperature=0.7,
        timeout=5,
        user="async langfuse_latency_test_user",
    )

    print("response", response)
    await asyncio.sleep(3)
