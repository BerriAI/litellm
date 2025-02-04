import json
import os
import sys
from datetime import datetime
import pytest

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system-path

import litellm


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
