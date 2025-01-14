import asyncio
import os
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import concurrent

from dotenv import load_dotenv
import asyncio
import litellm


@pytest.mark.asyncio
async def test_acompletion_fallbacks_basic():
    response = await litellm.acompletion(
        model="openai/unknown-model",
        messages=[{"role": "user", "content": "Hello, world!"}],
        fallbacks=["openai/gpt-4o-mini"],
    )
    print(response)
    assert response is not None


@pytest.mark.asyncio
async def test_acompletion_fallbacks_bad_models():
    response = await litellm.acompletion(
        model="openai/unknown-model",
        messages=[{"role": "user", "content": "Hello, world!"}],
        fallbacks=["openai/bad-model", "openai/unknown-model"],
    )
    assert response is not None
