import asyncio
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


import litellm.types
import litellm.types.utils
from litellm.router import Router
from typing import Optional
from unittest.mock import MagicMock, patch

import asyncio
import pytest
import psutil
import os
import litellm
from typing import Callable, Any


async def get_memory_usage() -> float:
    """Get current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


async def run_memory_test(request_func: Callable, name: str) -> None:
    """
    Generic memory test function
    Args:
        request_func: Async function that makes the API request
        name: Name of the test for logging
    """
    memory_before = await get_memory_usage()
    print(f"\n{name} - Initial memory usage: {memory_before:.2f}MB")

    for i in range(400):
        await request_func()
        if i % 10 == 0:
            current_memory = await get_memory_usage()
            print(f"Request {i}: Current memory usage: {current_memory:.2f}MB")

    memory_after = await get_memory_usage()
    print(f"Final memory usage: {memory_after:.2f}MB")

    memory_diff = memory_after - memory_before
    print(f"Memory difference: {memory_diff:.2f}MB")

    assert memory_diff < 10, f"Memory increased by {memory_diff:.2f}MB"


async def make_completion_request():
    return await litellm.acompletion(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Test message for memory usage"}],
        api_base="https://exampleopenaiendpoint-production.up.railway.app/",
    )


async def make_text_completion_request():
    return await litellm.atext_completion(
        model="openai/gpt-4o",
        prompt="Test message for memory usage",
        api_base="https://exampleopenaiendpoint-production.up.railway.app/",
    )


@pytest.mark.asyncio
async def test_acompletion_memory():
    """Test memory usage for litellm.acompletion"""
    await run_memory_test(make_completion_request, "acompletion")


@pytest.mark.asyncio
async def test_atext_completion_memory():
    """Test memory usage for litellm.atext_completion"""
    await run_memory_test(make_text_completion_request, "atext_completion")


litellm_router = Router(
    model_list=[
        {
            "model_name": "text-gpt-4o",
            "litellm_params": {
                "model": "text-completion-openai/gpt-3.5-turbo-instruct-unlimited",
                "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
            },
        },
    ]
)


async def make_router_atext_completion_request():
    return await litellm_router.atext_completion(
        model="text-gpt-4o",
        temperature=0.5,
        frequency_penalty=0.5,
        prompt="<|fim prefix|> Test message for memory usage<fim suffix> <|fim prefix|> Test message for memory usage<fim suffix>",
        api_base="https://exampleopenaiendpoint-production.up.railway.app/",
        max_tokens=500,
    )


@pytest.mark.asyncio
async def test_router_atext_completion_memory():
    """Test memory usage for litellm.atext_completion"""
    await run_memory_test(
        make_router_atext_completion_request, "router_atext_completion"
    )


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    import gc

    yield
    gc.collect()
