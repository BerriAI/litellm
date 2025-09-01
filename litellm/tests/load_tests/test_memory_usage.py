import asyncio
import os
import sys
import traceback
import tracemalloc

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
import os
import litellm
from typing import Callable, Any

import tracemalloc
import gc
from typing import Type
from pydantic import BaseModel

from litellm.proxy.proxy_server import app


async def get_memory_usage() -> float:
    """Get current memory usage of the process in MB"""
    import psutil

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

    for i in range(60 * 4):  # 4 minutes
        all_tasks = [request_func() for _ in range(100)]
        await asyncio.gather(*all_tasks)
        current_memory = await get_memory_usage()
        print(f"Request {i * 100}: Current memory usage: {current_memory:.2f}MB")

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
@pytest.mark.skip(
    reason="This test is too slow to run on every commit. We can use this after nightly release"
)
async def test_acompletion_memory():
    """Test memory usage for litellm.acompletion"""
    await run_memory_test(make_completion_request, "acompletion")


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="This test is too slow to run on every commit. We can use this after nightly release"
)
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
        {
            "model_name": "chat-gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
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
@pytest.mark.skip(
    reason="This test is too slow to run on every commit. We can use this after nightly release"
)
async def test_router_atext_completion_memory():
    """Test memory usage for litellm.atext_completion"""
    await run_memory_test(
        make_router_atext_completion_request, "router_atext_completion"
    )


async def make_router_acompletion_request():
    return await litellm_router.acompletion(
        model="chat-gpt-4o",
        messages=[{"role": "user", "content": "Test message for memory usage"}],
        api_base="https://exampleopenaiendpoint-production.up.railway.app/",
    )


def get_pydantic_objects():
    """Get all Pydantic model instances in memory"""
    return [obj for obj in gc.get_objects() if isinstance(obj, BaseModel)]


def analyze_pydantic_snapshot():
    """Analyze current Pydantic objects"""
    objects = get_pydantic_objects()
    type_counts = {}

    for obj in objects:
        type_name = type(obj).__name__
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    print("\nPydantic Object Count:")
    for type_name, count in sorted(
        type_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{type_name}: {count}")
        # Print an example object if helpful
        if count > 1000:  # Only look at types with many instances
            example = next(obj for obj in objects if type(obj).__name__ == type_name)
            print(f"Example fields: {example.dict().keys()}")


from collections import defaultdict


def get_blueprint_stats():
    # Dictionary to collect lists of blueprint objects by their type name.
    blueprint_objects = defaultdict(list)

    for obj in gc.get_objects():
        try:
            # Check for attributes that are typically present on Pydantic model blueprints.
            if (
                hasattr(obj, "__pydantic_fields__")
                or hasattr(obj, "__pydantic_validator__")
                or hasattr(obj, "__pydantic_core_schema__")
            ):
                typename = type(obj).__name__
                blueprint_objects[typename].append(obj)
        except Exception:
            # Some objects might cause issues when inspected; skip them.
            continue

    # Now calculate count and total shallow size for each type.
    stats = []
    for typename, objs in blueprint_objects.items():
        total_size = sum(sys.getsizeof(o) for o in objs)
        stats.append((typename, len(objs), total_size))
    return stats


def print_top_blueprints(top_n=10):
    stats = get_blueprint_stats()
    # Sort by total_size in descending order.
    stats.sort(key=lambda x: x[2], reverse=True)

    print(f"Top {top_n} Pydantic blueprint objects by memory usage (shallow size):")
    for typename, count, total_size in stats[:top_n]:
        print(
            f"{typename}: count = {count}, total shallow size = {total_size / 1024:.2f} KiB"
        )

        # Get one instance of the blueprint object for this type (if available)
        blueprint_objs = [
            obj for obj in gc.get_objects() if type(obj).__name__ == typename
        ]
        if blueprint_objs:
            obj = blueprint_objs[0]
            # Ensure that tracemalloc is enabled and tracking this allocation.
            tb = tracemalloc.get_object_traceback(obj)
            if tb:
                print("Allocation traceback (most recent call last):")
                for frame in tb.format():
                    print(frame)
            else:
                print("No allocation traceback available for this object.")
        else:
            print("No blueprint objects found for this type.")


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    import gc

    yield
    gc.collect()
