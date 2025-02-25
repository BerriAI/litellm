import sys

import os

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import litellm
from litellm._logging import verbose_logger
import logging
import time
import pytest


def test_otel_logging_async():
    try:
        os.environ["OTEL_EXPORTER"] = "otlp_http"
        os.environ["OTEL_ENDPOINT"] = (
            "https://exampleopenaiendpoint-production.up.railway.app/traces"
        )
        os.environ["OTEL_HEADERS"] = "Authorization=K0BSwd"

        def single_run():
            litellm.callbacks = []
            start_time_empty = asyncio.run(make_async_calls())
            print(f"Time with empty callback: {start_time_empty}")

            litellm.callbacks = ["otel"]
            start_time_otel = asyncio.run(make_async_calls())
            print(f"Time with otel callback: {start_time_otel}")

            percent_diff = (
                abs(start_time_otel - start_time_empty) / start_time_empty * 100
            )
            print(f"Run performance difference: {percent_diff:.2f}%")
            return percent_diff

        percent_diffs = [single_run() for _ in range(3)]
        avg_percent_diff = sum(percent_diffs) / len(percent_diffs)

        print(f"Percentage differences: {percent_diffs}")
        print(f"Average performance difference: {avg_percent_diff:.2f}%")

        assert (
            avg_percent_diff < 30
        ), f"Average performance difference of {avg_percent_diff:.2f}% exceeds 30% threshold"

    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")


async def make_async_calls(metadata=None, **completion_kwargs):
    total_start_time = asyncio.get_event_loop().time()
    tasks = []

    async def create_and_run_task():
        task = create_async_task(**completion_kwargs)
        response = await task
        print(f"Response: {response}")

    for _ in range(3):  # Run for 10 seconds
        # Create 100 tasks
        tasks = []
        for _ in range(100):
            tasks.append(asyncio.create_task(create_and_run_task()))

        # Wait for any remaining tasks to complete
        await asyncio.gather(*tasks)

        await asyncio.sleep(1)

    # Calculate the total time taken
    total_time = asyncio.get_event_loop().time() - total_start_time

    return total_time


def create_async_task(**completion_kwargs):
    """
    Creates an async task for the litellm.acompletion function.
    This is just the task, but it is not run here.
    To run the task it must be awaited or used in other asyncio coroutine execution functions like asyncio.gather.
    Any kwargs passed to this function will be passed to the litellm.acompletion function.
    By default a standard set of arguments are used for the litellm.acompletion function.
    """
    completion_args = {
        "model": "openai/chatgpt-v-2",
        "api_version": "2024-02-01",
        "messages": [{"role": "user", "content": "This is a test" * 100}],
        "max_tokens": 5,
        "temperature": 0.7,
        "timeout": 5,
        "user": "langfuse_latency_test_user",
        "mock_response": "Mock response",
    }
    completion_args.update(completion_kwargs)
    return asyncio.create_task(litellm.acompletion(**completion_args))
