import sys

import os

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import litellm
from litellm._logging import verbose_logger
import logging
import time
import pytest


def test_langsmith_logging_async():
    # this tests time added to make langsmith logging calls, vs just acompletion calls
    try:

        os.environ["LANGSMITH_API_KEY"] = "lsv2_anything"
        os.environ["LANGSMITH_PROJECT"] = "pr-b"
        os.environ["LANGSMITH_BASE_URL"] = "http://0.0.0.0:8090"

        # Make 5 calls with an empty success_callback
        litellm.success_callback = []
        litellm.callbacks = []
        litellm._async_success_callback = []
        litellm._async_failure_callback = []
        litellm._async_failure_callback = []
        litellm.failure_callback = []
        start_time_empty_callback = asyncio.run(make_async_calls())
        print("done with no callback test")

        print("starting langsmith test")
        # Make 5 calls with success_callback set to "langsmith"
        litellm.success_callback = ["langsmith"]
        start_time_langsmith = asyncio.run(make_async_calls())
        print("done with langsmith test")

        # Compare the time for both scenarios
        print(f"Time taken with success_callback='langsmith': {start_time_langsmith}")
        print(f"Time taken with empty success_callback: {start_time_empty_callback}")

        # Calculate the percentage difference
        percentage_diff = (
            abs(start_time_langsmith - start_time_empty_callback)
            / start_time_empty_callback
            * 100
        )

        # Assert that the difference is not more than 10%
        assert (
            percentage_diff < 10
        ), f"Performance difference of {percentage_diff:.2f}% exceeds 10% threshold"

        print(f"Performance difference: {percentage_diff:.2f}%")

    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")


async def make_async_calls(metadata=None, **completion_kwargs):
    tasks = []
    for _ in range(100):
        tasks.append(create_async_task())

    # Measure the start time before running the tasks
    start_time = asyncio.get_event_loop().time()

    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)

    # Print the responses when tasks return
    for idx, response in enumerate(responses):
        print(f"Response from Task {idx + 1}: {response}")

    await asyncio.sleep(1)

    for _ in range(100):
        tasks.append(create_async_task())

    # Measure the start time before running the tasks
    start_time = asyncio.get_event_loop().time()

    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)

    # Print the responses when tasks return
    for idx, response in enumerate(responses):
        print(f"Response from Task {idx + 1}: {response}")

    await asyncio.sleep(1)

    for _ in range(100):
        tasks.append(create_async_task())

    # Measure the start time before running the tasks
    start_time = asyncio.get_event_loop().time()

    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)

    # Print the responses when tasks return
    for idx, response in enumerate(responses):
        print(f"Response from Task {idx + 1}: {response}")

    # Calculate the total time taken
    total_time = asyncio.get_event_loop().time() - start_time

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
        "messages": [{"role": "user", "content": "This is a test"}],
        "max_tokens": 5,
        "temperature": 0.7,
        "timeout": 5,
        "user": "langfuse_latency_test_user",
        "mock_response": "hello from my load test",
    }
    completion_args.update(completion_kwargs)
    return asyncio.create_task(litellm.acompletion(**completion_args))
