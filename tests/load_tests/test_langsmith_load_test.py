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
    try:
        os.environ["LANGSMITH_API_KEY"] = "lsv2_anything"
        os.environ["LANGSMITH_PROJECT"] = "pr-b"
        os.environ["LANGSMITH_BASE_URL"] = (
            "https://exampleopenaiendpoint-production.up.railway.app"
        )

        percentage_diffs = []

        for run in range(3):
            print(f"\nRun {run + 1}:")

            # Test with empty success_callback
            litellm.success_callback = []
            litellm.callbacks = []
            litellm._async_success_callback = []
            litellm._async_failure_callback = []
            litellm.failure_callback = []
            start_time_empty_callback = asyncio.run(make_async_calls())
            print("Done with no callback test")

            # Test with langsmith callback
            print("Starting langsmith test")
            litellm.success_callback = ["langsmith"]
            start_time_langsmith = asyncio.run(make_async_calls())
            print("Done with langsmith test")

            # Compare times and calculate percentage difference
            print(f"Time with success_callback='langsmith': {start_time_langsmith}")
            print(f"Time with empty success_callback: {start_time_empty_callback}")

            percentage_diff = (
                abs(start_time_langsmith - start_time_empty_callback)
                / start_time_empty_callback
                * 100
            )
            percentage_diffs.append(percentage_diff)
            print(f"Performance difference: {percentage_diff:.2f}%")
        print("percentage_diffs", percentage_diffs)
        # Calculate average percentage difference
        avg_percentage_diff = sum(percentage_diffs) / len(percentage_diffs)
        print(f"\nAverage performance difference: {avg_percentage_diff:.2f}%")

        # Assert that the average difference is not more than 10%
        assert (
            avg_percentage_diff < 10
        ), f"Average performance difference of {avg_percentage_diff:.2f}% exceeds 10% threshold"

    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")

    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")


async def make_async_calls(metadata=None, **completion_kwargs):
    total_tasks = 300
    batch_size = 100
    total_time = 0

    for batch in range(3):
        tasks = [create_async_task() for _ in range(batch_size)]

        start_time = asyncio.get_event_loop().time()
        responses = await asyncio.gather(*tasks)

        for idx, response in enumerate(responses):
            print(f"Response from Task {batch * batch_size + idx + 1}: {response}")

        await asyncio.sleep(1)

        batch_time = asyncio.get_event_loop().time() - start_time
        total_time += batch_time

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
