import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import litellm
import pytest
import logging
from litellm._logging import verbose_logger


def test_datadog_logging_async():
    try:
        # litellm.set_verbose = True
        os.environ["DD_API_KEY"] = "anything"
        os.environ["_DATADOG_BASE_URL"] = (
            "https://exampleopenaiendpoint-production.up.railway.app"
        )

        os.environ["DD_SITE"] = "us5.datadoghq.com"
        os.environ["DD_API_KEY"] = "xxxxxx"

        litellm.success_callback = ["datadog"]

        percentage_diffs = []

        for run in range(1):
            print(f"\nRun {run + 1}:")

            # Test with empty success_callback
            litellm.success_callback = []
            litellm.callbacks = []
            start_time_empty_callback = asyncio.run(make_async_calls())
            print("Done with no callback test")

            # Test with datadog callback
            print("Starting datadog test")
            litellm.success_callback = ["datadog"]
            start_time_datadog = asyncio.run(make_async_calls())
            print("Done with datadog test")

            # Compare times and calculate percentage difference
            print(f"Time with success_callback='datadog': {start_time_datadog}")
            print(f"Time with empty success_callback: {start_time_empty_callback}")

            percentage_diff = (
                abs(start_time_datadog - start_time_empty_callback)
                / start_time_empty_callback
                * 100
            )
            percentage_diffs.append(percentage_diff)
            print(f"Performance difference: {percentage_diff:.2f}%")

        print("percentage_diffs", percentage_diffs)
        avg_percentage_diff = sum(percentage_diffs) / len(percentage_diffs)
        print(f"\nAverage performance difference: {avg_percentage_diff:.2f}%")

        assert (
            avg_percentage_diff < 10
        ), f"Average performance difference of {avg_percentage_diff:.2f}% exceeds 10% threshold"

    except litellm.Timeout:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")


async def make_async_calls(metadata=None, **completion_kwargs):
    total_tasks = 300
    batch_size = 100
    total_time = 0

    for batch in range(1):
        tasks = [create_async_task() for _ in range(batch_size)]

        start_time = asyncio.get_event_loop().time()
        responses = await asyncio.gather(*tasks)

        for idx, response in enumerate(responses):
            print(f"Response from Task {batch * batch_size + idx + 1}: {response}")

        await asyncio.sleep(7)

        batch_time = asyncio.get_event_loop().time() - start_time
        total_time += batch_time

    return total_time


def create_async_task(**completion_kwargs):
    litellm.set_verbose = True
    completion_args = {
        "model": "openai/chatgpt-v-2",
        "api_version": "2024-02-01",
        "messages": [{"role": "user", "content": "This is a test"}],
        "max_tokens": 5,
        "temperature": 0.7,
        "timeout": 5,
        "user": "datadog_latency_test_user",
        "mock_response": "hello from my load test",
    }
    completion_args.update(completion_kwargs)
    return asyncio.create_task(litellm.acompletion(**completion_args))
