import asyncio
import copy
import logging
import os
import sys
import time
from typing import Any
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import completion

litellm.num_retries = 3
litellm.success_callback = ["helicone"]
os.environ["HELICONE_DEBUG"] = "True"
os.environ["LITELLM_LOG"] = "DEBUG"

import pytest


def pre_helicone_setup():
    """
    Set up the logging for the 'pre_helicone_setup' function.
    """
    import logging

    logging.basicConfig(filename="helicone.log", level=logging.DEBUG)
    logger = logging.getLogger()

    file_handler = logging.FileHandler("helicone.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return


def test_helicone_logging_async():
    try:
        pre_helicone_setup()
        litellm.success_callback = []
        start_time_empty_callback = asyncio.run(make_async_calls())
        print("done with no callback test")

        print("starting helicone test")
        litellm.success_callback = ["helicone"]
        start_time_helicone = asyncio.run(make_async_calls())
        print("done with helicone test")

        print(f"Time taken with success_callback='helicone': {start_time_helicone}")
        print(f"Time taken with empty success_callback: {start_time_empty_callback}")

        assert abs(start_time_helicone - start_time_empty_callback) < 1

    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")


async def make_async_calls(metadata=None, **completion_kwargs):
    tasks = []
    for _ in range(5):
        tasks.append(create_async_task())

    start_time = asyncio.get_event_loop().time()

    responses = await asyncio.gather(*tasks)

    for idx, response in enumerate(responses):
        print(f"Response from Task {idx + 1}: {response}")

    total_time = asyncio.get_event_loop().time() - start_time

    return total_time


def create_async_task(**completion_kwargs):
    completion_args = {
        "model": "azure/chatgpt-v-3",
        "api_version": "2024-02-01",
        "messages": [{"role": "user", "content": "This is a test"}],
        "max_tokens": 5,
        "temperature": 0.7,
        "timeout": 5,
        "user": "helicone_latency_test_user",
        "mock_response": "It's simple to use and easy to get started",
    }
    completion_args.update(completion_kwargs)
    return asyncio.create_task(litellm.acompletion(**completion_args))


@pytest.mark.asyncio
@pytest.mark.skipif(
    condition=not os.environ.get("OPENAI_API_KEY", False),
    reason="Authentication missing for openai",
)
async def test_helicone_logging_metadata():
    from litellm._uuid import uuid

    litellm.success_callback = ["helicone"]

    request_id = str(uuid.uuid4())
    trace_common_metadata = {"Helicone-Property-Request-Id": request_id}

    metadata = copy.deepcopy(trace_common_metadata)
    metadata["Helicone-Property-Conversation"] = "support_issue"
    metadata["Helicone-Auth"] = os.getenv("HELICONE_API_KEY")
    response = await create_async_task(
        model="gpt-3.5-turbo",
        mock_response="Hey! how's it going?",
        messages=[
            {
                "role": "user",
                "content": f"{request_id}",
            }
        ],
        max_tokens=100,
        temperature=0.2,
        metadata=copy.deepcopy(metadata),
    )
    print(response)

    time.sleep(3)
