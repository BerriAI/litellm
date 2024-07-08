import asyncio
import copy
import logging
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import completion

litellm.num_retries = 3
litellm.success_callback = ["helicone"]
os.environ["HELICONE_DEBUG"] = "True"

import pytest

def search_logs(log_file_path, num_good_logs=1):
    import re

    print("\n searching logs")
    bad_logs = []
    good_logs = []
    all_logs = []
    try:
        with open(log_file_path, "r") as log_file:
            lines = log_file.readlines()
            print(f"searching logslines: {lines}")
            for line in lines:
                all_logs.append(line.strip())
                if "/v1/request/query" in line:
                    print("Found log with /v1/request/query:")
                    print(line.strip())
                    print("\n\n")
                    match = re.search(
                        r'"POST /v1/request/query HTTP/1.1" (\d+) (\d+)',
                        line,
                    )
                    if match:
                        status_code = int(match.group(1))
                        print("STATUS CODE", status_code)
                        if status_code != 200:
                            print("got a BAD log")
                            bad_logs.append(line.strip())
                        else:
                            good_logs.append(line.strip())
        print("\nBad Logs")
        print(bad_logs)
        if len(bad_logs) > 0:
            raise Exception(f"bad logs, Bad logs = {bad_logs}")
        assert (
            len(good_logs) == num_good_logs
        ), f"Did not get expected number of good logs, expected {num_good_logs}, got {len(good_logs)}. All logs \n {all_logs}"
        print("\nGood Logs")
        print(good_logs)
        if len(good_logs) <= 0:
            raise Exception(
                f"There were no Good Logs from Helicone. No logs with /v1/request/query status 200. \nAll logs:{all_logs}"
            )

    except Exception as e:
        raise e


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
        litellm.set_verbose = True

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
        "model": "azure/chatgpt-v-2",
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
async def test_helicone_logging_metadata():
    import uuid

    litellm.set_verbose = True
    litellm.success_callback = ["helicone"]

    run_id = str(uuid.uuid4())
    request_id = f"litellm-test-session-{run_id}"
    trace_common_metadata = {
        "Helicone-Property-Request-Id": request_id
    }
    for request_num in range(1, 3):
        metadata = copy.deepcopy(trace_common_metadata)
        metadata["Helicone-Property-Conversation"] = "support_issue"
        response = await create_async_task(
            model="gpt-3.5-turbo",
            mock_response=f"{request_id}",
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

        await asyncio.sleep(2)

    # Check log file for entries
    search_logs("helicone.log")
