import json
import sys
import os
import io, asyncio

import logging

logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))

from litellm import completion
import litellm

litellm.num_retries = 3
import time
import pytest


async def custom_callback(
    kwargs,  # kwargs to completion
    completion_response,  # response from completion
    start_time,
    end_time,  # start/end time
):
    # Your custom code here
    print("LITELLM: in custom callback function")
    print("kwargs", kwargs)
    print("completion_response", completion_response)
    print("start_time", start_time)
    print("end_time", end_time)
    time.sleep(1)

    return


def test_time_to_run_10_completions():
    litellm.callbacks = [custom_callback]
    start = time.time()

    asyncio.run(
        litellm.acompletion(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "hello"}]
        )
    )
    end = time.time()
    print(f"Time to run 10 completions: {end - start}")


test_time_to_run_10_completions()
