import json
import os
import sys
import time
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm


@pytest.mark.asyncio
async def test_litellm_overhead():

    litellm._turn_on_debug()
    start_time = datetime.now()
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    end_time = datetime.now()
    total_time_ms = (end_time - start_time).total_seconds() * 1000
    print(response)
    print(response._hidden_params)
    litellm_overhead_ms = response._hidden_params["litellm_overhead_time_ms"]
    # calculate percent of overhead caused by litellm
    overhead_percent = litellm_overhead_ms * 100 / total_time_ms
    print("##########################\n")
    print("total_time_ms", total_time_ms)
    print("response litellm_overhead_ms", litellm_overhead_ms)
    print("litellm overhead_percent {}%".format(overhead_percent))
    print("##########################\n")
    assert litellm_overhead_ms > 0
    assert litellm_overhead_ms < 1000

    # latency overhead should be less than total request time
    assert litellm_overhead_ms < (end_time - start_time).total_seconds() * 1000

    # latency overhead should be under 30% of total request time
    assert overhead_percent < 30

    pass
