import json
import os
import sys
import time
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "bedrock/mistral.mistral-7b-instruct-v0:2",
        "openai/gpt-4o",
        "openai/self_hosted",
        "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
    ],
)
async def test_litellm_overhead(model):

    litellm._turn_on_debug()
    start_time = datetime.now()
    if model == "openai/self_hosted":
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Hello, world!"}],
            api_base="https://exampleopenaiendpoint-production.up.railway.app/",
        )
    else:
        response = await litellm.acompletion(
            model=model,
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

    # latency overhead should be under 40% of total request time
    assert overhead_percent < 40

    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "bedrock/mistral.mistral-7b-instruct-v0:2",
        "openai/gpt-4o",
        "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
        "openai/self_hosted",
    ],
)
async def test_litellm_overhead_stream(model):

    litellm._turn_on_debug()
    start_time = datetime.now()
    if model == "openai/self_hosted":
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Hello, world!"}],
            api_base="https://exampleopenaiendpoint-production.up.railway.app/",
            stream=True,
        )
    else:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Hello, world!"}],
            stream=True,
        )

    async for chunk in response:
        print()

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

    # latency overhead should be under 40% of total request time
    assert overhead_percent < 40

    pass


@pytest.mark.asyncio
async def test_track_llm_api_timing_async():
    # Create a mock logging object
    mock_logging = MagicMock()
    mock_logging.model_call_details = {}

    # Create an async function decorated with track_llm_api_timing
    @track_llm_api_timing()
    async def sample_async_function(logging_obj):
        await asyncio.sleep(0.1)  # Simulate some async work
        return "async result"

    # Call the function
    result = await sample_async_function(logging_obj=mock_logging)

    # Verify the result
    assert result == "async result"

    # Verify timing was recorded
    assert "llm_api_duration_ms" in mock_logging.model_call_details
    duration = mock_logging.model_call_details["llm_api_duration_ms"]
    assert duration >= 100  # Should be at least 100ms due to sleep
    assert duration < 200  # Shouldn't be too much longer than the sleep


def test_track_llm_api_timing_sync():
    # Create a mock logging object
    mock_logging = MagicMock()
    mock_logging.model_call_details = {}

    # Create a sync function decorated with track_llm_api_timing
    @track_llm_api_timing()
    def sample_sync_function(logging_obj):
        time.sleep(0.1)  # Simulate some work
        return "sync result"

    # Call the function
    result = sample_sync_function(logging_obj=mock_logging)

    # Verify the result
    assert result == "sync result"

    # Verify timing was recorded
    assert "llm_api_duration_ms" in mock_logging.model_call_details
    duration = mock_logging.model_call_details["llm_api_duration_ms"]
    assert duration >= 100  # Should be at least 100ms due to sleep
    assert duration < 200  # Shouldn't be too much longer than the sleep


def test_track_llm_api_timing_no_logging_obj():
    # Test behavior when logging_obj is not provided
    @track_llm_api_timing()
    def sample_function():
        time.sleep(0.1)
        return "result"

    # Should not raise any errors when logging_obj is missing
    result = sample_function()
    assert result == "result"
