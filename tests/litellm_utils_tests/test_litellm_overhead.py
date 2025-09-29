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
@pytest.mark.parametrize(
    "model",
    [
        "bedrock/mistral.mistral-7b-instruct-v0:2",
        "openai/gpt-4o",
        "openai/self_hosted",
        "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
        "vertex_ai/gemini-1.5-flash",
    ],
)
async def test_litellm_overhead_non_streaming(model):
    """
    - Test we can see the litellm overhead and that it is less than 40% of the total request time
    """

    litellm._turn_on_debug()
    start_time = datetime.now()
    kwargs ={
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "model": model
    }
    #########################################################
    # Specific cases for models
    #########################################################
    if model == "vertex_ai/gemini-1.5-flash":
        kwargs["api_base"] = "https://exampleopenaiendpoint-production.up.railway.app/v1/projects/pathrise-convert-1606954137718/locations/us-central1/publishers/google/models/gemini-1.0-pro-vision-001"
        # warmup call for auth validation on vertex_ai models
        await litellm.acompletion(**kwargs)
    if model == "openai/self_hosted":
        kwargs["api_base"] = "https://exampleopenaiendpoint-production.up.railway.app/"


    response = await litellm.acompletion(
        **kwargs
    )
    #########################################################
    # End of specific cases for models
    #########################################################
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
    kwargs ={
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "model": model,
        "stream": True,
    }
    #########################################################
    # Specific cases for models
    #########################################################
    if model == "openai/self_hosted":
        kwargs["api_base"] = "https://exampleopenaiendpoint-production.up.railway.app/"
        # warmup call for auth validation on vertex_ai models
        await litellm.acompletion(**kwargs)
    
    response = await litellm.acompletion(
        **kwargs
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
