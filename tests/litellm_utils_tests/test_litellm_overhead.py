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
async def test_text_acompletion_codestral_emits_overhead_time_ms():
    import time
    # use %time 
    start_time = time.time()

    litellm._turn_on_debug()
    response = await litellm.atext_completion(
        model="text-completion-codestral/codestral-2405",
        prompt="hello",
    )
    print(response.choices)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    print("response hidden params=", response._hidden_params)

    # assert overhead is in the response
    assert response._hidden_params.get("litellm_overhead_time_ms") is not None

    # assert response_ms is in the response and in hidden params
    assert response._response_ms is not None
    assert response._hidden_params.get("_response_ms") is not None
    assert response._response_ms == response._hidden_params.get("_response_ms")

    print("total response time ms=", response._response_ms)
    print("total litellm overhead time ms=", response._hidden_params.get("litellm_overhead_time_ms"))
