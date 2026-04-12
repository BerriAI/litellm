import json
import os
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
import pytest
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm


# Fake Vertex AI Gemini response for mocking
FAKE_VERTEX_GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "Hello! How can I help you today?"}],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 5,
        "candidatesTokenCount": 8,
        "totalTokenCount": 13,
    },
}


def _make_fake_httpx_response(url: str) -> httpx.Response:
    """Create a fake httpx.Response that looks like a Vertex AI Gemini response."""
    response = httpx.Response(
        status_code=200,
        json=FAKE_VERTEX_GEMINI_RESPONSE,
        request=httpx.Request("POST", url),
    )
    return response


@asynccontextmanager
async def _vertex_ai_mocks():
    """Context manager that mocks Vertex AI auth and HTTP calls.

    Mocks at the httpx.AsyncClient.send level so that the
    @track_llm_api_timing decorator on AsyncHTTPHandler.post still runs,
    preserving the overhead measurement.
    """
    fake_response = _make_fake_httpx_response(
        "https://fake-vertex-endpoint/v1/models/gemini-1.5-flash:generateContent"
    )

    async def fake_send(self, request, **kwargs):
        await asyncio.sleep(0.2)  # simulate ~200ms network latency
        return fake_response

    with patch(
        "litellm.llms.vertex_ai.vertex_llm_base.VertexBase._ensure_access_token_async",
        new_callable=AsyncMock,
        return_value=("Bearer fake-token", "fake-project"),
    ), patch.object(
        httpx.AsyncClient,
        "send",
        new=fake_send,
    ):
        yield


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
        kwargs["vertex_project"] = "fake-project"
        kwargs["vertex_location"] = "us-central1"
    if model == "openai/self_hosted":
        kwargs["api_base"] = "https://exampleopenaiendpoint-production.up.railway.app/"

    async def _run():
        return await litellm.acompletion(**kwargs)

    if model == "vertex_ai/gemini-1.5-flash":
        async with _vertex_ai_mocks():
            response = await _run()
    else:
        response = await _run()
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


@pytest.mark.asyncio
async def test_litellm_overhead_cache_hit():
    """
    Test that litellm overhead is tracked on cache hits.
    Makes two identical requests and checks that the second one (cache hit) has overhead in hidden params.
    """
    from litellm.caching.caching import Cache
    
    litellm._turn_on_debug()
    litellm.cache = Cache()
    print("test2 for caching")
    litellm.set_verbose = True
    messages = [{"role": "user", "content": "Hello, world! Cache test"}]
    response1 = await litellm.acompletion(model="gpt-4.1-nano", messages=messages, caching=True)
    await asyncio.sleep(2)
    # Wait for any pending background tasks to complete
    pending_tasks = [task for task in asyncio.all_tasks() if not task.done()]
    print("all pending tasks", pending_tasks)
    if pending_tasks:
        await asyncio.wait(pending_tasks, timeout=1.0)
    
    response2 = await litellm.acompletion(model="gpt-4.1-nano", messages=messages, caching=True)
    print("RESPONSE 1", response1)
    print("RESPONSE 2", response2)
    assert response1.id == response2.id

    print("response 2 hidden params", response2._hidden_params)


    assert "_response_ms" in response2._hidden_params
    total_time_ms = response2._hidden_params["_response_ms"]
    assert response2._hidden_params["litellm_overhead_time_ms"] > 0 and response2._hidden_params["litellm_overhead_time_ms"] < total_time_ms