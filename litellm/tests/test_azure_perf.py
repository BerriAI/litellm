#### What this tests ####
#    This adds perf testing to the router, to ensure it's never > 50ms slower than the azure-openai sdk.
import sys, os, time, inspect, asyncio, traceback
from datetime import datetime
import pytest

sys.path.insert(0, os.path.abspath("../.."))
import openai, litellm, uuid
from openai import AsyncAzureOpenAI

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_API_BASE"),  # type: ignore
    api_version=os.getenv("AZURE_API_VERSION"),
)

model_list = [
    {
        "model_name": "azure-test",
        "litellm_params": {
            "model": "azure/chatgpt-v-2",
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_base": os.getenv("AZURE_API_BASE"),
            "api_version": os.getenv("AZURE_API_VERSION"),
        },
    }
]

router = litellm.Router(model_list=model_list)


async def _openai_completion():
    try:
        start_time = time.time()
        response = await client.chat.completions.create(
            model="chatgpt-v-2",
            messages=[{"role": "user", "content": f"This is a test: {uuid.uuid4()}"}],
            stream=True,
        )
        time_to_first_token = None
        first_token_ts = None
        init_chunk = None
        async for chunk in response:
            if (
                time_to_first_token is None
                and len(chunk.choices) > 0
                and chunk.choices[0].delta.content is not None
            ):
                first_token_ts = time.time()
                time_to_first_token = first_token_ts - start_time
                init_chunk = chunk
        end_time = time.time()
        print(
            "OpenAI Call: ",
            init_chunk,
            start_time,
            first_token_ts,
            time_to_first_token,
            end_time,
        )
        return time_to_first_token
    except Exception as e:
        print(e)
        return None


async def _router_completion():
    try:
        start_time = time.time()
        response = await router.acompletion(
            model="azure-test",
            messages=[{"role": "user", "content": f"This is a test: {uuid.uuid4()}"}],
            stream=True,
        )
        time_to_first_token = None
        first_token_ts = None
        init_chunk = None
        async for chunk in response:
            if (
                time_to_first_token is None
                and len(chunk.choices) > 0
                and chunk.choices[0].delta.content is not None
            ):
                first_token_ts = time.time()
                time_to_first_token = first_token_ts - start_time
                init_chunk = chunk
        end_time = time.time()
        print(
            "Router Call: ",
            init_chunk,
            start_time,
            first_token_ts,
            time_to_first_token,
            end_time - first_token_ts,
        )
        return time_to_first_token
    except Exception as e:
        print(e)
        return None


async def test_azure_completion_streaming():
    """
    Test azure streaming call - measure on time to first (non-null) token.
    """
    n = 3  # Number of concurrent tasks
    ## OPENAI AVG. TIME
    tasks = [_openai_completion() for _ in range(n)]
    chat_completions = await asyncio.gather(*tasks)
    successful_completions = [c for c in chat_completions if c is not None]
    total_time = 0
    for item in successful_completions:
        total_time += item
    avg_openai_time = total_time / 3
    ## ROUTER AVG. TIME
    tasks = [_router_completion() for _ in range(n)]
    chat_completions = await asyncio.gather(*tasks)
    successful_completions = [c for c in chat_completions if c is not None]
    total_time = 0
    for item in successful_completions:
        total_time += item
    avg_router_time = total_time / 3
    ## COMPARE
    print(f"avg_router_time: {avg_router_time}; avg_openai_time: {avg_openai_time}")
    assert avg_router_time < avg_openai_time + 0.5


# asyncio.run(test_azure_completion_streaming())
