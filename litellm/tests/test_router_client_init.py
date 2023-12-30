#### What this tests ####
#    This tests client initialization + reinitialization on the router

#### What this tests ####
#    This tests caching on the router
import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router


async def test_router_init():
    """
    1. Initializes clients on the router with 0
    2. Checks if client is still valid
    3. Checks if new client was initialized
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo-0613",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "1234"},
            "tpm": 100000,
            "rpm": 10000,
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/chatgpt-v-2",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_base": os.getenv("AZURE_API_BASE"),
                "api_version": os.getenv("AZURE_API_VERSION"),
            },
            "tpm": 100000,
            "rpm": 10000,
        },
    ]

    messages = [
        {"role": "user", "content": f"write a one sentence poem {time.time()}?"}
    ]
    client_ttl_time = 2
    router = Router(
        model_list=model_list,
        redis_host=os.environ["REDIS_HOST"],
        redis_password=os.environ["REDIS_PASSWORD"],
        redis_port=os.environ["REDIS_PORT"],
        cache_responses=True,
        timeout=30,
        routing_strategy="simple-shuffle",
        client_ttl=client_ttl_time,
    )
    model = "gpt-3.5-turbo"
    cache_key = f"1234_async_client"
    ## ASSERT IT EXISTS AT THE START ##
    assert router.cache.get_cache(key=cache_key) is not None
    response1 = await router.acompletion(model=model, messages=messages, temperature=1)
    await asyncio.sleep(client_ttl_time)
    ## ASSERT IT'S CLEARED FROM CACHE ##
    assert router.cache.get_cache(key=cache_key, local_only=True) is None
    ## ASSERT IT EXISTS AFTER RUNNING __GET_CLIENT() ##
    assert (
        router._get_client(
            deployment=model_list[0], client_type="async", kwargs={"stream": False}
        )
        is not None
    )


# asyncio.run(test_router_init())
