#### What this tests ####
#    This tests caching on the router
import asyncio
import os
import sys
import time
import traceback
from unittest.mock import patch
from typing import Union
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from litellm.caching import RedisCache, RedisClusterCache


## Scenarios
## 1. 2 models - openai + azure - 1 model group "gpt-3.5-turbo",
## 2. 2 models - openai, azure - 2 diff model groups, 1 caching group


@pytest.mark.asyncio
async def test_router_async_caching_with_ssl_url():
    """
    Tests when a redis url is passed to the router, if caching is correctly setup
    """
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    },
                    "tpm": 100000,
                    "rpm": 10000,
                },
            ],
            redis_url=os.getenv("REDIS_SSL_URL"),
        )

        response = await router.cache.redis_cache.ping()
        print(f"response: {response}")
        assert response == True
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


def test_router_sync_caching_with_ssl_url():
    """
    Tests when a redis url is passed to the router, if caching is correctly setup
    """
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    },
                    "tpm": 100000,
                    "rpm": 10000,
                },
            ],
            redis_url=os.getenv("REDIS_SSL_URL"),
        )

        response = router.cache.redis_cache.sync_ping()
        print(f"response: {response}")
        assert response == True
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_acompletion_caching_on_router():
    # tests acompletion + caching on router
    try:
        litellm.set_verbose = True
        model_list = [
            {
                "model_name": "gpt-4.1-nano",
                "litellm_params": {
                    "model": "gpt-4.1-nano",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "mock_response": "Hello world",
                },
                "tpm": 100000,
                "rpm": 10000,
            },
        ]

        messages = [
            {"role": "user", "content": f"write a one sentence poem {time.time()}?"}
        ]
        start_time = time.time()
        router = Router(
            model_list=model_list,
            redis_host=os.environ["REDIS_HOST"],
            redis_password=os.environ["REDIS_PASSWORD"],
            redis_port=os.environ["REDIS_PORT"],
            cache_responses=True,
            timeout=30,
            routing_strategy="simple-shuffle",
        )
        response1 = await router.acompletion(
            model="gpt-4.1-nano", messages=messages, temperature=1
        )
        print(f"response1: {response1}")
        await asyncio.sleep(5)  # add cache is async, async sleep for cache to get set

        response2 = await router.acompletion(
            model="gpt-4.1-nano", messages=messages, temperature=1
        )
        print(f"response2: {response2}")
        assert response1.id == response2.id
        assert len(response1.choices[0].message.content) > 0
        assert (
            response1.choices[0].message.content == response2.choices[0].message.content
        )
        router.reset()
    except litellm.Timeout as e:
        end_time = time.time()
        print(f"timeout error occurred: {end_time - start_time}")
        pass
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_completion_caching_on_router():
    # tests completion + caching on router
    try:
        litellm.set_verbose = True
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 1000,
                "rpm": 1,
            },
        ]

        messages = [
            {"role": "user", "content": f"write a one sentence poem {time.time()}?"}
        ]
        router = Router(
            model_list=model_list,
            redis_host=os.environ["REDIS_HOST"],
            redis_password=os.environ["REDIS_PASSWORD"],
            redis_port=os.environ["REDIS_PORT"],
            cache_responses=True,
            timeout=30,
            routing_strategy_args={"ttl": 10},
            routing_strategy="usage-based-routing",
        )
        response1 = await router.acompletion(
            model="gpt-3.5-turbo", messages=messages, temperature=1
        )
        print(f"response1: {response1}")
        await asyncio.sleep(10)
        response2 = await router.acompletion(
            model="gpt-3.5-turbo", messages=messages, temperature=1
        )
        print(f"response2: {response2}")
        assert len(response1.choices[0].message.content) > 0
        assert len(response2.choices[0].message.content) > 0

        router.reset()
    except litellm.Timeout as e:
        pass
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_acompletion_caching_with_ttl_on_router():
    # tests acompletion + caching on router
    try:
        litellm.set_verbose = True
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 100000,
                "rpm": 10000,
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-4.1-nano",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 100000,
                "rpm": 10000,
            },
        ]

        messages = [
            {"role": "user", "content": f"write a one sentence poem {time.time()}?"}
        ]
        start_time = time.time()
        router = Router(
            model_list=model_list,
            redis_host=os.environ["REDIS_HOST"],
            redis_password=os.environ["REDIS_PASSWORD"],
            redis_port=os.environ["REDIS_PORT"],
            cache_responses=True,
            timeout=30,
            routing_strategy="simple-shuffle",
        )
        response1 = await router.acompletion(
            model="gpt-3.5-turbo", messages=messages, temperature=1, ttl=0
        )
        print(f"response1: {response1}")
        await asyncio.sleep(1)  # add cache is async, async sleep for cache to get set
        response2 = await router.acompletion(
            model="gpt-3.5-turbo", messages=messages, temperature=1, ttl=0
        )
        print(f"response2: {response2}")
        assert response1.id != response2.id
        assert len(response1.choices[0].message.content) > 0
        assert (
            response1.choices[0].message.content != response2.choices[0].message.content
        )
        router.reset()
    except litellm.Timeout as e:
        end_time = time.time()
        print(f"timeout error occurred: {end_time - start_time}")
        pass
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_acompletion_caching_on_router_caching_groups():
    # tests acompletion + caching on router
    try:
        litellm.set_verbose = True
        model_list = [
            {
                "model_name": "openai-gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "mock_response": "Hello world",
                },
                "tpm": 100000,
                "rpm": 10000,
            },
            {
                "model_name": "azure-gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-nano",
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
        start_time = time.time()
        router = Router(
            model_list=model_list,
            redis_host=os.environ["REDIS_HOST"],
            redis_password=os.environ["REDIS_PASSWORD"],
            redis_port=os.environ["REDIS_PORT"],
            cache_responses=True,
            timeout=30,
            routing_strategy="simple-shuffle",
            caching_groups=[("openai-gpt-3.5-turbo", "azure-gpt-3.5-turbo")],
        )
        response1 = await router.acompletion(
            model="openai-gpt-3.5-turbo", messages=messages, temperature=1
        )
        print(f"response1: {response1}")
        await asyncio.sleep(1)  # add cache is async, async sleep for cache to get set
        response2 = await router.acompletion(
            model="azure-gpt-3.5-turbo", messages=messages, temperature=1
        )
        print(f"response2: {response2}")
        assert response1.id == response2.id
        assert len(response1.choices[0].message.content) > 0
        assert (
            response1.choices[0].message.content == response2.choices[0].message.content
        )
        router.reset()
    except litellm.Timeout as e:
        end_time = time.time()
        print(f"timeout error occurred: {end_time - start_time}")
        pass
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "startup_nodes, expected_cache_type",
    [
        pytest.param(
            [dict(host="node1.localhost", port=6379)],
            RedisClusterCache,
            id="Expects a RedisClusterCache instance when startup_nodes provided",
        ),
        pytest.param(
            None,
            RedisCache,
            id="Expects a RedisCache instance when there is no startup nodes",
        ),
    ],
)
def test_create_correct_redis_cache_instance(
    startup_nodes: Union[list[dict], None],
    expected_cache_type: Union[type[RedisClusterCache], type[RedisCache]],
):
    cache_config = dict(
        host="mockhost",
        port=6379,
        password="mock-password",
        startup_nodes=startup_nodes,
    )

    def _mock_redis_cache_init(*args, **kwargs): ...

    with patch.object(RedisCache, "__init__", _mock_redis_cache_init):
        redis_cache = Router._create_redis_cache(cache_config)
        assert isinstance(redis_cache, expected_cache_type)
