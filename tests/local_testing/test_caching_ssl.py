#### What this tests ####
#    This tests using caching w/ litellm which requires SSL=True

import sys, os
import time
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import embedding, completion, Router
from litellm.caching.caching import Cache

messages = [{"role": "user", "content": f"who is ishaan {time.time()}"}]


def test_caching_v2():  # test in memory cache
    try:
        litellm.cache = Cache(
            type="redis",
            host="os.environ/REDIS_HOST_2",
            port="os.environ/REDIS_PORT_2",
            password="os.environ/REDIS_PASSWORD_2",
            ssl="os.environ/REDIS_SSL_2",
        )
        response1 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
        response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        litellm.cache = None  # disable cache
        if (
            response2["choices"][0]["message"]["content"]
            != response1["choices"][0]["message"]["content"]
        ):
            print(f"response1: {response1}")
            print(f"response2: {response2}")
            raise Exception()
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


# test_caching_v2()


def test_caching_router():
    """
    Test scenario where litellm.cache is set but kwargs("caching") is not. This should still return a cache hit.
    """
    try:
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-nano",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            }
        ]
        litellm.cache = Cache(
            type="redis",
            host="os.environ/REDIS_HOST_2",
            port="os.environ/REDIS_PORT_2",
            password="os.environ/REDIS_PASSWORD_2",
            ssl="os.environ/REDIS_SSL_2",
        )
        router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            set_verbose=False,
            num_retries=1,
        )  # type: ignore
        response1 = completion(model="gpt-3.5-turbo", messages=messages)
        response2 = completion(model="gpt-3.5-turbo", messages=messages)
        if (
            response2["choices"][0]["message"]["content"]
            != response1["choices"][0]["message"]["content"]
        ):
            print(f"response1: {response1}")
            print(f"response2: {response2}")
        litellm.cache = None  # disable cache
        assert (
            response2["choices"][0]["message"]["content"]
            == response1["choices"][0]["message"]["content"]
        )
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


# test_caching_router()
@pytest.mark.skip(reason="redis cloud auth errors - need to re-enable")
@pytest.mark.asyncio
async def test_redis_with_ssl():
    """
    Test connecting to redis connection pool when ssl=None


    Relevant issue:
        User was seeing this error: `TypeError: AbstractConnection.__init__() got an unexpected keyword argument 'ssl'`
    """
    from litellm._redis import get_redis_connection_pool, get_redis_async_client

    # Get the connection pool with SSL
    # REDIS_HOST_WITH_SSL is just a redis cloud instance with Transport layer security (TLS) enabled
    pool = get_redis_connection_pool(
        host=os.environ.get("REDIS_HOST_WITH_SSL"),
        port=os.environ.get("REDIS_PORT_WITH_SSL"),
        password=os.environ.get("REDIS_PASSWORD_WITH_SSL"),
        ssl=None,
    )

    # Create Redis client with the pool
    redis_client = get_redis_async_client(connection_pool=pool)

    print("pinging redis")
    print(await redis_client.ping())
    print("pinged redis")
