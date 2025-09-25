import os
import sys
import time
import traceback
from litellm._uuid import uuid

from dotenv import load_dotenv

load_dotenv()
import os
import json

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import hashlib
import random

import pytest

import litellm
from litellm import aembedding, completion, embedding
from litellm.caching.caching import Cache
from redis.asyncio import RedisCluster
from litellm.caching.redis_cluster_cache import RedisClusterCache
from unittest.mock import AsyncMock, patch, MagicMock, call
import datetime
from datetime import timedelta

# litellm.set_verbose=True

messages = [{"role": "user", "content": "who is ishaan Github?  "}]
# comment

import random
import string


def generate_random_word(length=4):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


messages = [{"role": "user", "content": "who is ishaan 5222"}]


@pytest.mark.asyncio
async def test_dual_cache_async_batch_get_cache():
    """
    Unit testing for Dual Cache async_batch_get_cache()
    - 2 item query
    - in_memory result has a partial hit (1/2)
    - hit redis for the other -> expect to return None
    - expect result = [in_memory_result, None]
    """
    from litellm.caching.caching import DualCache, InMemoryCache, RedisCache

    in_memory_cache = InMemoryCache()
    redis_cache = RedisCache()  # get credentials from environment
    dual_cache = DualCache(in_memory_cache=in_memory_cache, redis_cache=redis_cache)

    with patch.object(
        dual_cache.redis_cache, "async_batch_get_cache", new=AsyncMock()
    ) as mock_redis_cache:
        mock_redis_cache.return_value = {"test_value_2": None, "test_value": "hello"}

        await dual_cache.async_batch_get_cache(keys=["test_value", "test_value_2"])
        await dual_cache.async_batch_get_cache(keys=["test_value", "test_value_2"])

        assert mock_redis_cache.call_count == 1


def test_dual_cache_batch_get_cache():
    """
    Unit testing for Dual Cache batch_get_cache()
    - 2 item query
    - in_memory result has a partial hit (1/2)
    - hit redis for the other -> expect to return None
    - expect result = [in_memory_result, None]
    """
    from litellm.caching.caching import DualCache, InMemoryCache, RedisCache

    in_memory_cache = InMemoryCache()
    redis_cache = RedisCache()  # get credentials from environment
    dual_cache = DualCache(in_memory_cache=in_memory_cache, redis_cache=redis_cache)

    in_memory_cache.set_cache(key="test_value", value="hello world")

    result = dual_cache.batch_get_cache(
        keys=["test_value", "test_value_2"], parent_otel_span=None
    )

    assert result[0] == "hello world"
    assert result[1] == None


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_batch_get_cache_with_none_keys(sync_mode):
    """
    Unit testing for RedisCache batch_get_cache() and async_batch_get_cache()
    - test with None keys. Ensure it can safely handle when keys are None.
    - expect result = {key: None}
    """
    from litellm.caching.caching import RedisCache

    litellm._turn_on_debug()

    redis_cache = RedisCache(
        host=os.environ.get("REDIS_HOST"),
        port=os.environ.get("REDIS_PORT"),
        password=os.environ.get("REDIS_PASSWORD"),
    )
    keys_to_lookup = [
        None,
        f"test_value_{uuid.uuid4()}",
        None,
        f"test_value_2_{uuid.uuid4()}",
        None,
        f"test_value_3_{uuid.uuid4()}",
    ]
    if sync_mode:
        result = redis_cache.batch_get_cache(key_list=keys_to_lookup)
        print("result from batch_get_cache=", result)
    else:
        result = await redis_cache.async_batch_get_cache(key_list=keys_to_lookup)
        print("result from async_batch_get_cache=", result)
    expected_result = {}
    for key in keys_to_lookup:
        if key is None:
            continue
        expected_result[key] = None
    assert result == expected_result


# @pytest.mark.skip(reason="")
def test_caching_dynamic_args():  # test in memory cache
    try:
        litellm.set_verbose = True
        _redis_host_env = os.environ.pop("REDIS_HOST")
        _redis_port_env = os.environ.pop("REDIS_PORT")
        _redis_password_env = os.environ.pop("REDIS_PASSWORD")
        litellm.cache = Cache(
            type="redis",
            host=_redis_host_env,
            port=_redis_port_env,
            password=_redis_password_env,
        )
        response1 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
        response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        litellm.cache = None  # disable cache
        litellm.success_callback = []
        litellm._async_success_callback = []
        if (
            response2["choices"][0]["message"]["content"]
            != response1["choices"][0]["message"]["content"]
        ):
            print(f"response1: {response1}")
            print(f"response2: {response2}")
            pytest.fail(f"Error occurred:")
        os.environ["REDIS_HOST"] = _redis_host_env
        os.environ["REDIS_PORT"] = _redis_port_env
        os.environ["REDIS_PASSWORD"] = _redis_password_env
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


def test_caching_v2():  # test in memory cache
    try:
        litellm.set_verbose = True
        litellm.cache = Cache()
        response1 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
        response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        litellm.cache = None  # disable cache
        litellm.success_callback = []
        litellm._async_success_callback = []
        if (
            response2["choices"][0]["message"]["content"]
            != response1["choices"][0]["message"]["content"]
        ):
            print(f"response1: {response1}")
            print(f"response2: {response2}")
            pytest.fail(f"Error occurred:")
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


# test_caching_v2()


def test_caching_with_ttl():
    try:
        litellm.set_verbose = True
        litellm.cache = Cache()
        response1 = completion(
            model="gpt-3.5-turbo", messages=messages, caching=True, ttl=0
        )
        response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        litellm.cache = None  # disable cache
        litellm.success_callback = []
        litellm._async_success_callback = []
        assert (
            response2["choices"][0]["message"]["content"]
            != response1["choices"][0]["message"]["content"]
        )
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


def test_caching_with_default_ttl():
    try:
        litellm.set_verbose = True
        litellm.cache = Cache(ttl=0)
        response1 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
        response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        litellm.cache = None  # disable cache
        litellm.success_callback = []
        litellm._async_success_callback = []
        assert response2["id"] != response1["id"]
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "sync_flag",
    [True, False],
)
@pytest.mark.asyncio
async def test_caching_with_cache_controls(sync_flag):
    try:
        litellm.set_verbose = True
        litellm.cache = Cache()
        message = [{"role": "user", "content": f"Hey, how's it going? {uuid.uuid4()}"}]
        if sync_flag:
            ## TTL = 0
            response1 = completion(
                model="gpt-3.5-turbo", messages=messages, cache={"ttl": 0}
            )
            response2 = completion(
                model="gpt-3.5-turbo", messages=messages, cache={"s-maxage": 10}
            )

            assert response2["id"] != response1["id"]
        else:
            ## TTL = 0
            response1 = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=messages,
                cache={"ttl": 0},
                mock_response="Hello world",
            )
            await asyncio.sleep(10)
            response2 = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=messages,
                cache={"s-maxage": 10},
                mock_response="Hello world",
            )

            assert response2["id"] != response1["id"]

        message = [{"role": "user", "content": f"Hey, how's it going? {uuid.uuid4()}"}]
        ## TTL = 5
        if sync_flag:
            response1 = completion(
                model="gpt-3.5-turbo",
                messages=messages,
                cache={"ttl": 5},
                mock_response="Hello world",
            )
            response2 = completion(
                model="gpt-3.5-turbo",
                messages=messages,
                cache={"s-maxage": 5},
                mock_response="Hello world",
            )
            print(f"response1: {response1}")
            print(f"response2: {response2}")
            assert response2["id"] == response1["id"]
        else:
            response1 = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=messages,
                cache={"ttl": 25},
                mock_response="Hello world",
            )
            await asyncio.sleep(10)
            response2 = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=messages,
                cache={"s-maxage": 25},
                mock_response="Hello world",
            )
            print(f"response1: {response1}")
            print(f"response2: {response2}")
            assert response2["id"] == response1["id"]
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


# test_caching_with_cache_controls()


@pytest.mark.flaky(retries=3, delay=1)
def test_caching_with_models_v2():
    messages = [
        {"role": "user", "content": "who is ishaan CTO of litellm from litellm 2023"}
    ]
    litellm.cache = Cache()
    print("test2 for caching")
    litellm.set_verbose = True
    response1 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
    response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
    response3 = completion(model="azure/chatgpt-v-3", messages=messages, caching=True)
    print(f"response1: {response1}")
    print(f"response2: {response2}")
    print(f"response3: {response3}")
    litellm.cache = None
    litellm.success_callback = []
    litellm._async_success_callback = []
    if (
        response3["choices"][0]["message"]["content"]
        == response2["choices"][0]["message"]["content"]
    ):
        # if models are different, it should not return cached response
        print(f"response2: {response2}")
        print(f"response3: {response3}")
        pytest.fail(f"Error occurred:")
    if (
        response1["choices"][0]["message"]["content"]
        != response2["choices"][0]["message"]["content"]
    ):
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        pytest.fail(f"Error occurred:")


# test_caching_with_models_v2()


def c():
    litellm.enable_caching_on_provider_specific_optional_params = True
    messages = [
        {"role": "user", "content": "who is ishaan CTO of litellm from litellm 2023"}
    ]
    litellm.cache = Cache()
    print("test2 for caching")
    litellm.set_verbose = True

    response1 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        top_k=10,
        caching=True,
        mock_response="Hello: {}".format(uuid.uuid4()),
    )
    response2 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        top_k=10,
        caching=True,
        mock_response="Hello: {}".format(uuid.uuid4()),
    )
    response3 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        top_k=9,
        caching=True,
        mock_response="Hello: {}".format(uuid.uuid4()),
    )
    print(f"response1: {response1}")
    print(f"response2: {response2}")
    print(f"response3: {response3}")
    litellm.cache = None
    litellm.success_callback = []
    litellm._async_success_callback = []
    if (
        response3["choices"][0]["message"]["content"]
        == response2["choices"][0]["message"]["content"]
    ):
        # if models are different, it should not return cached response
        print(f"response2: {response2}")
        print(f"response3: {response3}")
        pytest.fail(f"Error occurred:")
    if (
        response1["choices"][0]["message"]["content"]
        != response2["choices"][0]["message"]["content"]
    ):
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        pytest.fail(f"Error occurred:")
    litellm.enable_caching_on_provider_specific_optional_params = False


embedding_large_text = (
    """
small text
"""
    * 5
)


# # test_caching_with_models()
def test_embedding_caching():
    import time

    # litellm.set_verbose = True

    litellm.cache = Cache()
    text_to_embed = [embedding_large_text]
    start_time = time.time()
    embedding1 = embedding(
        model="text-embedding-ada-002", input=text_to_embed, caching=True
    )
    end_time = time.time()
    print(f"Embedding 1 response time: {end_time - start_time} seconds")

    time.sleep(1)
    start_time = time.time()
    embedding2 = embedding(
        model="text-embedding-ada-002", input=text_to_embed, caching=True
    )
    end_time = time.time()
    # print(f"embedding2: {embedding2}")
    print(f"Embedding 2 response time: {end_time - start_time} seconds")

    litellm.cache = None
    litellm.success_callback = []
    litellm._async_success_callback = []
    assert end_time - start_time <= 0.1  # ensure 2nd response comes in in under 0.1 s
    if embedding2["data"][0]["embedding"] != embedding1["data"][0]["embedding"]:
        print(f"embedding1: {embedding1}")
        print(f"embedding2: {embedding2}")
        pytest.fail("Error occurred: Embedding caching failed")


# test_embedding_caching()


@pytest.mark.asyncio
async def test_embedding_caching_individual_items_and_then_list():
    litellm._turn_on_debug()
    litellm.cache = Cache()
    text_to_embed = [
        "hello",
        "world",
    ]
    embedding1 = await aembedding(
        model="text-embedding-ada-002", input=text_to_embed[0], caching=True
    )
    initial_prompt_tokens = embedding1.usage.prompt_tokens
    await asyncio.sleep(1)
    embedding2 = await aembedding(
        model="text-embedding-ada-002", input=text_to_embed[1], caching=True
    )
    await asyncio.sleep(1)
    embedding3 = await aembedding(
        model="text-embedding-ada-002", input=text_to_embed, caching=True
    )
    final_prompt_tokens = embedding3.usage.prompt_tokens
    assert embedding3["data"][0]["embedding"] == embedding1["data"][0]["embedding"]
    assert embedding3["data"][1]["embedding"] == embedding2["data"][0]["embedding"]
    assert embedding3._hidden_params["cache_hit"] == True
    assert embedding3.usage.prompt_tokens != 0

    ## with new input, check that prompt tokens increase
    additional_text = "this is a new text"
    text_to_embed.append(additional_text)
    embedding4 = await aembedding(
        model="text-embedding-ada-002", input=text_to_embed, caching=True
    )
    assert embedding4.usage.prompt_tokens > embedding3.usage.prompt_tokens


@pytest.mark.asyncio
async def test_embedding_caching_individual_items():
    litellm.cache = Cache()
    text_to_embed = "hello"
    embedding1 = await aembedding(
        model="text-embedding-ada-002", input=text_to_embed, caching=True
    )

    await asyncio.sleep(1)

    embedding3 = await aembedding(
        model="text-embedding-ada-002", input=text_to_embed, caching=True
    )
    final_prompt_tokens = embedding3.usage.prompt_tokens
    assert embedding3["data"][0]["embedding"] == embedding1["data"][0]["embedding"]
    assert len(embedding3.data) == 1
    assert embedding3._hidden_params["cache_hit"] == True
    assert embedding3.usage.prompt_tokens != 0


def test_embedding_caching_azure():
    print("Testing azure embedding caching")
    import time

    litellm.cache = Cache()
    text_to_embed = [embedding_large_text]

    api_key = os.environ["AZURE_API_KEY"]
    api_base = os.environ["AZURE_API_BASE"]
    api_version = os.environ["AZURE_API_VERSION"]

    os.environ["AZURE_API_VERSION"] = ""
    os.environ["AZURE_API_BASE"] = ""
    os.environ["AZURE_API_KEY"] = ""

    start_time = time.time()
    print("AZURE CONFIGS")
    print(api_version)
    print(api_key)
    print(api_base)
    embedding1 = embedding(
        model="azure/azure-embedding-model",
        input=["good morning from litellm", "this is another item"],
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
        caching=True,
    )
    end_time = time.time()
    print(f"Embedding 1 response time: {end_time - start_time} seconds")

    time.sleep(1)
    start_time = time.time()
    embedding2 = embedding(
        model="azure/azure-embedding-model",
        input=["good morning from litellm", "this is another item"],
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
        caching=True,
    )
    end_time = time.time()
    print(f"Embedding 2 response time: {end_time - start_time} seconds")

    litellm.cache = None
    litellm.success_callback = []
    litellm._async_success_callback = []
    assert end_time - start_time <= 0.1  # ensure 2nd response comes in in under 0.1 s
    if embedding2["data"][0]["embedding"] != embedding1["data"][0]["embedding"]:
        print(f"embedding1: {embedding1}")
        print(f"embedding2: {embedding2}")
        pytest.fail("Error occurred: Embedding caching failed")

    os.environ["AZURE_API_VERSION"] = api_version
    os.environ["AZURE_API_BASE"] = api_base
    os.environ["AZURE_API_KEY"] = api_key


# test_embedding_caching_azure()


@pytest.mark.asyncio
async def test_embedding_caching_azure_individual_items():
    """
    Tests caching for individual items in an embedding list

    - Cache an item
    - call aembedding(..) with the item + 1 unique item
    - compare to a 2nd aembedding (...) with 2 unique items
    ```
    embedding_1 = ["hey how's it going", "I'm doing well"]
    embedding_val_1 = embedding(...)

    embedding_2 = ["hey how's it going", "I'm fine"]
    embedding_val_2 = embedding(...)

    assert embedding_val_1[0]["id"] == embedding_val_2[0]["id"]
    ```
    """
    litellm.cache = Cache()
    common_msg = f"hey how's it going {uuid.uuid4()}"
    common_msg_2 = f"hey how's it going {uuid.uuid4()}"
    embedding_1 = [common_msg]
    embedding_2 = [
        common_msg,
        f"I'm fine {uuid.uuid4()}",
    ]

    embedding_val_1 = await aembedding(
        model="azure/azure-embedding-model", input=embedding_1, caching=True
    )
    embedding_val_2 = await aembedding(
        model="azure/azure-embedding-model", input=embedding_2, caching=True
    )
    print(f"embedding_val_2._hidden_params: {embedding_val_2._hidden_params}")
    assert embedding_val_2._hidden_params["cache_hit"] == True


@pytest.mark.asyncio
async def test_embedding_caching_azure_individual_items_reordered():
    """
    Tests caching for individual items in an embedding list

    - Cache an item
    - call aembedding(..) with the item + 1 unique item
    - compare to a 2nd aembedding (...) with 2 unique items
    ```
    embedding_1 = ["hey how's it going", "I'm doing well"]
    embedding_val_1 = embedding(...)

    embedding_2 = ["hey how's it going", "I'm fine"]
    embedding_val_2 = embedding(...)

    assert embedding_val_1[0]["id"] == embedding_val_2[0]["id"]
    ```
    """
    litellm.set_verbose = True
    litellm.cache = Cache()
    common_msg = f"{uuid.uuid4()}"
    common_msg_2 = f"hey how's it going {uuid.uuid4()}"
    embedding_1 = [common_msg_2, common_msg]
    embedding_2 = [
        common_msg,
        f"I'm fine {uuid.uuid4()}",
    ]

    embedding_val_1 = await aembedding(
        model="azure/azure-embedding-model", input=embedding_1, caching=True
    )
    print("embedding val 1", embedding_val_1)
    embedding_val_2 = await aembedding(
        model="azure/azure-embedding-model", input=embedding_2, caching=True
    )
    print("embedding val 2", embedding_val_2)
    print(f"embedding_val_2._hidden_params: {embedding_val_2._hidden_params}")
    assert embedding_val_2._hidden_params["cache_hit"] == True

    assert embedding_val_2.data[0]["embedding"] == embedding_val_1.data[1]["embedding"]
    assert embedding_val_2.data[0]["index"] != embedding_val_1.data[1]["index"]
    assert embedding_val_2.data[0]["index"] == 0
    assert embedding_val_1.data[1]["index"] == 1


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_embedding_caching_base_64():
    """ """
    litellm.set_verbose = True
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
    )
    from litellm._uuid import uuid

    inputs = [
        f"{uuid.uuid4()} hello this is ishaan",
        f"{uuid.uuid4()} hello this is ishaan again",
    ]

    embedding_val_1 = await aembedding(
        model="azure/azure-embedding-model",
        input=inputs,
        caching=True,
        encoding_format="base64",
    )
    await asyncio.sleep(5)
    print("\n\nCALL2\n\n")
    embedding_val_2 = await aembedding(
        model="azure/azure-embedding-model",
        input=inputs,
        caching=True,
        encoding_format="base64",
    )

    assert embedding_val_2._hidden_params["cache_hit"] == True
    print(embedding_val_2)
    print(embedding_val_1)
    assert embedding_val_2.data[0]["embedding"] == embedding_val_1.data[0]["embedding"]
    assert embedding_val_2.data[1]["embedding"] == embedding_val_1.data[1]["embedding"]


@pytest.mark.asyncio
async def test_embedding_caching_redis_ttl():
    """
    Test default_in_redis_ttl is used for embedding caching

    issue: https://github.com/BerriAI/litellm/issues/6010
    """
    litellm.set_verbose = True

    # Create a mock for the pipeline
    mock_pipeline = AsyncMock()
    mock_set = AsyncMock()
    mock_pipeline.__aenter__.return_value.set = mock_set
    # Patch the Redis class to return our mock
    with patch("redis.asyncio.Redis.pipeline", return_value=mock_pipeline):
        # Simulate the context manager behavior for the pipeline
        litellm.cache = Cache(
            type="redis",
            host="dummy_host",
            password="dummy_password",
            default_in_redis_ttl=2,
        )

        inputs = [
            f"{uuid.uuid4()} hello this is ishaan",
            f"{uuid.uuid4()} hello this is ishaan again",
        ]

        # Call the embedding method
        embedding_val_1 = await litellm.aembedding(
            model="azure/azure-embedding-model",
            input=inputs,
            encoding_format="base64",
        )

        await asyncio.sleep(3)  # Wait for TTL to expire

        # Check if set was called on the pipeline
        mock_set.assert_called()

        # Check if the TTL was set correctly
        for call in mock_set.call_args_list:
            args, kwargs = call
            print(f"redis pipeline set args: {args}")
            print(f"redis pipeline set kwargs: {kwargs}")
            assert kwargs.get("ex") == datetime.timedelta(
                seconds=2
            )  # Check if TTL is set to 2.5 seconds


@pytest.mark.asyncio
async def test_redis_cache_basic():
    """
    Init redis client
    - write to client
    - read from client
    """
    litellm.set_verbose = False

    random_number = random.randint(
        1, 100000
    )  # add a random number to ensure it's always adding / reading from cache
    messages = [
        {"role": "user", "content": f"write a one sentence poem about: {random_number}"}
    ]
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
    )
    response1 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    cache_key = litellm.cache.get_cache_key(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    print(f"cache_key: {cache_key}")
    litellm.cache.add_cache(result=response1, cache_key=cache_key)
    print(f"cache key pre async get: {cache_key}")
    stored_val = await litellm.cache.async_get_cache(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    print(f"stored_val: {stored_val}")
    assert stored_val["id"] == response1.id


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_redis_batch_cache_write():
    """
    Init redis client
    - write to client
    - read from client
    """
    litellm.set_verbose = True
    from litellm._uuid import uuid

    messages = [
        {"role": "user", "content": f"write a one sentence poem about: {uuid.uuid4()}"},
    ]
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
        redis_flush_size=2,
    )
    response1 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    response2 = await litellm.acompletion(
        model="anthropic/claude-3-opus-20240229",
        messages=messages,
        mock_response="good morning from this test",
    )

    # we hit the flush size, this will now send to redis
    await asyncio.sleep(2)

    response4 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    assert response1.id == response4.id


def test_redis_cache_completion():
    litellm.set_verbose = False

    random_number = random.randint(
        1, 100000
    )  # add a random number to ensure it's always adding / reading from cache
    messages = [
        {"role": "user", "content": f"write a one sentence poem about: {random_number}"}
    ]
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
    )
    print("test2 for Redis Caching - non streaming")
    response1 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        caching=True,
        max_tokens=20,
    )
    response2 = completion(
        model="gpt-3.5-turbo", messages=messages, caching=True, max_tokens=20
    )
    response3 = completion(
        model="gpt-3.5-turbo", messages=messages, caching=True, temperature=0.5
    )
    response4 = completion(model="gpt-4o-mini", messages=messages, caching=True)

    print("\nresponse 1", response1)
    print("\nresponse 2", response2)
    print("\nresponse 3", response3)
    print("\nresponse 4", response4)
    litellm.cache = None
    litellm.success_callback = []
    litellm._async_success_callback = []

    """
    1 & 2 should be exactly the same 
    1 & 3 should be different, since input params are diff
    1 & 4 should be diff, since models are diff
    """
    if (
        response1["choices"][0]["message"]["content"]
        != response2["choices"][0]["message"]["content"]
    ):  # 1 and 2 should be the same
        # 1&2 have the exact same input params. This MUST Be a CACHE HIT
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        pytest.fail(f"Error occurred:")
    if (
        response1["choices"][0]["message"]["content"]
        == response3["choices"][0]["message"]["content"]
    ):
        # if input params like seed, max_tokens are diff it should NOT be a cache hit
        print(f"response1: {response1}")
        print(f"response3: {response3}")
        pytest.fail(
            f"Response 1 == response 3. Same model, diff params shoudl not cache Error occurred:"
        )
    if (
        response1["choices"][0]["message"]["content"]
        == response4["choices"][0]["message"]["content"]
    ):
        # if models are different, it should not return cached response
        print(f"response1: {response1}")
        print(f"response4: {response4}")
        pytest.fail(f"Error occurred:")

    assert response1.id == response2.id
    assert response1.created == response2.created
    assert response1.choices[0].message.content == response2.choices[0].message.content


# test_redis_cache_completion()


def test_redis_cache_completion_stream():
    try:
        litellm.success_callback = []
        litellm._async_success_callback = []
        litellm.callbacks = []
        litellm.set_verbose = True
        random_number = random.randint(
            1, 100000
        )  # add a random number to ensure it's always adding / reading from cache
        messages = [
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_number}",
            }
        ]
        litellm.cache = Cache(
            type="redis",
            host=os.environ["REDIS_HOST"],
            port=os.environ["REDIS_PORT"],
            password=os.environ["REDIS_PASSWORD"],
        )
        print("test for caching, streaming + completion")
        response1 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=40,
            temperature=0.2,
            stream=True,
        )
        response_1_id = ""
        for chunk in response1:
            print(chunk)
            response_1_id = chunk.id
        time.sleep(0.5)
        response2 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=40,
            temperature=0.2,
            stream=True,
        )
        response_2_id = ""
        for chunk in response2:
            print(chunk)
            response_2_id = chunk.id
        assert (
            response_1_id == response_2_id
        ), f"Response 1 != Response 2. Same params, Response 1{response_1_id} != Response 2{response_2_id}"
        litellm.success_callback = []
        litellm.cache = None
        litellm.success_callback = []
        litellm._async_success_callback = []
    except Exception as e:
        print(e)
        litellm.success_callback = []
        raise e
    """

    1 & 2 should be exactly the same 
    """


# test_redis_cache_completion_stream()


@pytest.mark.skip(reason="Local test. Requires running redis cluster locally.")
@pytest.mark.asyncio
async def test_redis_cache_cluster_init_unit_test():
    try:
        from redis.asyncio import RedisCluster as AsyncRedisCluster
        from redis.cluster import RedisCluster

        from litellm.caching.caching import RedisCache

        litellm.set_verbose = True

        # List of startup nodes
        startup_nodes = [
            {"host": "127.0.0.1", "port": "7001"},
        ]

        resp = RedisCache(startup_nodes=startup_nodes)

        assert isinstance(resp.redis_client, RedisCluster)
        assert isinstance(resp.init_async_client(), AsyncRedisCluster)

        resp = litellm.Cache(type="redis", redis_startup_nodes=startup_nodes)

        assert isinstance(resp.cache, RedisCache)
        assert isinstance(resp.cache.redis_client, RedisCluster)
        assert isinstance(resp.cache.init_async_client(), AsyncRedisCluster)

    except Exception as e:
        print(f"{str(e)}\n\n{traceback.format_exc()}")
        raise e


@pytest.mark.asyncio
@pytest.mark.skip(reason="Local test. Requires running redis cluster locally.")
async def test_redis_cache_cluster_init_with_env_vars_unit_test():
    try:
        import json

        from redis.asyncio import RedisCluster as AsyncRedisCluster
        from redis.cluster import RedisCluster

        from litellm.caching.caching import RedisCache

        litellm.set_verbose = True

        # List of startup nodes
        startup_nodes = [
            {"host": "127.0.0.1", "port": "7001"},
            {"host": "127.0.0.1", "port": "7003"},
            {"host": "127.0.0.1", "port": "7004"},
            {"host": "127.0.0.1", "port": "7005"},
            {"host": "127.0.0.1", "port": "7006"},
            {"host": "127.0.0.1", "port": "7007"},
        ]

        # set startup nodes in environment variables
        os.environ["REDIS_CLUSTER_NODES"] = json.dumps(startup_nodes)
        print("REDIS_CLUSTER_NODES", os.environ["REDIS_CLUSTER_NODES"])

        # unser REDIS_HOST, REDIS_PORT, REDIS_PASSWORD
        os.environ.pop("REDIS_HOST", None)
        os.environ.pop("REDIS_PORT", None)
        os.environ.pop("REDIS_PASSWORD", None)

        resp = RedisCache()
        print("response from redis cache", resp)
        assert isinstance(resp.redis_client, RedisCluster)
        assert isinstance(resp.init_async_client(), AsyncRedisCluster)

        resp = litellm.Cache(type="redis")

        assert isinstance(resp.cache, RedisCache)
        assert isinstance(resp.cache.redis_client, RedisCluster)
        assert isinstance(resp.cache.init_async_client(), AsyncRedisCluster)

    except Exception as e:
        print(f"{str(e)}\n\n{traceback.format_exc()}")
        raise e


@pytest.mark.asyncio
async def test_redis_cache_acompletion_stream():
    try:
        litellm.set_verbose = True
        random_word = generate_random_word()
        messages = [
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_word}",
            }
        ]
        litellm.cache = Cache(
            type="redis",
            host=os.environ["REDIS_HOST"],
            port=os.environ["REDIS_PORT"],
            password=os.environ["REDIS_PASSWORD"],
        )
        print("test for caching, streaming + completion")
        response_1_content = ""
        response_2_content = ""

        response1 = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=40,
            temperature=1,
            stream=True,
        )
        async for chunk in response1:
            response_1_content += chunk.choices[0].delta.content or ""
        print(response_1_content)

        await asyncio.sleep(0.5)
        print("\n\n Response 1 content: ", response_1_content, "\n\n")

        response2 = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=40,
            temperature=1,
            stream=True,
        )
        async for chunk in response2:
            response_2_content += chunk.choices[0].delta.content or ""
        print(response_2_content)

        print("\nresponse 1", response_1_content)
        print("\nresponse 2", response_2_content)
        assert (
            response_1_content == response_2_content
        ), f"Response 1 != Response 2. Same params, Response 1{response_1_content} != Response 2{response_2_content}"
        litellm.cache = None
        litellm.success_callback = []
        litellm._async_success_callback = []
    except Exception as e:
        print(f"{str(e)}\n\n{traceback.format_exc()}")
        raise e


# test_redis_cache_acompletion_stream()


@pytest.mark.asyncio
async def test_redis_cache_atext_completion():
    try:
        litellm.set_verbose = True
        prompt = f"write a one sentence poem about: {uuid.uuid4()}"
        litellm.cache = Cache(
            type="redis",
            host=os.environ["REDIS_HOST"],
            port=os.environ["REDIS_PORT"],
            password=os.environ["REDIS_PASSWORD"],
            supported_call_types=["atext_completion"],
        )
        print("test for caching, atext_completion")

        response1 = await litellm.atext_completion(
            model="gpt-3.5-turbo-instruct", prompt=prompt, max_tokens=40, temperature=1
        )

        await asyncio.sleep(0.5)
        print("\n\n Response 1 content: ", response1, "\n\n")

        response2 = await litellm.atext_completion(
            model="gpt-3.5-turbo-instruct", prompt=prompt, max_tokens=40, temperature=1
        )

        print(response2)

        assert response1.id == response2.id
    except Exception as e:
        print(f"{str(e)}\n\n{traceback.format_exc()}")
        raise e


@pytest.mark.asyncio
async def test_redis_cache_acompletion_stream_bedrock():
    import asyncio

    try:
        litellm.set_verbose = True
        random_word = generate_random_word()
        messages = [
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_word}",
            }
        ]
        litellm.cache = Cache(type="redis")
        print("test for caching, streaming + completion")
        response_1_content = ""
        response_2_content = ""

        response1 = await litellm.acompletion(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=messages,
            max_tokens=40,
            temperature=1,
            stream=True,
        )
        async for chunk in response1:
            print(chunk)
            response_1_content += chunk.choices[0].delta.content or ""
        print(response_1_content)

        await asyncio.sleep(1)
        print("\n\n Response 1 content: ", response_1_content, "\n\n")

        response2 = await litellm.acompletion(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=messages,
            max_tokens=40,
            temperature=1,
            stream=True,
        )
        async for chunk in response2:
            print(chunk)
            response_2_content += chunk.choices[0].delta.content or ""
        print(response_2_content)

        print("\nfinal response 1", response_1_content)
        print("\nfinal response 2", response_2_content)
        assert (
            response_1_content == response_2_content
        ), f"Response 1 != Response 2. Same params, Response 1{response_1_content} != Response 2{response_2_content}"

        litellm.cache = None
        litellm.success_callback = []
        litellm._async_success_callback = []
    except Exception as e:
        print(e)
        raise e


# @pytest.mark.skip(reason="AWS Suspended Account")
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_s3_cache_stream_azure(sync_mode):
    try:
        litellm.set_verbose = True
        random_word = generate_random_word()
        messages = [
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_word}",
            }
        ]
        litellm.cache = Cache(
            type="s3",
            s3_bucket_name="litellm-proxy",
            s3_region_name="us-west-2",
        )
        print("s3 Cache: test for caching, streaming + completion")
        response_1_content = ""
        response_2_content = ""

        response_1_created = ""
        response_2_created = ""

        if sync_mode:
            response1 = litellm.completion(
                model="azure/chatgpt-v-3",
                messages=messages,
                max_tokens=40,
                temperature=1,
                stream=True,
            )
            for chunk in response1:
                print(chunk)
                response_1_created = chunk.created
                response_1_content += chunk.choices[0].delta.content or ""
            print(response_1_content)
        else:
            response1 = await litellm.acompletion(
                model="azure/chatgpt-v-3",
                messages=messages,
                max_tokens=40,
                temperature=1,
                stream=True,
            )
            async for chunk in response1:
                print(chunk)
                response_1_created = chunk.created
                response_1_content += chunk.choices[0].delta.content or ""
            print(response_1_content)

        if sync_mode:
            time.sleep(0.5)
        else:
            await asyncio.sleep(0.5)
        print("\n\n Response 1 content: ", response_1_content, "\n\n")

        if sync_mode:
            response2 = litellm.completion(
                model="azure/chatgpt-v-3",
                messages=messages,
                max_tokens=40,
                temperature=1,
                stream=True,
            )
            for chunk in response2:
                print(chunk)
                response_2_content += chunk.choices[0].delta.content or ""
                response_2_created = chunk.created
            print(response_2_content)
        else:
            response2 = await litellm.acompletion(
                model="azure/chatgpt-v-3",
                messages=messages,
                max_tokens=40,
                temperature=1,
                stream=True,
            )
            async for chunk in response2:
                print(chunk)
                response_2_content += chunk.choices[0].delta.content or ""
                response_2_created = chunk.created
            print(response_2_content)

        print("\nresponse 1", response_1_content)
        print("\nresponse 2", response_2_content)

        assert (
            response_1_content == response_2_content
        ), f"Response 1 != Response 2. Same params, Response 1{response_1_content} != Response 2{response_2_content}"

        # prioritizing getting a new deploy out - will look at this in the next deploy
        # print("response 1 created", response_1_created)
        # print("response 2 created", response_2_created)

        # assert response_1_created == response_2_created

        litellm.cache = None
        litellm.success_callback = []
        litellm._async_success_callback = []
    except Exception as e:
        print(e)
        raise e


# test_s3_cache_acompletion_stream_azure()


@pytest.mark.skip(reason="AWS Suspended Account")
@pytest.mark.asyncio
async def test_s3_cache_acompletion_azure():
    import asyncio
    import logging
    import tracemalloc

    tracemalloc.start()
    logging.basicConfig(level=logging.DEBUG)

    try:
        litellm.set_verbose = True
        random_word = generate_random_word()
        messages = [
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_word}",
            }
        ]
        litellm.cache = Cache(
            type="s3",
            s3_bucket_name="litellm-my-test-bucket-2",
            s3_region_name="us-east-1",
        )
        print("s3 Cache: test for caching, streaming + completion")

        response1 = await litellm.acompletion(
            model="azure/chatgpt-v-3",
            messages=messages,
            max_tokens=40,
            temperature=1,
        )
        print(response1)

        time.sleep(2)

        response2 = await litellm.acompletion(
            model="azure/chatgpt-v-3",
            messages=messages,
            max_tokens=40,
            temperature=1,
        )

        print(response2)

        assert response1.id == response2.id

        litellm.cache = None
        litellm.success_callback = []
        litellm._async_success_callback = []
    except Exception as e:
        print(e)
        raise e


# test_redis_cache_acompletion_stream_bedrock()
# redis cache with custom keys
def custom_get_cache_key(*args, **kwargs):
    # return key to use for your cache:
    key = (
        kwargs.get("model", "")
        + str(kwargs.get("messages", ""))
        + str(kwargs.get("temperature", ""))
        + str(kwargs.get("logit_bias", ""))
    )
    return key


def test_custom_redis_cache_with_key():
    messages = [{"role": "user", "content": "write a one line story"}]
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
    )
    litellm.cache.get_cache_key = custom_get_cache_key

    local_cache = {}

    def set_cache(key, value):
        local_cache[key] = value

    def get_cache(key):
        if key in local_cache:
            return local_cache[key]

    litellm.cache.cache.set_cache = set_cache
    litellm.cache.cache.get_cache = get_cache

    # patch this redis cache get and set call

    response1 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        caching=True,
        num_retries=3,
    )
    response2 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        caching=True,
        num_retries=3,
    )
    response3 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        caching=False,
        num_retries=3,
    )

    print(f"response1: {response1}")
    print(f"response2: {response2}")
    print(f"response3: {response3}")

    if (
        response3["choices"][0]["message"]["content"]
        == response2["choices"][0]["message"]["content"]
    ):
        pytest.fail(f"Error occurred:")
    litellm.cache = None
    litellm.success_callback = []
    litellm._async_success_callback = []


# test_custom_redis_cache_with_key()


def test_cache_override():
    # test if we can override the cache, when `caching=False` but litellm.cache = Cache() is set
    # in this case it should not return cached responses
    litellm.cache = Cache()
    print("Testing cache override")
    litellm.set_verbose = True

    # test embedding
    response1 = embedding(
        model="text-embedding-ada-002", input=["hello who are you"], caching=False
    )

    start_time = time.time()

    response2 = embedding(
        model="text-embedding-ada-002", input=["hello who are you"], caching=False
    )

    end_time = time.time()
    print(f"Embedding 2 response time: {end_time - start_time} seconds")

    assert (
        end_time - start_time > 0.05
    )  # ensure 2nd response comes in over 0.05s. This should not be cached.


# test_cache_override()


@pytest.mark.asyncio
async def test_cache_control_overrides():
    # we use the cache controls to ensure there is no cache hit on this test
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
    )
    print("Testing cache override")
    litellm.set_verbose = True
    from litellm._uuid import uuid

    unique_num = str(uuid.uuid4())

    start_time = time.time()

    response1 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "hello who are you" + unique_num,
            }
        ],
        caching=True,
    )

    print(response1)

    await asyncio.sleep(2)

    response2 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "hello who are you" + unique_num,
            }
        ],
        caching=True,
        cache={"no-cache": True},
    )

    print(response2)

    assert response1.id != response2.id


def test_sync_cache_control_overrides():
    # we use the cache controls to ensure there is no cache hit on this test
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
    )
    print("Testing cache override")
    litellm.set_verbose = True
    from litellm._uuid import uuid

    unique_num = str(uuid.uuid4())

    start_time = time.time()

    response1 = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "hello who are you" + unique_num,
            }
        ],
        caching=True,
    )

    print(response1)

    time.sleep(2)

    response2 = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "hello who are you" + unique_num,
            }
        ],
        caching=True,
        cache={"no-cache": True},
    )

    print(response2)

    assert response1.id != response2.id


def test_custom_redis_cache_params():
    # test if we can init redis with **kwargs
    try:
        litellm.cache = Cache(
            type="redis",
            host=os.environ["REDIS_HOST"],
            port=os.environ["REDIS_PORT"],
            password=os.environ["REDIS_PASSWORD"],
            db=0,
        )

        print(litellm.cache.cache.redis_client)
        litellm.cache = None
        litellm.success_callback = []
        litellm._async_success_callback = []
    except Exception as e:
        pytest.fail(f"Error occurred: {str(e)}")


def test_get_cache_key():
    from litellm.caching.caching import Cache

    try:
        print("Testing get_cache_key")
        cache_instance = Cache()
        cache_key = cache_instance.get_cache_key(
            **{
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": "write a one sentence poem about: 7510"}
                ],
                "max_tokens": 40,
                "temperature": 0.2,
                "stream": True,
                "litellm_call_id": "ffe75e7e-8a07-431f-9a74-71a5b9f35f0b",
                "litellm_logging_obj": {},
            }
        )
        cache_key_2 = cache_instance.get_cache_key(
            **{
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": "write a one sentence poem about: 7510"}
                ],
                "max_tokens": 40,
                "temperature": 0.2,
                "stream": True,
                "litellm_call_id": "ffe75e7e-8a07-431f-9a74-71a5b9f35f0b",
                "litellm_logging_obj": {},
            }
        )
        cache_key_str = "model: gpt-3.5-turbomessages: [{'role': 'user', 'content': 'write a one sentence poem about: 7510'}]max_tokens: 40temperature: 0.2stream: True"
        hash_object = hashlib.sha256(cache_key_str.encode())
        # Hexadecimal representation of the hash
        hash_hex = hash_object.hexdigest()
        assert cache_key == hash_hex
        assert (
            cache_key_2 == hash_hex
        ), f"{cache_key} != {cache_key_2}. The same kwargs should have the same cache key across runs"

        embedding_cache_key = cache_instance.get_cache_key(
            **{
                "model": "azure/azure-embedding-model",
                "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
                "api_key": "",
                "api_version": "2023-07-01-preview",
                "timeout": None,
                "max_retries": 0,
                "input": ["hi who is ishaan"],
                "caching": True,
                "client": "<openai.lib.azure.AsyncAzureOpenAI object at 0x12b6a1060>",
            }
        )

        print(embedding_cache_key)

        embedding_cache_key_str = (
            "model: azure/azure-embedding-modelinput: ['hi who is ishaan']"
        )
        hash_object = hashlib.sha256(embedding_cache_key_str.encode())
        # Hexadecimal representation of the hash
        hash_hex = hash_object.hexdigest()
        assert (
            embedding_cache_key == hash_hex
        ), f"{embedding_cache_key} != 'model: azure/azure-embedding-modelinput: ['hi who is ishaan']'. The same kwargs should have the same cache key across runs"

        # Proxy - embedding cache, test if embedding key, gets model_group and not model
        embedding_cache_key_2 = cache_instance.get_cache_key(
            **{
                "model": "azure/azure-embedding-model",
                "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
                "api_key": "",
                "api_version": "2023-07-01-preview",
                "timeout": None,
                "max_retries": 0,
                "input": ["hi who is ishaan"],
                "caching": True,
                "client": "<openai.lib.azure.AsyncAzureOpenAI object at 0x12b6a1060>",
                "proxy_server_request": {
                    "url": "http://0.0.0.0:8000/embeddings",
                    "method": "POST",
                    "headers": {
                        "host": "0.0.0.0:8000",
                        "user-agent": "curl/7.88.1",
                        "accept": "*/*",
                        "content-type": "application/json",
                        "content-length": "80",
                    },
                    "body": {
                        "model": "azure-embedding-model",
                        "input": ["hi who is ishaan"],
                    },
                },
                "user": None,
                "metadata": {
                    "user_api_key": None,
                    "headers": {
                        "host": "0.0.0.0:8000",
                        "user-agent": "curl/7.88.1",
                        "accept": "*/*",
                        "content-type": "application/json",
                        "content-length": "80",
                    },
                    "model_group": "EMBEDDING_MODEL_GROUP",
                    "deployment": "azure/azure-embedding-model-ModelID-azure/azure-embedding-modelhttps://openai-gpt-4-test-v-1.openai.azure.com/2023-07-01-preview",
                },
                "model_info": {
                    "mode": "embedding",
                    "base_model": "text-embedding-ada-002",
                    "id": "20b2b515-f151-4dd5-a74f-2231e2f54e29",
                },
                "litellm_call_id": "2642e009-b3cd-443d-b5dd-bb7d56123b0e",
                "litellm_logging_obj": "<litellm.utils.Logging object at 0x12f1bddb0>",
            }
        )

        print(embedding_cache_key_2)
        embedding_cache_key_str_2 = (
            "model: EMBEDDING_MODEL_GROUPinput: ['hi who is ishaan']"
        )
        hash_object = hashlib.sha256(embedding_cache_key_str_2.encode())
        # Hexadecimal representation of the hash
        hash_hex = hash_object.hexdigest()
        assert embedding_cache_key_2 == hash_hex
        print("passed!")
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred:", e)


# test_get_cache_key()


def test_cache_context_managers():
    litellm.set_verbose = True
    litellm.cache = Cache(type="redis")

    # cache is on, disable it
    litellm.disable_cache()
    assert litellm.cache == None
    assert "cache" not in litellm.success_callback
    assert "cache" not in litellm._async_success_callback

    # disable a cache that is off
    litellm.disable_cache()
    assert litellm.cache == None
    assert "cache" not in litellm.success_callback
    assert "cache" not in litellm._async_success_callback

    litellm.enable_cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
    )

    assert litellm.cache != None
    assert litellm.cache.type == "redis"

    print("VARS of litellm.cache", vars(litellm.cache))


def test_redis_semantic_cache_completion():
    litellm.set_verbose = True
    import logging

    logging.basicConfig(level=logging.DEBUG)

    print("testing semantic caching")
    litellm.cache = Cache(
        type="redis-semantic",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
        similarity_threshold=0.8,
        redis_semantic_cache_embedding_model="text-embedding-ada-002",
    )
    response1 = completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "write a one sentence poem about summer",
            }
        ],
        max_tokens=20,
    )
    print(f"response1: {response1}")

    response2 = completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "write a one sentence poem about summertime",
            }
        ],
        max_tokens=20,
    )
    print(f"response2: {response2}")
    assert response1.id == response2.id


# test_redis_cache_completion()


@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_redis_semantic_cache_acompletion():
    litellm.set_verbose = True
    import logging

    logging.basicConfig(level=logging.DEBUG)

    print("testing semantic caching")
    litellm.cache = Cache(
        type="redis-semantic",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
        similarity_threshold=0.7,
    )
    response1 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "write a one sentence poem about summer",
            }
        ],
        max_tokens=5,
    )
    print(f"response1: {response1}")

    await asyncio.sleep(2)

    response2 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "write a one sentence poem about summertime",
            }
        ],
        max_tokens=5,
    )
    print(f"response2: {response2}")
    assert response1.id == response2.id


def test_caching_redis_simple(caplog, capsys):
    """
    Relevant issue - https://github.com/BerriAI/litellm/issues/4511
    """
    litellm.set_verbose = True  ## REQUIRED FOR TEST.
    litellm.cache = Cache(
        type="redis", url=os.getenv("REDIS_SSL_URL")
    )  # passing `supported_call_types = ["completion"]` has no effect

    s = time.time()

    uuid_str = str(uuid.uuid4())
    x = completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Hello, how are you? Wink {uuid_str}"}],
        stream=True,
    )
    for m in x:
        print(m)
    print(time.time() - s)

    s2 = time.time()
    x = completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Hello, how are you? Wink {uuid_str}"}],
        stream=True,
    )
    for m in x:
        print(m)
    print(time.time() - s2)

    redis_async_caching_error = False
    redis_service_logging_error = False
    captured = capsys.readouterr()
    captured_logs = [rec.message for rec in caplog.records]

    print(f"captured_logs: {captured_logs}")
    for item in captured_logs:
        if (
            "Error connecting to Async Redis client" in item
            or "Set ASYNC Redis Cache" in item
        ):
            redis_async_caching_error = True

        if "ServiceLogging.async_service_success_hook" in item:
            redis_service_logging_error = True

    assert redis_async_caching_error is False
    assert redis_service_logging_error is False
    assert "async success_callback: reaches cache for logging" not in captured.out


@pytest.mark.asyncio()
async def test_cache_default_off_acompletion():
    litellm.set_verbose = True
    import logging

    from litellm._logging import verbose_logger

    verbose_logger.setLevel(logging.DEBUG)

    from litellm.caching.caching import CacheMode

    random_number = random.randint(
        1, 100000
    )  # add a random number to ensure it's always adding /reading from cache
    litellm.cache = Cache(
        type="local",
        mode=CacheMode.default_off,
    )

    ### No Cache hits when it's default off

    response1 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_number}",
            }
        ],
        mock_response="hello",
        max_tokens=20,
    )
    print(f"Response1: {response1}")

    response2 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_number}",
            }
        ],
        max_tokens=20,
    )
    print(f"Response2: {response2}")
    assert response1.id != response2.id

    ## Cache hits when it's default off and then opt in

    response3 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_number}",
            }
        ],
        mock_response="hello",
        cache={"use-cache": True},
        metadata={"key": "value"},
        max_tokens=20,
    )
    print(f"Response3: {response3}")

    await asyncio.sleep(2)

    response4 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_number}",
            }
        ],
        cache={"use-cache": True},
        metadata={"key": "value"},
        max_tokens=20,
    )
    print(f"Response4: {response4}")
    assert response3.id == response4.id


@pytest.mark.skip(reason="local test. Requires sentinel setup.")
@pytest.mark.asyncio
async def test_redis_sentinel_caching():
    """
    Init redis client
    - write to client
    - read from client
    """
    litellm.set_verbose = False

    random_number = random.randint(
        1, 100000
    )  # add a random number to ensure it's always adding / reading from cache
    messages = [
        {"role": "user", "content": f"write a one sentence poem about: {random_number}"}
    ]

    litellm.cache = Cache(
        type="redis",
        # host=os.environ["REDIS_HOST"],
        # port=os.environ["REDIS_PORT"],
        # password=os.environ["REDIS_PASSWORD"],
        service_name="mymaster",
        sentinel_nodes=[("localhost", 26379)],
    )
    response1 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    cache_key = litellm.cache.get_cache_key(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    print(f"cache_key: {cache_key}")
    litellm.cache.add_cache(result=response1, cache_key=cache_key)
    print(f"cache key pre async get: {cache_key}")
    stored_val = litellm.cache.get_cache(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    print(f"stored_val: {stored_val}")
    assert stored_val["id"] == response1.id

    stored_val_2 = await litellm.cache.async_get_cache(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    print(f"stored_val: {stored_val}")
    assert stored_val_2["id"] == response1.id


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=2)
async def test_redis_proxy_batch_redis_get_cache():
    """
    Tests batch_redis_get.py

    - make 1st call -> expect miss
    - make 2nd call -> expect hit
    """

    from litellm.caching.caching import Cache, DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.batch_redis_get import _PROXY_BatchRedisRequests

    litellm.cache = Cache(
        type="redis",
        host=os.getenv("REDIS_HOST"),
        port=os.getenv("REDIS_PORT"),
        password=os.getenv("REDIS_PASSWORD"),
        namespace="test_namespace",
    )

    batch_redis_get_obj = (
        _PROXY_BatchRedisRequests()
    )  # overrides the .async_get_cache method

    user_api_key_cache = DualCache()

    from litellm._uuid import uuid

    batch_redis_get_obj.in_memory_cache = user_api_key_cache.in_memory_cache

    messages = [{"role": "user", "content": "hi {}".format(uuid.uuid4())}]
    # 1st call -> expect miss
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=messages,
        mock_response="hello",
    )

    assert response is not None
    assert "cache_key" not in response._hidden_params
    print(response._hidden_params)

    await asyncio.sleep(1)

    # 2nd call -> expect hit
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=messages,
        mock_response="hello",
    )

    print(response._hidden_params)
    assert "cache_key" in response._hidden_params


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_logging_turn_off_message_logging_streaming(sync_mode):
    litellm.turn_off_message_logging = True
    mock_obj = Cache(type="local")
    litellm.cache = mock_obj

    with patch.object(mock_obj, "add_cache") as mock_client, patch.object(
        mock_obj, "async_add_cache"
    ) as mock_async_client:
        print(f"mock_obj.add_cache: {mock_obj.add_cache}")

        if sync_mode is True:
            resp = litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="hello",
                stream=True,
            )

            for chunk in resp:
                continue

            time.sleep(1)
            mock_client.assert_called_once()
            print(f"mock_client.call_args: {mock_client.call_args}")
            assert mock_client.call_args.args[0].choices[0].message.content == "hello"
        else:
            resp = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="hello",
                stream=True,
            )

            async for chunk in resp:
                continue

            await asyncio.sleep(1)

            mock_async_client.assert_called_once()
            print(f"mock_async_client.call_args: {mock_async_client.call_args.args[0]}")
            print(
                f"mock_async_client.call_args: {json.loads(mock_async_client.call_args.args[0])}"
            )
            json_mock = json.loads(mock_async_client.call_args.args[0])
            try:
                assert json_mock["choices"][0]["message"]["content"] == "hello"
            except Exception as e:
                print(
                    f"mock_async_client.call_args.args[0]: {mock_async_client.call_args.args[0]}"
                )
                print(
                    f"mock_async_client.call_args.args[0]['choices']: {mock_async_client.call_args.args[0]['choices']}"
                )
                raise e


def test_basic_caching_import():
    from litellm.caching import Cache

    assert Cache is not None
    print("Cache imported successfully")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio()
async def test_caching_kwargs_input(sync_mode):
    from litellm import acompletion
    from litellm.caching.caching_handler import LLMCachingHandler
    from litellm.types.utils import (
        Choices,
        EmbeddingResponse,
        Message,
        ModelResponse,
        Usage,
        CompletionTokensDetailsWrapper,
        PromptTokensDetailsWrapper,
    )
    from datetime import datetime

    llm_caching_handler = LLMCachingHandler(
        original_function=acompletion, request_kwargs={}, start_time=datetime.now()
    )

    input = {
        "result": ModelResponse(
            id="chatcmpl-AJ119H5XsDnYiZPp5axJ5d7niwqeR",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Hello! I'm just a computer program, so I don't have feelings, but I'm here to assist you. How can I help you today?",
                        role="assistant",
                        tool_calls=None,
                        function_call=None,
                    ),
                )
            ],
            created=1729095507,
            model="gpt-3.5-turbo-0125",
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(
                completion_tokens=31,
                prompt_tokens=16,
                total_tokens=47,
                completion_tokens_details=CompletionTokensDetailsWrapper(
                    audio_tokens=None, reasoning_tokens=0
                ),
                prompt_tokens_details=PromptTokensDetailsWrapper(
                    audio_tokens=None, cached_tokens=0
                ),
            ),
            service_tier=None,
        ),
        "kwargs": {
            "messages": [{"role": "user", "content": "42HHey, how's it going?"}],
            "caching": True,
            "litellm_call_id": "fae2aa4f-9f75-4f11-8c9c-63ab8d9fae26",
            "preset_cache_key": "2f69f5640d5e0f25315d0e132f1278bb643554d14565d2c61d61564b10ade90f",
        },
        "args": ("gpt-3.5-turbo",),
    }
    if sync_mode is True:
        llm_caching_handler.sync_set_cache(**input)
    else:
        input["original_function"] = acompletion
        await llm_caching_handler.async_set_cache(**input)


@pytest.mark.skip(reason="audio caching not supported yet")
@pytest.mark.parametrize("stream", [False])  # True,
@pytest.mark.asyncio()
async def test_audio_caching(stream):
    litellm.cache = Cache(type="local")

    ## CALL 1 - no cache hit
    completion = await litellm.acompletion(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "pcm16"},
        messages=[{"role": "user", "content": "response in 1 word - yes or no"}],
        stream=stream,
    )

    assert "cache_hit" not in completion._hidden_params

    ## CALL 2 - cache hit
    completion = await litellm.acompletion(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "pcm16"},
        messages=[{"role": "user", "content": "response in 1 word - yes or no"}],
        stream=stream,
    )

    assert "cache_hit" in completion._hidden_params


def test_redis_caching_default_ttl():
    """
    Ensure that the default redis cache TTL is 60s
    """
    from litellm.caching.redis_cache import RedisCache

    litellm.default_redis_ttl = 120

    cache_obj = RedisCache()
    assert cache_obj.default_ttl == 120


@pytest.mark.asyncio()
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_redis_caching_llm_caching_ttl(sync_mode):
    """
    Ensure default redis cache ttl is used for a sample redis cache object
    """
    from litellm.caching.redis_cache import RedisCache

    litellm.default_redis_ttl = 120
    cache_obj = RedisCache()
    assert cache_obj.default_ttl == 120

    if sync_mode is False:
        # Create an AsyncMock for the Redis client
        mock_redis_instance = AsyncMock()

        # Make sure the mock can be used as an async context manager
        mock_redis_instance.__aenter__.return_value = mock_redis_instance
        mock_redis_instance.__aexit__.return_value = None

    ## Set cache
    if sync_mode is True:
        with patch.object(cache_obj.redis_client, "set") as mock_set:
            cache_obj.set_cache(key="test", value="test")
            mock_set.assert_called_once_with(name="test", value="test", ex=120)
    else:

        # Patch self.init_async_client to return our mock Redis client
        with patch.object(
            cache_obj, "init_async_client", return_value=mock_redis_instance
        ):
            # Call async_set_cache
            await cache_obj.async_set_cache(key="test", value="test_value")

            # Verify that the set method was called on the mock Redis instance
            mock_redis_instance.set.assert_called_once_with(
                name="test", value='"test_value"', ex=120, nx=False
            )

    ## Increment cache
    if sync_mode is True:
        with patch.object(cache_obj.redis_client, "ttl") as mock_incr:
            cache_obj.increment_cache(key="test", value=1)
            mock_incr.assert_called_once_with("test")
    else:
        # Patch self.init_async_client to return our mock Redis client
        with patch.object(
            cache_obj, "init_async_client", return_value=mock_redis_instance
        ):
            # Call async_set_cache
            await cache_obj.async_increment(key="test", value="test_value")

            # Verify that the set method was called on the mock Redis instance
            mock_redis_instance.ttl.assert_called_once_with("test")


@pytest.mark.asyncio()
async def test_redis_caching_ttl_pipeline():
    """
    Ensure that a default ttl is set for all redis functions
    """

    from litellm.caching.redis_cache import RedisCache

    litellm.default_redis_ttl = 120
    expected_timedelta = timedelta(seconds=120)
    cache_obj = RedisCache()

    ## TEST 1 - async_set_cache_pipeline
    # Patch self.init_async_client to return our mock Redis client
    # Call async_set_cache
    mock_pipe_instance = AsyncMock()
    with patch.object(mock_pipe_instance, "set", return_value=None) as mock_set:
        await cache_obj._pipeline_helper(
            pipe=mock_pipe_instance,
            cache_list=[("test_key1", "test_value1"), ("test_key2", "test_value2")],
            ttl=None,
        )

        # Verify that the set method was called on the mock Redis instance
        mock_set.assert_has_calls(
            [
                call.set(
                    name="test_key1", value='"test_value1"', ex=expected_timedelta
                ),
                call.set(
                    name="test_key2", value='"test_value2"', ex=expected_timedelta
                ),
            ]
        )


@pytest.mark.asyncio()
async def test_redis_caching_ttl_sadd():
    """
    Ensure that a default ttl is set for all redis functions
    """
    from litellm.caching.redis_cache import RedisCache

    litellm.default_redis_ttl = 120
    expected_timedelta = timedelta(seconds=120)
    cache_obj = RedisCache()
    redis_client = AsyncMock()

    with patch.object(redis_client, "expire", return_value=None) as mock_expire:
        await cache_obj._set_cache_sadd_helper(
            redis_client=redis_client, key="test_key", value=["test_value"], ttl=None
        )
        print(f"expected_timedelta: {expected_timedelta}")
        assert mock_expire.call_args.args[1] == expected_timedelta


@pytest.mark.asyncio()
async def test_dual_cache_caching_batch_get_cache():
    """
    - check redis cache called for initial batch get cache
    - check redis cache not called for consecutive batch get cache with same keys
    """
    from litellm.caching.dual_cache import DualCache
    from litellm.caching.redis_cache import RedisCache

    dc = DualCache(redis_cache=MagicMock(spec=RedisCache))

    with patch.object(
        dc.redis_cache,
        "async_batch_get_cache",
        new=AsyncMock(
            return_value={"test_key1": "test_value1", "test_key2": "test_value2"}
        ),
    ) as mock_async_get_cache:
        await dc.async_batch_get_cache(keys=["test_key1", "test_key2"])

        assert mock_async_get_cache.call_count == 1

        await dc.async_batch_get_cache(keys=["test_key1", "test_key2"])

        assert mock_async_get_cache.call_count == 1


@pytest.mark.asyncio
async def test_redis_increment_pipeline():
    """Test Redis increment pipeline functionality"""
    try:
        from litellm.caching.redis_cache import RedisCache

        litellm.set_verbose = True
        litellm._turn_on_debug()
        redis_cache = RedisCache(
            host=os.environ["REDIS_HOST"],
            port=os.environ["REDIS_PORT"],
            password=os.environ["REDIS_PASSWORD"],
        )

        # Create test increment operations
        increment_list = [
            {"key": "test_key1", "increment_value": 1.5, "ttl": 60},
            {"key": "test_key1", "increment_value": 1.1, "ttl": 58},
            {"key": "test_key1", "increment_value": 0.4, "ttl": 55},
            {"key": "test_key2", "increment_value": 2.5, "ttl": 60},
        ]

        # Test pipeline increment
        results = await redis_cache.async_increment_pipeline(increment_list)

        # Verify results
        assert len(results) == 4

        # Verify the values were actually set in Redis
        value1 = await redis_cache.async_get_cache("test_key1")
        print("result in cache for key=test_key1", value1)
        value2 = await redis_cache.async_get_cache("test_key2")
        print("result in cache for key=test_key2", value2)

        assert float(value1) == 3.0
        assert float(value2) == 2.5

        # Clean up
        await redis_cache.async_delete_cache("test_key1")
        await redis_cache.async_delete_cache("test_key2")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e


@pytest.mark.asyncio
async def test_redis_get_ttl():
    """
    Test Redis get TTL functionality

    Redis returns -2 if the key does not exist and -1 if the key exists but has no associated expire.

    test that litellm redis caching wrapper handles -1 and -2 values and returns them as None
    """
    try:
        from litellm.caching.redis_cache import RedisCache

        redis_cache = RedisCache(
            host=os.environ["REDIS_HOST"],
            port=os.environ["REDIS_PORT"],
            password=os.environ["REDIS_PASSWORD"],
        )

        # Test case 1: Key does not exist
        result = await redis_cache.async_get_ttl("nonexistent_key")
        print("ttl for nonexistent key: ", result)
        assert result is None, f"Expected None for nonexistent key, got {result}"

        # Test case 2: Key exists with TTL
        test_key = "test_key_ttl"
        test_value = "test_value"
        ttl = 10  # 10 seconds TTL

        # Set a key with TTL
        _redis_client = await redis_cache.init_async_client()
        async with _redis_client as redis_client:
            await redis_client.set(test_key, test_value, ex=ttl)

            # Get TTL and verify it's close to what we set
            result = await redis_cache.async_get_ttl(test_key)
            print("ttl for test_key: ", result)
            assert (
                result is not None and 0 <= result <= ttl
            ), f"Expected TTL between 0 and {ttl}, got {result}"

            # Clean up
            await redis_client.delete(test_key)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e


def test_redis_caching_multiple_namespaces():
    """
    Test that redis caching works with multiple namespaces

    If client side request specifies a namespace, it should be used for caching

    The same request with different namespaces should not be cached under the same key
    """
    from litellm._uuid import uuid
    from unittest.mock import patch, MagicMock
    import litellm
    from litellm.caching import Cache
    from litellm import completion

    # Use a fixed uuid to ensure consistent cache keys
    test_uuid = "12345678-1234-1234-1234-123456789abc"
    messages = [{"role": "user", "content": f"what is litellm? {test_uuid}"}]

    # Mock the Redis client creation from the _redis module
    with patch("litellm._redis.get_redis_client") as mock_get_redis_client, patch(
        "litellm._redis.get_redis_connection_pool"
    ) as mock_get_redis_connection_pool:
        # Create a mock Redis client that simulates real Redis behavior
        mock_redis_client = MagicMock()
        mock_get_redis_client.return_value = mock_redis_client

        # Mock the connection pool
        mock_connection_pool = MagicMock()
        mock_get_redis_connection_pool.return_value = mock_connection_pool

        # Dictionary to simulate Redis storage with namespace support
        redis_storage = {}

        def mock_redis_get(key):
            print(f"Redis GET: {key}")
            value = redis_storage.get(key, None)
            # Convert to bytes to match real Redis behavior
            if value is not None:
                import json

                return json.dumps(value).encode("utf-8")
            return None

        def mock_redis_set(name, value, ex=None, **kwargs):
            print(f"Redis SET: {name} = {value}")
            redis_storage[name] = value
            return True

        def mock_redis_ping():
            return True

        def mock_redis_info():
            return {"redis_version": "7.0.0"}

        mock_redis_client.get = mock_redis_get
        mock_redis_client.set = mock_redis_set
        mock_redis_client.ping = mock_redis_ping
        mock_redis_client.info = mock_redis_info

        # Initialize the cache
        litellm.cache = Cache(type="redis")

        namespace_1 = "org-id1"
        namespace_2 = "org-id2"

        # Use mock_response to ensure deterministic responses without external API calls
        response_1 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            cache={"namespace": namespace_1},
            mock_response="Response for namespace 1",
        )

        response_2 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            cache={"namespace": namespace_2},
            mock_response="Response for namespace 2",
        )

        response_3 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            cache={"namespace": namespace_1},
            mock_response="This should be cached",
        )

        response_4 = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            mock_response="Response without namespace",
        )

        print(
            f"Response 1 type: {type(response_1)} - ID: {getattr(response_1, 'id', 'N/A')}"
        )
        print(
            f"Response 2 type: {type(response_2)} - ID: {getattr(response_2, 'id', 'N/A')}"
        )
        print(
            f"Response 3 type: {type(response_3)} - Cache hit: {isinstance(response_3, str)}"
        )
        print(
            f"Response 4 type: {type(response_4)} - ID: {getattr(response_4, 'id', 'N/A')}"
        )

        print(f"Redis storage keys: {list(redis_storage.keys())}")

        # Verify that different namespaces created different cache keys
        cache_keys = list(redis_storage.keys())
        namespace_1_keys = [k for k in cache_keys if k.startswith(f"{namespace_1}:")]
        namespace_2_keys = [k for k in cache_keys if k.startswith(f"{namespace_2}:")]
        no_namespace_keys = [
            k
            for k in cache_keys
            if not k.startswith(f"{namespace_1}:")
            and not k.startswith(f"{namespace_2}:")
        ]

        print(f"Namespace 1 keys: {namespace_1_keys}")
        print(f"Namespace 2 keys: {namespace_2_keys}")
        print(f"No namespace keys: {no_namespace_keys}")

        # Should have at least one key for each namespace
        assert len(namespace_1_keys) > 0, "Should have cache keys for namespace 1"
        assert len(namespace_2_keys) > 0, "Should have cache keys for namespace 2"
        assert len(no_namespace_keys) > 0, "Should have cache keys for no namespace"

        # The main test: response 3 should be a cache hit (string) because it uses same namespace as response 1
        assert isinstance(
            response_3, str
        ), "Response 3 should be a cache hit (string) for same namespace"

        # response 1 & 2 should be ModelResponse objects (cache misses)
        assert hasattr(response_1, "id"), "Response 1 should be a ModelResponse object"
        assert hasattr(response_2, "id"), "Response 2 should be a ModelResponse object"
        assert hasattr(response_4, "id"), "Response 4 should be a ModelResponse object"

        # response 1 & 2 should have different IDs (different namespaces)
        assert (
            response_1.id != response_2.id
        ), f"Expected different response ID for different namespace. Got {response_1.id} and {response_2.id}"

        # response 1 & 4 should have different IDs (different namespaces)
        assert (
            response_1.id != response_4.id
        ), f"Expected different response ID for no namespace vs namespaced. Got {response_1.id} and {response_4.id}"


def test_caching_with_reasoning_content():
    """
    Test that reasoning content is cached
    """

    from litellm._uuid import uuid

    messages = [{"role": "user", "content": f"what is litellm? {uuid.uuid4()}"}]
    litellm.cache = Cache()

    response_1 = completion(
        model="anthropic/claude-3-7-sonnet-latest",
        messages=messages,
        thinking={"type": "enabled", "budget_tokens": 1024},
    )

    response_2 = completion(
        model="anthropic/claude-3-7-sonnet-latest",
        messages=messages,
        thinking={"type": "enabled", "budget_tokens": 1024},
    )

    print(f"response 2: {response_2.model_dump_json(indent=4)}")
    assert response_2._hidden_params["cache_hit"] == True
    assert response_2.choices[0].message.reasoning_content is not None


def test_caching_reasoning_args_miss():  # test in memory cache
    try:
        # litellm._turn_on_debug()
        litellm.set_verbose = True
        litellm.cache = Cache()
        response1 = completion(
            model="claude-3-7-sonnet-latest",
            messages=messages,
            caching=True,
            reasoning_effort="low",
            mock_response="My response",
        )
        response2 = completion(
            model="claude-3-7-sonnet-latest",
            messages=messages,
            caching=True,
            mock_response="My response",
        )
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        assert response1.id != response2.id
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


def test_caching_reasoning_args_hit():  # test in memory cache
    try:
        # litellm._turn_on_debug()
        litellm.set_verbose = True
        litellm.cache = Cache()
        response1 = completion(
            model="claude-3-7-sonnet-latest",
            messages=messages,
            caching=True,
            reasoning_effort="low",
            mock_response="My response",
        )
        response2 = completion(
            model="claude-3-7-sonnet-latest",
            messages=messages,
            caching=True,
            reasoning_effort="low",
            mock_response="My response",
        )
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        assert response1.id == response2.id
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


def test_caching_thinking_args_miss():  # test in memory cache
    try:
        # litellm._turn_on_debug()
        litellm.set_verbose = True
        litellm.cache = Cache()
        response1 = completion(
            model="claude-3-7-sonnet-latest",
            messages=messages,
            caching=True,
            thinking={"type": "enabled", "budget_tokens": 1024},
            mock_response="My response",
        )
        response2 = completion(
            model="claude-3-7-sonnet-latest",
            messages=messages,
            caching=True,
            mock_response="My response",
        )
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        assert response1.id != response2.id
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


def test_caching_thinking_args_hit():  # test in memory cache
    try:
        # litellm._turn_on_debug()
        litellm.set_verbose = True
        litellm.cache = Cache()
        response1 = completion(
            model="claude-3-7-sonnet-latest",
            messages=messages,
            caching=True,
            thinking={"type": "enabled", "budget_tokens": 1024},
            mock_response="My response",
        )
        response2 = completion(
            model="claude-3-7-sonnet-latest",
            messages=messages,
            caching=True,
            thinking={"type": "enabled", "budget_tokens": 1024},
            mock_response="My response",
        )
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        assert response1.id == response2.id
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")
