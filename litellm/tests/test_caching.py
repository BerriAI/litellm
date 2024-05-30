import sys, os, uuid
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
from litellm import embedding, completion, aembedding
from litellm.caching import Cache
import random
import hashlib, asyncio

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
    from litellm.caching import DualCache, InMemoryCache, RedisCache

    in_memory_cache = InMemoryCache()
    redis_cache = RedisCache()  # get credentials from environment
    dual_cache = DualCache(in_memory_cache=in_memory_cache, redis_cache=redis_cache)

    in_memory_cache.set_cache(key="test_value", value="hello world")

    result = await dual_cache.async_batch_get_cache(keys=["test_value", "test_value_2"])

    assert result[0] == "hello world"
    assert result[1] == None


def test_dual_cache_batch_get_cache():
    """
    Unit testing for Dual Cache batch_get_cache()
    - 2 item query
    - in_memory result has a partial hit (1/2)
    - hit redis for the other -> expect to return None
    - expect result = [in_memory_result, None]
    """
    from litellm.caching import DualCache, InMemoryCache, RedisCache

    in_memory_cache = InMemoryCache()
    redis_cache = RedisCache()  # get credentials from environment
    dual_cache = DualCache(in_memory_cache=in_memory_cache, redis_cache=redis_cache)

    in_memory_cache.set_cache(key="test_value", value="hello world")

    result = dual_cache.batch_get_cache(keys=["test_value", "test_value_2"])

    assert result[0] == "hello world"
    assert result[1] == None


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
                model="gpt-3.5-turbo", messages=messages, cache={"ttl": 0}
            )
            await asyncio.sleep(10)
            response2 = await litellm.acompletion(
                model="gpt-3.5-turbo", messages=messages, cache={"s-maxage": 10}
            )

            assert response2["id"] != response1["id"]

        message = [{"role": "user", "content": f"Hey, how's it going? {uuid.uuid4()}"}]
        ## TTL = 5
        if sync_flag:
            response1 = completion(
                model="gpt-3.5-turbo", messages=messages, cache={"ttl": 5}
            )
            response2 = completion(
                model="gpt-3.5-turbo", messages=messages, cache={"s-maxage": 5}
            )
            print(f"response1: {response1}")
            print(f"response2: {response2}")
            assert response2["id"] == response1["id"]
        else:
            response1 = await litellm.acompletion(
                model="gpt-3.5-turbo", messages=messages, cache={"ttl": 25}
            )
            await asyncio.sleep(10)
            response2 = await litellm.acompletion(
                model="gpt-3.5-turbo", messages=messages, cache={"s-maxage": 25}
            )
            print(f"response1: {response1}")
            print(f"response2: {response2}")
            assert response2["id"] == response1["id"]
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")


# test_caching_with_cache_controls()


def test_caching_with_models_v2():
    messages = [
        {"role": "user", "content": "who is ishaan CTO of litellm from litellm 2023"}
    ]
    litellm.cache = Cache()
    print("test2 for caching")
    litellm.set_verbose = True
    response1 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
    response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
    response3 = completion(model="azure/chatgpt-v-2", messages=messages, caching=True)
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
    embedding_val_2 = await aembedding(
        model="azure/azure-embedding-model", input=embedding_2, caching=True
    )
    print(f"embedding_val_2._hidden_params: {embedding_val_2._hidden_params}")
    assert embedding_val_2._hidden_params["cache_hit"] == True

    assert embedding_val_2.data[0]["embedding"] == embedding_val_1.data[1]["embedding"]
    assert embedding_val_2.data[0]["index"] != embedding_val_1.data[1]["index"]
    assert embedding_val_2.data[0]["index"] == 0
    assert embedding_val_1.data[1]["index"] == 1


@pytest.mark.asyncio
async def test_embedding_caching_base_64():
    """ """
    litellm.set_verbose = True
    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
    )
    import uuid

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
async def test_redis_batch_cache_write():
    """
    Init redis client
    - write to client
    - read from client
    """
    litellm.set_verbose = True
    import uuid

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
    response4 = completion(model="azure/chatgpt-v-2", messages=messages, caching=True)

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

        time.sleep(0.5)
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
            model="bedrock/anthropic.claude-v2",
            messages=messages,
            max_tokens=40,
            temperature=1,
            stream=True,
        )
        async for chunk in response1:
            print(chunk)
            response_1_content += chunk.choices[0].delta.content or ""
        print(response_1_content)

        time.sleep(0.5)
        print("\n\n Response 1 content: ", response_1_content, "\n\n")

        response2 = await litellm.acompletion(
            model="bedrock/anthropic.claude-v2",
            messages=messages,
            max_tokens=40,
            temperature=1,
            stream=True,
        )
        async for chunk in response2:
            print(chunk)
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
        print(e)
        raise e

def test_disk_cache_completion():
    litellm.set_verbose = False

    random_number = random.randint(
        1, 100000
    )  # add a random number to ensure it's always adding / reading from cache
    messages = [
        {"role": "user", "content": f"write a one sentence poem about: {random_number}"}
    ]
    litellm.cache = Cache(
        type="disk",
    )

    response1 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        caching=True,
        max_tokens=20,
        mock_response="This number is so great!",
    )
    # response2 is mocked to a different response from response1,
    # but the completion from the cache should be used instead of the mock
    # response since the input is the same as response1
    response2 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        caching=True,
        max_tokens=20,
        mock_response="This number is awful!",
    )
    # Since the parameters are not the same as response1, response3 should actually
    # be the mock response
    response3 = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        caching=True,
        temperature=0.5,
        mock_response="This number is awful!",
    )

    print("\nresponse 1", response1)
    print("\nresponse 2", response2)
    print("\nresponse 3", response3)
    # print("\nresponse 4", response4)
    litellm.cache = None
    litellm.success_callback = []
    litellm._async_success_callback = []

    # 1 & 2 should be exactly the same 
    # 1 & 3 should be different, since input params are diff
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
        # if input params like max_tokens, temperature are diff it should NOT be a cache hit
        print(f"response1: {response1}")
        print(f"response3: {response3}")
        pytest.fail(
            f"Response 1 == response 3. Same model, diff params shoudl not cache Error"
            f" occurred:"
        )

    assert response1.id == response2.id
    assert response1.created == response2.created
    assert response1.choices[0].message.content == response2.choices[0].message.content


@pytest.mark.skip(reason="AWS Suspended Account")
@pytest.mark.asyncio
async def test_s3_cache_acompletion_stream_azure():
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
        litellm.cache = Cache(
            type="s3",
            s3_bucket_name="litellm-my-test-bucket-2",
            s3_region_name="us-east-1",
        )
        print("s3 Cache: test for caching, streaming + completion")
        response_1_content = ""
        response_2_content = ""

        response_1_created = ""
        response_2_created = ""

        response1 = await litellm.acompletion(
            model="azure/chatgpt-v-2",
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

        time.sleep(0.5)
        print("\n\n Response 1 content: ", response_1_content, "\n\n")

        response2 = await litellm.acompletion(
            model="azure/chatgpt-v-2",
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
            model="azure/chatgpt-v-2",
            messages=messages,
            max_tokens=40,
            temperature=1,
        )
        print(response1)

        time.sleep(2)

        response2 = await litellm.acompletion(
            model="azure/chatgpt-v-2",
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
    import uuid

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
    import uuid

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
    from litellm.caching import Cache

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
        cache_key_str = "model: gpt-3.5-turbomessages: [{'role': 'user', 'content': 'write a one sentence poem about: 7510'}]temperature: 0.2max_tokens: 40"
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


# test_cache_context_managers()


@pytest.mark.skip(reason="beta test - new redis semantic cache")
def test_redis_semantic_cache_completion():
    litellm.set_verbose = True
    import logging

    logging.basicConfig(level=logging.DEBUG)

    random_number = random.randint(
        1, 100000
    )  # add a random number to ensure it's always adding /reading from cache

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
                "content": f"write a one sentence poem about: {random_number}",
            }
        ],
        max_tokens=20,
    )
    print(f"response1: {response1}")

    random_number = random.randint(1, 100000)

    response2 = completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_number}",
            }
        ],
        max_tokens=20,
    )
    print(f"response2: {response1}")
    assert response1.id == response2.id


# test_redis_cache_completion()


@pytest.mark.skip(reason="beta test - new redis semantic cache")
@pytest.mark.asyncio
async def test_redis_semantic_cache_acompletion():
    litellm.set_verbose = True
    import logging

    logging.basicConfig(level=logging.DEBUG)

    random_number = random.randint(
        1, 100000
    )  # add a random number to ensure it's always adding / reading from cache

    print("testing semantic caching")
    litellm.cache = Cache(
        type="redis-semantic",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
        similarity_threshold=0.8,
        redis_semantic_cache_use_async=True,
    )
    response1 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_number}",
            }
        ],
        max_tokens=5,
    )
    print(f"response1: {response1}")

    random_number = random.randint(1, 100000)
    response2 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_number}",
            }
        ],
        max_tokens=5,
    )
    print(f"response2: {response2}")
    assert response1.id == response2.id


def test_dual_semantic_cache_completion():
    litellm.set_verbose = True
    import logging

    logging.basicConfig(level=logging.DEBUG)

    random_number = random.randint(
        1, 100000
    )  # add a random number to ensure it's always adding /reading from cache

    print("testing semantic caching")
    litellm.cache = Cache(
        type="dual-semantic",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
        similarity_threshold=0.8,
        redis_semantic_cache_embedding_model="local/multi-qa-MiniLM-L6-cos-v1",
        semantic_cache_embedding_length=384
    )
    response1 = completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_number}",
            }
        ],
        max_tokens=20,
    )
    print(f"response1: {response1}")

    random_number = random.randint(1, 100000)

    response2 = completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"please write a one sentence poem about: {random_number}",
            }
        ],
        max_tokens=20,
    )
    print(f"response2: {response1}")
    assert response1.id == response2.id


@pytest.mark.asyncio
async def test_dual_semantic_cache_acompletion():
    litellm.set_verbose = True
    import logging

    logging.basicConfig(level=logging.DEBUG)

    random_number = random.randint(
        1, 100000
    )  # add a random number to ensure it's always adding / reading from cache

    print("testing semantic caching")
    litellm.cache = Cache(
        type="dual-semantic",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
        similarity_threshold=0.8,
        redis_semantic_cache_embedding_model="local/multi-qa-MiniLM-L6-cos-v1",
        semantic_cache_embedding_length=384,
        redis_semantic_cache_use_async=True
    )
    response1 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_number}",
            }
        ],
        max_tokens=5,
    )
    print(f"response1: {response1}")

    random_number = random.randint(1, 100000)
    response2 = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"please write a one sentence poem about: {random_number}",
            }
        ],
        max_tokens=5,
    )
    print(f"response2: {response2}")
    assert response1.id == response2.id
