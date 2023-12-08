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
from litellm import embedding, completion
from litellm.caching import Cache
import random
# litellm.set_verbose=True

messages = [{"role": "user", "content": "who is ishaan Github?  "}]
# comment

messages = [{"role": "user", "content": "who is ishaan 5222"}]
def test_caching_v2(): # test in memory cache
    try:
        litellm.cache = Cache()
        response1 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
        response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        litellm.cache = None # disable cache
        if response2['choices'][0]['message']['content'] != response1['choices'][0]['message']['content']:
            print(f"response1: {response1}")
            print(f"response2: {response2}")
            pytest.fail(f"Error occurred: {e}")
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
        pytest.fail(f"Error occurred: {e}")

# test_caching_v2()



def test_caching_with_models_v2():
    messages = [{"role": "user", "content": "who is ishaan CTO of litellm from litellm 2023"}]
    litellm.cache = Cache()
    print("test2 for caching")
    response1 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
    response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
    response3 = completion(model="command-nightly", messages=messages, caching=True)
    print(f"response1: {response1}")
    print(f"response2: {response2}")
    print(f"response3: {response3}")
    litellm.cache = None
    if response3['choices'][0]['message']['content'] == response2['choices'][0]['message']['content']:
        # if models are different, it should not return cached response
        print(f"response2: {response2}")
        print(f"response3: {response3}")
        pytest.fail(f"Error occurred:")
    if response1['choices'][0]['message']['content'] != response2['choices'][0]['message']['content']:
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        pytest.fail(f"Error occurred:")
# test_caching_with_models_v2()

embedding_large_text = """
small text
""" * 5

# # test_caching_with_models()
def test_embedding_caching():
    import time
    litellm.cache = Cache()
    text_to_embed = [embedding_large_text]
    start_time = time.time()
    embedding1 = embedding(model="text-embedding-ada-002", input=text_to_embed, caching=True)
    end_time = time.time()
    print(f"Embedding 1 response time: {end_time - start_time} seconds")

    time.sleep(1)
    start_time = time.time()
    embedding2 = embedding(model="text-embedding-ada-002", input=text_to_embed, caching=True)
    end_time = time.time()
    print(f"embedding2: {embedding2}")
    print(f"Embedding 2 response time: {end_time - start_time} seconds")

    litellm.cache = None
    assert end_time - start_time <= 0.1 # ensure 2nd response comes in in under 0.1 s
    if embedding2['data'][0]['embedding'] != embedding1['data'][0]['embedding']:
        print(f"embedding1: {embedding1}")
        print(f"embedding2: {embedding2}")
        pytest.fail("Error occurred: Embedding caching failed")

# test_embedding_caching()


def test_embedding_caching_azure():
    print("Testing azure embedding caching")
    import time
    litellm.cache = Cache()
    text_to_embed = [embedding_large_text]

    api_key = os.environ['AZURE_API_KEY']
    api_base = os.environ['AZURE_API_BASE']
    api_version = os.environ['AZURE_API_VERSION']

    os.environ['AZURE_API_VERSION'] = ""
    os.environ['AZURE_API_BASE'] = ""
    os.environ['AZURE_API_KEY'] = ""


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
        caching=True
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
        caching=True
    )
    end_time = time.time()
    print(f"Embedding 2 response time: {end_time - start_time} seconds")

    litellm.cache = None
    assert end_time - start_time <= 0.1 # ensure 2nd response comes in in under 0.1 s
    if embedding2['data'][0]['embedding'] != embedding1['data'][0]['embedding']:
        print(f"embedding1: {embedding1}")
        print(f"embedding2: {embedding2}")
        pytest.fail("Error occurred: Embedding caching failed")

    os.environ['AZURE_API_VERSION'] = api_version
    os.environ['AZURE_API_BASE'] = api_base
    os.environ['AZURE_API_KEY'] = api_key

# test_embedding_caching_azure()


def test_redis_cache_completion():
    litellm.set_verbose = False

    random_number = random.randint(1, 100000) # add a random number to ensure it's always adding / reading from cache
    messages = [{"role": "user", "content": f"write a one sentence poem about: {random_number}"}]
    litellm.cache = Cache(type="redis", host=os.environ['REDIS_HOST'], port=os.environ['REDIS_PORT'], password=os.environ['REDIS_PASSWORD'])
    print("test2 for caching")
    response1 = completion(model="gpt-3.5-turbo", messages=messages, caching=True, max_tokens=20)
    response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True, max_tokens=20)
    response3 = completion(model="gpt-3.5-turbo", messages=messages, caching=True, temperature=0.5)
    response4 = completion(model="command-nightly", messages=messages, caching=True)

    print("\nresponse 1", response1)
    print("\nresponse 2", response2)
    print("\nresponse 3", response3)
    print("\nresponse 4", response4)
    litellm.cache = None

    """
    1 & 2 should be exactly the same 
    1 & 3 should be different, since input params are diff
    1 & 4 should be diff, since models are diff
    """
    if response1['choices'][0]['message']['content'] != response2['choices'][0]['message']['content']: # 1 and 2 should be the same
        # 1&2 have the exact same input params. This MUST Be a CACHE HIT
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        pytest.fail(f"Error occurred:")
    if response1['choices'][0]['message']['content'] == response3['choices'][0]['message']['content']:
        # if input params like seed, max_tokens are diff it should NOT be a cache hit
        print(f"response1: {response1}")
        print(f"response3: {response3}")
        pytest.fail(f"Response 1 == response 3. Same model, diff params shoudl not cache Error occurred:")
    if response1['choices'][0]['message']['content'] == response4['choices'][0]['message']['content']:
        # if models are different, it should not return cached response
        print(f"response1: {response1}")
        print(f"response4: {response4}")
        pytest.fail(f"Error occurred:")

# test_redis_cache_completion()

def test_redis_cache_completion_stream():
    try:
        litellm.set_verbose = False
        random_number = random.randint(1, 100000) # add a random number to ensure it's always adding / reading from cache
        messages = [{"role": "user", "content": f"write a one sentence poem about: {random_number}"}]
        litellm.cache = Cache(type="redis", host=os.environ['REDIS_HOST'], port=os.environ['REDIS_PORT'], password=os.environ['REDIS_PASSWORD'])
        print("test for caching, streaming + completion")
        response1 = completion(model="gpt-3.5-turbo", messages=messages, max_tokens=40, temperature=0.2, stream=True)
        response_1_content = ""
        for chunk in response1:
            print(chunk)
            response_1_content += chunk.choices[0].delta.content or ""
        print(response_1_content)
        time.sleep(0.5)
        response2 = completion(model="gpt-3.5-turbo", messages=messages, max_tokens=40, temperature=0.2, stream=True)
        response_2_content = ""
        for chunk in response2:
            print(chunk)
            response_2_content += chunk.choices[0].delta.content or ""
        print("\nresponse 1", response_1_content)
        print("\nresponse 2", response_2_content)
        assert response_1_content == response_2_content, f"Response 1 != Response 2. Same params, Response 1{response_1_content} != Response 2{response_2_content}"
        litellm.cache = None
    except Exception as e:
        print(e)
        raise e
    """

    1 & 2 should be exactly the same 
    """
# test_redis_cache_completion_stream()


# redis cache with custom keys
def custom_get_cache_key(*args, **kwargs):
    # return key to use for your cache:
    key = kwargs.get("model", "") + str(kwargs.get("messages", "")) + str(kwargs.get("temperature", "")) + str(kwargs.get("logit_bias", ""))
    return key

def test_custom_redis_cache_with_key():
    messages = [{"role": "user", "content": "write a one line story"}]
    litellm.cache = Cache(type="redis", host=os.environ['REDIS_HOST'], port=os.environ['REDIS_PORT'], password=os.environ['REDIS_PASSWORD'])
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

    response1 = completion(model="gpt-3.5-turbo", messages=messages, temperature=1, caching=True, num_retries=3)
    response2 = completion(model="gpt-3.5-turbo", messages=messages, temperature=1, caching=True, num_retries=3)
    response3 = completion(model="gpt-3.5-turbo", messages=messages, temperature=1, caching=False, num_retries=3)

    print(f"response1: {response1}")
    print(f"response2: {response2}")
    print(f"response3: {response3}")

    if response3['choices'][0]['message']['content'] == response2['choices'][0]['message']['content']:
        pytest.fail(f"Error occurred:")
    litellm.cache = None

# test_custom_redis_cache_with_key()


def test_custom_redis_cache_params():
    # test if we can init redis with **kwargs
    try:
        litellm.cache = Cache(
            type="redis",
            host=os.environ['REDIS_HOST'],
            port=os.environ['REDIS_PORT'],
            password=os.environ['REDIS_PASSWORD'],
            db = 0,
            ssl=True,
            ssl_certfile="./redis_user.crt",
            ssl_keyfile="./redis_user_private.key",
            ssl_ca_certs="./redis_ca.pem",
        )

        print(litellm.cache.cache.redis_client) 
        litellm.cache = None
    except Exception as e:
        pytest.fail(f"Error occurred:", e)

# test_custom_redis_cache_params()

# def test_redis_cache_with_ttl():
#     cache = Cache(type="redis", host=os.environ['REDIS_HOST'], port=os.environ['REDIS_PORT'], password=os.environ['REDIS_PASSWORD'])
#     sample_model_response_object_str = """{
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
#       }
#     }
#   ],
#   "created": 1691429984.3852863,
#   "model": "claude-instant-1",
#   "usage": {
#     "prompt_tokens": 18,
#     "completion_tokens": 23,
#     "total_tokens": 41
#   }
# }"""
#     sample_model_response_object = {
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
#       }
#     }
#   ],
#   "created": 1691429984.3852863,
#   "model": "claude-instant-1",
#   "usage": {
#     "prompt_tokens": 18,
#     "completion_tokens": 23,
#     "total_tokens": 41
#   }
# }
#     cache.add_cache(cache_key="test_key", result=sample_model_response_object_str, ttl=1)
#     cached_value = cache.get_cache(cache_key="test_key")
#     print(f"cached-value: {cached_value}")
#     assert cached_value['choices'][0]['message']['content'] == sample_model_response_object['choices'][0]['message']['content']
#     time.sleep(2)
#     assert cache.get_cache(cache_key="test_key") is None

# # test_redis_cache_with_ttl()

# def test_in_memory_cache_with_ttl():
#     cache = Cache(type="local")
#     sample_model_response_object_str = """{
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
#       }
#     }
#   ],
#   "created": 1691429984.3852863,
#   "model": "claude-instant-1",
#   "usage": {
#     "prompt_tokens": 18,
#     "completion_tokens": 23,
#     "total_tokens": 41
#   }
# }"""
#     sample_model_response_object = {
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
#       }
#     }
#   ],
#   "created": 1691429984.3852863,
#   "model": "claude-instant-1",
#   "usage": {
#     "prompt_tokens": 18,
#     "completion_tokens": 23,
#     "total_tokens": 41
#   }
# }
#     cache.add_cache(cache_key="test_key", result=sample_model_response_object_str, ttl=1)
#     cached_value = cache.get_cache(cache_key="test_key")
#     assert cached_value['choices'][0]['message']['content'] == sample_model_response_object['choices'][0]['message']['content']
#     time.sleep(2)
#     assert cache.get_cache(cache_key="test_key") is None
# # test_in_memory_cache_with_ttl()