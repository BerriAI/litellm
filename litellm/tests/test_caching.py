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
# litellm.set_verbose=True

messages = [{"role": "user", "content": "who is ishaan Github?  "}]
# comment


def test_gpt_cache():
    try: 
      # INIT GPT Cache #
      from gptcache import cache
      import gptcache

      from gptcache.processor.pre import last_content_without_prompt
      from litellm.gpt_cache import completion
      from typing import Dict, Any

      def pre_cache_func(data: Dict[str, Any], **params: Dict[str, Any]) -> Any:
          # use this to set cache key
          print("in do nothing")
          last_content_without_prompt_val = last_content_without_prompt(data, **params)
          print("last content without prompt", last_content_without_prompt_val)
          print("model", data["model"])
          cache_key = last_content_without_prompt_val + data["model"]
          print("cache_key", cache_key)
          return cache_key


      cache.init(pre_func=pre_cache_func)
      cache.set_openai_key()

      messages = [{"role": "user", "content": "why should I use LiteLLM today"}]
      response1 = completion(model="gpt-3.5-turbo", messages=messages)
      response2 = completion(model="gpt-3.5-turbo", messages=messages)
      response3 = completion(model="command-nightly", messages=messages)

      if response1["choices"] != response2["choices"]: # same models should cache
          print(f"response1: {response1}")
          print(f"response2: {response2}")
          pytest.fail(f"Error occurred:")

      if response3["choices"] == response2["choices"]: # different models, don't cache
          # if models are different, it should not return cached response
          print(f"response2: {response2}")
          print(f"response3: {response3}")
          pytest.fail(f"Error occurred:")
    except: 
        pass


# test_gpt_cache() 


####### Updated Caching as of Aug 28, 2023 ###################
messages = [{"role": "user", "content": "who is ishaan 5222"}]
def test_caching_v2():
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


# test caching with streaming
# flaky test on circle ci for some reason? 
# def test_caching_v2_stream_basic():
#     try:
#         litellm.cache = Cache()
#         messages = [{"role": "user", "content": "tell me a story in 2 sentences"}]
#         response1 = completion(model="gpt-3.5-turbo", messages=messages, stream=True)

#         result_string = ""
#         for chunk in response1:
#             print(chunk)
#             result_string+=chunk['choices'][0]['delta']['content']
#             # response1_id = chunk['id']

#         print("current cache")
#         print(litellm.cache.cache.cache_dict)

#         result2_string=""
#         import time
#         time.sleep(1)
#         response2 = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
#         for chunk in response2:
#             print(chunk)
#             result2_string+=chunk['choices'][0]['delta']['content']
#         if result_string != result2_string:
#             print(result_string)
#             print(result2_string)
#             pytest.fail(f"Error occurred: Caching with streaming failed, strings diff")
#         litellm.cache = None

#     except Exception as e:
#         print(f"error occurred: {traceback.format_exc()}")
#         pytest.fail(f"Error occurred: {e}")

# test_caching_v2_stream_basic()

# def test_caching_v2_stream():
#     try:
#         litellm.cache = Cache()
#         # litellm.token="ishaan@berri.ai"
#         messages = [{"role": "user", "content": "tell me a story in 2 sentences"}]
#         response1 = completion(model="gpt-3.5-turbo", messages=messages, stream=True)

#         messages = [{"role": "user", "content": "tell me a chair"}]
#         response7 = completion(model="command-nightly", messages=messages)
#         messages = [{"role": "user", "content": "sing a song"}]
#         response8 = completion(model="gpt-3.5-turbo", messages=messages, stream=True)

#         result_string = ""
#         for chunk in response1:
#             print(chunk)
#             result_string+=chunk['choices'][0]['delta']['content']
#             # response1_id = chunk['id']

#         print("current cache")
#         messages = [{"role": "user", "content": "tell me a story in 2 sentences"}]
#         print(litellm.cache.cache.cache_dict)

#         result2_string=""
#         response2 = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
#         for chunk in response2:
#             print(chunk)
#             result2_string+=chunk['choices'][0]['delta']['content']
#         if result_string != result2_string:
#             print(result_string)
#             print(result2_string)
#             pytest.fail(f"Error occurred: Caching with streaming failed, strings diff")
#         litellm.cache = None

#     except Exception as e:
#         print(f"error occurred: {traceback.format_exc()}")
#         pytest.fail(f"Error occurred: {e}")

# test_caching_v2_stream()


def test_redis_cache_completion():
    litellm.set_verbose = True
    messages = [{"role": "user", "content": "who is ishaan CTO of litellm from litellm 2023"}]
    litellm.cache = Cache(type="redis", host=os.environ['REDIS_HOST'], port=os.environ['REDIS_PORT'], password=os.environ['REDIS_PASSWORD'])
    print("test2 for caching")
    response1 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
    response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
    response3 = completion(model="command-nightly", messages=messages, caching=True)
    litellm.cache = None
    if response3['choices'][0]['message']['content'] == response2['choices'][0]['message']['content']:
        # if models are different, it should not return cached response
        print(f"response2: {response2}")
        print(f"response3: {response3}")
        pytest.fail(f"Error occurred:")
    if response1['choices'][0]['message']['content'] != response2['choices'][0]['message']['content']: # 1 and 2 should be the same
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        pytest.fail(f"Error occurred:")

# test_redis_cache_completion()

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

def test_hosted_cache():
    litellm.cache = Cache(type="hosted") # use api.litellm.ai for caching

    messages = [{"role": "user", "content": "what is litellm arr today?"}]
    response1 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
    print("response1", response1)

    response2 = completion(model="gpt-3.5-turbo", messages=messages, caching=True)
    print("response2", response2)

    if response1['choices'][0]['message']['content'] != response2['choices'][0]['message']['content']: # 1 and 2 should be the same
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        pytest.fail(f"Hosted cache: Response2 is not cached and the same as response 1")
    litellm.cache = None

# test_hosted_cache()


def test_redis_cache_with_ttl():
    cache = Cache(type="redis", host=os.environ['REDIS_HOST'], port=os.environ['REDIS_PORT'], password=os.environ['REDIS_PASSWORD'])
    sample_model_response_object_str = """{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
      }
    }
  ],
  "created": 1691429984.3852863,
  "model": "claude-instant-1",
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 23,
    "total_tokens": 41
  }
}"""
    sample_model_response_object = {
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
      }
    }
  ],
  "created": 1691429984.3852863,
  "model": "claude-instant-1",
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 23,
    "total_tokens": 41
  }
}
    cache.add_cache(cache_key="test_key", result=sample_model_response_object_str, ttl=1)
    cached_value = cache.get_cache(cache_key="test_key")
    print(f"cached-value: {cached_value}")
    assert cached_value['choices'][0]['message']['content'] == sample_model_response_object['choices'][0]['message']['content']
    time.sleep(2)
    assert cache.get_cache(cache_key="test_key") is None

# test_redis_cache_with_ttl()

def test_in_memory_cache_with_ttl():
    cache = Cache(type="local")
    sample_model_response_object_str = """{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
      }
    }
  ],
  "created": 1691429984.3852863,
  "model": "claude-instant-1",
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 23,
    "total_tokens": 41
  }
}"""
    sample_model_response_object = {
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
      }
    }
  ],
  "created": 1691429984.3852863,
  "model": "claude-instant-1",
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 23,
    "total_tokens": 41
  }
}
    cache.add_cache(cache_key="test_key", result=sample_model_response_object_str, ttl=1)
    cached_value = cache.get_cache(cache_key="test_key")
    assert cached_value['choices'][0]['message']['content'] == sample_model_response_object['choices'][0]['message']['content']
    time.sleep(2)
    assert cache.get_cache(cache_key="test_key") is None
# test_in_memory_cache_with_ttl()