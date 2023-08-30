# Using GPTCache with LiteLLM

GPTCache is a Library for Creating Semantic Cache for LLM Queries

GPTCache Docs: https://gptcache.readthedocs.io/en/latest/index.html#

GPTCache Github: https://github.com/zilliztech/GPTCache

In this document we cover:
* Quick Start Usage
* Advanced Usage - Set Custom Cache Keys


## Quick Start Usage
👉 Jump to [Colab Notebook Example](https://github.com/BerriAI/litellm/blob/main/cookbook/LiteLLM_GPTCache.ipynb)

### Install GPTCache
```
pip install gptcache
```

### Using GPT Cache with Litellm Completion()

#### Using GPTCache
In order to use GPTCache the following lines are used to instantiate it
```python
from gptcache import cache
# set API keys in .env / os.environ
cache.init()
cache.set_openai_key()
```

#### Full Code using GPTCache and LiteLLM
By default GPT Cache uses the content in `messages` as the cache key

```python
from gptcache import cache
from litellm.gpt_cache import completion # import completion from litellm.cache
import time

# Set your .env keys 
os.environ['OPENAI_API_KEY'] = ""
cache.init()
cache.set_openai_key()

question = "what's LiteLLM"
for _ in range(2):
    start_time = time.time()
    response = completion(
      model='gpt-3.5-turbo',
      messages=[
        {
            'role': 'user',
            'content': question
        }
      ],
    )
    print(f'Question: {question}')
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
```

## Advanced Usage - Set Custom Cache Keys

By default gptcache uses the `messages` as the cache key

GPTCache allows you to set custom cache keys by setting
```python
cache.init(pre_func=pre_cache_func)
```

In this code snippet below we define a `pre_func` that returns message content + model as key 

### Defining a `pre_func` for GPTCache
```python
### using / setting up gpt cache
from gptcache import cache
from gptcache.processor.pre import last_content_without_prompt
from typing import Dict, Any

# use this function to set your cache keys -> gptcache
# data are all the args passed to your completion call 
def pre_cache_func(data: Dict[str, Any], **params: Dict[str, Any]) -> Any:
        # use this to set cache key
        print("in pre_cache_func")
        last_content_without_prompt_val = last_content_without_prompt(data, **params)
        print("last content without prompt", last_content_without_prompt_val)
        print("model", data["model"])
        cache_key = last_content_without_prompt_val + data["model"]
        print("cache_key", cache_key)
        return cache_key # using this as cache_key
        
```

### Init Cache with `pre_func` to set custom keys

```python
# init GPT Cache with custom pre_func
cache.init(pre_func=pre_cache_func)
cache.set_openai_key()
```

## Using Cache 
* Cache key is `message` + `model`

We make 3 LLM API calls
* 2 to OpenAI 
* 1 to Cohere command nightly 

```python
messages = [{"role": "user", "content": "why should I use LiteLLM for completions()"}]
response1 = completion(model="gpt-3.5-turbo", messages=messages)
response2 = completion(model="gpt-3.5-turbo", messages=messages)
response3 = completion(model="command-nightly", messages=messages) # calling cohere command nightly

if response1["choices"] != response2["choices"]: # same models should cache 
    print(f"Error occurred: Caching for same model+prompt failed")

if response3["choices"] == response2["choices"]: # different models, don't cache 
    # if models are different, it should not return cached response
    print(f"Error occurred: Caching for different model+prompt failed")

print("response1", response1)
print("response2", response2)
print("response3", response3)
```


