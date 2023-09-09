# LiteLLM - Caching

## Caching `completion()` and `embedding()` calls when switched on

liteLLM implements exact match caching and supports the following Caching:
* In-Memory Caching [Default]
* Redis Caching Local
* Redis Caching Hosted
* GPTCache 

## Quick Start Usage - Completion
Caching - cache
Keys in the cache are `model`, the following example will lead to a cache hit
```python
import litellm
from litellm import completion
from litellm.caching import Cache
litellm.cache = Cache()

# Make completion calls
response1 = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke."}])
response2 = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke."}])

# response1 == response2, response 1 is cached
```

## Using Redis Cache with LiteLLM
### Pre-requisites
Install redis
```
pip install redis
```
For the hosted version you can setup your own Redis DB here: https://app.redislabs.com/
### Usage
```python
import litellm
from litellm import completion
from litellm.caching import Cache
litellm.cache = Cache(type="redis", host=<host>, port=<port>, password=<password>)

# Make completion calls
response1 = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke."}])
response2 = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke."}])

# response1 == response2, response 1 is cached
```

### Custom Cache Keys:

Define function to return cache key
```python
# this function takes in *args, **kwargs and returns the key you want to use for caching
def custom_get_cache_key(*args, **kwargs):
    # return key to use for your cache:
    key = kwargs.get("model", "") + str(kwargs.get("messages", "")) + str(kwargs.get("temperature", "")) + str(kwargs.get("logit_bias", ""))
    print("key for cache", key)
    return key

```

Set your function as litellm.cache.get_cache_key
```python
from litellm.caching import Cache

cache = Cache(type="redis", host=os.environ['REDIS_HOST'], port=os.environ['REDIS_PORT'], password=os.environ['REDIS_PASSWORD'])

cache.get_cache_key = custom_get_cache_key # set get_cache_key function for your cache

litellm.cache = cache # set litellm.cache to your cache 

```

### Controlling Caching for each litellm.completion call

`completion()` lets you pass in `caching` (bool) [default False] to control whether to returned cached responses or not

Using the caching flag
**Ensure you have initialized litellm.cache to your cache object**

```python
from litellm import completion

response2 = completion(model="gpt-3.5-turbo", messages=messages, temperature=0.1, caching=True)

response3 = completion(model="gpt-3.5-turbo", messages=messages, temperature=0.1, caching=False)
    
```
### Detecting Cached Responses
For resposes that were returned as cache hit, the response includes a param `cache` = True 

Example response with cache hit
```
{
    'cache': True,
    'id': 'chatcmpl-7wggdzd6OXhgE2YhcLJHJNZsEWzZ2', 
    'created': 1694221467, 
    'model': 'gpt-3.5-turbo-0613', 
    'choices': [
        {
            'index': 0, 'message': {'role': 'assistant', 'content': 'I\'m sorry, but I couldn\'t find any information about "litellm" or how many stars it has. It is possible that you may be referring to a specific product, service, or platform that I am not familiar with. Can you please provide more context or clarify your question?'
        }, 'finish_reason': 'stop'}
    ], 
    'usage': {'prompt_tokens': 17, 'completion_tokens': 59, 'total_tokens': 76}, 
}

```
## Caching with Streaming 
LiteLLM can cache your streamed responses for you

### Usage
```python
import litellm
from litellm import completion
from litellm.caching import Cache
litellm.cache = Cache()

# Make completion calls
response1 = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke."}], stream=True)
for chunk in response1:
    print(chunk)
response2 = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke."}], stream=True)
for chunk in response2:
    print(chunk)
```

## Usage - Embedding()
1. Caching - cache
Keys in the cache are `model`, the following example will lead to a cache hit
```python
import time
import litellm
from litellm import completion
from litellm.caching import Cache
litellm.cache = Cache()

start_time = time.time()
embedding1 = embedding(model="text-embedding-ada-002", input=["hello from litellm"*5])
end_time = time.time()
print(f"Embedding 1 response time: {end_time - start_time} seconds")

start_time = time.time()
embedding2 = embedding(model="text-embedding-ada-002", input=["hello from litellm"*5])
end_time = time.time()
print(f"Embedding 2 response time: {end_time - start_time} seconds")
```