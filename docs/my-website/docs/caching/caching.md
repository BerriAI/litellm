# LiteLLM - Caching

## Caching `completion()` and `embedding()` calls when switched on

liteLLM implements exact match caching and supports the following Caching:
* In-Memory Caching [Default]
* Redis Caching Local
* Redic Caching Hosted
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