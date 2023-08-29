# LiteLLM - Caching

## LiteLLM Caches `completion()` and `embedding()` calls when switched on

liteLLM implements exact match caching and supports the following Caching:
* In-Memory Caching [Default]
* Redis Caching Local
* Redic Caching Hosted
* GPTCache 

## Usage
1. Caching - cache
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
