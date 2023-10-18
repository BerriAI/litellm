# Redis Cache
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
response1 = completion(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": "Tell me a joke."}],
    caching=True
)
response2 = completion(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": "Tell me a joke."}],
    caching=True
)

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

### Detecting Cached Responses
For resposes that were returned as cache hit, the response includes a param `cache` = True 

Example response with cache hit
```python
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