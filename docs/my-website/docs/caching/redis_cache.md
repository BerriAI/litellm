# Redis Cache

[**See Code**](https://github.com/BerriAI/litellm/blob/4d7ff1b33b9991dcf38d821266290631d9bcd2dd/litellm/caching.py#L71)

### Pre-requisites
Install redis
```shell
pip install redis
```
For the hosted version you can setup your own Redis DB here: https://app.redislabs.com/
### Quick Start
```python
import litellm
from litellm import completion
from litellm.caching import Cache
litellm.cache = Cache(type="redis", host=<host>, port=<port>, password=<password>)

# Make completion calls
response1 = completion(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": "Tell me a joke."}]
)
response2 = completion(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": "Tell me a joke."}]
)

# response1 == response2, response 1 is cached
```

## Cache Context Manager - Enable, Disable, Update Cache
Use the context manager for easily enabling, disabling & updating the litellm cache 

### Enabling Cache

Quick Start Enable
```python
litellm.enable_cache()
```

Advanced Params

```python
litellm.enable_cache(
    type: Optional[Literal["local", "redis"]] = "local",
    host: Optional[str] = None,
    port: Optional[str] = None,
    password: Optional[str] = None,
    supported_call_types: Optional[
        List[Literal["completion", "acompletion", "embedding", "aembedding"]]
    ] = ["completion", "acompletion", "embedding", "aembedding"],
    **kwargs,
)
```

### Disabling Cache

Switch caching off 
```python
litellm.disable_cache()
```

## Custom Cache Keys:
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

## Cache Initialization Parameters

#### `type` (str, optional)

The type of cache to initialize. It can be either "local" or "redis". Defaults to "local".

#### `host` (str, optional)

The host address for the Redis cache. This parameter is required if the `type` is set to "redis".

#### `port` (int, optional)

The port number for the Redis cache. This parameter is required if the `type` is set to "redis".

#### `password` (str, optional)

The password for the Redis cache. This parameter is required if the `type` is set to "redis".

#### `supported_call_types` (list, optional)

A list of call types to cache for. Defaults to caching for all call types. The available call types are:

- "completion"
- "acompletion"
- "embedding"
- "aembedding"

#### `**kwargs` (additional keyword arguments)

Additional keyword arguments are accepted for the initialization of the Redis cache using the `redis.Redis()` constructor. These arguments allow you to fine-tune the Redis cache configuration based on your specific needs.


## Logging 

Cache hits are logged in success events as `kwarg["cache_hit"]`. 

Here's an example of accessing it: 

  ```python
  import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm import completion, acompletion, Cache

# create custom callback for success_events
class MyCustomHandler(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Success")
        print(f"Value of Cache hit: {kwargs['cache_hit']"})

async def test_async_completion_azure_caching():
    # set custom callback
    customHandler_caching = MyCustomHandler()
    litellm.callbacks = [customHandler_caching]

    # init cache 
    litellm.cache = Cache(type="redis", host=os.environ['REDIS_HOST'], port=os.environ['REDIS_PORT'], password=os.environ['REDIS_PASSWORD'])
    unique_time = time.time()
    response1 = await litellm.acompletion(model="azure/chatgpt-v-2",
                            messages=[{
                                "role": "user",
                                "content": f"Hi 👋 - i'm async azure {unique_time}"
                            }],
                            caching=True)
    await asyncio.sleep(1)
    print(f"customHandler_caching.states pre-cache hit: {customHandler_caching.states}")
    response2 = await litellm.acompletion(model="azure/chatgpt-v-2",
                            messages=[{
                                "role": "user",
                                "content": f"Hi 👋 - i'm async azure {unique_time}"
                            }],
                            caching=True)
    await asyncio.sleep(1) # success callbacks are done in parallel
  ```
