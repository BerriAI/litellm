import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Caching - In-Memory, Redis, s3,  Redis Semantic Cache

[**See Code**](https://github.com/BerriAI/litellm/blob/main/litellm/caching.py)

:::info

Need to use Caching on LiteLLM Proxy Server? Doc here: [Caching Proxy Server](https://docs.litellm.ai/docs/proxy/caching)

:::

## Initialize Cache - In Memory, Redis, s3 Bucket, Redis Semantic Cache


<Tabs>

<TabItem value="redis" label="redis-cache">

Install redis
```shell
pip install redis
```

For the hosted version you can setup your own Redis DB here: https://app.redislabs.com/

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

</TabItem>


<TabItem value="s3" label="s3-cache">

Install boto3
```shell
pip install boto3
```

Set AWS environment variables

```shell
AWS_ACCESS_KEY_ID = "AKI*******"
AWS_SECRET_ACCESS_KEY = "WOl*****"
```

```python
import litellm
from litellm import completion
from litellm.caching import Cache

# pass s3-bucket name
litellm.cache = Cache(type="s3", s3_bucket_name="cache-bucket-litellm", s3_region_name="us-west-2")

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

</TabItem>


<TabItem value="redis-sem" label="redis-semantic cache">

Install redis
```shell
pip install redisvl==0.0.7
```

For the hosted version you can setup your own Redis DB here: https://app.redislabs.com/

```python
import litellm
from litellm import completion
from litellm.caching import Cache

random_number = random.randint(
    1, 100000
)  # add a random number to ensure it's always adding / reading from cache

print("testing semantic caching")
litellm.cache = Cache(
    type="redis-semantic",
    host=os.environ["REDIS_HOST"],
    port=os.environ["REDIS_PORT"],
    password=os.environ["REDIS_PASSWORD"],
    similarity_threshold=0.8, # similarity threshold for cache hits, 0 == no similarity, 1 = exact matches, 0.5 == 50% similarity
    redis_semantic_cache_embedding_model="text-embedding-ada-002", # this model is passed to litellm.embedding(), any litellm.embedding() model is supported here
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
# response1 == response2, response 1 is cached
```

</TabItem>



<TabItem value="in-mem" label="in memory cache">

### Quick Start

```python
import litellm
from litellm import completion
from litellm.caching import Cache
litellm.cache = Cache()

# Make completion calls
response1 = completion(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": "Tell me a joke."}]
    caching=True
)
response2 = completion(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": "Tell me a joke."}],
    caching=True
)

# response1 == response2, response 1 is cached

```

</TabItem>


</Tabs>

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

### Updating Cache Params (Redis Host, Port etc)

Update the Cache params

```python
litellm.update_cache(
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
## How to write custom add/get cache functions 
### 1. Init Cache 
```python
from litellm.caching import Cache
cache = Cache()
``` 

### 2. Define custom add/get cache functions 
```python
def add_cache(self, result, *args, **kwargs):
  your logic
  
def get_cache(self, *args, **kwargs):
  your logic
```

### 3. Point cache add/get functions to your add/get functions 
```python
cache.add_cache = add_cache
cache.get_cache = get_cache
```

## Cache Initialization Parameters

```python
def __init__(
    self,
    type: Optional[Literal["local", "redis", "s3"]] = "local",
    supported_call_types: Optional[
        List[Literal["completion", "acompletion", "embedding", "aembedding"]]
    ] = ["completion", "acompletion", "embedding", "aembedding"], # A list of litellm call types to cache for. Defaults to caching for all litellm call types.
    
    # redis cache params
    host: Optional[str] = None,
    port: Optional[str] = None,
    password: Optional[str] = None,


    # s3 Bucket, boto3 configuration
    s3_bucket_name: Optional[str] = None,
    s3_region_name: Optional[str] = None,
    s3_api_version: Optional[str] = None,
    s3_path: Optional[str] = None, # if you wish to save to a spefic path
    s3_use_ssl: Optional[bool] = True,
    s3_verify: Optional[Union[bool, str]] = None,
    s3_endpoint_url: Optional[str] = None,
    s3_aws_access_key_id: Optional[str] = None,
    s3_aws_secret_access_key: Optional[str] = None,
    s3_aws_session_token: Optional[str] = None,
    s3_config: Optional[Any] = None,
    **kwargs,
):
```

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
