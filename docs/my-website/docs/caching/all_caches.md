import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Caching - In-Memory, Redis, s3, gcs, Redis Semantic Cache, Disk

[**See Code**](https://github.com/BerriAI/litellm/blob/main/litellm/caching/caching.py)

:::info

- For Proxy Server? Doc here: [Caching Proxy Server](https://docs.litellm.ai/docs/proxy/caching)

- For OpenAI/Anthropic Prompt Caching, go [here](../completion/prompt_caching.md)


:::

## Initialize Cache - In Memory, Redis, s3 Bucket, gcs Bucket, Redis Semantic, Disk Cache, Qdrant Semantic


<Tabs>

<TabItem value="redis" label="redis-cache">

Install redis
```shell
pip install redis
```

For the hosted version you can setup your own Redis DB here: https://redis.io/try-free/

**Basic Redis Cache**

```python
import litellm
from litellm import completion
from litellm.caching.caching import Cache

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

**GCP IAM Redis Authentication**

For GCP Memorystore Redis with IAM authentication:

```shell
pip install google-cloud-iam
```

```python
import litellm
from litellm import completion
# For Redis Cluster with GCP IAM
from litellm.caching.redis_cluster_cache import RedisClusterCache

litellm.cache = RedisClusterCache(
    startup_nodes=[
        {"host": "10.128.0.2", "port": 6379},
        {"host": "10.128.0.2", "port": 11008},
    ],
    gcp_service_account="projects/-/serviceAccounts/your-sa@project.iam.gserviceaccount.com",
    ssl=True,
    ssl_cert_reqs=None,
    ssl_check_hostname=False,
)

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

**Environment Variables for GCP IAM Redis**

You can also set these as environment variables:

```shell
export REDIS_HOST="10.128.0.2"
export REDIS_PORT="6379"
export REDIS_GCP_SERVICE_ACCOUNT="projects/-/serviceAccounts/your-sa@project.iam.gserviceaccount.com"
export REDIS_SSL="False"
```

Then simply initialize:

```python
litellm.cache = Cache(type="redis")
```

:::info
Use `REDIS_*` environment variables as the primary mechanism for configuring all Redis client library parameters. This approach automatically maps environment variables to Redis client kwargs and is the suggested way to toggle Redis settings.
:::

:::warning
If you need to pass non-string Redis parameters (integers, booleans, complex objects), avoid `REDIS_*` environment variables as they may fail during Redis client initialization. Instead, pass them directly as kwargs to the `Cache()` constructor.
:::

</TabItem>

<TabItem value="gcs" label="gcs-cache">

Set environment variables

```shell
GCS_BUCKET_NAME="my-cache-bucket"
GCS_PATH_SERVICE_ACCOUNT="/path/to/service_account.json"
```

```python
import litellm
from litellm import completion
from litellm.caching.caching import Cache

litellm.cache = Cache(type="gcs", gcs_bucket_name="my-cache-bucket", gcs_path_service_account="/path/to/service_account.json")

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
from litellm.caching.caching import Cache

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

<TabItem value="azureblob" label="azure-blob-cache">

Install azure-storage-blob and azure-identity
```shell
pip install azure-storage-blob azure-identity
```

```python
import litellm
from litellm import completion
from litellm.caching.caching import Cache
from azure.identity import DefaultAzureCredential

# pass Azure Blob Storage account URL and container name
litellm.cache = Cache(type="azure-blob", azure_account_url="https://example.blob.core.windows.net", azure_blob_container="litellm")

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

Install redisvl client
```shell
pip install redisvl==0.4.1
```

For the hosted version you can setup your own Redis DB here: https://redis.io/try-free/

```python
import litellm
from litellm import completion
from litellm.caching.caching import Cache

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
    ttl=120,
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

<TabItem value="qdrant-sem" label="qdrant-semantic cache">

You can set up your own cloud Qdrant cluster by following this: https://qdrant.tech/documentation/quickstart-cloud/

To set up a Qdrant cluster locally follow: https://qdrant.tech/documentation/quickstart/
```python
import litellm
from litellm import completion
from litellm.caching.caching import Cache

random_number = random.randint(
    1, 100000
)  # add a random number to ensure it's always adding / reading from cache

print("testing semantic caching")
litellm.cache = Cache(
    type="qdrant-semantic",
    qdrant_api_base=os.environ["QDRANT_API_BASE"], 
    qdrant_api_key=os.environ["QDRANT_API_KEY"],
    qdrant_collection_name="your_collection_name", # any name of your collection
    similarity_threshold=0.7, # similarity threshold for cache hits, 0 == no similarity, 1 = exact matches, 0.5 == 50% similarity
    qdrant_quantization_config ="binary", # can be one of 'binary', 'product' or 'scalar' quantizations that is supported by qdrant
    qdrant_semantic_cache_embedding_model="text-embedding-ada-002", # this model is passed to litellm.embedding(), any litellm.embedding() model is supported here
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
print(f"response2: {response2}")
assert response1.id == response2.id
# response1 == response2, response 1 is cached
```

</TabItem>

<TabItem value="in-mem" label="in memory cache">

### Quick Start

```python
import litellm
from litellm import completion
from litellm.caching.caching import Cache
litellm.cache = Cache()

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

</TabItem>

<TabItem value="disk" label="disk cache">

### Quick Start

Install the disk caching extra:

```shell
pip install "litellm[caching]"
```

Then you can use the disk cache as follows.

```python
import litellm
from litellm import completion
from litellm.caching.caching import Cache
litellm.cache = Cache(type="disk")

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

If you run the code two times, response1 will use the cache from the first run that was stored in a cache file.

</TabItem>

</Tabs>

## Switch Cache On / Off Per LiteLLM Call 

LiteLLM supports 4 cache-controls:

- `no-cache`: *Optional(bool)* When `True`, Will not return a cached response, but instead call the actual endpoint. 
- `no-store`: *Optional(bool)* When `True`, Will not cache the response. 
- `ttl`: *Optional(int)* - Will cache the response for the user-defined amount of time (in seconds).
- `s-maxage`: *Optional(int)* Will only accept cached responses that are within user-defined range (in seconds).

[Let us know if you need more](https://github.com/BerriAI/litellm/issues/1218)
<Tabs>
<TabItem value="no-cache" label="No-Cache">

Example usage `no-cache` - When `True`, Will not return a cached response

```python
response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "hello who are you"
            }
        ],
        cache={"no-cache": True},
    )
```

</TabItem>

<TabItem value="no-store" label="No-Store">

Example usage `no-store` - When `True`, Will not cache the response. 

```python
response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "hello who are you"
            }
        ],
        cache={"no-store": True},
    )
```

</TabItem>

<TabItem value="ttl" label="ttl">
Example usage `ttl` - cache the response for 10 seconds

```python
response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "hello who are you"
            }
        ],
        cache={"ttl": 10},
    )
```

</TabItem>

<TabItem value="s-maxage" label="s-maxage">
Example usage `s-maxage` - Will only accept cached responses for 60 seconds

```python
response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "hello who are you"
            }
        ],
        cache={"s-maxage": 60},
    )
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
    type: Optional[Literal["local", "redis", "s3", "gcs", "disk"]] = "local",
    host: Optional[str] = None,
    port: Optional[str] = None,
    password: Optional[str] = None,
    supported_call_types: Optional[
        List[Literal["completion", "acompletion", "embedding", "aembedding", "atranscription", "transcription"]]
    ] = ["completion", "acompletion", "embedding", "aembedding", "atranscription", "transcription"],
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
    type: Optional[Literal["local", "redis", "s3", "gcs", "disk"]] = "local",
    host: Optional[str] = None,
    port: Optional[str] = None,
    password: Optional[str] = None,
    supported_call_types: Optional[
        List[Literal["completion", "acompletion", "embedding", "aembedding", "atranscription", "transcription"]]
    ] = ["completion", "acompletion", "embedding", "aembedding", "atranscription", "transcription"],
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
from litellm.caching.caching import Cache

cache = Cache(type="redis", host=os.environ['REDIS_HOST'], port=os.environ['REDIS_PORT'], password=os.environ['REDIS_PASSWORD'])

cache.get_cache_key = custom_get_cache_key # set get_cache_key function for your cache

litellm.cache = cache # set litellm.cache to your cache 

```
## How to write custom add/get cache functions 
### 1. Init Cache 
```python
from litellm.caching.caching import Cache
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
    type: Optional[Literal["local", "redis", "redis-semantic", "s3", "gcs", "disk"]] = "local",
    supported_call_types: Optional[
        List[Literal["completion", "acompletion", "embedding", "aembedding", "atranscription", "transcription"]]
    ] = ["completion", "acompletion", "embedding", "aembedding", "atranscription", "transcription"],
    ttl: Optional[float] = None,
    default_in_memory_ttl: Optional[float] = None,

    # redis cache params
    host: Optional[str] = None,
    port: Optional[str] = None,
    password: Optional[str] = None,
    namespace: Optional[str] = None,
    default_in_redis_ttl: Optional[float] = None,
    redis_flush_size=None,
    
    # GCP IAM Redis authentication params
    gcp_service_account: Optional[str] = None,
    gcp_ssl_ca_certs: Optional[str] = None,
    ssl: Optional[bool] = None,
    ssl_cert_reqs: Optional[Union[str, None]] = None,
    ssl_check_hostname: Optional[bool] = None,

    # redis semantic cache params
    similarity_threshold: Optional[float] = None,
    redis_semantic_cache_embedding_model: str = "text-embedding-ada-002",
    redis_semantic_cache_index_name: Optional[str] = None,

    # s3 Bucket, boto3 configuration
    s3_bucket_name: Optional[str] = None,
    s3_region_name: Optional[str] = None,
    s3_api_version: Optional[str] = None,
    s3_path: Optional[str] = None, # if you wish to save to a specific path
    s3_use_ssl: Optional[bool] = True,
    s3_verify: Optional[Union[bool, str]] = None,
    s3_endpoint_url: Optional[str] = None,
    s3_aws_access_key_id: Optional[str] = None,
    s3_aws_secret_access_key: Optional[str] = None,
    s3_aws_session_token: Optional[str] = None,
    s3_config: Optional[Any] = None,

    # disk cache params
    disk_cache_dir=None,

    # qdrant cache params
    qdrant_api_base: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    qdrant_collection_name: Optional[str] = None,
    qdrant_quantization_config: Optional[str] = None,
    qdrant_semantic_cache_embedding_model="text-embedding-ada-002",

    **kwargs
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
