import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Caching 
Cache LLM Responses

LiteLLM supports:
- In Memory Cache
- Redis Cache 
- Redis Semantic Cache
- Dual Semantic Cache
- s3 Bucket Cache 

## Quick Start - Redis, s3 Cache, Semantic Cache, Dual Semantic Cache
<Tabs>

<TabItem value="redis" label="redis cache">

Caching can be enabled by adding the `cache` key in the `config.yaml`

#### Step 1: Add `cache` to the config.yaml
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
  - model_name: text-embedding-ada-002
    litellm_params:
      model: text-embedding-ada-002

litellm_settings:
  set_verbose: True
  cache: True          # set cache responses to True, litellm defaults to using a redis cache
```

#### [OPTIONAL] Step 1.5: Add redis namespaces, default ttl 

## Namespace
If you want to create some folder for your keys, you can set a namespace, like this:

```yaml
litellm_settings:
  cache: true 
  cache_params:        # set cache params for redis
    type: redis
    namespace: "litellm_caching"
```

and keys will be stored like:

```
litellm_caching:<hash>
```

## TTL

```yaml
litellm_settings:
  cache: true 
  cache_params:        # set cache params for redis
    type: redis
    ttl: 600 # will be cached on redis for 600s
```


## SSL

just set `REDIS_SSL="True"` in your .env, and LiteLLM will pick this up. 

```env
REDIS_SSL="True"
```

For quick testing, you can also use REDIS_URL, eg.:

```
REDIS_URL="rediss://.."
```

but we **don't** recommend using REDIS_URL in prod. We've noticed a performance difference between using it vs. redis_host, port, etc. 
#### Step 2: Add Redis Credentials to .env
Set either `REDIS_URL` or the `REDIS_HOST` in your os environment, to enable caching.

  ```shell
  REDIS_URL = ""        # REDIS_URL='redis://username:password@hostname:port/database'
  ## OR ## 
  REDIS_HOST = ""       # REDIS_HOST='redis-18841.c274.us-east-1-3.ec2.cloud.redislabs.com'
  REDIS_PORT = ""       # REDIS_PORT='18841'
  REDIS_PASSWORD = ""   # REDIS_PASSWORD='liteLlmIsAmazing'
  ```

**Additional kwargs**  
You can pass in any additional redis.Redis arg, by storing the variable + value in your os environment, like this: 
```shell
REDIS_<redis-kwarg-name> = ""
``` 

[**See how it's read from the environment**](https://github.com/BerriAI/litellm/blob/4d7ff1b33b9991dcf38d821266290631d9bcd2dd/litellm/_redis.py#L40)
#### Step 3: Run proxy with config
```shell
$ litellm --config /path/to/config.yaml
```
</TabItem>

<TabItem value="s3" label="s3 cache">

#### Step 1: Add `cache` to the config.yaml
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
  - model_name: text-embedding-ada-002
    litellm_params:
      model: text-embedding-ada-002

litellm_settings:
  set_verbose: True
  cache: True          # set cache responses to True
  cache_params:        # set cache params for s3
    type: s3
    s3_bucket_name: cache-bucket-litellm   # AWS Bucket Name for S3
    s3_region_name: us-west-2              # AWS Region Name for S3
    s3_aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID  # us os.environ/<variable name> to pass environment variables. This is AWS Access Key ID for S3
    s3_aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY  # AWS Secret Access Key for S3
    s3_endpoint_url: https://s3.amazonaws.com  # [OPTIONAL] S3 endpoint URL, if you want to use Backblaze/cloudflare s3 buckets
```

#### Step 2: Run proxy with config
```shell
$ litellm --config /path/to/config.yaml
```
</TabItem>


<TabItem value="redis-sem" label="redis semantic cache">

Caching can be enabled by adding the `cache` key in the `config.yaml`

#### Step 1: Add `cache` to the config.yaml
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
  - model_name: azure-embedding-model
    litellm_params:
      model: azure/azure-embedding-model
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"

litellm_settings:
  set_verbose: True
  cache: True          # set cache responses to True, litellm defaults to using a redis cache
  cache_params:
    type: "redis-semantic"  
    similarity_threshold: 0.8   # similarity threshold for semantic cache
    redis_semantic_cache_embedding_model: azure-embedding-model # set this to a model_name set in model_list
```

#### Step 2: Add Redis Credentials to .env
Set either `REDIS_URL` or the `REDIS_HOST` in your os environment, to enable caching.

  ```shell
  REDIS_URL = ""        # REDIS_URL='redis://username:password@hostname:port/database'
  ## OR ## 
  REDIS_HOST = ""       # REDIS_HOST='redis-18841.c274.us-east-1-3.ec2.cloud.redislabs.com'
  REDIS_PORT = ""       # REDIS_PORT='18841'
  REDIS_PASSWORD = ""   # REDIS_PASSWORD='liteLlmIsAmazing'
  ```

**Additional kwargs**  
You can pass in any additional redis.Redis arg, by storing the variable + value in your os environment, like this: 
```shell
REDIS_<redis-kwarg-name> = ""
``` 

#### Step 3: Run proxy with config
```shell
$ litellm --config /path/to/config.yaml
```
</TabItem>

<TabItem value="dual-sem" label="dual semantic cache">

Caching can be enabled by adding the `cache` key in the `config.yaml`

#### Step 1: Add `cache` to the config.yaml
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
  - model_name: azure-embedding-model
    litellm_params:
      model: azure/azure-embedding-model
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"

litellm_settings:
  set_verbose: True
  cache: True          # set cache responses to True, litellm defaults to using a redis cache
  cache_params:
    type: "dual-semantic"
    host: "localhost" # The host address for the Redis cache. Required if type is "dual-semantic".
    port: "6379" # The port number for the Redis cache. Required if type is "dual-semantic".
    password: "sk-4321" # The password for the Redis cache. Required if type is "dual-semantic".
    similarity_threshold: 0.75 # Similarity threshold for semantic cache
    redis_semantic_cache_embedding_model: gpt-3.5-turbo # this model is passed to litellm.embedding(), any litellm.embedding() model is supported here. Optional.
    semantic_cache_embedding_length: 384 # Length of the embedding vector produced by the embedding model. Optional.
```

#### Step 2: Add Redis Credentials to .env
Set either `REDIS_URL` or the `REDIS_HOST` in your os environment, to enable caching.

  ```shell
  REDIS_URL = ""        # REDIS_URL='redis://username:password@hostname:port/database'
  ## OR ## 
  REDIS_HOST = ""       # REDIS_HOST='redis-18841.c274.us-east-1-3.ec2.cloud.redislabs.com'
  REDIS_PORT = ""       # REDIS_PORT='18841'
  REDIS_PASSWORD = ""   # REDIS_PASSWORD='liteLlmIsAmazing'
  ```

**Additional kwargs**  
You can pass in any additional redis.Redis arg, by storing the variable + value in your os environment, like this: 
```shell
REDIS_<redis-kwarg-name> = ""
``` 

#### Step 3: Run proxy with config
```shell
$ litellm --config /path/to/config.yaml
```
</TabItem>
</Tabs>


## Using Caching - /chat/completions

<Tabs>
<TabItem value="chat_completions" label="/chat/completions">

Send the same request twice:
```shell
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "write a poem about litellm!"}],
     "temperature": 0.7
   }'

curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "write a poem about litellm!"}],
     "temperature": 0.7
   }'
```
</TabItem>
<TabItem value="embeddings" label="/embeddings">

Send the same request twice:
```shell
curl --location 'http://0.0.0.0:4000/embeddings' \
  --header 'Content-Type: application/json' \
  --data ' {
  "model": "text-embedding-ada-002",
  "input": ["write a litellm poem"]
  }'

curl --location 'http://0.0.0.0:4000/embeddings' \
  --header 'Content-Type: application/json' \
  --data ' {
  "model": "text-embedding-ada-002",
  "input": ["write a litellm poem"]
  }'
```
</TabItem>
</Tabs>

## Debugging Caching - `/cache/ping`
LiteLLM Proxy exposes a `/cache/ping` endpoint to test if the cache is working as expected

**Usage**
```shell
curl --location 'http://0.0.0.0:4000/cache/ping'  -H "Authorization: Bearer sk-1234"
```

**Expected Response - when cache healthy**
```shell
{
    "status": "healthy",
    "cache_type": "redis",
    "ping_response": true,
    "set_cache_response": "success",
    "litellm_cache_params": {
        "supported_call_types": "['completion', 'acompletion', 'embedding', 'aembedding', 'atranscription', 'transcription']",
        "type": "redis",
        "namespace": "None"
    },
    "redis_cache_params": {
        "redis_client": "Redis<ConnectionPool<Connection<host=redis-16337.c322.us-east-1-2.ec2.cloud.redislabs.com,port=16337,db=0>>>",
        "redis_kwargs": "{'url': 'redis://:******@redis-16337.c322.us-east-1-2.ec2.cloud.redislabs.com:16337'}",
        "async_redis_conn_pool": "BlockingConnectionPool<Connection<host=redis-16337.c322.us-east-1-2.ec2.cloud.redislabs.com,port=16337,db=0>>",
        "redis_version": "7.2.0"
    }
}
```

## Advanced
### Set Cache Params on config.yaml
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
  - model_name: text-embedding-ada-002
    litellm_params:
      model: text-embedding-ada-002

litellm_settings:
  set_verbose: True
  cache: True          # set cache responses to True, litellm defaults to using a redis cache
  cache_params:         # cache_params are optional
    type: "redis"  # The type of cache to initialize. Can be "local" or "redis". Defaults to "local".
    host: "localhost"  # The host address for the Redis cache. Required if type is "redis".
    port: 6379  # The port number for the Redis cache. Required if type is "redis".
    password: "your_password"  # The password for the Redis cache. Required if type is "redis".
    
    # Optional configurations
    supported_call_types: ["acompletion", "completion", "embedding", "aembedding"] # defaults to all litellm call types
```

### Turn on / off caching per request.  

The proxy support 4 cache-controls:

- `ttl`: *Optional(int)* - Will cache the response for the user-defined amount of time (in seconds).
- `s-maxage`: *Optional(int)* Will only accept cached responses that are within user-defined range (in seconds).
- `no-cache`: *Optional(bool)* Will not return a cached response, but instead call the actual endpoint. 
- `no-store`: *Optional(bool)* Will not cache the response. 

[Let us know if you need more](https://github.com/BerriAI/litellm/issues/1218)

**Turn off caching**

```python
import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
		base_url="http://0.0.0.0:4000"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
    extra_body = {        # OpenAI python accepts extra args in extra_body
        cache: {
          "no-cache": True # will not return a cached response 
      }
    }
)
```

**Turn on caching**

```python
import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
		base_url="http://0.0.0.0:4000"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
    extra_body = {        # OpenAI python accepts extra args in extra_body
        cache: {
          "ttl": 600 # caches response for 10 minutes 
      }
    }
)
```

```python
import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
		base_url="http://0.0.0.0:4000"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
    extra_body = {        # OpenAI python accepts extra args in extra_body
        cache: {
          "s-maxage": 600 # only get responses cached within last 10 minutes 
      }
    }
)
```

### Turn on / off caching per Key.

1. Add cache params when creating a key [full list](#turn-on--off-caching-per-key)

```bash 
curl -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-D '{
    "user_id": "222",
    "metadata": {
        "cache": {
            "no-cache": true
        }
    }
}'
```

2. Test it! 

```bash 
curl -X POST 'http://localhost:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer <YOUR_NEW_KEY>' \
-D '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "bom dia"}]}'
```

### Deleting Cache Keys - `/cache/delete` 
In order to delete a cache key, send a request to `/cache/delete` with the `keys` you want to delete

Example 
```shell
curl -X POST "http://0.0.0.0:4000/cache/delete" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"keys": ["586bf3f3c1bf5aecb55bd9996494d3bbc69eb58397163add6d49537762a7548d", "key2"]}'
```

```shell
# {"status":"success"}
```

#### Viewing Cache Keys from responses
You can view the cache_key in the response headers, on cache hits the cache key is sent as the `x-litellm-cache-key` response headers
```shell
curl -i --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "user": "ishan",
    "messages": [
        {
        "role": "user",
        "content": "what is litellm"
        }
    ],
}'
```

Response from litellm proxy 
```json
date: Thu, 04 Apr 2024 17:37:21 GMT
content-type: application/json
x-litellm-cache-key: 586bf3f3c1bf5aecb55bd9996494d3bbc69eb58397163add6d49537762a7548d

{
    "id": "chatcmpl-9ALJTzsBlXR9zTxPvzfFFtFbFtG6T",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "I'm sorr.."
                "role": "assistant"
            }
        }
    ],
    "created": 1712252235,
}
             
```


### Turn on `batch_redis_requests` 

**What it does?**
When a request is made:

- Check if a key starting with `litellm:<hashed_api_key>:<call_type>:` exists in-memory, if no - get the last 100 cached requests for this key and store it

- New requests are stored with this `litellm:..` as the namespace

**Why?**
Reduce number of redis GET requests. This improved latency by 46% in prod load tests. 

**Usage**

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis
    ... # remaining redis args (host, port, etc.)
  callbacks: ["batch_redis_requests"] # 👈 KEY CHANGE!
```

[**SEE CODE**](https://github.com/BerriAI/litellm/blob/main/litellm/proxy/hooks/batch_redis_get.py)

## Supported `cache_params` on proxy config.yaml

```yaml
cache_params:
  # Type of cache (options: "local", "redis", "s3")
  type: s3

  # List of litellm call types to cache for
  # Options: "completion", "acompletion", "embedding", "aembedding"
  supported_call_types:
    - completion
    - acompletion
    - embedding
    - aembedding

  # Redis cache parameters
  host: localhost  # Redis server hostname or IP address
  port: "6379"  # Redis server port (as a string)
  password: secret_password  # Redis server password

  # S3 cache parameters
  s3_bucket_name: your_s3_bucket_name  # Name of the S3 bucket
  s3_region_name: us-west-2  # AWS region of the S3 bucket
  s3_api_version: 2006-03-01  # AWS S3 API version
  s3_use_ssl: true  # Use SSL for S3 connections (options: true, false)
  s3_verify: true  # SSL certificate verification for S3 connections (options: true, false)
  s3_endpoint_url: https://s3.amazonaws.com  # S3 endpoint URL
  s3_aws_access_key_id: your_access_key  # AWS Access Key ID for S3
  s3_aws_secret_access_key: your_secret_key  # AWS Secret Access Key for S3
  s3_aws_session_token: your_session_token  # AWS Session Token for temporary credentials

```

## Advanced - user api key cache ttl 

Configure how long the in-memory cache stores the key object (prevents db requests)

```yaml
general_settings:
  user_api_key_cache_ttl: <your-number> #time in seconds
```

By default this value is set to 60s.