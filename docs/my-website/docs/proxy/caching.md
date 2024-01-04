import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Caching 
Cache LLM Responses

LiteLLM supports:
- In Memory Cache
- Redis Cache 
- s3 Bucket Cache 

## Quick Start - Redis, s3 Cache
<Tabs>

<TabItem value="redis" label="redis cache">

Caching can be enabled by adding the `cache` key in the `config.yaml`

### Step 1: Add `cache` to the config.yaml
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

### Step 2: Add Redis Credentials to .env
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
### Step 3: Run proxy with config
```shell
$ litellm --config /path/to/config.yaml
```
</TabItem>

<TabItem value="s3" label="s3 cache">

### Step 1: Add `cache` to the config.yaml
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
    s3_aws_access_key_id: your_access_key  # AWS Access Key ID for S3
    s3_aws_secret_access_key: your_secret_key  # AWS Secret Access Key for S3
    s3_endpoint_url: https://s3.amazonaws.com  # [OPTIONAL] S3 endpoint URL, if you want to use Backblaze/cloudflare s3 buckets
```

### Step 2: Run proxy with config
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
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "write a poem about litellm!"}],
     "temperature": 0.7
   }'

curl http://0.0.0.0:8000/v1/chat/completions \
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
curl --location 'http://0.0.0.0:8000/embeddings' \
  --header 'Content-Type: application/json' \
  --data ' {
  "model": "text-embedding-ada-002",
  "input": ["write a litellm poem"]
  }'

curl --location 'http://0.0.0.0:8000/embeddings' \
  --header 'Content-Type: application/json' \
  --data ' {
  "model": "text-embedding-ada-002",
  "input": ["write a litellm poem"]
  }'
```
</TabItem>
</Tabs>

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

The proxy support 3 cache-controls:

- `ttl`: Will cache the response for the user-defined amount of time (in seconds).
- `s-maxage`: Will only accept cached responses that are within user-defined range (in seconds).
- `no-cache`: Will not return a cached response, but instead call the actual endpoint. 

[Let us know if you need more](https://github.com/BerriAI/litellm/issues/1218)

**Turn off caching**

```python
import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
		base_url="http://0.0.0.0:8000"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
    cache={
			"no-cache": True # will not return a cached response 
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
		base_url="http://0.0.0.0:8000"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
    cache={
			"ttl": 600 # caches response for 10 minutes 
		}
)
```

```python
import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
		base_url="http://0.0.0.0:8000"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
    cache={
			"s-maxage": 600 # only get responses cached within last 10 minutes 
		}
)
```

## Supported `cache_params`

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
