import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Router - Load Balancing, Fallbacks

LiteLLM manages:
- Load-balance across multiple deployments (e.g. Azure/OpenAI)
- Prioritizing important requests to ensure they don't fail (i.e. Queueing)
- Basic reliability logic - cooldowns, fallbacks, timeouts and retries (fixed + exponential backoff) across multiple deployments/providers.

In production, litellm supports using Redis as a way to track cooldown server and usage (managing tpm/rpm limits).

:::info

If you want a server to load balance across different LLM APIs, use our [OpenAI Proxy Server](./simple_proxy#load-balancing---multiple-instances-of-1-model)

:::


## Load Balancing
(s/o [@paulpierre](https://www.linkedin.com/in/paulpierre/) and [sweep proxy](https://docs.sweep.dev/blogs/openai-proxy) for their contributions to this implementation)
[**See Code**](https://github.com/BerriAI/litellm/blob/main/litellm/router.py)

### Quick Start

```python
from litellm import Router

model_list = [{ # list of model deployments 
	"model_name": "gpt-3.5-turbo", # model alias 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-v-2", # actual model name
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE")
	}
}, {
    "model_name": "gpt-3.5-turbo", 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-functioncalling", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE")
	}
}, {
    "model_name": "gpt-3.5-turbo", 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "gpt-3.5-turbo", 
		"api_key": os.getenv("OPENAI_API_KEY"),
	}
}]

router = Router(model_list=model_list)

# openai.ChatCompletion.create replacement
response = await router.acompletion(model="gpt-3.5-turbo", 
				messages=[{"role": "user", "content": "Hey, how's it going?"}])

print(response)
```

### Available Endpoints
- `router.completion()` - chat completions endpoint to call 100+ LLMs
- `router.acompletion()` - async chat completion calls
- `router.embeddings()` - embedding endpoint for Azure, OpenAI, Huggingface endpoints
- `router.aembeddings()` - async embeddings endpoint
- `router.text_completion()` - completion calls in the old OpenAI `/v1/completions` endpoint format

### Advanced
#### Routing Strategies - Weighted Pick, Rate Limit Aware

Router provides 2 strategies for routing your calls across multiple deployments: 

<Tabs>
<TabItem value="simple-shuffle" label="Weighted Pick">

**Default** Picks a deployment based on the provided **Requests per minute (rpm) or Tokens per minute (tpm)**

If `rpm` or `tpm` is not provided, it randomly picks a deployment

```python
from litellm import Router 
import asyncio

model_list = [{ # list of model deployments 
	"model_name": "gpt-3.5-turbo", # model alias 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-v-2", # actual model name
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE"),
		"rpm": 900,			# requests per minute for this API
	}
}, {
    "model_name": "gpt-3.5-turbo", 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-functioncalling", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE"),
		"rpm": 10,
	}
}, {
    "model_name": "gpt-3.5-turbo", 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "gpt-3.5-turbo", 
		"api_key": os.getenv("OPENAI_API_KEY"),
		"rpm": 10,
	}
}]

# init router
router = Router(model_list=model_list, routing_strategy="simple-shuffle")
async def router_acompletion():
	response = await router.acompletion(
		model="gpt-3.5-turbo", 
		messages=[{"role": "user", "content": "Hey, how's it going?"}]
	)
	print(response)
	return response

asyncio.run(router_acompletion())
```
</TabItem>
<TabItem value="usage-based" label="Rate-Limit Aware">

This will route to the deployment with the lowest TPM usage for that minute. 

In production, we use Redis to track usage (TPM/RPM) across multiple deployments. 

If you pass in the deployment's tpm/rpm limits, this will also check against that, and filter out any who's limits would be exceeded. 

For Azure, your RPM = TPM/6. 


```python
from litellm import Router 


model_list = [{ # list of model deployments 
	"model_name": "gpt-3.5-turbo", # model alias 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-v-2", # actual model name
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE")
	}, 
    "tpm": 100000,
	"rpm": 10000,
}, {
    "model_name": "gpt-3.5-turbo", 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-functioncalling", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE")
	},
    "tpm": 100000,
	"rpm": 1000,
}, {
    "model_name": "gpt-3.5-turbo", 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "gpt-3.5-turbo", 
		"api_key": os.getenv("OPENAI_API_KEY"),
	},
    "tpm": 100000,
	"rpm": 1000,
}]
router = Router(model_list=model_list, 
                redis_host=os.environ["REDIS_HOST"], 
				redis_password=os.environ["REDIS_PASSWORD"], 
				redis_port=os.environ["REDIS_PORT"], 
                routing_strategy="simple-shuffle")


response = await router.acompletion(model="gpt-3.5-turbo", 
				messages=[{"role": "user", "content": "Hey, how's it going?"}]

print(response)
```


</TabItem>
</Tabs>

## Basic Reliability

### Timeouts 

The timeout set in router is for the entire length of the call, and is passed down to the completion() call level as well. 

```python
from litellm import Router 

model_list = [{...}]

router = Router(model_list=model_list, 
                timeout=30) # raise timeout error if call takes > 30s 

print(response)
```

### Cooldowns

Set the limit for how many calls a model is allowed to fail in a minute, before being cooled down for a minute. 

```python
from litellm import Router

model_list = [{...}]

router = Router(model_list=model_list, 
                allowed_fails=1) # cooldown model if it fails > 1 call in a minute. 

user_message = "Hello, whats the weather in San Francisco??"
messages = [{"content": user_message, "role": "user"}]

# normal call 
response = router.completion(model="gpt-3.5-turbo", messages=messages)

print(f"response: {response}")

```

### Retries

For both async + sync functions, we support retrying failed requests. 

For RateLimitError we implement exponential backoffs 

For generic errors, we retry immediately 

Here's a quick look at how we can set `num_retries = 3`: 

```python 
from litellm import Router

model_list = [{...}]

router = Router(model_list=model_list,  
                num_retries=3)

user_message = "Hello, whats the weather in San Francisco??"
messages = [{"content": user_message, "role": "user"}]

# normal call 
response = router.completion(model="gpt-3.5-turbo", messages=messages)

print(f"response: {response}")
```

### Fallbacks 

If a call fails after num_retries, fall back to another model group. 

If the error is a context window exceeded error, fall back to a larger model group (if given). 

```python
from litellm import Router

model_list = [
    { # list of model deployments 
		"model_name": "azure/gpt-3.5-turbo", # openai model name 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "azure/chatgpt-v-2", 
			"api_key": "bad-key",
			"api_version": os.getenv("AZURE_API_VERSION"),
			"api_base": os.getenv("AZURE_API_BASE")
		},
		"tpm": 240000,
		"rpm": 1800
	}, 
    { # list of model deployments 
		"model_name": "azure/gpt-3.5-turbo-context-fallback", # openai model name 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "azure/chatgpt-v-2", 
			"api_key": "bad-key",
			"api_version": os.getenv("AZURE_API_VERSION"),
			"api_base": os.getenv("AZURE_API_BASE")
		},
		"tpm": 240000,
		"rpm": 1800
	}, 
	{
		"model_name": "azure/gpt-3.5-turbo", # openai model name 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "azure/chatgpt-functioncalling", 
			"api_key": "bad-key",
			"api_version": os.getenv("AZURE_API_VERSION"),
			"api_base": os.getenv("AZURE_API_BASE")
		},
		"tpm": 240000,
		"rpm": 1800
	}, 
	{
		"model_name": "gpt-3.5-turbo", # openai model name 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "gpt-3.5-turbo", 
			"api_key": os.getenv("OPENAI_API_KEY"),
		},
		"tpm": 1000000,
		"rpm": 9000
	},
    {
		"model_name": "gpt-3.5-turbo-16k", # openai model name 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "gpt-3.5-turbo-16k", 
			"api_key": os.getenv("OPENAI_API_KEY"),
		},
		"tpm": 1000000,
		"rpm": 9000
	}
]


router = Router(model_list=model_list, 
                fallbacks=[{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}], 
                context_window_fallbacks=[{"azure/gpt-3.5-turbo-context-fallback": ["gpt-3.5-turbo-16k"]}, {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]}],
                set_verbose=True)


user_message = "Hello, whats the weather in San Francisco??"
messages = [{"content": user_message, "role": "user"}]

# normal fallback call 
response = router.completion(model="azure/gpt-3.5-turbo", messages=messages)

# context window fallback call
response = router.completion(model="azure/gpt-3.5-turbo-context-fallback", messages=messages)

print(f"response: {response}")
```

### Caching

In production, we recommend using a Redis cache. For quickly testing things locally, we also support simple in-memory caching. 

**In-memory Cache**

```python
router = Router(model_list=model_list, 
                cache_responses=True)

print(response)
```

**Redis Cache**
```python
router = Router(model_list=model_list, 
                redis_host=os.getenv("REDIS_HOST"), 
                redis_password=os.getenv("REDIS_PASSWORD"), 
                redis_port=os.getenv("REDIS_PORT"),
                cache_responses=True)

print(response)
```

**Pass in Redis URL, additional kwargs** 
```python 
router = Router(model_list: Optional[list] = None,
                 ## CACHING ## 
                 redis_url=os.getenv("REDIS_URL")",
				 cache_kwargs= {}, # additional kwargs to pass to RedisCache (see caching.py)
				 cache_responses=True)
```

#### Default litellm.completion/embedding params

You can also set default params for litellm completion/embedding calls. Here's how to do that: 

```python 
from litellm import Router

fallback_dict = {"gpt-3.5-turbo": "gpt-3.5-turbo-16k"}

router = Router(model_list=model_list, 
                default_litellm_params={"context_window_fallback_dict": fallback_dict})

user_message = "Hello, whats the weather in San Francisco??"
messages = [{"content": user_message, "role": "user"}]

# normal call 
response = router.completion(model="gpt-3.5-turbo", messages=messages)

print(f"response: {response}")
```


## Deploy Router 

If you want a server to load balance across different LLM APIs, use our [OpenAI Proxy Server](./simple_proxy#load-balancing---multiple-instances-of-1-model)

## Queuing (Beta)

**Never fail a request due to rate limits**

The LiteLLM Queuing endpoints can handle 100+ req/s. We use Celery workers to process requests. 

:::info

This is pretty new, and might have bugs. Any contributions to improving our implementation are welcome

:::


[**See Code**](https://github.com/BerriAI/litellm/blob/fbf9cab5b9e35df524e2c9953180c58d92e4cd97/litellm/proxy/proxy_server.py#L589)


### Quick Start 

1. Add Redis credentials in a .env file

```python
REDIS_HOST="my-redis-endpoint"
REDIS_PORT="my-redis-port"
REDIS_PASSWORD="my-redis-password" # [OPTIONAL] if self-hosted
REDIS_USERNAME="default" # [OPTIONAL] if self-hosted
```

2. Start litellm server with your model config

```bash
$ litellm --config /path/to/config.yaml --use_queue
```

Here's an example config for `gpt-3.5-turbo`

**config.yaml** (This will load balance between OpenAI + Azure endpoints)
```yaml
model_list: 
  - model_name: gpt-3.5-turbo
    litellm_params: 
      model: gpt-3.5-turbo
      api_key: 
  - model_name: gpt-3.5-turbo
    litellm_params: 
      model: azure/chatgpt-v-2 # actual model name
      api_key: 
      api_version: 2023-07-01-preview
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
```

3. Test (in another window) → sends 100 simultaneous requests to the queue 

```bash
$ litellm --test_async --num_requests 100
```


### Available Endpoints
- `/queue/request` - Queues a /chat/completions request. Returns a job id. 
- `/queue/response/{id}` - Returns the status of a job. If completed, returns the response as well. Potential status's are: `queued` and `finished`.


## Hosted Request Queing api.litellm.ai
Queue your LLM API requests to ensure you're under your rate limits
- Step 1: Step 1 Add a config to the proxy, generate a temp key 
- Step 2: Queue a request to the proxy, using your generated_key
- Step 3: Poll the request


### Step 1 Add a config to the proxy, generate a temp key 
```python
import requests
import time
import os

# Set the base URL as needed
base_url = "https://api.litellm.ai"

# Step 1 Add a config to the proxy, generate a temp key
# use the same model_name to load balance
config = {
  "model_list": [
    {
      "model_name": "gpt-3.5-turbo",
      "litellm_params": {
        "model": "gpt-3.5-turbo",
        "api_key": os.environ['OPENAI_API_KEY'],
      }
    },
    {
      "model_name": "gpt-3.5-turbo",
      "litellm_params": {
        "model": "azure/chatgpt-v-2",
        "api_key": "",
        "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
        "api_version": "2023-07-01-preview"
      }
    }
  ]
}

response = requests.post(
    url=f"{base_url}/key/generate",
    json={
        "config": config,
        "duration": "30d"  # default to 30d, set it to 30m if you want a temp 30 minute key
    },
    headers={
        "Authorization": "Bearer sk-hosted-litellm" # this is the key to use api.litellm.ai
    }
)

print("\nresponse from generating key", response.text)
print("\n json response from gen key", response.json())

generated_key = response.json()["key"]
print("\ngenerated key for proxy", generated_key)
```

#### Output
```shell
response from generating key {"key":"sk-...,"expires":"2023-12-22T03:43:57.615000+00:00"}
```

### Step 2: Queue a request to the proxy, using your generated_key
```python
print("Creating a job on the proxy")
job_response = requests.post(
    url=f"{base_url}/queue/request",
    json={
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'system', 'content': f'You are a helpful assistant. What is your name'},
        ],
    },
    headers={
        "Authorization": f"Bearer {generated_key}"
    }
)
print(job_response.status_code)
print(job_response.text)
print("\nResponse from creating job", job_response.text)
job_response = job_response.json()
job_id = job_response["id"]
polling_url = job_response["url"]
polling_url = f"{base_url}{polling_url}"
print("\nCreated Job, Polling Url", polling_url)
```

#### Output
```shell
Response from creating job 
{"id":"0e3d9e98-5d56-4d07-9cc8-c34b7e6658d7","url":"/queue/response/0e3d9e98-5d56-4d07-9cc8-c34b7e6658d7","eta":5,"status":"queued"}
```

### Step 3: Poll the request
```python
while True:
    try:
        print("\nPolling URL", polling_url)
        polling_response = requests.get(
            url=polling_url,
            headers={
                "Authorization": f"Bearer {generated_key}"
            }
        )
        print("\nResponse from polling url", polling_response.text)
        polling_response = polling_response.json()
        status = polling_response.get("status", None)
        if status == "finished":
            llm_response = polling_response["result"]
            print("LLM Response")
            print(llm_response)
            break
        time.sleep(0.5)
    except Exception as e:
        print("got exception in polling", e)
        break
```

#### Output
```shell
Polling URL https://api.litellm.ai/queue/response/0e3d9e98-5d56-4d07-9cc8-c34b7e6658d7

Response from polling url {"status":"queued"}

Polling URL https://api.litellm.ai/queue/response/0e3d9e98-5d56-4d07-9cc8-c34b7e6658d7

Response from polling url {"status":"queued"}

Polling URL https://api.litellm.ai/queue/response/0e3d9e98-5d56-4d07-9cc8-c34b7e6658d7

Response from polling url 
{"status":"finished","result":{"id":"chatcmpl-8NYRce4IeI4NzYyodT3NNp8fk5cSW","choices":[{"finish_reason":"stop","index":0,"message":{"content":"I am an AI assistant and do not have a physical presence or personal identity. You can simply refer to me as \"Assistant.\" How may I assist you today?","role":"assistant"}}],"created":1700624639,"model":"gpt-3.5-turbo-0613","object":"chat.completion","system_fingerprint":null,"usage":{"completion_tokens":33,"prompt_tokens":17,"total_tokens":50}}}

```