import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Router - Load Balancing, Queueing

LiteLLM manages:
- Load-balance across multiple deployments (e.g. Azure/OpenAI)
- Prioritizing important requests to ensure they don't fail (i.e. Queueing)

## Load Balancing
(s/o [@paulpierre](https://www.linkedin.com/in/paulpierre/) for his contribution to this implementation)
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
#### Routing Strategies - Shuffle, Rate Limit Aware

Router provides 2 strategies for routing your calls across multiple deployments: 

<Tabs>
<TabItem value="simple-shuffle" label="Simple Shuffle">

**Default** Randomly picks a deployment to route a call too.

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


router = Router(model_list=model_list, routing_strategy="simple-shuffle")


response = await router.acompletion(model="gpt-3.5-turbo", 
				messages=[{"role": "user", "content": "Hey, how's it going?"}])

print(response)
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
                routing_strategy="usage-based-routing")


response = await router.acompletion(model="gpt-3.5-turbo", 
				messages=[{"role": "user", "content": "Hey, how's it going?"}]

print(response)
```


</TabItem>
</Tabs>

#### Caching + Request Timeouts 

In production, we recommend using a Redis cache. For quickly testing things locally, we also support simple in-memory caching. 

**In-memory Cache + Timeouts**

```python
router = Router(model_list=model_list, 
                cache_responses=True, 
                timeout=30) # timeout set to 30s 

print(response)
```

**Redis Cache + Timeouts**
```python
router = Router(model_list=model_list, 
                redis_host=os.getenv("REDIS_HOST"), 
                redis_password=os.getenv("REDIS_PASSWORD"), 
                redis_port=os.getenv("REDIS_PORT"),
                cache_responses=True, 
                timeout=30)

print(response)
```

#### Retry failed requests

For both async + sync functions, we support retrying failed requests. 

If it's a RateLimitError we implement exponential backoffs 

If it's a generic OpenAI API Error, we retry immediately 

For any other exception types, we don't retry

Here's a quick look at how we can set `num_retries = 3`: 

```python 
from litellm import Router

router = Router(model_list=model_list, 
                cache_responses=True, 
                timeout=30, 
                num_retries=3)

user_message = "Hello, whats the weather in San Francisco??"
messages = [{"content": user_message, "role": "user"}]

# normal call 
response = router.completion(model="gpt-3.5-turbo", messages=messages)

print(f"response: {response}")
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


#### Deploy Router 

If you want a server to just route requests to different LLM APIs, use our [OpenAI Proxy Server](./simple_proxy.md#multiple-instances-of-1-model)

## Queuing

### Quick Start 

This requires a [Redis DB](https://redis.com/) to work. 

Our implementation uses LiteLLM's proxy server + Celery workers to process up to 100 req./s

[**See Code**](https://github.com/BerriAI/litellm/blob/fbf9cab5b9e35df524e2c9953180c58d92e4cd97/litellm/proxy/proxy_server.py#L589)

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

3. Test (in another window) → sends 100 simultaneous requests to the queue 

```bash
$ litellm --test_async --num_requests 100
```


### Available Endpoints
- `/queue/request` - Queues a /chat/completions request. Returns a job id. 
- `/queue/response/{id}` - Returns the status of a job. If completed, returns the response as well. Potential status's are: `queued` and `finished`.


## Hosted Router + Request Queing api.litellm.ai
Queue your LLM API requests to ensure you're under your rate limits
- Step 1: Make a POST request `/queue/request` (this follows the same input format as an openai `/chat/completions` call, and returns a job id).
- Step 2: Make a GET request, `queue/response` to check if it's completed


## Step 1 Add a config to the proxy, generate a temp key 
```python
import requests
import time
config = {
  "model_list": [
    {
      "model_name": "gpt-3.5-turbo",
      "litellm_params": {
        "model": "gpt-3.5-turbo",
        "api_key": "sk-"
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
    url = "http://0.0.0.0:8000/key/generate",
    json={
        "config": config,
        "duration": "30d" # default to 30d, set it to 30m if you want a temp key
    },
    headers={
        "Authorization": "Bearer sk-hosted-litellm"
    }
)

print("\nresponse from generating key", response.json())

generated_key = response.json()["key"]
print("\ngenerated key for proxy", generated_key)
```

# Step 2: Queue a request to the proxy, using your generated_key
```python
job_response = requests.post(
    url = "http://0.0.0.0:8000/queue/request",
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

job_response = job_response.json()
job_id  = job_response["id"]
polling_url = job_response["url"]
polling_url = f"http://0.0.0.0:8000{polling_url}"
print("\nCreated Job, Polling Url", polling_url)
```

# Step 3: Poll the request
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
        polling_response = polling_response.json()
        print("\nResponse from polling url", polling_response)
        status = polling_response["status"]
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