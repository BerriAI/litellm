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

If you want a server to load balance across different LLM APIs, use our [OpenAI Proxy Server](./proxy/load_balancing.md)

:::


## Load Balancing
(s/o [@paulpierre](https://www.linkedin.com/in/paulpierre/) and [sweep proxy](https://docs.sweep.dev/blogs/openai-proxy) for their contributions to this implementation)
[**See Code**](https://github.com/BerriAI/litellm/blob/main/litellm/router.py)

### Quick Start

```python
from litellm import Router

model_list = [{ # list of model deployments 
	"model_name": "gpt-3.5-turbo", # model alias -> loadbalance between models with same `model_name`
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
}, {
    "model_name": "gpt-4", 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/gpt-4", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_base": os.getenv("AZURE_API_BASE"),
		"api_version": os.getenv("AZURE_API_VERSION"),
	}
}, {
    "model_name": "gpt-4", 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "gpt-4", 
		"api_key": os.getenv("OPENAI_API_KEY"),
	}
},

]

router = Router(model_list=model_list)

# openai.ChatCompletion.create replacement
# requests with model="gpt-3.5-turbo" will pick a deployment where model_name="gpt-3.5-turbo"
response = await router.acompletion(model="gpt-3.5-turbo", 
				messages=[{"role": "user", "content": "Hey, how's it going?"}])

print(response)

# openai.ChatCompletion.create replacement
# requests with model="gpt-4" will pick a deployment where model_name="gpt-4"
response = await router.acompletion(model="gpt-4", 
				messages=[{"role": "user", "content": "Hey, how's it going?"}])

print(response)
```

### Available Endpoints
- `router.completion()` - chat completions endpoint to call 100+ LLMs
- `router.acompletion()` - async chat completion calls
- `router.embeddings()` - embedding endpoint for Azure, OpenAI, Huggingface endpoints
- `router.aembeddings()` - async embeddings calls
- `router.text_completion()` - completion calls in the old OpenAI `/v1/completions` endpoint format
- `router.atext_completion()` - async text completion calls
- `router.image_generation()` - completion calls in OpenAI `/v1/images/generations` endpoint format
- `router.aimage_generation()` - async image generation calls

## Advanced - Routing Strategies
#### Routing Strategies - Weighted Pick, Rate Limit Aware, Least Busy, Latency Based

Router provides 4 strategies for routing your calls across multiple deployments: 

<Tabs>
<TabItem value="usage-based-v2" label="Rate-Limit Aware v2 (ASYNC)">

**🎉 NEW** This is an async implementation of usage-based-routing.

**Filters out deployment if tpm/rpm limit exceeded** - If you pass in the deployment's tpm/rpm limits.

Routes to **deployment with lowest TPM usage** for that minute. 

In production, we use Redis to track usage (TPM/RPM) across multiple deployments. This implementation uses **async redis calls** (redis.incr and redis.mget).

For Azure, your RPM = TPM/6. 

<Tabs>
<TabItem value="sdk" label="sdk">

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
                routing_strategy="usage-based-routing-v2" # 👈 KEY CHANGE
				enable_pre_call_check=True, # enables router rate limits for concurrent calls
				)

response = await router.acompletion(model="gpt-3.5-turbo", 
				messages=[{"role": "user", "content": "Hey, how's it going?"}]

print(response)
```
</TabItem>
<TabItem value="proxy" label="proxy">

**1. Set strategy in config**

```yaml
model_list:
	- model_name: gpt-3.5-turbo # model alias 
	  litellm_params: # params for litellm completion/embedding call 
		model: azure/chatgpt-v-2 # actual model name
		api_key: os.environ/AZURE_API_KEY
		api_version: os.environ/AZURE_API_VERSION
		api_base: os.environ/AZURE_API_BASE
      tpm: 100000
	  rpm: 10000
	- model_name: gpt-3.5-turbo 
	  litellm_params: # params for litellm completion/embedding call 
		model: gpt-3.5-turbo 
		api_key: os.getenv(OPENAI_API_KEY)
      tpm: 100000
	  rpm: 1000

router_settings:
  routing_strategy: usage-based-routing-v2 # 👈 KEY CHANGE
  redis_host: <your-redis-host>
  redis_password: <your-redis-password>
  redis_port: <your-redis-port>
  enable_pre_call_check: true

general_settings:
  master_key: sk-1234
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://localhost:4000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
    "model": "gpt-3.5-turbo", 
    "messages": [{"role": "user", "content": "Hey, how's it going?"}]
}'
```

</TabItem>
</Tabs>


</TabItem>
<TabItem value="latency-based" label="Latency-Based">


Picks the deployment with the lowest response time.

It caches, and updates the response times for deployments based on when a request was sent and received from a deployment.

[**How to test**](https://github.com/BerriAI/litellm/blob/main/litellm/tests/test_lowest_latency_routing.py)

```python
from litellm import Router 
import asyncio

model_list = [{ ... }]

# init router
router = Router(model_list=model_list,
				routing_strategy="latency-based-routing",# 👈 set routing strategy
				enable_pre_call_check=True, # enables router rate limits for concurrent calls
				)

## CALL 1+2
tasks = []
response = None
final_response = None
for _ in range(2):
	tasks.append(router.acompletion(model=model, messages=messages))
response = await asyncio.gather(*tasks)

if response is not None:
	## CALL 3 
	await asyncio.sleep(1)  # let the cache update happen
	picked_deployment = router.lowestlatency_logger.get_available_deployments(
		model_group=model, healthy_deployments=router.healthy_deployments
	)
	final_response = await router.acompletion(model=model, messages=messages)
	print(f"min deployment id: {picked_deployment}")
	print(f"model id: {final_response._hidden_params['model_id']}")
	assert (
		final_response._hidden_params["model_id"]
		== picked_deployment["model_info"]["id"]
	)
```

### Set Time Window 

Set time window for how far back to consider when averaging latency for a deployment. 

**In Router**
```python 
router = Router(..., routing_strategy_args={"ttl": 10})
```

**In Proxy**

```yaml
router_settings:
	routing_strategy_args: {"ttl": 10}
```

</TabItem>
<TabItem value="simple-shuffle" label="(Default) Weighted Pick (Async)">

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
                routing_strategy="usage-based-routing"
				enable_pre_call_check=True, # enables router rate limits for concurrent calls
				)

response = await router.acompletion(model="gpt-3.5-turbo", 
				messages=[{"role": "user", "content": "Hey, how's it going?"}]

print(response)
```


</TabItem>
<TabItem value="least-busy" label="Least-Busy">


Picks a deployment with the least number of ongoing calls, it's handling.

[**How to test**](https://github.com/BerriAI/litellm/blob/main/litellm/tests/test_least_busy_routing.py)

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
	}
}, {
    "model_name": "gpt-3.5-turbo", 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-functioncalling", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE"),
	}
}, {
    "model_name": "gpt-3.5-turbo", 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "gpt-3.5-turbo", 
		"api_key": os.getenv("OPENAI_API_KEY"),
	}
}]

# init router
router = Router(model_list=model_list, routing_strategy="least-busy")
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

</Tabs>

## Basic Reliability

### Max Parallel Requests (ASYNC)

Used in semaphore for async requests on router. Limit the max concurrent calls made to a deployment. Useful in high-traffic scenarios. 

If tpm/rpm is set, and no max parallel request limit given, we use the RPM or calculated RPM (tpm/1000/6) as the max parallel request limit. 


```python
from litellm import Router 

model_list = [{
	"model_name": "gpt-4",
	"litellm_params": {
		"model": "azure/gpt-4",
		...
		"max_parallel_requests": 10 # 👈 SET PER DEPLOYMENT
	}
}]

### OR ### 

router = Router(model_list=model_list, default_max_parallel_requests=20) # 👈 SET DEFAULT MAX PARALLEL REQUESTS 


# deployment max parallel requests > default max parallel requests
```

[**See Code**](https://github.com/BerriAI/litellm/blob/a978f2d8813c04dad34802cb95e0a0e35a3324bc/litellm/utils.py#L5605)

### Timeouts 

The timeout set in router is for the entire length of the call, and is passed down to the completion() call level as well. 

**Global Timeouts**
```python
from litellm import Router 

model_list = [{...}]

router = Router(model_list=model_list, 
                timeout=30) # raise timeout error if call takes > 30s 

print(response)
```

**Timeouts per model**

```python
from litellm import Router 
import asyncio

model_list = [{
	"model_name": "gpt-3.5-turbo",
	"litellm_params": {
		"model": "azure/chatgpt-v-2",
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE"),
		"timeout": 300 # sets a 5 minute timeout
		"stream_timeout": 30 # sets a 30s timeout for streaming calls
	}
}]

# init router
router = Router(model_list=model_list, routing_strategy="least-busy")
async def router_acompletion():
	response = await router.acompletion(
		model="gpt-3.5-turbo", 
		messages=[{"role": "user", "content": "Hey, how's it going?"}]
	)
	print(response)
	return response

asyncio.run(router_acompletion())
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

We also support setting minimum time to wait before retrying a failed request. This is via the `retry_after` param. 

```python 
from litellm import Router

model_list = [{...}]

router = Router(model_list=model_list,  
                num_retries=3, retry_after=5) # waits min 5s before retrying request

user_message = "Hello, whats the weather in San Francisco??"
messages = [{"content": user_message, "role": "user"}]

# normal call 
response = router.completion(model="gpt-3.5-turbo", messages=messages)

print(f"response: {response}")
```

### Fallbacks 

If a call fails after num_retries, fall back to another model group. 

If the error is a context window exceeded error, fall back to a larger model group (if given). 

Fallbacks are done in-order - ["gpt-3.5-turbo, "gpt-4", "gpt-4-32k"], will do 'gpt-3.5-turbo' first, then 'gpt-4', etc.

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

## Pre-Call Checks (Context Window)

Enable pre-call checks to filter out:
1. deployments with context window limit < messages for a call.
2. deployments that have exceeded rate limits when making concurrent calls. (eg. `asyncio.gather(*[
        router.acompletion(model="gpt-3.5-turbo", messages=m) for m in list_of_messages
    ])`)

<Tabs>
<TabItem value="sdk" label="SDK">

**1. Enable pre-call checks**
```python 
from litellm import Router 
# ...
router = Router(model_list=model_list, enable_pre_call_checks=True) # 👈 Set to True
```


**2. Set Model List**

For azure deployments, set the base model. Pick the base model from [this list](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json), all the azure models start with `azure/`. 

<Tabs>
<TabItem value="same-group" label="Same Group">

```python
model_list = [
            {
                "model_name": "gpt-3.5-turbo", # model group name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
				"model_info": {
					"base_model": "azure/gpt-35-turbo", # 👈 (Azure-only) SET BASE MODEL
				}
            },
            {
                "model_name": "gpt-3.5-turbo", # model group name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo-1106",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ]

router = Router(model_list=model_list, enable_pre_call_checks=True) 
```

</TabItem>

<TabItem value="different-group" label="Context Window Fallbacks (Different Groups)">

```python
model_list = [
            {
                "model_name": "gpt-3.5-turbo-small", # model group name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
				"model_info": {
					"base_model": "azure/gpt-35-turbo", # 👈 (Azure-only) SET BASE MODEL
				}
            },
            {
                "model_name": "gpt-3.5-turbo-large", # model group name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-3.5-turbo-1106",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "claude-opus", 
                "litellm_params": {  call
                    "model": "claude-3-opus-20240229",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
            },
        ]

router = Router(model_list=model_list, enable_pre_call_checks=True, context_window_fallbacks=[{"gpt-3.5-turbo-small": ["gpt-3.5-turbo-large", "claude-opus"]}]) 
```

</TabItem>

</Tabs>

**3. Test it!**

```python
"""
- Give a gpt-3.5-turbo model group with different context windows (4k vs. 16k)
- Send a 5k prompt
- Assert it works
"""
from litellm import Router
import os

try:
model_list = [
	{
		"model_name": "gpt-3.5-turbo",  # model group name
		"litellm_params": {  # params for litellm completion/embedding call
			"model": "azure/chatgpt-v-2",
			"api_key": os.getenv("AZURE_API_KEY"),
			"api_version": os.getenv("AZURE_API_VERSION"),
			"api_base": os.getenv("AZURE_API_BASE"),
		},
		"model_info": {
			"base_model": "azure/gpt-35-turbo", 
		}
	},
	{
		"model_name": "gpt-3.5-turbo",  # model group name
		"litellm_params": {  # params for litellm completion/embedding call
			"model": "gpt-3.5-turbo-1106",
			"api_key": os.getenv("OPENAI_API_KEY"),
		},
	},
]

router = Router(model_list=model_list, enable_pre_call_checks=True) 

text = "What is the meaning of 42?" * 5000

response = router.completion(
	model="gpt-3.5-turbo",
	messages=[
		{"role": "system", "content": text},
		{"role": "user", "content": "Who was Alexander?"},
	],
)

print(f"response: {response}")
```
</TabItem>
<TabItem value="proxy" label="Proxy">

:::info
Go [here](./proxy/reliability.md#advanced---context-window-fallbacks) for how to do this on the proxy
:::
</TabItem>
</Tabs>

## Caching across model groups

If you want to cache across 2 different model groups (e.g. azure deployments, and openai), use caching groups. 

```python
import litellm, asyncio, time
from litellm import Router 

# set os env
os.environ["OPENAI_API_KEY"] = ""
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

async def test_acompletion_caching_on_router_caching_groups(): 
	# tests acompletion + caching on router 
	try:
		litellm.set_verbose = True
		model_list = [
			{
				"model_name": "openai-gpt-3.5-turbo",
				"litellm_params": {
					"model": "gpt-3.5-turbo-0613",
					"api_key": os.getenv("OPENAI_API_KEY"),
				},
			},
			{
				"model_name": "azure-gpt-3.5-turbo",
				"litellm_params": {
					"model": "azure/chatgpt-v-2",
					"api_key": os.getenv("AZURE_API_KEY"),
					"api_base": os.getenv("AZURE_API_BASE"),
					"api_version": os.getenv("AZURE_API_VERSION")
				},
			}
		]

		messages = [
			{"role": "user", "content": f"write a one sentence poem {time.time()}?"}
		]
		start_time = time.time()
		router = Router(model_list=model_list, 
				cache_responses=True, 
				caching_groups=[("openai-gpt-3.5-turbo", "azure-gpt-3.5-turbo")])
		response1 = await router.acompletion(model="openai-gpt-3.5-turbo", messages=messages, temperature=1)
		print(f"response1: {response1}")
		await asyncio.sleep(1) # add cache is async, async sleep for cache to get set
		response2 = await router.acompletion(model="azure-gpt-3.5-turbo", messages=messages, temperature=1)
		assert response1.id == response2.id
		assert len(response1.choices[0].message.content) > 0
		assert response1.choices[0].message.content == response2.choices[0].message.content
	except Exception as e:
		traceback.print_exc()

asyncio.run(test_acompletion_caching_on_router_caching_groups())
```

## Track cost for Azure Deployments

**Problem**: Azure returns `gpt-4` in the response when `azure/gpt-4-1106-preview` is used. This leads to inaccurate cost tracking

**Solution** ✅ :  Set `model_info["base_model"]` on your router init so litellm uses the correct model for calculating azure cost

Step 1. Router Setup

```python
from litellm import Router

model_list = [
	{ # list of model deployments 
		"model_name": "gpt-4-preview", # model alias 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "azure/chatgpt-v-2", # actual model name
			"api_key": os.getenv("AZURE_API_KEY"),
			"api_version": os.getenv("AZURE_API_VERSION"),
			"api_base": os.getenv("AZURE_API_BASE")
		},
		"model_info": {
			"base_model": "azure/gpt-4-1106-preview" # azure/gpt-4-1106-preview will be used for cost tracking, ensure this exists in litellm model_prices_and_context_window.json
		}
	}, 
	{
		"model_name": "gpt-4-32k", 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "azure/chatgpt-functioncalling", 
			"api_key": os.getenv("AZURE_API_KEY"),
			"api_version": os.getenv("AZURE_API_VERSION"),
			"api_base": os.getenv("AZURE_API_BASE")
		},
		"model_info": {
			"base_model": "azure/gpt-4-32k" # azure/gpt-4-32k will be used for cost tracking, ensure this exists in litellm model_prices_and_context_window.json
		}
	}
]

router = Router(model_list=model_list)

```

Step 2. Access `response_cost` in the custom callback, **litellm calculates the response cost for you**

```python
import litellm
from litellm.integrations.custom_logger import CustomLogger

class MyCustomHandler(CustomLogger):        
	def log_success_event(self, kwargs, response_obj, start_time, end_time): 
		print(f"On Success")
		response_cost = kwargs.get("response_cost")
		print("response_cost=", response_cost)

customHandler = MyCustomHandler()
litellm.callbacks = [customHandler]

# router completion call
response = router.completion(
	model="gpt-4-32k", 
	messages=[{ "role": "user", "content": "Hi who are you"}]
)
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

## Custom Callbacks - Track API Key, API Endpoint, Model Used 

If you need to track the api_key, api endpoint, model, custom_llm_provider used for each completion call, you can setup a [custom callback](https://docs.litellm.ai/docs/observability/custom_callback) 

### Usage

```python
import litellm
from litellm.integrations.custom_logger import CustomLogger

class MyCustomHandler(CustomLogger):        
	def log_success_event(self, kwargs, response_obj, start_time, end_time): 
		print(f"On Success")
		print("kwargs=", kwargs)
		litellm_params= kwargs.get("litellm_params")
		api_key = litellm_params.get("api_key")
		api_base = litellm_params.get("api_base")
		custom_llm_provider= litellm_params.get("custom_llm_provider")
		response_cost = kwargs.get("response_cost")

		# print the values
		print("api_key=", api_key)
		print("api_base=", api_base)
		print("custom_llm_provider=", custom_llm_provider)
		print("response_cost=", response_cost)

	def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
		print(f"On Failure")
		print("kwargs=")

customHandler = MyCustomHandler()

litellm.callbacks = [customHandler]

# Init Router
router = Router(model_list=model_list, routing_strategy="simple-shuffle")

# router completion call
response = router.completion(
	model="gpt-3.5-turbo", 
	messages=[{ "role": "user", "content": "Hi who are you"}]
)
```

## Deploy Router 

If you want a server to load balance across different LLM APIs, use our [OpenAI Proxy Server](./simple_proxy#load-balancing---multiple-instances-of-1-model)


## Init Params for the litellm.Router

```python
def __init__(
	model_list: Optional[list] = None,
	
	## CACHING ##
	redis_url: Optional[str] = None,
	redis_host: Optional[str] = None,
	redis_port: Optional[int] = None,
	redis_password: Optional[str] = None,
	cache_responses: Optional[bool] = False,
	cache_kwargs: dict = {},  # additional kwargs to pass to RedisCache (see caching.py)
	caching_groups: Optional[
		List[tuple]
	] = None,  # if you want to cache across model groups
	client_ttl: int = 3600,  # ttl for cached clients - will re-initialize after this time in seconds

	## RELIABILITY ##
	num_retries: int = 0,
	timeout: Optional[float] = None,
	default_litellm_params={},  # default params for Router.chat.completion.create
	fallbacks: List = [],
	allowed_fails: Optional[int] = None, # Number of times a deployment can failbefore being added to cooldown
	cooldown_time: float = 1,  # (seconds) time to cooldown a deployment after failure
	context_window_fallbacks: List = [],
	model_group_alias: Optional[dict] = {},
	retry_after: int = 0,  # (min) time to wait before retrying a failed request
	routing_strategy: Literal[
		"simple-shuffle",
		"least-busy",
		"usage-based-routing",
		"latency-based-routing",
	] = "simple-shuffle",

	## DEBUGGING ##
	set_verbose: bool = False,	# set this to True for seeing logs
    debug_level: Literal["DEBUG", "INFO"] = "INFO", # set this to "DEBUG" for detailed debugging
):
```

## Debugging Router
### Basic Debugging
Set `Router(set_verbose=True)`

```python
from litellm import Router

router = Router(
    model_list=model_list,
    set_verbose=True
)
```

### Detailed Debugging
Set `Router(set_verbose=True,debug_level="DEBUG")`

```python
from litellm import Router

router = Router(
    model_list=model_list,
    set_verbose=True,
    debug_level="DEBUG"  # defaults to INFO
)
```

### Very Detailed Debugging
Set `litellm.set_verbose=True` and `Router(set_verbose=True,debug_level="DEBUG")`

```python
from litellm import Router
import litellm

litellm.set_verbose = True

router = Router(
    model_list=model_list,
    set_verbose=True,
    debug_level="DEBUG"  # defaults to INFO
)
```
