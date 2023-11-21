import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Router - Load Balancing, Queing

- Load-balance across multiple deployments (e.g. Azure/OpenAI): Pick the deployment which is below rate-limit and has the least amount of tokens used. 
- Queuing Requests to ensure requests don't fail

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

## Advanced
### Routing Strategies - Shuffle, Rate Limit Aware

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

### Caching + Request Timeouts 

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

### Retry failed requests

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

### Default litellm.completion/embedding params

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


### Deploy Router 

If you want a server to just route requests to different LLM APIs, use our [OpenAI Proxy Server](./simple_proxy.md#multiple-instances-of-1-model)

## Hosted Router + Request Queing api.litellm.ai
Queue your LLM API requests to ensure you're under your rate limits
- Step 1: Create a `/queue/reques` request
- Step 2: Poll your request, to check if it's completed

### Step 1: Queue a `/chat/completion` request

```python
import requests
# args to litellm.completion()
data = {
    'model': 'gpt-3.5-turbo',
    'messages': [
        {'role': 'system', 'content': f'You are a helpful assistant. What llm are you?'},
    ],
}
response = requests.post("http://0.0.0.0:8000/queue/request", json=data)
response = response.json()
polling_url = response["url"]
```

### Step 2: Poll your `/chat/completion` request
```python
 while True:
    try:
        polling_url = f"http://0.0.0.0:8000{polling_url}"
        polling_response = requests.get(polling_url)
        polling_response = polling_response.json()
        print("\n RESPONSE FROM POLLING JOB", polling_response)
        status = polling_response["status"]
        if status == "finished":
            llm_response = polling_response["result"]
            print(llm_response)
```


<!-- ## litellm.completion() 

If you're calling litellm.completion(), here's the different reliability options you can enable. 

## Retry failed requests

Call it in completion like this `completion(..num_retries=2)`.


Here's a quick look at how you can use it: 

```python 
from litellm import completion

user_message = "Hello, whats the weather in San Francisco??"
messages = [{"content": user_message, "role": "user"}]

# normal call 
response = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            num_retries=2
        )
```

## Fallbacks 

## Helper utils 
LiteLLM supports the following functions for reliability:
* `litellm.longer_context_model_fallback_dict`: Dictionary which has a mapping for those models which have larger equivalents  
* `num_retries`: use tenacity retries
* `completion()` with fallbacks: switch between models/keys/api bases in case of errors. 


### Context Window Fallbacks
```python 
from litellm import completion

fallback_dict = {"gpt-3.5-turbo": "gpt-3.5-turbo-16k"}
messages = [{"content": "how does a court case get to the Supreme Court?" * 500, "role": "user"}]

completion(model="gpt-3.5-turbo", messages=messages, context_window_fallback_dict=fallback_dict)
```

### Fallbacks - Switch Models/API Keys/API Bases

LLM APIs can be unstable, completion() with fallbacks ensures you'll always get a response from your calls

#### Usage 
To use fallback models with `completion()`, specify a list of models in the `fallbacks` parameter. 

The `fallbacks` list should include the primary model you want to use, followed by additional models that can be used as backups in case the primary model fails to provide a response.

#### switch models 
```python
response = completion(model="bad-model", messages=messages, 
    fallbacks=["gpt-3.5-turbo" "command-nightly"])
```

#### switch api keys/bases (E.g. azure deployment)
Switch between different keys for the same azure deployment, or use another deployment as well. 

```python
api_key="bad-key"
response = completion(model="azure/gpt-4", messages=messages, api_key=api_key,
    fallbacks=[{"api_key": "good-key-1"}, {"api_key": "good-key-2", "api_base": "good-api-base-2"}])
```

[Check out this section for implementation details](#fallbacks-1)

## Implementation Details 

### Fallbacks
#### Output from calls
```
Completion with 'bad-model': got exception Unable to map your input to a model. Check your input - {'model': 'bad-model'



completion call gpt-3.5-turbo
{
  "id": "chatcmpl-7qTmVRuO3m3gIBg4aTmAumV1TmQhB",
  "object": "chat.completion",
  "created": 1692741891,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I apologize, but as an AI, I do not have the capability to provide real-time weather updates. However, you can easily check the current weather in San Francisco by using a search engine or checking a weather website or app."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 16,
    "completion_tokens": 46,
    "total_tokens": 62
  }
}

```

#### How does fallbacks work

When you pass `fallbacks` to `completion`, it makes the first `completion` call using the primary model specified as `model` in `completion(model=model)`. If the primary model fails or encounters an error, it automatically tries the `fallbacks` models in the specified order. This ensures a response even if the primary model is unavailable.


#### Key components of Model Fallbacks implementation:
* Looping through `fallbacks`
* Cool-Downs for rate-limited models

#### Looping through `fallbacks`
Allow `45seconds` for each request. In the 45s this function tries calling the primary model set as `model`. If model fails it loops through the backup `fallbacks` models and attempts to get a response in the allocated `45s` time set here: 
```python
while response == None and time.time() - start_time < 45:
        for model in fallbacks:
```

#### Cool-Downs for rate-limited models
If a model API call leads to an error - allow it to cooldown for `60s`
```python
except Exception as e:
  print(f"got exception {e} for model {model}")
  rate_limited_models.add(model)
  model_expiration_times[model] = (
      time.time() + 60
  )  # cool down this selected model
  pass
```

Before making an LLM API call we check if the selected model is in `rate_limited_models`, if so skip making the API call
```python
if (
  model in rate_limited_models
):  # check if model is currently cooling down
  if (
      model_expiration_times.get(model)
      and time.time() >= model_expiration_times[model]
  ):
      rate_limited_models.remove(
          model
      )  # check if it's been 60s of cool down and remove model
  else:
      continue  # skip model

```

#### Full code of completion with fallbacks()
```python

    response = None
    rate_limited_models = set()
    model_expiration_times = {}
    start_time = time.time()
    fallbacks = [kwargs["model"]] + kwargs["fallbacks"]
    del kwargs["fallbacks"]  # remove fallbacks so it's not recursive

    while response == None and time.time() - start_time < 45:
        for model in fallbacks:
            # loop thru all models
            try:
                if (
                    model in rate_limited_models
                ):  # check if model is currently cooling down
                    if (
                        model_expiration_times.get(model)
                        and time.time() >= model_expiration_times[model]
                    ):
                        rate_limited_models.remove(
                            model
                        )  # check if it's been 60s of cool down and remove model
                    else:
                        continue  # skip model

                # delete model from kwargs if it exists
                if kwargs.get("model"):
                    del kwargs["model"]

                print("making completion call", model)
                response = litellm.completion(**kwargs, model=model)

                if response != None:
                    return response

            except Exception as e:
                print(f"got exception {e} for model {model}")
                rate_limited_models.add(model)
                model_expiration_times[model] = (
                    time.time() + 60
                )  # cool down this selected model
                pass
    return response
``` -->