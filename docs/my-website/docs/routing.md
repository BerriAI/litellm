# Reliability - Fallbacks, Azure Deployments, etc.

# Reliability

LiteLLM helps prevent failed requests in 3 ways: 
- Retries
- Fallbacks: Context Window + General
- RateLimitManager

## Helper utils 
LiteLLM supports the following functions for reliability:
* `litellm.longer_context_model_fallback_dict`: Dictionary which has a mapping for those models which have larger equivalents  
* `num_retries`: use tenacity retries
* `completion()` with fallbacks: switch between models/keys/api bases in case of errors. 
* `router()`: An abstraction on top of completion + embeddings to route the request to a deployment with capacity (available tpm/rpm).

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

### Context Window Fallbacks
```python 
from litellm import completion

fallback_dict = {"gpt-3.5-turbo": "gpt-3.5-turbo-16k"}
messages = [{"content": "how does a court case get to the Supreme Court?" * 500, "role": "user"}]

completion(model="gpt-3.5-turbo", messages=messages, context_window_fallback_dict=ctx_window_fallback_dict)
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

## Manage Multiple Deployments

Use this if you're trying to load-balance across multiple deployments (e.g. Azure/OpenAI). 

`Router` prevents failed requests, by picking the deployment which is below rate-limit and has the least amount of tokens used. 

In production, [Router connects to a Redis Cache](#redis-queue) to track usage across multiple deployments.

### Quick Start

```python
from litellm import Router

model_list = [{ # list of model deployments 
	"model_name": "gpt-3.5-turbo", # openai model name 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-v-2", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE")
	},
	"tpm": 240000,
	"rpm": 1800
}, {
    "model_name": "gpt-3.5-turbo", # openai model name 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-functioncalling", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE")
	},
	"tpm": 240000,
	"rpm": 1800
}, {
    "model_name": "gpt-3.5-turbo", # openai model name 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "gpt-3.5-turbo", 
		"api_key": os.getenv("OPENAI_API_KEY"),
	},
	"tpm": 1000000,
	"rpm": 9000
}]

router = Router(model_list=model_list)

# openai.ChatCompletion.create replacement
response = router.completion(model="gpt-3.5-turbo", 
				messages=[{"role": "user", "content": "Hey, how's it going?"}]

print(response)
```

### Redis Queue 

In production, we use Redis to track usage across multiple Azure deployments.

```python
router = Router(model_list=model_list, 
                redis_host=os.getenv("REDIS_HOST"), 
                redis_password=os.getenv("REDIS_PASSWORD"), 
                redis_port=os.getenv("REDIS_PORT"))

print(response)
```

### Deploy Router 

1. Clone repo
```shell
 git clone https://github.com/BerriAI/litellm
```

2. Create + Modify router_config.yaml (save your azure/openai/etc. deployment info)

```shell
cp ./router_config_template.yaml ./router_config.yaml
```

3. Build + Run docker image 

```shell
docker build -t litellm-proxy . --build-arg CONFIG_FILE=./router_config.yaml 
```

```shell
docker run --name litellm-proxy -e PORT=8000 -p 8000:8000 litellm-proxy
```

### Test 

```curl
curl 'http://0.0.0.0:8000/router/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hey"}]
}'
```


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
```