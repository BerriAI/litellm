# Reliability
## Helper utils 
LiteLLM supports the following functions for reliability:
* `litellm.longer_context_model_fallback_dict`: Dictionary which has a mapping for those models which have larger equivalents  
* `completion_with_retries`: use tenacity retries
* `completion()` with fallbacks: switch between models/keys/api bases in case of errors. 

## Context Window Errors 

```python 
from litellm import longer_context_model_fallback_dict, ContextWindowExceededError

sample_text = "how does a court case get to the Supreme Court?" * 1000
messages = [{"content": user_message, "role": "user"}]
model = "gpt-3.5-turbo"
try: 
    # try the original model
    response = completion(model=model, messages=messages) 
# catch the context window error
except ContextWindowExceededError as e:
    if model in longer_context_model_fallback_dict: 
        # switch to the equivalent larger model -> gpt.3.5-turbo-16k 
        new_model = longer_context_model_fallback_dict[model]
        response = completion(new_model, messages)

print(response)
```


## Retry failed requests

You can use this as a drop-in replacement for the `completion()` function to use tenacity retries - by default we retry the call 3 times. 

Here's a quick look at how you can use it: 

```python 
from litellm import completion_with_retries

user_message = "Hello, whats the weather in San Francisco??"
messages = [{"content": user_message, "role": "user"}]

# normal call 
def test_completion_custom_provider_model_name():
    try:
        response = completion_with_retries(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        printf"Error occurred: {e}")
```

## Switch Models/API Keys/API Bases

LLM APIs can be unstable, completion() with fallbacks ensures you'll always get a response from your calls

### Usage 
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


### Key components of Model Fallbacks implementation:
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
