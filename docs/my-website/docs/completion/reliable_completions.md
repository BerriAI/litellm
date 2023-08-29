# Reliability for Completions()

LiteLLM supports the following functions for reliability:
* `completion_with_retries`: use tenacity retries
* `completion()` with fallback models: set `fallback_models=['gpt-3.5-turbo', 'command-nightly', 'llama2`]. If primary model fails try fallback models

## Completion with Retries

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

## Completion with Fallbacks
LLM APIs can be unstable, completion() with fallbacks ensures you'll always get a response from your calls

### Usage 
To use fallback models with `completion()`, specify a list of models in the `fallbacks` parameter. 

The `fallbacks` list should include the primary model you want to use, followed by additional models that can be used as backups in case the primary model fails to provide a response.

```python
response = completion(model="bad-model", fallbacks=["gpt-3.5-turbo" "command-nightly"], messages=messages)
```

## How does `completion_with_fallbacks()` work

The `completion_with_fallbacks()` function attempts a completion call using the primary model specified as `model` in `completion()`. If the primary model fails or encounters an error, it automatically tries the fallback models in the specified order. This ensures a response even if the primary model is unavailable.

### Output from calls
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
