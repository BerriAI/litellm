# Custom API Server (OpenAI Format)

LiteLLM allows you to call your custom endpoint in the OpenAI ChatCompletion format

## API KEYS
No api keys required

## Set up your Custom API Server
Your server should have the following Endpoints:

Here's an example OpenAI proxy server with routes: https://replit.com/@BerriAI/openai-proxy#main.py

### Required Endpoints
- POST `/chat/completions` - chat completions endpoint 

### Optional Endpoints
- POST `/completions` - completions endpoint 
- Get `/models` - available models on server
- POST `/embeddings` - creates an embedding vector representing the input text.


## Example Usage

### Call `/chat/completions`
In order to use your custom OpenAI Chat Completion proxy with LiteLLM, ensure you set

* `api_base` to your proxy url, example "https://openai-proxy.berriai.repl.co"
* `custom_llm_provider` to `openai` this ensures litellm uses the `openai.ChatCompletion` to your api_base

```python
import os
from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "anything" #key is not used for proxy

messages = [{ "content": "Hello, how are you?","role": "user"}]

response = completion(
    model="command-nightly", 
    messages=[{ "content": "Hello, how are you?","role": "user"}],
    api_base="https://openai-proxy.berriai.repl.co",
    custom_llm_provider="openai" # litellm will use the openai.ChatCompletion to make the request

)
print(response)
```

#### Response
```json
{
    "object":
    "chat.completion",
    "choices": [{
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content":
        "The sky, a canvas of blue,\nA work of art, pure and true,\nA",
        "role": "assistant"
      }
    }],
    "id":
    "chatcmpl-7fbd6077-de10-4cb4-a8a4-3ef11a98b7c8",
    "created":
    1699290237.408061,
    "model":
    "togethercomputer/llama-2-70b-chat",
    "usage": {
      "completion_tokens": 18,
      "prompt_tokens": 14,
      "total_tokens": 32
    }
  }
```


### Call `/completions`
In order to use your custom OpenAI Completion proxy with LiteLLM, ensure you set

* `api_base` to your proxy url, example "https://openai-proxy.berriai.repl.co"
* `custom_llm_provider` to `text-completion-openai` this ensures litellm uses the `openai.Completion` to your api_base

```python
import os
from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "anything" #key is not used for proxy

messages = [{ "content": "Hello, how are you?","role": "user"}]

response = completion(
    model="command-nightly", 
    messages=[{ "content": "Hello, how are you?","role": "user"}],
    api_base="https://openai-proxy.berriai.repl.co",
    custom_llm_provider="text-completion-openai" # litellm will use the openai.Completion to make the request

)
print(response)
```

#### Response 
```json
{
    "warning":
    "This model version is deprecated. Migrate before January 4, 2024 to avoid disruption of service. Learn more https://platform.openai.com/docs/deprecations",
    "id":
    "cmpl-8HxHqF5dymQdALmLplS0dWKZVFe3r",
    "object":
    "text_completion",
    "created":
    1699290166,
    "model":
    "text-davinci-003",
    "choices": [{
      "text":
      "\n\nThe weather in San Francisco varies depending on what time of year and time",
      "index": 0,
      "logprobs": None,
      "finish_reason": "length"
    }],
    "usage": {
      "prompt_tokens": 7,
      "completion_tokens": 16,
      "total_tokens": 23
    }
  }
```