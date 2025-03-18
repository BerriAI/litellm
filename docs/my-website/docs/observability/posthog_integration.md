# 🦔 PostHog - Tracking LLM Usage Analytics

## What is PostHog?

PostHog is an open-source product analytics platform that helps you track and analyze how users interact with your product. For LLM applications, PostHog provides specialized AI features to track model usage, performance, and user interactions with your AI features.

## Usage with LiteLLM Proxy (LLM Gateway)

👉 [**Follow this link to start sending logs to PostHog with LiteLLM Proxy server**](../proxy/logging)

## Usage with LiteLLM Python SDK

### Pre-Requisites
Ensure you have run `pip install posthog` for this integration
```shell
pip install posthog litellm
```

### Quick Start
Use just 2 lines of code, to instantly log your responses **across all providers** with PostHog:

```python
litellm.success_callback = ["posthog"]
litellm.failure_callback = ["posthog"] # logs errors to posthog
```
```python
# pip install posthog
import litellm
import os

# from PostHog
os.environ["POSTHOG_API_KEY"] = ""
# Optional, defaults to https://us.i.posthog.com
os.environ["POSTHOG_API_URL"] = "" # optional

# LLM API Keys
os.environ['OPENAI_API_KEY']=""

# set posthog as a callback, litellm will send the data to posthog
litellm.success_callback = ["posthog"] 
 
# openai call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ],
  metadata = {
    "user_id": "user-123", # set posthog user ID
  }
)
```

### Advanced
#### Set User ID and Custom Metadata

Pass `user_id` in `metadata` to associate events with specific users in PostHog

```python
import litellm
from litellm import completion
import os

# from PostHog
os.environ["POSTHOG_API_KEY"] = "phc_..."
os.environ["POSTHOG_API_URL"] = "https://us.i.posthog.com" # optional

# OpenAI API key
os.environ['OPENAI_API_KEY']="sk-..."

# set posthog as a callback, litellm will send the data to posthog
litellm.success_callback = ["posthog"] 
 
# openai call
response = completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ],
  metadata = {
    "user_id": "user-123", # set posthog user ID
    # custom metadata fields
    "project": "litellm-proxy",
    "environment": "production"
  }
)
 
print(response)
```

### What Gets Logged to PostHog

LiteLLM logs the following information to PostHog:

#### For Completions (Chat or Text)
- `$ai_trace_id`: Unique identifier for the request
- `$ai_provider`: The provider of the model (e.g., "openai", "anthropic")
- `$ai_model`: The model name used
- `$ai_input`: The input messages or prompt
- `$ai_output_choices`: The model's response(s)
- `$ai_latency`: Request duration in seconds
- `$ai_input_tokens`: Number of input tokens
- `$ai_output_tokens`: Number of output tokens
- `$ai_tools`: Any tools provided to the model (if applicable)
- `$ai_model_parameters`: Model parameters like temperature, max_tokens, etc.
- Any custom metadata you provide

#### For Embeddings
- `$ai_trace_id`: Unique identifier for the request
- `$ai_provider`: The provider of the model
- `$ai_model`: The model name used
- `$ai_input`: The input text(s)
- `$ai_latency`: Request duration in seconds
- `$ai_input_tokens`: Number of input tokens
- `$ai_embedding_dimensions`: Dimensions of the generated embeddings
- `$ai_model_parameters`: Model parameters
- Any custom metadata you provide

#### For Errors
- `$ai_trace_id`: Unique identifier for the request
- `$ai_provider`: The provider of the model
- `$ai_model`: The model name used
- `$ai_input`: The input that caused the error
- `$ai_latency`: Request duration before error
- `$ai_http_status`: HTTP status code (if available)
- `$ai_is_error`: Set to true
- `$ai_error`: Error message
- Any custom metadata you provide

### Use LangChain ChatLiteLLM + PostHog
Pass `user_id` in model_kwargs
```python
import os
from langchain.chat_models import ChatLiteLLM
from langchain.schema import HumanMessage
import litellm

# from PostHog
os.environ["POSTHOG_API_KEY"] = "phc_..."
os.environ["POSTHOG_API_URL"] = "https://us.i.posthog.com" # optional

os.environ['OPENAI_API_KEY']="sk-..."

# set posthog as a callback, litellm will send the data to posthog
litellm.success_callback = ["posthog"] 

chat = ChatLiteLLM(
  model="gpt-3.5-turbo",
  model_kwargs={
      "metadata": {
        "user_id": "user-123", # set posthog user ID
        "environment": "production"
      }
    }
  )
messages = [
    HumanMessage(
        content="what model are you"
    )
]
chat(messages)
```

### Disable Logging - Specific Calls

To disable logging for specific calls use the `no-log` flag. 

`completion(messages = ..., model = ...,  **{"no-log": True})`

## Troubleshooting & Errors
### Data not getting logged to PostHog?
- Ensure you're on the latest version of posthog `pip install posthog -U`
- Verify your PostHog API key is correctly set in the environment variables
- Check that the PostHog host URL is correct if you're using a self-hosted or EU instance
- Enable verbose logging in LiteLLM to see more details: `litellm.set_verbose=True`

## Support

This integration is maintained by the Posthog team. For support, please reach out to us at [peter@posthog.com](mailto:peter@posthog.com).
