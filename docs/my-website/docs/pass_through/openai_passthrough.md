# OpenAI Passthrough


Pass-through endpoints for `/openai`

## Overview

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ❌ | Not supported |
| Logging | ✅ | works across all integrations |
| Streaming | ✅ | |

### When to use this?

- For 90% of your use cases, you should use the [native Litellm OpenAI Integration](https://docs.litellm.ai/docs/providers/openai). (`/chat/completions`, `/embeddings`, `/completions`, `/images`, `/batches`, etc.)
- Use this to call less popular or newer OpenAI endpoints that LiteLLM doesn't support yet. eg. `/assistants`, `/threads`, `/vector_stores`


Just replace `https://api.openai.com` with `LITELLM_PROXY_BASE_URL/openai`

## Usage Examples


### Assistants

### Create OpenAI Client

Make sure you do the following:
- point `base_url` to your `LITELLM_PROXY_BASE_URL/openai`
- use your `LITELLM_API_KEY` as the `api_key`

```python
import openai

client = openai.OpenAI(
    base_url="http://0.0.0.0:4000/openai",  # <your-proxy-url>/openai
    api_key="sk-anything"  # <your-proxy-api-key>
)
```

### Create an Assistant

```python

# Create an assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a math tutor. Help solve equations.",
    model="gpt-4o",
)
```

### Create a Thread
```python
# Create a thread
thread = client.beta.threads.create()
```

### Add a message to the thread
```python
# Add a message
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Solve 3x + 11 = 14",
)
```



### Delete the assistant

```python
# Delete the assistant
client.beta.assistants.delete(assistant.id)
```

