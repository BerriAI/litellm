# OpenAI Passthrough

Pass-through endpoints for `/openai`

## Overview

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ❌ | Not supported |
| Logging | ✅ | Works across all integrations |
| Streaming | ✅ | Fully supported |

### When to use this?

- For 90% of your use cases, you should use the [native LiteLLM OpenAI Integration](https://docs.litellm.ai/docs/providers/openai) (`/chat/completions`, `/embeddings`, `/completions`, `/images`, `/batches`, etc.)
- Use this passthrough to call less popular or newer OpenAI endpoints that LiteLLM doesn't fully support yet, such as `/assistants`, `/threads`, `/vector_stores`

Simply replace `https://api.openai.com` with `LITELLM_PROXY_BASE_URL/openai`

## Usage Examples

Requirements:
Set `OPENAI_API_KEY` in your environment variables.

### Assistants API

#### Create OpenAI Client

Make sure you do the following:
- Point `base_url` to your `LITELLM_PROXY_BASE_URL/openai`
- Use your `LITELLM_API_KEY` as the `api_key`

```python
import openai

client = openai.OpenAI(
    base_url="http://0.0.0.0:4000/openai",  # <your-proxy-url>/openai
    api_key="sk-anything"  # <your-proxy-api-key>
)
```

#### Create an Assistant

```python
# Create an assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a math tutor. Help solve equations.",
    model="gpt-4o",
)
```

#### Create a Thread
```python
# Create a thread
thread = client.beta.threads.create()
```

#### Add a Message to the Thread
```python
# Add a message
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Solve 3x + 11 = 14",
)
```

#### Run the Assistant
```python
# Create a run to get the assistant's response
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# Check run status
run_status = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
)
```

#### Retrieve Messages
```python
# List messages after the run completes
messages = client.beta.threads.messages.list(
    thread_id=thread.id
)
```

#### Delete the Assistant

```python
# Delete the assistant when done
client.beta.assistants.delete(assistant.id)
```

