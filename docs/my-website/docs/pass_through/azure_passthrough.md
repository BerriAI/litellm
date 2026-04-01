# Azure Passthrough

Pass-through endpoints for `/azure`

## Overview

| Feature | Supported | Notes |
|-------|-------|-------|
| Cost Tracking | ❌ | Not supported |
| Logging | ✅ | Works across all integrations |
| Streaming | ✅ | Fully supported |

### When to use this?

- For most use cases, you should use the [native LiteLLM Azure OpenAI Integration](../providers/azure/azure) (`/chat/completions`, `/embeddings`, `/completions`, `/images`, etc.)
- Use this passthrough to call newer or less common Azure OpenAI endpoints that LiteLLM doesn't fully support yet, such as `/assistants`, `/threads`, `/vector_stores`

Simply replace your Azure endpoint (e.g. `https://<your-resource-name>.openai.azure.com`) with `LITELLM_PROXY_BASE_URL/azure`

## Usage Examples

### Assistants API

#### Create Azure OpenAI Client

Make sure you do the following:
- Point `azure_endpoint` to your `LITELLM_PROXY_BASE_URL/azure`
- Use your `LITELLM_API_KEY` as the `api_key`

```python
import openai

client = openai.AzureOpenAI(
    azure_endpoint="http://0.0.0.0:4000/azure",  # <your-proxy-url>/azure
    api_key="sk-anything",  # <your-proxy-api-key>
    api_version="2024-05-01-preview"  # required Azure API version
)
```

#### Create an Assistant

```python
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a math tutor. Help solve equations.",
    model="gpt-4o",
)
```

#### Create a Thread
```python
thread = client.beta.threads.create()
```

#### Add a Message to the Thread
```python
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Solve 3x + 11 = 14",
)
```

#### Run the Assistant
```python
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
messages = client.beta.threads.messages.list(
    thread_id=thread.id
)
```

#### Delete the Assistant

```python
client.beta.assistants.delete(assistant.id)
```