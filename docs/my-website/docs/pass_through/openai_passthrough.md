# OpenAI Passthrough

Pass-through endpoints for `/openai`

## Overview

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ✅ | When using router models |
| Logging | ✅ | Works across all integrations |
| Streaming | ✅ | Fully supported |
| Load Balancing | ✅ | When using router models |

### When to use this?

- For 90% of your use cases, you should use the [native LiteLLM OpenAI Integration](https://docs.litellm.ai/docs/providers/openai) (`/chat/completions`, `/embeddings`, `/completions`, `/images`, `/batches`, etc.)
- Use this passthrough to call less popular or newer OpenAI endpoints that LiteLLM doesn't fully support yet, such as `/assistants`, `/threads`, `/vector_stores`

Simply replace `https://api.openai.com` with `LITELLM_PROXY_BASE_URL/openai`

## Quick Start

### 1. Setup config.yaml

```yaml showLineNumbers
model_list:
  # Deployment 1
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY_1  # reads from environment
  
  # Deployment 2
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY_2  # reads from environment
```

Set OpenAI API keys in your environment:

```bash showLineNumbers
export OPENAI_API_KEY_1="your-first-api-key"
export OPENAI_API_KEY_2="your-second-api-key"
```

### 2. Start Proxy

```bash showLineNumbers
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Use OpenAI SDK

Replace `https://api.openai.com` with your proxy URL:

```python showLineNumbers
import openai

client = openai.OpenAI(
    base_url="http://0.0.0.0:4000/openai",
    api_key="sk-1234"  # your litellm proxy api key
)

# Use any OpenAI endpoint
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a math tutor.",
    model="gpt-4"  # uses router model from config.yaml
)
```

## Load Balancing

Define multiple deployments with the same `model_name` for automatic load balancing:

```yaml showLineNumbers
model_list:
  # Deployment 1
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY_1
  
  # Deployment 2
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY_2
```

The proxy automatically distributes requests across both API keys.

## Specifying Router Models

For endpoints that don't have a `model` field in the request body (e.g., `/files/delete`), specify the router model using:

### Option 1: Request Header

```python showLineNumbers
import openai

client = openai.OpenAI(
    base_url="http://0.0.0.0:4000/openai",
    api_key="sk-1234",
    default_headers={"X-LiteLLM-Target-Model": "gpt-4"}
)

# Delete a file using the specified router model
client.files.delete(file_id="file-abc123")
```

### Option 2: Request Body

```python showLineNumbers
import openai

client = openai.OpenAI(
    base_url="http://0.0.0.0:4000/openai",
    api_key="sk-1234"
)

# Upload file with target model
file = client.files.create(
    file=open("data.jsonl", "rb"),
    purpose="batch",
    extra_body={"target_model_names": "gpt-4"}
)

# Or with a list (first model will be used)
file = client.files.create(
    file=open("data.jsonl", "rb"),
    purpose="batch",
    extra_body={"target_model_names": ["gpt-4", "gpt-3.5-turbo"]}
)
```

## Usage Examples

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

