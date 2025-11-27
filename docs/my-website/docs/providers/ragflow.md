import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# RAGFlow

## Overview

| Property | Details |
|-------|-------|
| Description | RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding. |
| Provider Route on LiteLLM | `ragflow/` |
| Link to Provider Doc | [RAGFlow ↗](https://github.com/infiniflow/ragflow) |
| Base URL | `http://localhost:9380/v1` (default for self-hosted) |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

https://github.com/infiniflow/ragflow

**We support ALL RAGFlow agents and datasets, just set `ragflow/` as a prefix when sending completion requests**

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["RAGFLOW_API_KEY"] = ""  # your RAGFlow API key
os.environ["RAGFLOW_API_BASE"] = "http://localhost:9380/v1"  # optional, defaults to localhost
```

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="RAGFlow Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["RAGFLOW_API_KEY"] = ""  # your RAGFlow API key
os.environ["RAGFLOW_API_BASE"] = "http://localhost:9380/v1"  # optional

messages = [{"content": "How does the deep doc understanding work?", "role": "user"}]

# RAGFlow call - use your agent ID or dataset ID as the model
response = completion(
    model="ragflow/<agent_id>", 
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="RAGFlow Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["RAGFLOW_API_KEY"] = ""  # your RAGFlow API key
os.environ["RAGFLOW_API_BASE"] = "http://localhost:9380/v1"  # optional

messages = [{"content": "How does the deep doc understanding work?", "role": "user"}]

# RAGFlow call with streaming
response = completion(
    model="ragflow/<agent_id>", 
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy

Add the following to your LiteLLM Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: my-rag-agent
    litellm_params:
      model: ragflow/<agent_id>  # Replace with your RAGFlow agent ID
      api_key: os.environ/RAGFLOW_API_KEY
      api_base: http://localhost:9380/v1  # Your RAGFlow instance URL

  - model_name: my-knowledge-base
    litellm_params:
      model: ragflow/<dataset_id>  # Replace with your RAGFlow dataset ID
      api_key: os.environ/RAGFLOW_API_KEY
      api_base: http://localhost:9380/v1
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="RAGFlow via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="my-rag-agent",
    messages=[{"role": "user", "content": "How does deep document understanding work?"}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="RAGFlow via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="my-rag-agent",
    messages=[{"role": "user", "content": "How does deep document understanding work?"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="RAGFlow via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/my-rag-agent",
    messages=[{"role": "user", "content": "How does deep document understanding work?"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="RAGFlow via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/my-rag-agent",
    messages=[{"role": "user", "content": "How does deep document understanding work?"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key",
    stream=True
)

for chunk in response:
    if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="curl" label="cURL">

```bash showLineNumbers title="RAGFlow via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "my-rag-agent",
    "messages": [{"role": "user", "content": "How does deep document understanding work?"}]
  }'
```

```bash showLineNumbers title="RAGFlow via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "my-rag-agent",
    "messages": [{"role": "user", "content": "How does deep document understanding work?"}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).

## About RAGFlow

RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine that:

- Provides deep document understanding through advanced parsing
- Supports multiple document formats (PDF, DOCX, PPT, Excel, etc.)
- Uses Elasticsearch or Infinity for vector storage
- Offers an OpenAI-compatible chat completion API
- Allows creation of conversational agents backed by knowledge bases

### Key Features

- **Deep Document Understanding**: Advanced document parsing and chunking strategies
- **Visual RAG**: Support for vision-language models and visual data processing
- **Flexible Architecture**: Modular design with support for various embedding models and LLMs
- **OpenAI Compatibility**: Fully compatible with OpenAI's chat completions API

### Setting Up RAGFlow

1. **Self-hosted Installation**: Follow the [RAGFlow installation guide](https://github.com/infiniflow/ragflow#-quick-start)
2. **Create a Knowledge Base**: Upload your documents and configure parsing settings
3. **Create an Agent**: Set up a conversational agent connected to your knowledge base
4. **Get API Key**: Generate an API key from the RAGFlow dashboard
5. **Get Agent/Dataset ID**: Find your agent or dataset ID in the RAGFlow UI

### Using with LiteLLM

RAGFlow exposes an OpenAI-compatible API at `/v1/chat/completions`. LiteLLM provides native support for RAGFlow, allowing you to:

- Use RAGFlow agents as if they were standard LLM models
- Switch between RAGFlow and other providers by changing the model prefix
- Leverage LiteLLM's proxy features (rate limiting, budgets, auth) with RAGFlow
- Track RAGFlow usage and costs through LiteLLM's observability integrations

## RAG-Specific Parameters

While RAGFlow is OpenAI-compatible, you can pass RAG-specific parameters through the standard API:

```python showLineNumbers title="RAGFlow with custom parameters"
response = completion(
    model="ragflow/<agent_id>",
    messages=[{"role": "user", "content": "Search for information about X"}],
    temperature=0.7,
    max_tokens=1000
)
```

## Troubleshooting

### Connection Issues

If you encounter connection errors:

1. Verify RAGFlow is running: `curl http://localhost:9380/health`
2. Check that the API base URL is correct
3. Ensure your API key is valid
4. Verify network connectivity to your RAGFlow instance

### API Key Issues

If you get authentication errors:

1. Generate a new API key in the RAGFlow dashboard
2. Ensure the API key is set correctly in environment variables
3. Check that the API key has appropriate permissions

## Additional Resources

- [RAGFlow GitHub Repository](https://github.com/infiniflow/ragflow)
- [RAGFlow Documentation](https://github.com/infiniflow/ragflow#-documentation)
- [RAGFlow API Reference](https://github.com/infiniflow/ragflow/blob/main/docs/references/api_reference.md)
