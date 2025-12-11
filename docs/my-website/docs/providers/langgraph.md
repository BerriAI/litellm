import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# LangGraph

Call LangGraph agents through LiteLLM using the OpenAI chat completions format.

| Property | Details |
|----------|---------|
| Description | LangGraph is a framework for building stateful, multi-actor applications with LLMs. LiteLLM supports calling LangGraph agents via their streaming and non-streaming endpoints. |
| Provider Route on LiteLLM | `langgraph/{agent_id}` |
| Provider Doc | [LangGraph Platform ↗](https://langchain-ai.github.io/langgraph/cloud/quick_start/) |

**Prerequisites:** You need a running LangGraph server. See [Setting Up a Local LangGraph Server](#setting-up-a-local-langgraph-server) below.

## Quick Start

### Model Format

```shell showLineNumbers title="Model Format"
langgraph/{agent_id}
```

**Example:**
- `langgraph/agent` - calls the default agent

### LiteLLM Python SDK

```python showLineNumbers title="Basic LangGraph Completion"
import litellm

response = litellm.completion(
    model="langgraph/agent",
    messages=[
        {"role": "user", "content": "What is 25 * 4?"}
    ],
    api_base="http://localhost:2024",
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Streaming LangGraph Response"
import litellm

response = litellm.completion(
    model="langgraph/agent",
    messages=[
        {"role": "user", "content": "What is the weather in Tokyo?"}
    ],
    api_base="http://localhost:2024",
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### LiteLLM Proxy

#### 1. Configure your model in config.yaml

<Tabs>
<TabItem value="config-yaml" label="config.yaml">

```yaml showLineNumbers title="LiteLLM Proxy Configuration"
model_list:
  - model_name: langgraph-agent
    litellm_params:
      model: langgraph/agent
      api_base: http://localhost:2024
```

</TabItem>
</Tabs>

#### 2. Start the LiteLLM Proxy

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml
```

#### 3. Make requests to your LangGraph agent

<Tabs>
<TabItem value="curl" label="Curl">

```bash showLineNumbers title="Basic Request"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "langgraph-agent",
    "messages": [
      {"role": "user", "content": "What is 25 * 4?"}
    ]
  }'
```

```bash showLineNumbers title="Streaming Request"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "langgraph-agent",
    "messages": [
      {"role": "user", "content": "What is the weather in Tokyo?"}
    ],
    "stream": true
  }'
```

</TabItem>

<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python showLineNumbers title="Using OpenAI SDK with LiteLLM Proxy"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-litellm-api-key"
)

response = client.chat.completions.create(
    model="langgraph-agent",
    messages=[
        {"role": "user", "content": "What is 25 * 4?"}
    ]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Streaming with OpenAI SDK"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-litellm-api-key"
)

stream = client.chat.completions.create(
    model="langgraph-agent",
    messages=[
        {"role": "user", "content": "What is the weather in Tokyo?"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>
</Tabs>

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LANGGRAPH_API_BASE` | Base URL of your LangGraph server (default: `http://localhost:2024`) |
| `LANGGRAPH_API_KEY` | Optional API key for authentication |

## Supported Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | The agent ID in format `langgraph/{agent_id}` |
| `messages` | array | Chat messages in OpenAI format |
| `stream` | boolean | Enable streaming responses |
| `api_base` | string | LangGraph server URL |
| `api_key` | string | Optional API key |


## Setting Up a Local LangGraph Server

Before using LiteLLM with LangGraph, you need a running LangGraph server.

### Prerequisites

- Python 3.11+
- An LLM API key (OpenAI or Google Gemini)

### 1. Install the LangGraph CLI

```bash
pip install "langgraph-cli[inmem]"
```

### 2. Create a new LangGraph project

```bash
langgraph new my-agent --template new-langgraph-project-python
cd my-agent
```

### 3. Install dependencies

```bash
pip install -e .
```

### 4. Set your API key

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 5. Start the server

```bash
langgraph dev
```

The server will start at `http://localhost:2024`.

### Verify the server is running

```bash
curl -s --request POST \
  --url "http://localhost:2024/runs/wait" \
  --header 'Content-Type: application/json' \
  --data '{
    "assistant_id": "agent",
    "input": {
      "messages": [{"role": "human", "content": "Hello!"}]
    }
  }'
```



## Further Reading

- [LangGraph Platform Documentation](https://langchain-ai.github.io/langgraph/cloud/quick_start/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)

