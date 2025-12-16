import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Vertex AI Agent Engine

Call Vertex AI Agent Engine (Reasoning Engines) in the OpenAI Request/Response format.

| Property | Details |
|----------|---------|
| Description | Vertex AI Agent Engine provides hosted agent runtimes that can execute agentic workflows with foundation models, tools, and custom logic. |
| Provider Route on LiteLLM | `vertex_ai/agent_engine/{RESOURCE_NAME}` |
| Supported Endpoints | `/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/a2a/message/send` |
| Provider Doc | [Vertex AI Agent Engine ↗](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/overview) |

## Authentication

Vertex AI Agent Engine requires **Google Cloud authentication**. You can authenticate using:

### Option 1: Application Default Credentials (Recommended)

```bash
gcloud auth application-default login
```

### Option 2: Service Account Key

Set the path to your service account JSON key:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### Option 3: Pass Credentials Directly

Pass credentials via `vertex_credentials` parameter:

```python
response = litellm.completion(
    model="vertex_ai/agent_engine/projects/123/locations/us-central1/reasoningEngines/456",
    messages=[{"role": "user", "content": "Hello!"}],
    vertex_credentials=json.dumps(service_account_dict),
)
```

## Quick Start

### Model Format to LiteLLM

To call a Vertex AI Agent Engine through LiteLLM, use the following model format.

Here the `model=vertex_ai/agent_engine/` tells LiteLLM to call the Vertex AI Reasoning Engine API.

```shell showLineNumbers title="Model Format to LiteLLM"
vertex_ai/agent_engine/{RESOURCE_NAME}
```

**Example:**
- `vertex_ai/agent_engine/projects/1060139831167/locations/us-central1/reasoningEngines/8263861224643493888`

You can find the Resource Name in your Google Cloud Console under Vertex AI > Agent Engine.

### LiteLLM Python SDK

```python showLineNumbers title="Basic Agent Completion"
import litellm

# Make a completion request to your Vertex AI Agent Engine
# Uses Application Default Credentials for auth
response = litellm.completion(
    model="vertex_ai/agent_engine/projects/1060139831167/locations/us-central1/reasoningEngines/8263861224643493888",
    messages=[
        {
            "role": "user", 
            "content": "Explain machine learning in simple terms"
        }
    ],
)

print(response.choices[0].message.content)
print(f"Usage: {response.usage}")
```

```python showLineNumbers title="Streaming Agent Responses"
import litellm

# Stream responses from your Vertex AI Agent Engine
response = await litellm.acompletion(
    model="vertex_ai/agent_engine/projects/1060139831167/locations/us-central1/reasoningEngines/8263861224643493888",
    messages=[
        {
            "role": "user",
            "content": "What are the key principles of software architecture?"
        }
    ],
    stream=True,
)

async for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### LiteLLM Proxy

#### 1. Configure your model in config.yaml

<Tabs>
<TabItem value="config-yaml" label="config.yaml">

```yaml showLineNumbers title="LiteLLM Proxy Configuration"
model_list:
  - model_name: vertex-agent-1
    litellm_params:
      model: vertex_ai/agent_engine/projects/1060139831167/locations/us-central1/reasoningEngines/8263861224643493888
      vertex_project: your-project-id
      vertex_location: us-central1
      # Uses Application Default Credentials by default
      # Or specify credentials path:
      # vertex_credentials: /path/to/service-account.json

  - model_name: vertex-agent-assistant
    litellm_params:
      model: vertex_ai/agent_engine/projects/1060139831167/locations/us-central1/reasoningEngines/9876543210123456789
      vertex_project: your-project-id
      vertex_location: us-central1
```

</TabItem>
</Tabs>

#### 2. Start the LiteLLM Proxy

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml
```

#### 3. Make requests to your Vertex AI Agent Engine

<Tabs>
<TabItem value="curl" label="Curl">

```bash showLineNumbers title="Basic Agent Request"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "vertex-agent-1",
    "messages": [
      {
        "role": "user", 
        "content": "Summarize the main benefits of cloud computing"
      }
    ]
  }'
```

```bash showLineNumbers title="Streaming Agent Request"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "vertex-agent-assistant",
    "messages": [
      {
        "role": "user",
        "content": "What is 25 * 4?"
      }
    ],
    "stream": true
  }'
```

</TabItem>

<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python showLineNumbers title="Using OpenAI SDK with LiteLLM Proxy"
from openai import OpenAI

# Initialize client with your LiteLLM proxy URL
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-litellm-api-key"
)

# Make a completion request to your Vertex AI Agent Engine
response = client.chat.completions.create(
    model="vertex-agent-1",
    messages=[
      {
        "role": "user",
        "content": "What are best practices for API design?"
      }
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

# Stream Agent responses
stream = client.chat.completions.create(
    model="vertex-agent-assistant",
    messages=[
      {
        "role": "user",
        "content": "Explain the Pythagorean theorem"
      }
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
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON key file |
| `VERTEXAI_PROJECT` | Google Cloud project ID |
| `VERTEXAI_LOCATION` | Google Cloud region (default: `us-central1`) |

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export VERTEXAI_PROJECT="your-project-id"
export VERTEXAI_LOCATION="us-central1"
```

## Session Management

Vertex AI Agent Engine supports session management for maintaining conversation context. You can pass a `session_id` to continue a conversation or a `user_id` for user-specific sessions.

```python showLineNumbers title="Using Sessions"
import litellm

# First message creates a new session
response1 = await litellm.acompletion(
    model="vertex_ai/agent_engine/projects/123/locations/us-central1/reasoningEngines/456",
    messages=[{"role": "user", "content": "My name is Alice"}],
    user="alice-user-123",  # Maps to user_id
)

# Continue the conversation with the same user
response2 = await litellm.acompletion(
    model="vertex_ai/agent_engine/projects/123/locations/us-central1/reasoningEngines/456",
    messages=[{"role": "user", "content": "What's my name?"}],
    user="alice-user-123",  # Same user_id maintains context
)

print(response2.choices[0].message.content)  # Should mention "Alice"
```

## Provider-specific Parameters

Vertex AI Agent Engine supports additional parameters that can be passed to customize the agent invocation.

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers title="Using Agent-specific parameters"
from litellm import completion

response = litellm.completion(
    model="vertex_ai/agent_engine/projects/123/locations/us-central1/reasoningEngines/456",
    messages=[
        {
            "role": "user",
            "content": "Analyze this data and provide insights",
        }
    ],
    user="user-123",  # Optional: User ID for session management
    session_id="session-abc",  # Optional: Continue existing session
)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

```yaml showLineNumbers title="LiteLLM Proxy Configuration with Parameters"
model_list:
  - model_name: vertex-agent-analyst
    litellm_params:
      model: vertex_ai/agent_engine/projects/123/locations/us-central1/reasoningEngines/456
      vertex_project: your-project-id
      vertex_location: us-central1
```

</TabItem>
</Tabs>

### Available Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `user` | string | User ID for session management (maps to `user_id` in Agent Engine) |
| `session_id` | string | Optional session ID to continue an existing conversation |
| `vertex_project` | string | Google Cloud project ID |
| `vertex_location` | string | Google Cloud region (default: `us-central1`) |
| `vertex_credentials` | string | Path to service account JSON or JSON string of credentials |

## Further Reading

- [Vertex AI Agent Engine Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/overview)
- [Create a Reasoning Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/create)
- [Vertex AI Provider](./vertex.md)

