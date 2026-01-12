import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /interactions

| Feature | Supported | Notes |
|---------|-----------|-------|
| Logging | ✅ | Works across all integrations |
| Streaming | ✅ | |
| Loadbalancing | ✅ | Between supported models |
| Supported LLM providers | **All LiteLLM supported CHAT COMPLETION providers** | `openai`, `anthropic`, `bedrock`, `vertex_ai`, `gemini`, `azure`, `azure_ai` etc. |

## **LiteLLM Python SDK Usage**

### Quick Start

```python showLineNumbers title="Create Interaction"
from litellm import create_interaction
import os

os.environ["GEMINI_API_KEY"] = "your-api-key"

response = create_interaction(
    model="gemini/gemini-2.5-flash",
    input="Tell me a short joke about programming."
)

print(response.outputs[-1].text)
```

### Async Usage

```python showLineNumbers title="Async Create Interaction"
from litellm import acreate_interaction
import os
import asyncio

os.environ["GEMINI_API_KEY"] = "your-api-key"

async def main():
    response = await acreate_interaction(
        model="gemini/gemini-2.5-flash",
        input="Tell me a short joke about programming."
    )
    print(response.outputs[-1].text)

asyncio.run(main())
```

### Streaming

```python showLineNumbers title="Streaming Interaction"
from litellm import create_interaction
import os

os.environ["GEMINI_API_KEY"] = "your-api-key"

response = create_interaction(
    model="gemini/gemini-2.5-flash",
    input="Write a 3 paragraph story about a robot.",
    stream=True
)

for chunk in response:
    print(chunk)
```

## **LiteLLM AI Gateway (Proxy) Usage**

### Setup

Add this to your litellm proxy config.yaml:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gemini-flash
    litellm_params:
      model: gemini/gemini-2.5-flash
      api_key: os.environ/GEMINI_API_KEY
```

Start litellm:

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### Test Request

<Tabs>
<TabItem value="curl" label="curl">

```bash showLineNumbers title="Create Interaction"
curl -X POST "http://localhost:4000/v1beta/interactions" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini/gemini-2.5-flash",
    "input": "Tell me a short joke about programming."
  }'
```

**Streaming:**

```bash showLineNumbers title="Streaming Interaction"
curl -N -X POST "http://localhost:4000/v1beta/interactions" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini/gemini-2.5-flash",
    "input": "Write a 3 paragraph story about a robot.",
    "stream": true
  }'
```

**Get Interaction:**

```bash showLineNumbers title="Get Interaction by ID"
curl "http://localhost:4000/v1beta/interactions/{interaction_id}" \
  -H "Authorization: Bearer sk-1234"
```

</TabItem>

<TabItem value="google-sdk" label="Google GenAI SDK">

Point the Google GenAI SDK to LiteLLM Proxy:

```python showLineNumbers title="Google GenAI SDK with LiteLLM Proxy"
from google import genai
import os

# Point SDK to LiteLLM Proxy
os.environ["GOOGLE_GENAI_BASE_URL"] = "http://localhost:4000"
os.environ["GEMINI_API_KEY"] = "sk-1234"  # Your LiteLLM API key

client = genai.Client()

# Create an interaction
interaction = client.interactions.create(
    model="gemini/gemini-2.5-flash",
    input="Tell me a short joke about programming."
)

print(interaction.outputs[-1].text)
```

**Streaming:**

```python showLineNumbers title="Google GenAI SDK Streaming"
from google import genai
import os

os.environ["GOOGLE_GENAI_BASE_URL"] = "http://localhost:4000"
os.environ["GEMINI_API_KEY"] = "sk-1234"

client = genai.Client()

for chunk in client.interactions.create_stream(
    model="gemini/gemini-2.5-flash",
    input="Write a story about space exploration.",
):
    print(chunk)
```

</TabItem>
</Tabs>

## **Request/Response Format**

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model to use (e.g., `gemini/gemini-2.5-flash`) |
| `input` | string | Yes | The input text for the interaction |
| `stream` | boolean | No | Enable streaming responses |
| `tools` | array | No | Tools available to the model |
| `system_instruction` | string | No | System instructions for the model |
| `generation_config` | object | No | Generation configuration |
| `previous_interaction_id` | string | No | ID of previous interaction for context |

### Response Format

```json
{
  "id": "interaction_abc123",
  "object": "interaction",
  "model": "gemini-2.5-flash",
  "status": "completed",
  "created": "2025-01-15T10:30:00Z",
  "updated": "2025-01-15T10:30:05Z",
  "role": "model",
  "outputs": [
    {
      "type": "text",
      "text": "Why do programmers prefer dark mode? Because light attracts bugs!"
    }
  ],
  "usage": {
    "total_input_tokens": 10,
    "total_output_tokens": 15,
    "total_tokens": 25
  }
}
```

## **Calling non-Interactions API endpoints (`/interactions` to `/responses` Bridge)**

LiteLLM allows you to call non-Interactions API models via a bridge to LiteLLM's `/responses` endpoint. This is useful for calling OpenAI, Anthropic, and other providers that don't natively support the Interactions API.

#### Python SDK Usage

```python showLineNumbers title="SDK Usage"
import litellm
import os

# Set API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Non-streaming interaction
response = litellm.interactions.create(
    model="gpt-4o",
    input="Tell me a short joke about programming."
)

print(response.outputs[-1].text)
```

#### LiteLLM Proxy Usage

**Setup Config:**

```yaml showLineNumbers title="Example Configuration"
model_list:
- model_name: openai-model
  litellm_params:
    model: gpt-4o
    api_key: os.environ/OPENAI_API_KEY
```

**Start Proxy:**

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**Make Request:**

```bash showLineNumbers title="non-Interactions API Model Request"
curl http://localhost:4000/v1beta/interactions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "openai-model",
    "input": "Tell me a short joke about programming."
  }'
```

## **Supported Providers**

| Provider | Link to Usage |
|----------|---------------|
| Google AI Studio | [Usage](#quick-start) |
| All other LiteLLM providers | [Bridge Usage](#calling-non-interactions-api-endpoints-interactions-to-responses-bridge) |
