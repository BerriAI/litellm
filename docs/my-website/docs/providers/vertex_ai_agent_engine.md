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

## Quick Start

### Model Format

```shell showLineNumbers title="Model Format"
vertex_ai/agent_engine/{RESOURCE_NAME}
```

**Example:**
- `vertex_ai/agent_engine/projects/1060139831167/locations/us-central1/reasoningEngines/8263861224643493888`

### LiteLLM Python SDK

```python showLineNumbers title="Basic Agent Completion"
import litellm

response = litellm.completion(
    model="vertex_ai/agent_engine/projects/1060139831167/locations/us-central1/reasoningEngines/8263861224643493888",
    messages=[
        {"role": "user", "content": "Explain machine learning in simple terms"}
    ],
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Streaming Agent Responses"
import litellm

response = await litellm.acompletion(
    model="vertex_ai/agent_engine/projects/1060139831167/locations/us-central1/reasoningEngines/8263861224643493888",
    messages=[
        {"role": "user", "content": "What are the key principles of software architecture?"}
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
      {"role": "user", "content": "Summarize the main benefits of cloud computing"}
    ]
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
    model="vertex-agent-1",
    messages=[
      {"role": "user", "content": "What are best practices for API design?"}
    ]
)

print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

## LiteLLM A2A Gateway

You can also connect to Vertex AI Agent Engine through LiteLLM's A2A (Agent-to-Agent) Gateway UI. This provides a visual way to register and test agents without writing code.

### 1. Navigate to Agents

From the sidebar, click "Agents" to open the agent management page, then click "+ Add New Agent".

![Click Agents](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/9a979927-ce6b-4168-9fba-e53e28f1c2c4/ascreenshot.jpeg?tl_px=0,14&br_px=1376,783&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=17,277)

![Add New Agent](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/a311750c-2e85-4589-99cb-2ce7e4021e77/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=195,257)

### 2. Select Vertex AI Agent Engine Type

Click "A2A Standard" to see available agent types, then select "Vertex AI Agent Engine".

![Select A2A Standard](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/5b1acc4c-dc3f-4639-b4a0-e64b35c228fd/ascreenshot.jpeg?tl_px=52,0&br_px=1428,769&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,271)

![Select Vertex AI Agent Engine](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/2f3bab61-3e02-4db7-84f0-82200a0f4136/ascreenshot.jpeg?tl_px=0,244&br_px=1376,1013&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=477,277)

### 3. Configure the Agent

Fill in the following fields:

- **Agent Name** - A friendly name for your agent (e.g., `my-vertex-agent`)
- **Reasoning Engine Resource ID** - The full resource path from Google Cloud Console (e.g., `projects/1060139831167/locations/us-central1/reasoningEngines/8263861224643493888`)
- **Vertex Project** - Your Google Cloud project ID
- **Vertex Location** - The region where your agent is deployed (e.g., `us-central1`)

![Enter Agent Name](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/695b84c7-9511-4337-bf19-f4505ab2b72b/ascreenshot.jpeg?tl_px=0,90&br_px=1376,859&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=480,276)

![Enter Resource ID](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/ddce64df-b3a3-4519-ab62-f137887bcea2/ascreenshot.jpeg?tl_px=0,294&br_px=1376,1063&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=440,277)

You can find the Resource ID in Google Cloud Console under Vertex AI > Agent Engine:

![Copy Resource ID from Google Cloud Console](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/185d7f17-cbaa-45de-948d-49d2091805ea/ascreenshot.jpeg?tl_px=0,165&br_px=1376,934&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=493,276)

![Enter Vertex Project](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/a64da441-3e61-4811-a1e3-9f0b12c949ff/ascreenshot.jpeg?tl_px=0,233&br_px=1376,1002&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=501,277)

You can find the Project ID in Google Cloud Console:

![Copy Project ID from Google Cloud Console](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/9ecad3bb-a534-42d6-9604-33906014fad6/user_cropped_screenshot.webp?tl_px=0,0&br_px=1728,1028&force_format=jpeg&q=100&width=1120.0)

![Enter Vertex Location](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/316d1f38-4fb7-4377-86b6-c0fe7ac24383/ascreenshot.jpeg?tl_px=0,330&br_px=1376,1099&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=423,277)

### 4. Create Agent

Click "Create Agent" to save your configuration.

![Create Agent](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/fb04b95d-793f-4eed-acf4-d1b3b5fa65e9/ascreenshot.jpeg?tl_px=352,347&br_px=1728,1117&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=623,498)

### 5. Test in Playground

Go to "Playground" in the sidebar to test your agent.

![Go to Playground](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/9e01369b-6102-4fe3-96a7-90082cadfd6e/ascreenshot.jpeg?tl_px=0,0&br_px=1376,769&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=55,226)

### 6. Select A2A Endpoint

Click the endpoint dropdown and select `/v1/a2a/message/send`.

![Select Endpoint](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/d5aeac35-531b-4cf0-af2d-88f0a71fd736/ascreenshot.jpeg?tl_px=0,146&br_px=1376,915&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=299,277)

### 7. Select Your Agent and Send a Message

Pick your Vertex AI Agent Engine from the dropdown and send a test message.

![Select Agent](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/353431f3-a0ba-4436-865d-ae11595e9cc4/ascreenshot.jpeg?tl_px=0,263&br_px=1376,1032&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=270,277)

![Send Message](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/fbfce72e-f50b-43e1-b6e5-0d41192d8e2d/ascreenshot.jpeg?tl_px=95,347&br_px=1471,1117&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,474)

![Agent Response](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/892dd826-fbf9-4530-8d82-95270889274a/ascreenshot.jpeg?tl_px=0,82&br_px=1376,851&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=485,277)

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

## Further Reading

- [Vertex AI Agent Engine Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/overview)
- [Create a Reasoning Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/create)
- [A2A Agent Gateway](../a2a.md)
- [Vertex AI Provider](./vertex.md)
