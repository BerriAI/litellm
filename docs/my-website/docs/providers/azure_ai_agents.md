import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure AI Foundry Agents

Call Azure AI Foundry Agents in the OpenAI Request/Response format.

| Property | Details |
|----------|---------|
| Description | Azure AI Foundry Agents provides hosted agent runtimes that can execute agentic workflows with foundation models, tools, and code interpreters. |
| Provider Route on LiteLLM | `azure_ai/agents/{AGENT_ID}` |
| Provider Doc | [Azure AI Foundry Agents ↗](https://learn.microsoft.com/en-us/rest/api/aifoundry/aiagents/create-thread-and-run/create-thread-and-run) |

## Quick Start

### Model Format to LiteLLM

To call an Azure AI Foundry Agent through LiteLLM, use the following model format.

Here the `model=azure_ai/agents/` tells LiteLLM to call the Azure AI Foundry Agent Service API.

```shell showLineNumbers title="Model Format to LiteLLM"
azure_ai/agents/{AGENT_ID}
```

**Example:**
- `azure_ai/agents/asst_abc123`

You can find the Agent ID in your Azure AI Foundry portal under Agents.

### LiteLLM Python SDK

```python showLineNumbers title="Basic Agent Completion"
import litellm

# Make a completion request to your Azure AI Foundry Agent
response = litellm.completion(
    model="azure_ai/agents/asst_abc123",
    messages=[
        {
            "role": "user", 
            "content": "Explain machine learning in simple terms"
        }
    ],
    api_base="https://your-project.services.ai.azure.com",
    api_key="your-api-key",
)

print(response.choices[0].message.content)
print(f"Usage: {response.usage}")
```

```python showLineNumbers title="Streaming Agent Responses"
import litellm

# Stream responses from your Azure AI Foundry Agent
response = await litellm.acompletion(
    model="azure_ai/agents/asst_abc123",
    messages=[
        {
            "role": "user",
            "content": "What are the key principles of software architecture?"
        }
    ],
    api_base="https://your-project.services.ai.azure.com",
    api_key="your-api-key",
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
  - model_name: azure-agent-1
    litellm_params:
      model: azure_ai/agents/asst_abc123
      api_base: https://your-project.services.ai.azure.com
      api_key: os.environ/AZURE_API_KEY

  - model_name: azure-agent-math-tutor
    litellm_params:
      model: azure_ai/agents/asst_def456
      api_base: https://your-project.services.ai.azure.com
      api_key: os.environ/AZURE_API_KEY
```

</TabItem>
</Tabs>

#### 2. Start the LiteLLM Proxy

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml
```

#### 3. Make requests to your Azure AI Foundry Agents

<Tabs>
<TabItem value="curl" label="Curl">

```bash showLineNumbers title="Basic Agent Request"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "azure-agent-1",
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
    "model": "azure-agent-math-tutor",
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

# Make a completion request to your Azure AI Foundry Agent
response = client.chat.completions.create(
    model="azure-agent-1",
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
    model="azure-agent-math-tutor",
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

You can set the following environment variables to configure Azure AI Foundry Agents:

| Variable | Description |
|----------|-------------|
| `AZURE_API_BASE` | The Azure AI Foundry project endpoint (e.g., `https://your-project.services.ai.azure.com`) |
| `AZURE_API_KEY` | Your Azure AI Foundry API key |

```bash
export AZURE_API_BASE="https://your-project.services.ai.azure.com"
export AZURE_API_KEY="your-api-key"
```

## Conversation Continuity (Thread Management)

Azure AI Foundry Agents use threads to maintain conversation context. LiteLLM automatically manages threads for you, but you can also pass an existing thread ID to continue a conversation.

```python showLineNumbers title="Continuing a Conversation"
import litellm

# First message creates a new thread
response1 = await litellm.acompletion(
    model="azure_ai/agents/asst_abc123",
    messages=[{"role": "user", "content": "My name is Alice"}],
    api_base="https://your-project.services.ai.azure.com",
    api_key="your-api-key",
)

# Get the thread_id from the response
thread_id = response1._hidden_params.get("thread_id")

# Continue the conversation using the same thread
response2 = await litellm.acompletion(
    model="azure_ai/agents/asst_abc123",
    messages=[{"role": "user", "content": "What's my name?"}],
    api_base="https://your-project.services.ai.azure.com",
    api_key="your-api-key",
    thread_id=thread_id,  # Pass the thread_id to continue conversation
)

print(response2.choices[0].message.content)  # Should mention "Alice"
```

## Provider-specific Parameters

Azure AI Foundry Agents support additional parameters that can be passed to customize the agent invocation.

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers title="Using Agent-specific parameters"
from litellm import completion

response = litellm.completion(
    model="azure_ai/agents/asst_abc123",
    messages=[
        {
            "role": "user",
            "content": "Analyze this data and provide insights",
        }
    ],
    api_base="https://your-project.services.ai.azure.com",
    api_key="your-api-key",
    thread_id="thread_abc123",  # Optional: Continue existing conversation
    instructions="Be concise and focus on key insights",  # Optional: Override agent instructions
)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

```yaml showLineNumbers title="LiteLLM Proxy Configuration with Parameters"
model_list:
  - model_name: azure-agent-analyst
    litellm_params:
      model: azure_ai/agents/asst_abc123
      api_base: https://your-project.services.ai.azure.com
      api_key: os.environ/AZURE_API_KEY
      instructions: "Be concise and focus on key insights"
```

</TabItem>
</Tabs>

### Available Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `thread_id` | string | Optional thread ID to continue an existing conversation |
| `instructions` | string | Optional instructions to override the agent's default instructions for this run |

## Further Reading

- [Azure AI Foundry Agents Documentation](https://learn.microsoft.com/en-us/azure/ai-services/agents/)
- [Create Thread and Run API Reference](https://learn.microsoft.com/en-us/rest/api/aifoundry/aiagents/create-thread-and-run/create-thread-and-run)
