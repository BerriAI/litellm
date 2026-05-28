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
uv add "langgraph-cli[inmem]"
```

### 2. Create a new LangGraph project

```bash
langgraph new my-agent --template new-langgraph-project-python
cd my-agent
```

### 3. Install dependencies

```bash
uv add -e .
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



## LiteLLM A2A Gateway

You can register LangGraph agents in LiteLLM's [A2A (Agent-to-Agent) Gateway](../a2a.md), discover their upstream agent card, curate skills and capabilities, and invoke them through the LiteLLM proxy.

### 1. Navigate to Agents

From the sidebar, click "Agents" to open the agent management page, then click "+ Add New Agent".

![Navigate to Agents](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/27429cae-f743-440a-a6aa-29fa7ee013db/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=211,114)

### 2. Select LangGraph Agent Type

Click "A2A Standard" to see available agent types, then search for "langgraph" and select "Connect to LangGraph agents via the LangGraph Platform API".

![Select A2A Standard](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/4add4088-683d-49ca-9374-23fd65dddf8e/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=511,139)

![Select LangGraph](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/fd197907-47c7-4e05-959c-c0d42264263c/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=431,246)

### 3. Configure the Agent

Fill in the following fields:

- **Agent Name** - A unique identifier (e.g., `lan-agent`)
- **LangGraph API Base** - Your LangGraph server URL, typically `http://127.0.0.1:2024/`
- **API Key** - Optional. LangGraph doesn't require an API key by default
- **Assistant ID** - Not used by LangGraph, you can enter any string here

![Enter Agent Name](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/adce3df9-a67c-4d23-b2b5-05120738bc46/ascreenshot.jpeg?tl_px=0,0&br_px=2617,1463&force_format=jpeg&q=100&width=1120.0)

![Enter API Base](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/6a6a03a7-f235-41db-b4ba-d32ced330f25/ascreenshot.jpeg?tl_px=0,251&br_px=2617,1714&force_format=jpeg&q=100&width=1120.0)

### 4: Discover the agent card

Discovery runs automatically once the base URL and assistant ID are filled in. You can also trigger it manually from the discovery panel. 

The preview is a form. You can:

- **Edit** the name, description, provider, icon URL, and documentation URL.
- **Add, remove, or reorder skills**, and edit each skill's name, description, tags, examples, and input/output modes.
- **Toggle capabilities** that LiteLLM supports.

Select or deselect skills and capabilities before saving. LiteLLM only persists what you keep in the form.

Fields LiteLLM does not proxy are not shown. For the full support matrix, see [Agent card support](../a2a_agent_card.md#agent-card-support).

![Agent Card Fields on UI](../../img/providers/langgraph/agent-card-fields-on-ui.png)

### 5: Save the agent

Click on Next to save. And complete the rest of the steps

![Click on Next](../../img/providers/langgraph/click-on-next.png)

### 6: Verify the served card

From your terminal, fetch the agent card LiteLLM is serving:

```bash
curl -H "Authorization: Bearer sk-1234" \
  http://localhost:4000/a2a/{agent_id}/.well-known/agent.json | jq
```

You should see the card you saved, with:

- `supportedInterfaces[0].url` pointing at LiteLLM, not the upstream
- `securitySchemes` showing `LiteLLMKey` (HTTP bearer)
- The skills you kept during registration

### 7. Test in Playground
Go to "Playground" in the sidebar to test your agent. Change the endpoint type to `/v1/a2a/message/send`.

![Go to Playground](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/c4262189-95ac-4fbc-b5af-8aba8126e4f7/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=41,104)

![Select A2A Endpoint](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/6cbc8e93-7d0c-47fc-9ad4-562663f759d5/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=324,265)

### 8. Select Your Agent and Send a Message
Pick your LangGraph agent from the dropdown and send a test message.

![Select Agent](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/d01da2f1-3b89-47d7-ba95-de2dd8efbc1e/ascreenshot.jpeg?tl_px=0,92&br_px=2201,1323&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=348,277)

![Send Message](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/79db724e-a99e-493a-9747-dc91cb398370/ascreenshot.jpeg?tl_px=51,653&br_px=2252,1883&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,444)

![Agent Response](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/82aa546a-0eb5-4836-b986-9aefcfe09e10/ascreenshot.jpeg?tl_px=295,28&br_px=2496,1259&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,277)

### 9: Invoke the agent manually

Send an A2A `message/send` request to the LiteLLM proxy URL:

```bash
curl -X POST http://localhost:4000/a2a/{agent_id} \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "req-1",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "msg-001",
        "role": "user",
        "parts": [{"kind": "text", "text": "My order is urgent and still not delivered"}],
        "metadata": {"skillId": "triage_ticket"}
      }
    }
  }'
```

For streaming, use `message/stream` and add `-N -H "Accept: text/event-stream"` to the curl.

See also [Invoking A2A Agents](../a2a_invoking_agents.md) for SDK examples.


## Further Reading

- [LangGraph Platform Documentation](https://langchain-ai.github.io/langgraph/cloud/quick_start/)
- [LangGraph A2A endpoint docs](https://docs.langchain.com/langsmith/server-a2a)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [A2A Agent Gateway](../a2a.md)
- [A2A Agent Card on LiteLLM](../a2a_agent_card.md)
- [A2A Cost Tracking](../a2a_cost_tracking.md)
- [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/)

