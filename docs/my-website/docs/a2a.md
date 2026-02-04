import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# Agent Gateway (A2A Protocol) - Overview

Add A2A Agents on LiteLLM AI Gateway, Invoke agents in A2A Protocol, track request/response logs in LiteLLM Logs. Manage which Teams, Keys can access which Agents onboarded.

<Image 
  img={require('../img/a2a_gateway.png')}
  style={{width: '80%', display: 'block', margin: '0', borderRadius: '8px'}}
/>

<br />
<br />

| Feature | Supported | 
|---------|-----------|
| Supported Agent Providers | A2A, Vertex AI Agent Engine, LangGraph, Azure AI Foundry, Bedrock AgentCore, Pydantic AI |
| Logging | ✅ |
| Load Balancing | ✅ |
| Streaming | ✅ |


:::tip

LiteLLM follows the [A2A (Agent-to-Agent) Protocol](https://github.com/google/A2A) for invoking agents.

:::

## Adding your Agent

### Add A2A Agents

You can add A2A-compatible agents through the LiteLLM Admin UI.

1. Navigate to the **Agents** tab
2. Click **Add Agent**
3. Enter the agent name (e.g., `ij-local`) and the URL of your A2A agent

<Image 
  img={require('../img/add_agent_1.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

The URL should be the invocation URL for your A2A agent (e.g., `http://localhost:10001`).


### Add Azure AI Foundry Agents

Follow [this guide, to add your azure ai foundry agent to LiteLLM Agent Gateway](./providers/azure_ai_agents#litellm-a2a-gateway)

### Add Vertex AI Agent Engine

Follow [this guide, to add your Vertex AI Agent Engine to LiteLLM Agent Gateway](./providers/vertex_ai_agent_engine)

### Add Bedrock AgentCore Agents

Follow [this guide, to add your bedrock agentcore agent to LiteLLM Agent Gateway](./providers/bedrock_agentcore#litellm-a2a-gateway)

### Add LangGraph Agents

Follow [this guide, to add your langgraph agent to LiteLLM Agent Gateway](./providers/langgraph#litellm-a2a-gateway)

### Add Pydantic AI Agents

Follow [this guide, to add your pydantic ai agent to LiteLLM Agent Gateway](./providers/pydantic_ai_agent#litellm-a2a-gateway)

## Invoking your Agents

See the [Invoking A2A Agents](./a2a_invoking_agents) guide to learn how to call your agents using:
- **A2A SDK** - Native A2A protocol with full support for tasks and artifacts
- **OpenAI SDK** - Familiar `/chat/completions` interface with `a2a/` model prefix

## Tracking Agent Logs

After invoking an agent, you can view the request logs in the LiteLLM **Logs** tab.

The logs show:
- **Request/Response content** sent to and received from the agent
- **User, Key, Team** information for tracking who made the request
- **Latency and cost** metrics

<Image 
  img={require('../img/agent2.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>


## Forwarding LiteLLM Context Headers

When LiteLLM invokes your A2A agent, it sends special headers that enable:
- **Trace Grouping**: All LLM calls from the same agent execution appear under one trace
- **Agent Spend Tracking**: Costs are attributed to the specific agent

| Header | Purpose |
|--------|---------|
| `X-LiteLLM-Trace-Id` | Links all LLM calls to the same execution flow |
| `X-LiteLLM-Agent-Id` | Attributes spend to the correct agent |


To enable these features, your A2A server must **forward these headers** to any LLM calls it makes back to LiteLLM.

### Implementation Steps

**Step 1: Extract headers from incoming A2A request**
```python def get_litellm_headers(request) -> dict:
    """Extract X-LiteLLM-* headers from incoming A2A request."""
    all_headers = request.call_context.state.get('headers', {})
    return {
        k: v for k, v in all_headers.items() 
        if k.lower().startswith('x-litellm-')
    }
```

**Step 2: Forward headers to your LLM calls**
Pass the extracted headers when making calls back to LiteLLM:
<Tabs>
<TabItem value="openai" label="OpenAI SDK" default>

```python from openai import OpenAI

headers = get_litellm_headers(request)

client = OpenAI(
    api_key="sk-your-litellm-key",
    base_url="http://localhost:4000",
    default_headers=headers,  # Forward headers
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
```
</TabItem>

<TabItem value="langchain" label="LangChain">

```python
from langchain_openai import ChatOpenAI

headers = get_litellm_headers(request)

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key="sk-your-litellm-key",
    base_url="http://localhost:4000",
    default_headers=headers,  # Forward headers
)
```
</TabItem>
<TabItem value="litellm" label="LiteLLM SDK">

```python
import litellm

headers = get_litellm_headers(request)

response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    api_base="http://localhost:4000",
    extra_headers=headers,  # Forward headers
)
```
</TabItem>
<TabItem value="requests" label="HTTP (requests/httpx)">

```python
import httpx

headers = get_litellm_headers(request)
headers["Authorization"] = "Bearer sk-your-litellm-key"

response = httpx.post(
    "http://localhost:4000/v1/chat/completions",
    headers=headers,
    json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}
)
```
</TabItem>
</Tabs>

### Result

With header forwarding enabled, you'll see:

**Trace Grouping in Langfuse:**

<Image
  img={require('../img/a2a_trace_grouping.png')}
  style={{width: '80%', display: 'block', margin: '0', borderRadius: '8px'}}
/>

**Agent Spend Attribution:**

<Image
  img={require('../img/a2a_agent_spend.png')}
  style={{width: '80%', display: 'block', margin: '0', borderRadius: '8px'}}
/>

## API Reference

### Endpoint

```
POST /a2a/{agent_name}/message/send
```

### Authentication

Include your LiteLLM Virtual Key in the `Authorization` header:

```
Authorization: Bearer sk-your-litellm-key
```

### Request Format

LiteLLM follows the [A2A JSON-RPC 2.0 specification](https://github.com/google/A2A):

```json title="Request Body"
{
  "jsonrpc": "2.0",
  "id": "unique-request-id",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Your message here"}],
      "messageId": "unique-message-id"
    }
  }
}
```

### Response Format

```json title="Response"
{
  "jsonrpc": "2.0",
  "id": "unique-request-id",
  "result": {
    "kind": "task",
    "id": "task-id",
    "contextId": "context-id",
    "status": {"state": "completed", "timestamp": "2025-01-01T00:00:00Z"},
    "artifacts": [
      {
        "artifactId": "artifact-id",
        "name": "response",
        "parts": [{"kind": "text", "text": "Agent response here"}]
      }
    ]
  }
}
```

## Agent Registry

Want to create a central registry so your team can discover what agents are available within your company?

Use the [AI Hub](./proxy/ai_hub) to make agents public and discoverable across your organization. This allows developers to browse available agents without needing to rebuild them.
