import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MCP + Agent Orchestration in /chat/completions

Use your registered MCP servers and A2A agents directly from `/v1/chat/completions` — no hardcoded URLs required.

## The problem it solves

Without this, every client has to know which MCP server to call:

```json
// ❌ Every request hardcodes a specific server URL
{
  "tools": [{"type": "mcp", "server_url": "http://my-zapier-server/mcp", ...}]
}
```

That means updating every client when servers change, no central access control, and no way to let the LLM pick across multiple servers.

## How it works

Register your servers once via `POST /v1/mcp/server`. Then point any chat request at the proxy's registry using `"server_url": "litellm_proxy/mcp"` — the proxy fetches available tools, injects them into the LLM context, and executes tool calls on the model's behalf.

```
POST /v1/chat/completions
         │
         ├── type:"mcp",  server_url:"litellm_proxy/mcp"
         │        └── expand → all servers registered via POST /v1/mcp/server
         │        └── fetch tool schemas from each server
         │        └── inject into LLM context
         │
         └── type:"a2a_agent", server_url:"litellm_proxy/agents"
                  └── expand → all agents registered via POST /v1/agents
                  └── wrap each agent as a callable function tool
         │
         ▼
    LLM decides which tools to call
         │
         ▼
    Proxy executes tool calls, returns results
         │
         ▼
    Follow-up LLM call → final answer
```

## Quickstart

### 1. Register an MCP server

```bash
curl -X POST http://localhost:4000/v1/mcp/server \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "server_name": "math",
    "url": "http://localhost:8001/mcp",
    "transport": "http"
  }'
```

### 2. Call `/v1/chat/completions` with `litellm_proxy/mcp`

<Tabs>
<TabItem value="curl" label="cURL">

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 1250 + 18?"}],
    "tools": [
      {
        "type": "mcp",
        "server_url": "litellm_proxy/mcp",
        "require_approval": "never"
      }
    ]
  }'
```

</TabItem>
<TabItem value="python" label="Python (OpenAI SDK)">

```python
import openai

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000",
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is 1250 + 18?"}],
    tools=[
        {
            "type": "mcp",
            "server_url": "litellm_proxy/mcp",
            "require_approval": "never",
        }
    ],
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="litellm" label="LiteLLM SDK">

```python
import litellm

response = await litellm.acompletion(
    model="gpt-4o-mini",
    api_base="http://localhost:4000",
    api_key="sk-1234",
    messages=[{"role": "user", "content": "What is 1250 + 18?"}],
    tools=[
        {
            "type": "mcp",
            "server_url": "litellm_proxy/mcp",
            "require_approval": "never",
        }
    ],
)
print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

## Target a specific server

Append the server name to `litellm_proxy/mcp/` to restrict tool injection to one server:

```json
{
  "type": "mcp",
  "server_url": "litellm_proxy/mcp/math",
  "require_approval": "never"
}
```

## A2A Agent orchestration

Agents registered via `POST /v1/agents` are exposed as callable function tools using the same pattern.

### 1. Register an agent

```bash
curl -X POST http://localhost:4000/v1/agents \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "FX_Converter",
    "agent_url": "http://my-fx-agent/a2a",
    "description": "Converts currency amounts using live exchange rates."
  }'
```

### 2. Use MCP tools and agents together

```json
{
  "model": "gpt-4o-mini",
  "messages": [{"role": "user", "content": "Add 500 + 250, then convert the result to EUR."}],
  "tools": [
    {
      "type": "mcp",
      "server_url": "litellm_proxy/mcp",
      "require_approval": "never"
    },
    {
      "type": "a2a_agent",
      "server_url": "litellm_proxy/agents",
      "require_approval": "never"
    }
  ]
}
```

The proxy wraps each registered agent as a function tool. When the LLM calls it, the proxy sends a JSON-RPC `message/send` to the agent and returns the result as a tool message.

## Semantic filter

Add `"semantic_filter": true` to only inject tools relevant to the user's query. Useful when you have many registered servers and want to keep the LLM context lean.

```json
{
  "type": "mcp",
  "server_url": "litellm_proxy/mcp",
  "require_approval": "never",
  "semantic_filter": true
}
```

Configure top-k and similarity threshold in your proxy config:

```yaml
mcp_semantic_tool_filter:
  top_k: 10
  similarity_threshold: 0.3
  embedding_model: "text-embedding-3-small"
```

See [MCP Semantic Filter](./mcp_semantic_filter) for setup details.

## Streaming

Works with `"stream": true` — tokens arrive as they're generated, tool execution happens between LLM turns.

```python
stream = await litellm.acompletion(
    model="gpt-4o-mini",
    api_base="http://localhost:4000",
    api_key="sk-1234",
    messages=[{"role": "user", "content": "What is 42 × 13?"}],
    tools=[{"type": "mcp", "server_url": "litellm_proxy/mcp", "require_approval": "never"}],
    stream=True,
)
async for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

## Access control

Tool visibility follows your existing key and team permissions. A virtual key scoped to specific MCP servers will only see those servers when it calls `litellm_proxy/mcp` — no extra config needed.

See [MCP Zero Trust](./mcp_zero_trust) for per-key and per-team tool restrictions.
