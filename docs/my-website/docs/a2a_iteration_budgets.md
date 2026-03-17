import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Agent Iteration Budgets

Control runaway costs from agentic loops with per-session iteration and budget caps.

## Overview

When agents run agentic loops, they can make unbounded LLM calls, causing unexpected costs. LiteLLM provides two controls:

| Control | Description |
|---------|-------------|
| **Max Iterations** | Hard cap on the number of LLM calls per session |
| **Max Budget Per Session** | Dollar cap per session (identified by `x-litellm-trace-id`) |

Both controls require a `session_id` (sent via `x-litellm-trace-id` header or `metadata.session_id`) to track calls within a session.

## Trace-ID Enforcement

LiteLLM supports two independent trace-id flags, configured in `litellm_params` on the agent:

| Flag | Description |
|------|-------------|
| `require_trace_id_on_calls_to_agent` | Requires callers invoking this agent to include `x-litellm-trace-id`. Use when the agent should only be called as a sub-agent with a trace context. Returns **400** if missing. |
| `require_trace_id_on_calls_by_agent` | Requires all LLM/MCP calls made **by** this agent (via its virtual key) to include `x-litellm-trace-id`. This is what enables `max_iterations` and `max_budget_per_session` tracking. Returns **400** if missing. |

## Configuring via UI

When creating an agent in the LiteLLM Admin UI:

1. Navigate to the **Agents** tab and click **Add Agent**
2. In the **Agent Settings** step, expand the **Tracing** section
3. Toggle **Require x-litellm-trace-id on calls BY this agent** to enable session tracking
4. Set **Max Iterations** to cap the number of LLM calls per session
5. Set **Max Budget Per Session ($)** to cap spend per session

The trace-id flags are stored on the agent's `litellm_params`. Budget controls (`max_iterations`, `max_budget_per_session`) are stored in the virtual key's metadata.

## Configuring via API

Set trace-id enforcement on the agent itself:

```bash
curl -X POST 'http://localhost:4000/v1/agents' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "agent_name": "my-research-agent",
    "agent_card_params": {
      "name": "my-research-agent",
      "description": "A research agent with budget controls",
      "url": "http://my-agent:8080",
      "version": "1.0.0"
    },
    "litellm_params": {
      "require_trace_id_on_calls_to_agent": true,
      "require_trace_id_on_calls_by_agent": true
    }
  }'
```

Budget controls are set on the agent's `litellm_params` (not on individual keys), so they apply across all keys for the agent:

```bash
curl -X POST 'http://localhost:4000/v1/agents' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "agent_name": "my-research-agent",
    "agent_card_params": {
      "name": "my-research-agent",
      "description": "A research agent with budget controls",
      "url": "http://my-agent:8080",
      "version": "1.0.0"
    },
    "litellm_params": {
      "require_trace_id_on_calls_by_agent": true,
      "max_iterations": 25,
      "max_budget_per_session": 5.00
    }
  }'
```

## How It Works

### Session Tracking

Callers identify their session by including a `session_id` in one of these ways:
- **Header**: `x-litellm-trace-id: my-session-123`
- **Metadata**: `{"metadata": {"session_id": "my-session-123"}}`

### Max Iterations

When `max_iterations` is set in agent `litellm_params`:
- Each LLM call for a session increments a counter
- When the counter exceeds `max_iterations`, the request receives a **429 Too Many Requests**
- Counters expire after 1 hour by default (configurable via `LITELLM_MAX_ITERATIONS_TTL` env var)

### Max Budget Per Session

When `max_budget_per_session` is set in agent `litellm_params`:
- After each successful LLM call, the response cost is accumulated for the session
- Before each call, the accumulated spend is checked against the budget
- When spend exceeds the budget, the request receives a **429 Too Many Requests**
- Session spend counters expire after 1 hour by default (configurable via `LITELLM_MAX_BUDGET_PER_SESSION_TTL` env var)

## Example

Create an agent with max 25 iterations and a $5 budget cap:

<Tabs>
<TabItem value="ui" label="Via UI">

1. Go to **Agents** → **Add Agent**
2. Configure your agent (name, model, etc.)
3. In **Agent Settings**, expand the **Tracing** section
4. Toggle on **Require x-litellm-trace-id on calls BY this agent**
5. Set **Max Iterations** to `25`
6. Set **Max Budget Per Session** to `5.00`
7. Proceed to create a new key for the agent
8. Click **Create Agent**

</TabItem>
<TabItem value="api" label="Via API">

```bash
# 1. Create the agent with trace-id enforcement
curl -X POST 'http://localhost:4000/v1/agents' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "agent_name": "my-research-agent",
    "agent_card_params": {
      "name": "my-research-agent",
      "description": "A research agent with budget controls",
      "url": "http://my-agent:8080",
      "version": "1.0.0"
    },
    "litellm_params": {
      "require_trace_id_on_calls_by_agent": true
    }
  }'

# 2. Create a key for the agent
curl -X POST 'http://localhost:4000/key/generate' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "agent_id": "<agent_id_from_step_1>",
    "key_alias": "my-research-agent-key"
  }'
```

</TabItem>
</Tabs>

### Making Calls with Session Tracking

```bash
curl -X POST 'http://localhost:4000/chat/completions' \
  -H 'Authorization: Bearer sk-agent-key-xxx' \
  -H 'x-litellm-trace-id: session-abc-123' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

After 25 calls or $5 spent within this session, subsequent requests will receive:

```json
{
  "error": {
    "message": "Session budget exceeded for session session-abc-123. Current spend: $5.0032, max_budget_per_session: $5.00.",
    "type": "budget_exceeded",
    "code": 429
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_MAX_ITERATIONS_TTL` | `3600` (1 hour) | TTL in seconds for session iteration counters |
| `LITELLM_MAX_BUDGET_PER_SESSION_TTL` | `3600` (1 hour) | TTL in seconds for session budget counters |
