import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Max Iterations (Agent Loop Limits)

Limit the number of LLM calls an agentic loop can make per session. Callers send a `session_id` with each request, and LiteLLM returns `429` when `max_iterations` is exceeded.

## Quick Start

### 1. Set `max_iterations` on an agent

Set `max_iterations` in the agent's `litellm_params` when creating the agent:

```bash
curl -L -X POST 'http://0.0.0.0:4000/agents' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "agent_name": "my-agent",
    "litellm_params": {"max_iterations": 25},
    "agent_card_params": {"name": "my-agent", "url": "http://agent:8000"}
}'
```

You can also set it per key as a fallback (agent config takes priority):

```bash
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-d '{"metadata": {"max_iterations": 25}}'
```

### 2. Send requests with `session_id`

Include the same `session_id` on every call in the agent loop via `x-litellm-session-id` header or `metadata.session_id`.

<Tabs>
<TabItem value="python" label="Python">

```python
from openai import OpenAI

client = OpenAI(api_key="sk-generated-key", base_url="http://0.0.0.0:4000")

session_id = "agent-run-abc123"

for step in range(50):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        extra_headers={"x-litellm-session-id": session_id},
    )
    # ... process response, execute tools ...
    # Returns 429 after 25 calls
```

</TabItem>
<TabItem value="curl" label="curl">

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Authorization: Bearer sk-generated-key' \
-H 'x-litellm-session-id: agent-run-abc123' \
-H 'Content-Type: application/json' \
-d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

</TabItem>
</Tabs>

Works on all proxy endpoints: `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/a2a/{agent_id}`.

## Priority Order

`max_iterations` is resolved in this order:

1. **Agent config** — `litellm_params.max_iterations` (looked up via `agent_id` in request metadata)
2. **Key metadata** — `metadata.max_iterations` (set via `/key/generate` or `/key/update`)

## Settings

Session counters auto-expire after 1 hour (configurable via `LITELLM_MAX_ITERATIONS_TTL` env var in seconds). Works across multiple proxy instances via Redis.
