import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Max Iterations (Agent Loop Limits)

Limit the number of LLM calls an agentic loop can make per session. Callers send a `session_id` with each request, and LiteLLM returns `429` when `max_iterations` is exceeded.

## Quick Start

### 1. Set `max_iterations` on a key

```bash
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{"metadata": {"max_iterations": 25}}'
```

### 2. Send requests with `session_id`

Include the same `session_id` on every call in the agent loop via the `x-litellm-session-id` header or `metadata.session_id`.

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

## Configuration

Set `max_iterations` in key metadata via `/key/generate` or `/key/update`:

```bash
# Update existing key
curl -L -X POST 'http://0.0.0.0:4000/key/update' \
-H 'Authorization: Bearer sk-1234' \
-d '{"key": "sk-existing-key", "metadata": {"max_iterations": 50}}'
```

Session counters auto-expire after 1 hour (configurable via `LITELLM_MAX_ITERATIONS_TTL` env var in seconds).

Works across multiple proxy instances via Redis.
