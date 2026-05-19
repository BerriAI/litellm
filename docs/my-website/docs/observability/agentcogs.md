# AgentCOGS - Per-customer margin

[AgentCOGS](https://github.com/vaibhav11123/agentcogs) tracks per-customer LLM cost and gross margin for B2B SaaS (cost + revenue), alongside your existing proxy and observability stack.

## Quick Start

Use one line to send successful completion cost to AgentCOGS:

Get your AgentCOGS [API key and workspace id](https://github.com/vaibhav11123/agentcogs/blob/main/docs/quickstart.md).

```python
import os
import litellm

os.environ["AGENTCOGS_API_KEY"] = ""
os.environ["AGENTCOGS_WORKSPACE_ID"] = ""
# optional — defaults to https://api.agentcogs.dev
os.environ["AGENTCOGS_ENDPOINT"] = "https://api.agentcogs.dev"

litellm.success_callback = ["agentcogs"]

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi"}],
    user="your_customer_id",  # 👈 B2B tenant id → AgentCOGS customer_id
    metadata={"agentcogs_workflow_id": "support_bot"},
)
```

- SDK
- PROXY

```yaml
litellm_settings:
  callbacks: ["agentcogs"]
```

Set `AGENTCOGS_API_KEY`, `AGENTCOGS_WORKSPACE_ID`, and optionally `AGENTCOGS_ENDPOINT` in the proxy environment.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AGENTCOGS_API_KEY` | Yes | Workspace API key (`acg_live_...`) |
| `AGENTCOGS_WORKSPACE_ID` | Yes | Workspace UUID |
| `AGENTCOGS_ENDPOINT` | No | API base URL (default `https://api.agentcogs.dev`) |

## Tenant attribution

Pass `user=` on each completion (same pattern as [Lago](./lago.md)). LiteLLM maps it to AgentCOGS `customer_id`.

Alternatively set `metadata.agentcogs_customer_id` if you cannot use `user`.

Completions without a customer id are skipped (proxy traffic is not blocked).

## What LiteLLM sends

Each successful or failed completion POSTs to `POST /v1/ingest`:

```json
{
  "run_id": "<uuid>",
  "workspace_id": "<AGENTCOGS_WORKSPACE_ID>",
  "customer_id": "<user or metadata.agentcogs_customer_id>",
  "workflow_id": "default",
  "ts": 1710000000,
  "status": "completed",
  "total_usd": 0.0012,
  "models": {
    "gpt-4o-mini": {
      "input_tokens": 10,
      "output_tokens": 5,
      "usd": 0.0012
    }
  },
  "metadata": { "source": "litellm" }
}
```

## Learn more

- [AgentCOGS quickstart](https://github.com/vaibhav11123/agentcogs/blob/main/docs/quickstart.md)
- [LiteLLM callback (user-landed)](https://github.com/vaibhav11123/agentcogs/blob/main/docs/integrations/litellm.md)
