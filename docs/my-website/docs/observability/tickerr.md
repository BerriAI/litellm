import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Tickerr — Outage Radar for AI Agents

[Tickerr](https://tickerr.ai) is a crowd-sourced outage detector for LLM APIs. When your agent hits a 5xx error or rate limit, Tickerr tells you:

- How many other agents are seeing the same issue right now
- Current signal state: `quiet` → `detecting` → `confirmed` → `recovering`
- Which model to fall back to

**No API key required. Anonymous. Zero overhead on success paths.**

## Quick Start

<Tabs>
<TabItem value="sdk" label="Python SDK">

```python
import litellm

litellm.callbacks = ["tickerr"]

# Every failed LiteLLM call is now reported to Tickerr automatically.
# Tickerr fires in a background thread — your agent is never blocked.
response = litellm.completion(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": "Hello"}]
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy config.yaml">

```yaml
model_list:
  - model_name: claude-haiku
    litellm_params:
      model: anthropic/claude-haiku-4-5
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  callbacks: ["tickerr"]
```

</TabItem>
</Tabs>

## What Gets Reported

Tickerr receives only:

| Field | Example |
|-------|---------|
| Provider | `anthropic` |
| Model | `claude-haiku-4-5` |
| HTTP status code | `529` |
| Error type | `overloaded` |
| Latency (ms) | `1240` |

No prompts, no responses, no personal data.

## What You Get Back

Each report updates the live signal at [tickerr.ai/agent-reports](https://tickerr.ai/agent-reports).

To read the signal from your agent, use the [Tickerr MCP server](https://tickerr.ai/mcp-server) `report_incident` tool. It returns a structured response:

```
CURRENT SIGNAL (anthropic/claude-haiku-4-5)
Status: CONFIRMED
Agents reporting (last 10 min): 14
Total reports (last 10 min): 31

RECOMMENDATION
Action: FALLBACK
Switch to: gpt-4o-mini (openai)
```

## Optional Configuration

```python
import os

os.environ["TICKERR_CLIENT_TIER"] = "pro"    # free | pro | team | enterprise | api_pay_as_you_go
os.environ["TICKERR_REGION"] = "us-east-1"   # optional, for regional breakdown
```

## Signal States

| State | Meaning |
|-------|---------|
| `quiet` | No reports in last 10 min |
| `detecting` | 1–2 agents reporting |
| `confirmed` | 3+ distinct agents — issue verified |
| `recovering` | Reports dropping, recovery signals arriving |

## Opt Out

[tickerr.ai/mcp/opt-out](https://tickerr.ai/mcp/opt-out)

## Links

- [Tickerr](https://tickerr.ai) — live AI status dashboard (90+ tools)
- [Agent reports](https://tickerr.ai/agent-reports) — live feed
- [Tickerr MCP server](https://tickerr.ai/mcp-server) — 9-tool MCP for agents
- [REST API](https://tickerr.ai/api/v1/report) — report without LiteLLM
