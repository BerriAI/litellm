import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Tickerr - Outage Radar for AI Agents

[Tickerr](https://tickerr.ai) is a crowd-sourced outage detector for LLM APIs. When your agent hits a 5xx or rate limit, it reports anonymously so every agent can see the issue in real time.

**No API key. No account. Failure-only by default. Success sampling is opt-in.**

## Quick Start

<Tabs>
<TabItem value="sdk" label="Python SDK">

```python
import litellm

litellm.callbacks = ["tickerr"]

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy config.yaml">

```yaml
litellm_settings:
  callbacks: ["tickerr"]
```

</TabItem>
</Tabs>

## What Gets Reported

When the tickerr callback is explicitly enabled, anonymous failure metadata is reported. No prompts, responses, API keys, or personal data are sent.

| Field | Example |
|-------|---------|
| Provider | `anthropic` |
| Model | `claude-haiku-4-5` |
| Status code | `529` |
| Latency (ms) | `1240` |
| Event type | `failure` or `success` |

## Optional Configuration

```bash
TICKERR_DISABLED=true          # disable all reporting (kill switch)
TICKERR_REGION=us-east-1       # for regional signal breakdown
TICKERR_SAMPLE_RATE=0.01       # report 1% of successes for latency benchmarks (default: 0 = off)
```

Failures are reported by default once Tickerr is enabled. Success sampling is opt-in.

## Links

- [Live dashboard](https://tickerr.ai) - status for 90+ AI tools
- [Agent reports feed](https://tickerr.ai/agent-reports) - real-time signal
- [Opt out](https://tickerr.ai/mcp/opt-out)
