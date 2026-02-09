# ClawRouter

Smart model routing for [OpenClaw](https://openclaw.ai). Routes your OpenClaw's requests to cheap, mid-range, or expensive models based on what you're doing so you stop burning credits on simple requests.

## Why

Why drive a Lamborgini to the grocery store when the Prius will do? ClawRouter classifies each message with regex patterns and sends it to the right tier based on the table below:

| Tier | What goes here | Example models |
|------|---------------|----------------|
| **Low** | Greetings, simple chat, factual lookups | DeepSeek, Gemini Flash Lite |
| **Mid** | Coding, translation, summarization, creative writing | Claude Haiku, GLM |
| **Top** | Math proofs, multi-step reasoning, deep analysis | Claude Sonnet, GPT-4o |

## Install (OpenClaw Plugin)

```bash
openclaw plugins install clawrouter
```

Restart the gateway. The plugin will:
1. Clone the repo and set up a Python venv
2. Pull your API keys from your existing OpenClaw auth profiles
3. Generate routing configs and start the proxy
4. Register `clawrouter/auto` as a model provider

From here, messages sent to `clawrouter/auto` are automatically classified and routed to the models you have configured.

## Configuration

### Routing Rules

Edit `routing_rules.yaml` to customize tier models, category-to-tier mappings, or add domain-specific patterns:

```yaml
tiers:
  low:
    model: "deepseek/deepseek-chat"
  mid:
    model: "anthropic/claude-haiku-4-5-20251001"
  top:
    model: "anthropic/claude-sonnet-4-5-20250929"

routing:
  heartbeat: low
  simple-chat: low
  lookup: low
  translation: mid
  summarization: mid
  coding: mid
  creative: mid
  reasoning: top
  analysis: top
```

## User Controls

**Force a tier** by prefixing your message:
- `[low] what's 2+2` — force low tier model
- `[med] write a python script` — force medium tier model
- `[high] prove this theorem` — force top tier model

The tag is stripped before the model sees it.

## Standalone Usage (Without OpenClaw)

```bash
./setup.sh
```

This walks you through API key setup, tier model selection, and starts the proxy on port 4000. To test, run:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello!"}]}'
```
