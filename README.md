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
# 1. Clone the repo
git clone https://github.com/Counterweight-AI/clawrouter.git --depth 1
cd clawrouter

# 2. Install the plugin (link mode — points at your local copy)
openclaw plugins install -l ./openclaw-plugin

# 3. Restart the gateway to start the service
openclaw gateway restart
```

That's it. On first start, the service will automatically:
1. Clone the repo into `~/.openclaw/litellm/`
2. Create a Python 3.10+ venv and install LiteLLM
3. Read your existing API keys from OpenClaw auth profiles
4. Pick the best tier models based on your available keys
5. Generate `proxy_config.yaml` and `routing_rules.yaml`
6. Start the proxy on port 4000
7. Add `clawrouter/auto` as a provider in your OpenClaw config

Verify it works:

```bash
curl http://localhost:4000/health/liveliness    # → "I'm alive!"
```

Test routing:

```bash
# Simple → LOW tier
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-clawrouter" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'

# Coding → MID tier
# ... "write a Python function" ...

# Reasoning → TOP tier
# ... "prove that sqrt(2) is irrational" ...

# Force a tier with [low], [med], or [high] prefix
# ... "[high] hello" ...
```

## Configuration

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

This walks you through API key setup, tier model selection, and starts the proxy on port 4000.
