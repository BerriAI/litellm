---
slug: claude_opus_4_8
title: "Day 0 Support: Claude Opus 4.8"
date: 2026-05-28T10:00:00
authors:
  - mateo
  - krrish
  - ishaan-alt
description: "Day 0 support for Claude Opus 4.8 on the LiteLLM AI Gateway. Use it across Anthropic, Azure, Vertex AI, and Bedrock."
tags: [anthropic, claude, opus 4.8, day 0 support]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

LiteLLM now supports [Claude Opus 4.8](https://www.anthropic.com/news/claude-opus-4-8) on Day 0. Use it across Anthropic, Azure, Vertex AI, and Bedrock through the LiteLLM AI Gateway. Call it with the same OpenAI-compatible request you already use, and track spend, rate limits, and logging in one place.

{/* truncate */}

## What's new in Opus 4.8

Opus 4.8 builds on Opus 4.7 with gains across coding, agentic, and reasoning benchmarks, and ships at the **same price**. A few things stand out for teams running it through a gateway:

- **A sharper, more honest agent.** Anthropic reports Opus 4.8 is roughly **4× less likely** than Opus 4.7 to let flaws in code it wrote pass unremarked, and more likely to flag uncertainty than make unsupported claims. That reliability compounds when the model is driving multi-step tool calls behind your proxy. ([details from Anthropic](https://www.anthropic.com/news/claude-opus-4-8))
- **The full effort ladder, per request.** `low`, `medium`, `high` (default), `xhigh`, and `max`. Dial reasoning *up* for hard, long-running agentic work or *down* for fast, cheap responses. Set it per call via `reasoning_effort` or `output_config`.
- **Mid-task system messages.** The Messages API now accepts `system` entries *inside* the `messages` array, so an agent can update its instructions, permissions, or token budget mid-run without breaking the prompt cache, and it flows straight through LiteLLM's `/v1/messages` passthrough.
- **Same per-token price as Opus 4.7.** $5 / MTok input and $25 / MTok output, with prompt caching at $0.50 / MTok (read) and $6.25 / MTok (write). Better results, no price change.
- **1M-token context**, up to 128K output tokens.
- **One gateway, every surface.** Vision, PDF input, computer use, tool calling, prompt caching, adaptive thinking, and structured output, all available across Anthropic, Azure, Vertex AI, and Bedrock with unified spend tracking, logging, and fallbacks.

## Enabling Opus 4.8

Opus 4.8 ships in the nightly **`v1.88.0-dev.1`** image (and every release after it). How you pick it up depends on where your proxy reads pricing from:

- **Default (remote cost map): no upgrade needed.** In the LiteLLM UI, open the **Price Data** tab under **Models + Endpoints** and click **Reload Price Data** (or, as a proxy admin, `POST /reload/model_cost_map`). This refetches the latest pricing from LiteLLM's cost map **and** re-registers provider routing in one step, so `claude-opus-4-8` becomes available across Anthropic, Azure, Vertex AI, and Bedrock, even if you're on an older proxy version.
- **Running `LITELLM_LOCAL_MODEL_COST_MAP=true`?** The cost map is baked into the image, so the Reload button won't reach it. Pull `v1.88.0-dev.1` or later to get the bundled Opus 4.8 metadata:

  ```bash
  docker pull ghcr.io/berriai/litellm:v1.88.0-dev.1
  ```

## Usage - Anthropic

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-opus-4-8
    litellm_params:
      model: anthropic/claude-opus-4-8
      api_key: os.environ/ANTHROPIC_API_KEY
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/config.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:v1.88.0-dev.1 \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-8",
  "messages": [
    {
      "role": "user",
      "content": "what llm are you"
    }
  ]
}'
```

</TabItem>
</Tabs>

## Usage - Azure

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-opus-4-8
    litellm_params:
      model: azure_ai/claude-opus-4-8
      api_key: os.environ/AZURE_AI_API_KEY
      api_base: os.environ/AZURE_AI_API_BASE  # https://<resource>.services.ai.azure.com
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e AZURE_AI_API_KEY=$AZURE_AI_API_KEY \
  -e AZURE_AI_API_BASE=$AZURE_AI_API_BASE \
  -v $(pwd)/config.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:v1.88.0-dev.1 \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-8",
  "messages": [
    {
      "role": "user",
      "content": "what llm are you"
    }
  ]
}'
```

</TabItem>
</Tabs>

## Usage - Vertex AI

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-opus-4-8
    litellm_params:
      model: vertex_ai/claude-opus-4-8
      vertex_project: os.environ/VERTEX_PROJECT
      vertex_location: us-east5
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e VERTEX_PROJECT=$VERTEX_PROJECT \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/credentials.json:/app/credentials.json \
  ghcr.io/berriai/litellm:v1.88.0-dev.1 \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-8",
  "messages": [
    {
      "role": "user",
      "content": "what llm are you"
    }
  ]
}'
```

</TabItem>
</Tabs>

## Usage - Bedrock

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-opus-4-8
    litellm_params:
      model: bedrock/anthropic.claude-opus-4-8
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -v $(pwd)/config.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:v1.88.0-dev.1 \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-8",
  "messages": [
    {
      "role": "user",
      "content": "what llm are you"
    }
  ]
}'
```

</TabItem>
</Tabs>

## Advanced Features

### Adaptive Thinking

:::note
When using `reasoning_effort` with Claude Opus 4.8, all values (`low`, `medium`, `high`, `xhigh`, `max`) are mapped to `thinking: {type: "adaptive"}`. Opus 4.8 only supports adaptive thinking; explicit budgets via `thinking: {type: "enabled", budget_tokens: ...}` are rejected by the Anthropic API with a 400 error. To control thinking depth, pair adaptive thinking with `output_config.effort` (see [Effort Levels](#effort-levels) below) rather than a fixed budget.
:::

<Tabs>
<TabItem value="completions" label="/chat/completions">

LiteLLM supports adaptive thinking through the `reasoning_effort` parameter:

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-8",
  "messages": [
    {
      "role": "user",
      "content": "Solve this complex problem: What is the optimal strategy for..."
    }
  ],
  "reasoning_effort": "high"
}'
```

</TabItem>
<TabItem value="messages" label="/v1/messages">

Use the `thinking` parameter with `type: "adaptive"` to enable adaptive thinking mode:

```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
--header 'x-api-key: sk-12345' \
--header 'content-type: application/json' \
--data '{
    "model": "claude-opus-4-8",
    "max_tokens": 16000,
    "thinking": {
        "type": "adaptive"
    },
    "messages": [
        {
            "role": "user",
            "content": "Explain why the sum of two even numbers is always even."
        }
    ]
}'
```

</TabItem>
</Tabs>

### Effort Levels

Claude Opus 4.8 supports five effort levels: `low`, `medium`, `high` (default), `xhigh`, and `max`. These give you finer-grained control over how much reasoning the model applies to a task. Pass the effort level via the `output_config` parameter.

Opus 4.8 supports the full effort ladder. Both `xhigh` (introduced with Opus 4.7) and `max` (also available on Opus 4.6 and 4.7) are available.

<Tabs>
<TabItem value="completions" label="/chat/completions">

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-8",
  "messages": [
    {
      "role": "user",
      "content": "Explain quantum computing"
    }
  ],
  "output_config": {
    "effort": "max"
  }
}'
```

**Using OpenAI SDK:**

```python
import openai

client = openai.OpenAI(
    api_key="your-litellm-key",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="claude-opus-4-8",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    extra_body={"output_config": {"effort": "max"}}
)
```

**Using LiteLLM SDK:**

```python
from litellm import completion

response = completion(
    model="anthropic/claude-opus-4-8",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    output_config={"effort": "max"},
)
```

You can combine `reasoning_effort` with `output_config` for even more fine-grained control over the model's behavior.

</TabItem>
<TabItem value="messages" label="/v1/messages">

```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
--header 'x-api-key: sk-12345' \
--header 'content-type: application/json' \
--data '{
    "model": "claude-opus-4-8",
    "max_tokens": 4096,
    "messages": [
        {
            "role": "user",
            "content": "Explain quantum computing"
        }
    ],
    "output_config": {
        "effort": "max"
    }
}'
```

</TabItem>
</Tabs>

**Effort level guide:**

| Effort | When to use |
|--------|-------------|
| `low` | Short, fast responses for simple lookups, formatting, and classification |
| `medium` | Balanced tradeoff for everyday Q&A and light reasoning |
| `high` (default) | Complex reasoning, code generation, analysis |
| `xhigh` | Hard problems like multi-step math, deep research, and agentic planning |
| `max` | The hardest tasks where you want maximum reasoning depth regardless of latency |
