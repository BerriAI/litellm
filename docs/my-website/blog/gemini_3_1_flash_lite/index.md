---
slug: gemini_3_1_flash_lite_preview
title: "DAY 0 Support: Gemini 3.1 Flash Lite Preview on LiteLLM"
date: 2026-03-03T08:00:00
authors:
  - sameer
  - krrish
  - ishaan-alt
description: "Guide to using Gemini 3.1 Flash Lite Preview on LiteLLM Proxy and SDK with day 0 support."
tags: [gemini, day 0 support, llms, supernova]
hide_table_of_contents: false
---


import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Gemini 3.1 Flash Lite Preview Day 0 Support 

LiteLLM now supports `gemini-3.1-flash-lite-preview` with full day 0 support!

:::note
If you only want cost tracking, you need no change in your current Litellm version. But if you want the support for new features introduced along with it like thinking levels, you will need to use v1.80.8-stable.1 or above.
:::

{/* truncate */}

## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:main-v1.80.8-stable.1
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==v1.80.8-stable.1
```

</TabItem>
</Tabs>

## What's New

Supports all four thinking levels:
- **MINIMAL**: Ultra-fast responses with minimal reasoning
- **LOW**: Simple instruction following
- **MEDIUM**: Balanced reasoning for complex tasks
- **HIGH**: Maximum reasoning depth (dynamic)

---

## Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

**Basic Usage**

```python
from litellm import completion

response = completion(
    model="gemini/gemini-3.1-flash-lite-preview",
    messages=[{"role": "user", "content": "Extract key entities from this text: ..."}],
)

print(response.choices[0].message.content)
```

**With Thinking Levels**

```python
from litellm import completion

# Use MEDIUM thinking for complex reasoning tasks
response = completion(
    model="gemini/gemini-3.1-flash-lite-preview",
    messages=[{"role": "user", "content": "Analyze this dataset and identify patterns"}],
    reasoning_effort="medium",  # low, medium , high
)

print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="proxy" label="PROXY">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: gemini-3.1-flash-lite
    litellm_params:
      model: gemini/gemini-3.1-flash-lite-preview
      api_key: os.environ/GEMINI_API_KEY
  
  # Or use Vertex AI
  - model_name: vertex-gemini-3.1-flash-lite
    litellm_params:
      model: vertex_ai/gemini-3.1-flash-lite-preview
      vertex_project: your-project-id
      vertex_location: us-central1
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml
```

**3. Make requests**

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "gemini-3.1-flash-lite",
    "messages": [{"role": "user", "content": "Extract structured data from this text"}],
    "reasoning_effort": "low"
  }'
```

</TabItem>
</Tabs>

---

## Supported Endpoints

LiteLLM provides **full end-to-end support** for Gemini 3.1 Flash Lite Preview on:

- ✅ `/v1/chat/completions` - OpenAI-compatible chat completions endpoint
- ✅ `/v1/responses` - OpenAI Responses API endpoint (streaming and non-streaming)
- ✅ [`/v1/messages`](../../docs/anthropic_unified) - Anthropic-compatible messages endpoint
- ✅ `/v1/generateContent` – [Google Gemini API](../../docs/generateContent) compatible endpoint 

All endpoints support:
- Streaming and non-streaming responses
- Function calling with thought signatures
- Multi-turn conversations
- All Gemini 3-specific features (thinking levels, thought signatures)
- Full multimodal support (text, image, audio, video)

---

## `reasoning_effort` Mapping for Gemini 3.1

LiteLLM automatically maps OpenAI's `reasoning_effort` parameter to Gemini's `thinkingLevel`:

| reasoning_effort | thinking_level | Use Case |
|------------------|----------------|----------|
| `minimal` | `minimal` | Ultra-fast responses, simple queries |
| `low` | `low` | Basic instruction following |
| `medium` | `medium` | Balanced reasoning for moderate complexity |
| `high` | `high` | Maximum reasoning depth, complex problems |
| `disable` | `minimal` | Disable extended reasoning |
| `none` | `minimal` | No extended reasoning |
