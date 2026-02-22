---
slug: gemini_3_1_pro
title: "DAY 0 Support: Gemini 3.1 Pro on LiteLLM"
date: 2026-02-19T10:00:00
authors:
  - name: Sameer Kankute
    title: SWE @ LiteLLM (LLM Translation)
    url: https://www.linkedin.com/in/sameer-kankute/
    image_url: https://pbs.twimg.com/profile_images/2001352686994907136/ONgNuSk5_400x400.jpg
  - name: Krrish Dholakia
    title: "CEO, LiteLLM"
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: "CTO, LiteLLM"
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
description: "Guide to using Gemini 3.1 Pro on LiteLLM Proxy and SDK with day 0 support."
tags: [gemini, day 0 support, llms]
hide_table_of_contents: false
---


import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Gemini 3.1 Pro Day 0 Support 

LiteLLM now supports `gemini-3.1-pro-preview` and all the new API changes along with it.

## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:main-v1.81.9-stable.gemini.3.1-pro
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==v1.81.9-stable.gemini.3.1-pro
```

</TabItem>
</Tabs>

## What's New

### 1. New Thinking Levels: `thinkingLevel` with MINIMAL & MEDIUM

Gemini 3.1 Pro introduces support for **medium** thinking level

LiteLLM automatically maps the OpenAI `reasoning_effort` parameter to Gemini's `thinkingLevel`, so you can use familiar `reasoning_effort` values (`minimal`, `low`, `medium`, `high`) without changing your code!

---
## Supported Endpoints

LiteLLM provides **full end-to-end support** for Gemini 3.1 Pro on:

- ✅ `/v1/chat/completions` - OpenAI-compatible chat completions endpoint
- ✅ `/v1/responses` - OpenAI Responses API endpoint (streaming and non-streaming)
- ✅ [`/v1/messages`](../../docs/anthropic_unified) - Anthropic-compatible messages endpoint
- ✅ `/v1/generateContent` – [Google Gemini API](../../docs/generateContent.md) compatible endpoint 

All endpoints support:
- Streaming and non-streaming responses
- Function calling with thought signatures
- Multi-turn conversations
- All Gemini 3-specific features
- Conversion of provider specific thinking related param to thinkingLevel

## Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

**Basic Usage with MEDIUM thinking (NEW)**

```python
from litellm import completion

# No need to make any changes to your code as we map openai reasoning param to thinkingLevel
response = completion(
    model="gemini/gemini-3.1-pro-preview",
    messages=[{"role": "user", "content": "Solve this complex math problem: 25 * 4 + 10"}],
    reasoning_effort="medium",  # NEW: MEDIUM thinking level
)

print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="proxy" label="PROXY">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: gemini-3.1-pro-preview
    litellm_params:
      model: gemini/gemini-3.1-pro-preview
      api_key: os.environ/GEMINI_API_KEY
  - model_name: vertex-gemini-3.1-pro-preview
    litellm_params:
      model: vertex_ai/gemini-3.1-pro-preview
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml
```

**3. Call with MEDIUM thinking**

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "gemini-3.1-pro-preview",
    "messages": [{"role": "user", "content": "Complex reasoning task"}],
    "reasoning_effort": "medium"
  }'
```

</TabItem>
</Tabs>

---

## `reasoning_effort` Mapping for Gemini 3+

| reasoning_effort | thinking_level | 
|------------------|----------------|
| `minimal` | `minimal` |
| `low` | `low` |
| `medium` | `medium` |
| `high` | `high` |
| `disable` | `minimal` |
| `none` | `minimal` |

