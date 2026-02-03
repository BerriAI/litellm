---
slug: gemini_3_flash
title: "DAY 0 Support: Gemini 3 Flash on LiteLLM"
date: 2025-12-17T10:00:00
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
description: "Guide to using Gemini 3 Flash on LiteLLM Proxy and SDK with day 0 support."
tags: [gemini, day 0 support, llms]
hide_table_of_contents: false
---


import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Gemini 3 Flash Day 0 Support 

LiteLLM now supports `gemini-3-flash-preview` and all the new API changes along with it.

:::note
If you only want cost tracking, you need no change in your current Litellm version. But if you want the support for new features introduced along with it like thinking levels, you will need to use v1.80.8-stable.1 or above.
:::

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
pip install litellm==1.80.8.post1
```

</TabItem>
</Tabs>

## What's New

### 1. New Thinking Levels: `thinkingLevel` with MINIMAL & MEDIUM

Gemini 3 Flash introduces granular thinking control with `thinkingLevel` instead of `thinkingBudget`.
- **MINIMAL**: Ultra-lightweight thinking for fast responses
- **MEDIUM**: Balanced thinking for complex reasoning  
- **HIGH**: Maximum reasoning depth

LiteLLM automatically maps the OpenAI `reasoning_effort` parameter to Gemini's `thinkingLevel`, so you can use familiar `reasoning_effort` values (`minimal`, `low`, `medium`, `high`) without changing your code!

### 2. Thought Signatures

Like `gemini-3-pro`, this model also includes thought signatures for tool calls. LiteLLM handles signature extraction and embedding internally. [Learn more about thought signatures](../gemini_3/index.md#thought-signatures).

**Edge Case Handling**: If thought signatures are missing in the request, LiteLLM adds a dummy signature ensuring the API call doesn't break

---
## Supported Endpoints

LiteLLM provides **full end-to-end support** for Gemini 3 Flash on:

- ✅ `/v1/chat/completions` - OpenAI-compatible chat completions endpoint
- ✅ `/v1/responses` - OpenAI Responses API endpoint (streaming and non-streaming)
- ✅ [`/v1/messages`](../../docs/anthropic_unified) - Anthropic-compatible messages endpoint
- ✅ `/v1/generateContent` – [Google Gemini API](../../docs/generateContent.md) compatible endpoint 
All endpoints support:
- Streaming and non-streaming responses
- Function calling with thought signatures
- Multi-turn conversations
- All Gemini 3-specific features
- Converstion of provider specific thinking related param to thinkingLevel

## Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

**Basic Usage with MEDIUM thinking (NEW)**

```python
from litellm import completion

# No need to make any changes to your code as we map openai reasoning param to thinkingLevel
response = completion(
    model="gemini/gemini-3-flash-preview",
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
  - model_name: gemini-3-flash
    litellm_params:
      model: gemini/gemini-3-flash-preview
      api_key: os.environ/GEMINI_API_KEY
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
    "model": "gemini-3-flash",
    "messages": [{"role": "user", "content": "Complex reasoning task"}],
    "reasoning_effort": "medium"
  }'
``'

</TabItem>
</Tabs>

---

## All `reasoning_effort` Levels

<Tabs>
<TabItem value="minimal" label="MINIMAL">

**Ultra-fast, minimal reasoning**

```python
from litellm import completion

response = completion(
    model="gemini/gemini-3-flash-preview",
    messages=[{"role": "user", "content": "What's 2+2?"}],
    reasoning_effort="minimal",
)
```

</TabItem>

<TabItem value="low" label="LOW">

**Simple instruction following**

```python
response = completion(
    model="gemini/gemini-3-flash-preview",
    messages=[{"role": "user", "content": "Write a haiku about coding"}],
    reasoning_effort="low",
)
```

</TabItem>

<TabItem value="medium" label="MEDIUM (NEW)">

**Balanced reasoning for complex tasks** ✨

```python
response = completion(
    model="gemini/gemini-3-flash-preview",
    messages=[{"role": "user", "content": "Analyze this dataset and find patterns"}],
    reasoning_effort="medium",  # NEW!
)
```

</TabItem>

<TabItem value="high" label="HIGH">

**Maximum reasoning depth**

```python
response = completion(
    model="gemini/gemini-3-flash-preview",
    messages=[{"role": "user", "content": "Prove this mathematical theorem"}],
    reasoning_effort="high",
)
```

</TabItem>
</Tabs>

---

## Key Features

✅ **Thinking Levels**: MINIMAL, LOW, MEDIUM, HIGH  
✅ **Thought Signatures**: Track reasoning with unique identifiers  
✅ **Seamless Integration**: Works with existing OpenAI-compatible client  
✅ **Backward Compatible**: Gemini 2.5 models continue using `thinkingBudget`  

---

## Installation

```bash
pip install litellm --upgrade
```

```python
import litellm
from litellm import completion

response = completion(
    model="gemini/gemini-3-flash-preview",
    messages=[{"role": "user", "content": "Your question here"}],
    reasoning_effort="medium",  # Use MEDIUM thinking
)
print(response)
```

:::note
If using this model via vertex_ai, keep the location as global as this is the only supported location as of now.
:::


## `reasoning_effort` Mapping for Gemini 3+

| reasoning_effort | thinking_level | 
|------------------|----------------|
| `minimal` | `minimal` |
| `low` | `low` |
| `medium` | `medium` |
| `high` | `high` |
| `disable` | `minimal` |
| `none` | `minimal` |

