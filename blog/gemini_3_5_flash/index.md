---
slug: gemini_3_5_flash
title: "DAY 0 Support: Gemini 3.5 Flash on LiteLLM"
date: 2026-05-19T10:00:00
authors:
  - sameer
  - krrish
  - ishaan-alt
description: "Guide to using Gemini 3.5 Flash on LiteLLM Proxy and SDK with day 0 support."
tags: [gemini, day 0 support, llms]
hide_table_of_contents: false
---


import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Gemini 3.5 Flash Day 0 Support 

LiteLLM now supports `gemini-3.5-flash` with full day 0 support!

:::note
If you only want cost tracking, you need no change in your current LiteLLM version. But if you want support for new features introduced with this release — thinking levels, strict function-call IDs, and thought signatures — use `v1.87.0-dev.1` or above.
:::

{/* truncate */}

## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:v1.87.0-dev.1
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.87.0.dev1
```

</TabItem>
</Tabs>

## What's New

### 1. Minimal thinking level

Gemini 3.5 Flash supports the new "Minimal" level. LiteLLM maps OpenAI `reasoning_effort` to Gemini's `thinkingLevel` — use `reasoning_effort="minimal"`.
<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

response = completion(
    model="gemini/gemini-3.5-flash",
    messages=[{"role": "user", "content": "What's 2+2?"}],
    reasoning_effort="minimal",
)

print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="proxy" label="PROXY">

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "gemini-3.5-flash",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "reasoning_effort": "minimal"
  }'
```

</TabItem>
</Tabs>

| `reasoning_effort` | `thinkingLevel` |
|--------------------|-----------------|
| `minimal` | `minimal` |

### 2. Strict function calling

Gemini 3.5+ requires every `functionResponse` to include the same `id` as the originating `functionCall`, plus the matching function name. LiteLLM round-trips this through standard OpenAI fields: `tool_calls[].id` on the assistant message, and the same value as `tool_call_id` on the tool result.

**How the tool-call loop works**

**Step 1 : User submits a query that would trigger a tool call**

Send the user message and your tool definitions. The model responds with `tool_calls` — save the **`id`** from the first tool call (it may look like `5x450f94__thought__<signature>`; pass it back unchanged on the next request).

```bash
curl -sS http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "gemini-3.5-flash",
    "messages": [
      {
        "role": "user",
        "content": "What is the weather in Tokyo right now?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": { "type": "string" }
            },
            "required": ["city"]
          }
        }
      }
    ]
  }' | tee /tmp/gemini_tool_step1.json | jq .
```

Copy the tool call id from the response:

```bash
TOOL_CALL_ID=$(jq -r '.choices[0].message.tool_calls[0].id' /tmp/gemini_tool_step1.json)
echo "$TOOL_CALL_ID"
# e.g. 5x450f94__thought__EvACCu0CAQw51sdR...
```

**Step 2 : Run your tool, then send the result with the same `tool_call_id`**

Run `get_weather` locally, then call the proxy again with the full message history. Set **`tool_call_id`** to the exact **`id`** from Step 1 — LiteLLM uses it as the Gemini `functionResponse.id`.

```bash
# Result from your local get_weather("Tokyo") call
WEATHER_RESULT='{"temp_c": 18, "condition": "clear"}'

curl -sS http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d "$(jq -n \
    --arg id "$TOOL_CALL_ID" \
    --arg content "$WEATHER_RESULT" \
    '{
      model: "gemini-3.5-flash",
      messages: [
        {role: "user", content: "What is the weather in Tokyo right now?"},
        {
          role: "assistant",
          content: null,
          tool_calls: [{
            id: $id,
            type: "function",
            function: {name: "get_weather", arguments: "{\"city\": \"Tokyo\"}"}
          }]
        },
        {role: "tool", tool_call_id: $id, content: $content}
      ],
      tools: [{
        type: "function",
        function: {
          name: "get_weather",
          description: "Get current weather for a city",
          parameters: {
            type: "object",
            properties: {city: {type: "string"}},
            required: ["city"]
          }
        }
      }]
    }')" | jq .
```

The **`id`** on the assistant `tool_calls` entry and the **`tool_call_id`** on the `role: tool` message must match. The function **name** must match the tool definition (`get_weather`).

**Step 3 : Model produces the final answer**

LiteLLM sends the matching `id` and `name` on the Gemini `functionResponse` part. The model then returns a normal assistant message with the weather summary.

### 3. Sampling parameters (`temperature`, `top_p`, `top_k`)

Google has advised moving away from `temperature`, `top_p`, and `top_k` for Gemini 3.5+ and steering sampling behavior through **system instructions** instead. These parameters still work today, but may be removed in a future API release.

LiteLLM follows the same guidance: when you pass `temperature`, `top_p`, or `top_k` on Gemini 3+ models, you will see a deprecation warning in the logs recommending system-instruction-based sampling instead.

## Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

response = completion(
    model="gemini/gemini-3.5-flash",
    messages=[{"role": "user", "content": "Summarize this article in 3 bullet points."}],
)

print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="proxy" label="PROXY">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: gemini-3.5-flash
    litellm_params:
      model: gemini/gemini-3.5-flash
      api_key: os.environ/GEMINI_API_KEY

  # Or use Vertex AI
  - model_name: vertex-gemini-3.5-flash
    litellm_params:
      model: vertex_ai/gemini-3.5-flash
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
    "model": "gemini-3.5-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

</TabItem>
</Tabs>

## Supported Endpoints

LiteLLM provides **full end-to-end support** for Gemini 3.5 Flash on:

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
