---
slug: claude_opus_4_6
title: "Day 0 Support: Claude Opus 4.6"
date: 2026-02-05T10:00:00
authors:
  - name: Sameer Kankute
    title: SWE @ LiteLLM (LLM Translation)
    url: https://www.linkedin.com/in/sameer-kankute/
    image_url: https://pbs.twimg.com/profile_images/2001352686994907136/ONgNuSk5_400x400.jpg
  - name: Ishaan Jaff
    title: "CTO, LiteLLM"
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
  - name: Krrish Dholakia
    title: "CEO, LiteLLM"
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
description: "Day 0 support for Claude Opus 4.6 on LiteLLM AI Gateway - use across Anthropic, Azure, Vertex AI, and Bedrock."
tags: [anthropic, claude, opus 4.6]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

LiteLLM now supports Claude Opus 4.6 on Day 0. Use it across Anthropic, Azure, Vertex AI, and Bedrock through the LiteLLM AI Gateway.

## Docker Image

```bash
docker pull ghcr.io/berriai/litellm:litellm_stable_release_branch-v1.80.0-stable.opus-4-6
```

## Usage - Anthropic

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-opus-4-6
    litellm_params:
      model: anthropic/claude-opus-4-6
      api_key: os.environ/ANTHROPIC_API_KEY
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/config.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:litellm_stable_release_branch-v1.80.0-stable.opus-4-6 \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-6",
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
  - model_name: claude-opus-4-6
    litellm_params:
      model: azure_ai/claude-opus-4-6
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
  ghcr.io/berriai/litellm:litellm_stable_release_branch-v1.80.0-stable.opus-4-6 \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-6",
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
  - model_name: claude-opus-4-6
    litellm_params:
      model: vertex_ai/claude-opus-4-6
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
  ghcr.io/berriai/litellm:litellm_stable_release_branch-v1.80.0-stable.opus-4-6 \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-6",
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
  - model_name: claude-opus-4-6
    litellm_params:
      model: bedrock/anthropic.claude-opus-4-6-v1
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
  ghcr.io/berriai/litellm:litellm_stable_release_branch-v1.80.0-stable.opus-4-6 \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-6",
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

### Compaction

<Tabs>
<TabItem value="completions" label="/chat/completions">

Litellm supports enabling compaction for the new claude-opus-4-6.

**Enabling Compaction**

To enable compaction, add the `context_management` parameter with the `compact_20260112` edit type:

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-6",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather in San Francisco?"
    }
  ],
  "context_management": {
    "edits": [
      {
        "type": "compact_20260112"
      }
    ]
  },
  "max_tokens": 100
}'
```
All the parameters supported for context_management by anthropic are supported and can be directly added. Litellm automatically adds the `compact-2026-01-12` beta header in the request.

</TabItem>
<TabItem value="messages" label="/v1/messages">

Enable compaction to reduce context size while preserving key information. LiteLLM automatically adds the `compact-2026-01-12` beta header when compaction is enabled.

:::info
**Provider Support:** Compaction is supported on Anthropic, Azure AI, and Vertex AI. It is **not supported** on Bedrock (Invoke or Converse APIs).
:::

```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
--header 'x-api-key: sk-12345' \
--header 'content-type: application/json' \
--data '{
    "model": "claude-opus-4-6",
    "max_tokens": 4096,
    "messages": [
        {
            "role": "user",
            "content": "Hi"
        }
    ],
    "context_management": {
        "edits": [
            {
                "type": "compact_20260112"
            }
        ]
    }
}'
```

</TabItem>
</Tabs>


**Response with Compaction Block**

The response will include the compaction summary in `provider_specific_fields.compaction_blocks`:

```json
{
  "id": "chatcmpl-a6c105a3-4b25-419e-9551-c800633b6cb2",
  "created": 1770357619,
  "model": "claude-opus-4-6",
  "object": "chat.completion",
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "message": {
        "content": "I don't have access to real-time data, so I can't provide the current weather in San Francisco. To get up-to-date weather information, I'd recommend checking:\n\n- **Weather websites** like weather.com, accuweather.com, or wunderground.com\n- **Search engines** – just Google \"San Francisco weather\"\n- **Weather apps** on your phone (e.g., Apple Weather, Google Weather)\n- **National",
        "role": "assistant",
        "provider_specific_fields": {
          "compaction_blocks": [
            {
              "type": "compaction",
              "content": "Summary of the conversation: The user requested help building a web scraper..."
            }
          ]
        }
      }
    }
  ],
  "usage": {
    "completion_tokens": 100,
    "prompt_tokens": 86,
    "total_tokens": 186
  }
}
```

**Using Compaction Blocks in Follow-up Requests**

To continue the conversation with compaction, include the compaction block in the assistant message's `provider_specific_fields`:

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-6",
  "messages": [
    {
      "role": "user",
      "content": "How can I build a web scraper?"
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "Certainly! To build a basic web scraper, you'll typically use a programming language like Python along with libraries such as `requests` (for fetching web pages) and `BeautifulSoup` (for parsing HTML). Here's a basic example:\n\n```python\nimport requests\nfrom bs4 import BeautifulSoup\n\nurl = 'https://example.com'\nresponse = requests.get(url)\nsoup = BeautifulSoup(response.text, 'html.parser')\n\n# Extract and print all text\ntext = soup.get_text()\nprint(text)\n```\n\nLet me know what you're interested in scraping or if you need help with a specific website!"
        }
      ],
      "provider_specific_fields": {
        "compaction_blocks": [
          {
            "type": "compaction",
            "content": "Summary of the conversation: The user asked how to build a web scraper, and the assistant gave an overview using Python with requests and BeautifulSoup."
          }
        ]
      }
    },
    {
      "role": "user",
      "content": "How do I use it to scrape product prices?"
    }
  ],
  "context_management": {
    "edits": [
      {
        "type": "compact_20260112"
      }
    ]
  },
  "max_tokens": 100
}'
```

**Streaming Support**

Compaction blocks are also supported in streaming mode. You'll receive:
- `compaction_start` event when a compaction block begins
- `compaction_delta` events with the compaction content
- The accumulated `compaction_blocks` in `provider_specific_fields`

### Adaptive Thinking

:::note
When using `reasoning_effort` with Claude Opus 4.6, all values (`low`, `medium`, `high`) are mapped to `thinking: {type: "adaptive"}`. To use explicit thinking budgets with `type: "enabled"`, pass the native `thinking` parameter directly (see "Native thinking param" tab below).
:::

<Tabs>
<TabItem value="completions" label="/chat/completions">

LiteLLM supports adaptive thinking through the `reasoning_effort` parameter:

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-6",
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
    "model": "claude-opus-4-6",
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
<TabItem value="native" label="Native thinking param">

Use the `thinking` parameter directly for adaptive thinking via the SDK:

```python
import litellm

response = litellm.completion(
  model="anthropic/claude-opus-4-6",
  messages=[{"role": "user", "content": "Solve this complex problem: What is the optimal strategy for..."}],
  thinking={"type": "adaptive"},
)
```

</TabItem>
</Tabs>

### Effort Levels

<Tabs>
<TabItem value="completions" label="/chat/completions">

Four effort levels available: `low`, `medium`, `high` (default), and `max`. Pass directly via the `output_config` parameter:

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-6",
  "messages": [
    {
      "role": "user",
      "content": "Explain quantum computing"
    }
  ],
  "output_config": {
        "effort": "medium"
    }
}'
```

You can use reasoning effort plus output_config to have more control on the model.

</TabItem>
<TabItem value="messages" label="/v1/messages">

Four effort levels available: `low`, `medium`, `high` (default), and `max`. Pass directly via the `output_config` parameter:

```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
--header 'x-api-key: sk-12345' \
--header 'content-type: application/json' \
--data '{
    "model": "claude-opus-4-6",
    "max_tokens": 4096,
    "messages": [
        {
            "role": "user",
            "content": "Explain quantum computing"
        }
    ],
    "output_config": {
        "effort": "medium"
    }
}'
```

</TabItem>
</Tabs>

### 1M Token Context (Beta)

Opus 4.6 supports 1M token context. Premium pricing applies for prompts exceeding 200k tokens ($10/$37.50 per million input/output tokens). LiteLLM supports cost calculations for 1M token contexts.

<Tabs>
<TabItem value="completions" label="/chat/completions">

To use the 1M token context window, you need to forward the `anthropic-beta` header from your client to the LLM provider.

**Step 1: Enable header forwarding in your config**

```yaml
general_settings:
  forward_client_headers_to_llm_api: true
```

**Step 2: Send requests with the beta header**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--header 'anthropic-beta: context-1m-2025-08-07' \
--data '{
  "model": "claude-opus-4-6",
  "messages": [
    {
      "role": "user",
      "content": "Analyze this large document..."
    }
  ]
}'
```

</TabItem>
<TabItem value="messages" label="/v1/messages">

To use the 1M token context window, you need to forward the `anthropic-beta` header from your client to the LLM provider.

**Step 1: Enable header forwarding in your config**

```yaml
general_settings:
  forward_client_headers_to_llm_api: true
```

**Step 2: Send requests with the beta header**

```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
--header 'x-api-key: sk-12345' \
--header 'anthropic-beta: context-1m-2025-08-07' \
--header 'content-type: application/json' \
--data '{
    "model": "claude-opus-4-6",
    "max_tokens": 16000,
    "messages": [
        {
            "role": "user",
            "content": "Analyze this large document..."
        }
    ]
}'
```

:::tip
You can combine multiple beta headers by separating them with commas:
```bash
--header 'anthropic-beta: context-1m-2025-08-07,compact-2026-01-12'
```
:::

</TabItem>
</Tabs>

### US-Only Inference

Available at 1.1× token pricing. LiteLLM automatically tracks costs for US-only inference.

<Tabs>
<TabItem value="completions" label="/chat/completions">

Use the `inference_geo` parameter to specify US-only inference:

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-6",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "inference_geo": "us"
}'
```

LiteLLM will automatically apply the 1.1× pricing multiplier for US-only inference in cost tracking.

</TabItem>
<TabItem value="messages" label="/v1/messages">

Use the `inference_geo` parameter to specify US-only inference:

```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
--header 'x-api-key: sk-12345' \
--header 'content-type: application/json' \
--data '{
    "model": "claude-opus-4-6",
    "max_tokens": 4096,
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "inference_geo": "us"
}'
```

LiteLLM will automatically apply the 1.1× pricing multiplier for US-only inference in cost tracking.

</TabItem>
</Tabs>

### Fast Mode

:::info
Fast mode is **only supported on the Anthropic provider** (`anthropic/claude-opus-4-6`). It is not available on Azure AI, Vertex AI, or Bedrock.
:::

**Pricing:**
- Standard: $5 input / $25 output per MTok
- Fast: $30 input / $150 output per MTok (6× premium)

<Tabs>
<TabItem value="completions" label="/chat/completions">

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-opus-4-6",
  "messages": [
    {
      "role": "user",
      "content": "Refactor this module..."
    }
  ],
  "max_tokens": 4096,
  "speed": "fast"
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
    model="claude-opus-4-6",
    messages=[{"role": "user", "content": "Refactor this module..."}],
    max_tokens=4096,
    extra_body={"speed": "fast"}
)
```

**Using LiteLLM SDK:**

```python
from litellm import completion

response = completion(
    model="anthropic/claude-opus-4-6",
    messages=[{"role": "user", "content": "Refactor this module..."}],
    max_tokens=4096,
    speed="fast"
)
```

LiteLLM automatically tracks the higher costs for fast mode in usage and cost calculations.

</TabItem>
<TabItem value="messages" label="/v1/messages">

```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
--header 'x-api-key: sk-12345' \
--header 'content-type: application/json' \
--data '{
    "model": "claude-opus-4-6",
    "max_tokens": 4096,
    "speed": "fast",
    "messages": [
        {
            "role": "user",
            "content": "Refactor this module..."
        }
    ]
}'
```

LiteLLM automatically:
- Adds the `fast-mode-2026-02-01` beta header
- Tracks the 6× premium pricing in cost calculations

</TabItem>
</Tabs>
