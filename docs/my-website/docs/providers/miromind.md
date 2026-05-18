import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MiroMind

## Overview

| Property | Details |
|-------|-------|
| Description | MiroMind operates the [MiroThinker](https://miromind.ai) deep research model family — agent models that iteratively plan, search the web, fetch pages, and synthesize a final report. Exposed through an OpenAI-compatible Responses API. |
| Provider Route on LiteLLM | `miromind/` |
| Link to Provider Doc | [MiroMind API Documentation ↗](https://platform.miromind.ai/docs/responses-api) |
| Base URL | `https://api.miromind.ai/v1` |
| Supported Operations | [`/v1/responses`](#sample-usage), `/v1/chat/completions` |

<br />
<br />

MiroThinker models are **deep research agents** — they always run an internal planning loop with built-in `google_search` and URL-fetch tools. The Responses API is the recommended endpoint because it surfaces the full lifecycle (reasoning items, `web_search_call` events, final message) as typed SSE events. Chat Completions is supported as a thinner fallback that exposes only the final answer plus token usage.

## Available Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `miromind/mirothinker-1-7-deepresearch` | MiroThinker 1.7 flagship deep research agent | 262,144 tokens |
| `miromind/mirothinker-1-7-deepresearch-mini` | Smaller / faster variant | 262,144 tokens |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["MIROMIND_API_KEY"] = ""  # your MiroMind API key
```

## Usage - LiteLLM Python SDK

### Streaming Responses (recommended)

```python showLineNumbers title="MiroMind Deep Research — streaming"
import os
import asyncio
import litellm

os.environ["MIROMIND_API_KEY"] = ""

async def main():
    stream = await litellm.aresponses(
        model="miromind/mirothinker-1-7-deepresearch-mini",
        input="Find the latest news about LiteLLM and summarize in 3 bullets with sources.",
        stream=True,
    )
    async for event in stream:
        print(event.type, getattr(event, "delta", ""))

asyncio.run(main())
```

### Non-streaming Responses

```python showLineNumbers title="MiroMind Deep Research — non-streaming"
import os
import litellm

os.environ["MIROMIND_API_KEY"] = ""

resp = litellm.responses(
    model="miromind/mirothinker-1-7-deepresearch-mini",
    input="What were the most-cited papers on alignment in 2025?",
)
print(resp)
```

### Chat Completions (fallback)

```python showLineNumbers title="MiroMind via /v1/chat/completions"
import os
from litellm import completion

os.environ["MIROMIND_API_KEY"] = ""

response = completion(
    model="miromind/mirothinker-1-7-deepresearch-mini",
    messages=[{"role": "user", "content": "Summarize recent advances in retrieval-augmented generation."}],
)
print(response)
```

## Usage - LiteLLM Proxy Server

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: mirothinker
    litellm_params:
      model: miromind/mirothinker-1-7-deepresearch
      api_key: os.environ/MIROMIND_API_KEY
  - model_name: mirothinker-mini
    litellm_params:
      model: miromind/mirothinker-1-7-deepresearch-mini
      api_key: os.environ/MIROMIND_API_KEY
```

## Custom API Base

```python showLineNumbers title="Custom API Base"
import os
os.environ["MIROMIND_API_BASE"] = "https://api.miromind.ai/v1"
os.environ["MIROMIND_API_KEY"] = ""
```

Or pass `api_base=` directly on the call.

## Responses API event shape

MiroThinker emits OpenAI Responses-shaped SSE events. Key things to know when integrating:

- **Reasoning** is emitted as raw chain-of-thought via `response.reasoning_text.delta` / `.done` (GPT-OSS-style), not `response.reasoning_summary_text.*`.
- **Built-in tools** surface as `web_search_call` output items with `action.type`:
  - `google_search` → `action.type = "search"` (with `action.query`)
  - URL fetch / page extraction → `action.type = "open_page"` (with `action.url`)
- **Custom events** (`response.agent_summary.*`) carry the model's planning-phase scratch text and can be safely ignored — clients listening only to `response.output_text.*` will see the final answer correctly.

See [MiroMind Responses API docs](https://platform.miromind.ai/docs/responses-api) for the full event catalog.

## Supported OpenAI Parameters

- `temperature`
- `top_p`
- `max_output_tokens`
- `stream`
- `tools` (built-in tools are managed server-side; client-supplied tool definitions are accepted but executed by MiroMind's agent loop)
- `instructions`
