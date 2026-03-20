---
slug: gpt_5_3_codex
title: "Day 0 Support: GPT-5.3-Codex"
date: 2026-02-24T10:00:00
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
description: "Day 0 support for GPT-5.3-Codex on LiteLLM, including phase parameter handling for Responses API."
tags: [openai, gpt-5.3-codex, codex, day 0 support]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

LiteLLM now supports GPT-5.3-Codex on Day 0, including support for the new assistant `phase` metadata on Responses API output items.

## Why `phase` matters for GPT-5.3-Codex

`phase` appears on assistant output items and helps distinguish preamble/commentary turns from final closeout responses.

Reference: [Phase parameter docs](https://developers.openai.com/api/reference/overview)

Supported values:
- `null`
- `"commentary"`
- `"final_answer"`

Important:
- Persist assistant output items with `phase` exactly as returned.
- Send those assistant items back on the next turn.
- Do **not** add `phase` to user messages.

## Docker Image

```bash
docker pull ghcr.io/berriai/litellm:v1.81.12-stable.gpt-5.3
```

## Usage 

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: gpt-5.3-codex
    litellm_params:
      model: openai/gpt-5.3-codex
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e ANTHROPIC_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/config.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:v1.81.12-stable.gpt-5.3 \
  --config /app/config.yaml
```


**3. Test it**

```bash
curl -X POST "http://0.0.0.0:4000/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "gpt-5.3-codex",
    "input": "Write a Python script that checks if a number is prime."
  }'
```

</TabItem>
</Tabs>

## Python Example: Persist `phase` with OpenAI Client + LiteLLM Base URL

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:4000/v1",  # LiteLLM Proxy
    api_key="your-litellm-api-key",
)

items = []  # Persist this per conversation/thread


def _item_get(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def run_turn(user_text: str):
    global items

    # User message: no phase field
    items.append(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        }
    )

    resp = client.responses.create(
        model="gpt-5.3-codex",
        input=items,
    )

    # Persist assistant output items verbatim, including phase
    for out_item in (resp.output or []):
        items.append(out_item)

    # Optional: inspect latest phase for UI/telemetry routing
    latest_phase = None
    for out_item in reversed(resp.output or []):
        if _item_get(out_item, "type") == "output_item.done" and _item_get(out_item, "phase") is not None:
            latest_phase = _item_get(out_item, "phase")
            break

    return resp, latest_phase
```

## Notes

- Use `/v1/responses` for GPT Codex models.
- Preserve full assistant output history for best multi-turn behavior.
- If `phase` metadata is dropped during history reconstruction, output quality can degrade on long-running tasks.
