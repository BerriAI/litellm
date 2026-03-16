---
slug: gpt_5_4
title: "Day 0 Support: GPT-5.4"
date: 2026-03-05T10:00:00
authors:
  - sameer
  - krrish
  - ishaan-alt
description: "GPT-5.4 model support in LiteLLM"
tags: [openai, gpt-5.4, completion]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

LiteLLM now supports fully GPT-5.4!

## Docker Image

```bash
docker pull ghcr.io/berriai/litellm:v1.81.14-stable.gpt-5.4_patch
```

## Usage

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: gpt-5.4
    litellm_params:
      model: openai/gpt-5.4
      api_key: os.environ/OPENAI_API_KEY
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/config.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:v1.81.14-stable.gpt-5.4_patch \
  --config /app/config.yaml
```

**3. Test it**

```bash
curl -X POST "http://0.0.0.0:4000/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "gpt-5.4",
    "messages": [
      {"role": "user", "content": "Write a Python function to check if a number is prime."}
    ]
  }'
```

</TabItem>
<TabItem value="sdk" label="LiteLLM SDK">

```python
from litellm import completion

response = completion(
    model="openai/gpt-5.4",
    messages=[
        {"role": "user", "content": "Write a Python function to check if a number is prime."}
    ],
)

print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

## Notes

- Restart your container to get the cost tracking for this model.
- Use `/responses` for better model performance.
- GPT-5.4 supports reasoning, function calling, vision, and tool-use — see the [OpenAI provider docs](../../docs/providers/openai) for advanced usage.
