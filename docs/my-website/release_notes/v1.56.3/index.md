---
title: v1.56.3
slug: v1.56.3
date: 2024-12-28T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1743638400&v=beta&t=39KOXMUFedvukiWWVPHf3qI45fuQD7lNglICwN31DrI
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGiM7ZrUwqu_Q/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1675971026692?e=1741824000&v=beta&t=eQnRdXPJo4eiINWTZARoYTfqh064pgZ-E21pQTSy8jc
tags: [guardrails, logging, virtual key management, new models]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

`guardrails`, `logging`, `virtual key management`, `new models`

:::info

Get a 7 day free trial for LiteLLM Enterprise [here](https://litellm.ai/#trial).

**no call needed**

:::

## New Features

### ✨ Log Guardrail Traces 

Track guardrail failure rate and if a guardrail is going rogue and failing requests. [Start here](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)


#### Traced Guardrail Success

<Image img={require('../../img/gd_success.png')} />

#### Traced Guardrail Failure

<Image img={require('../../img/gd_fail.png')} />


### `/guardrails/list` 

`/guardrails/list` allows clients to view available guardrails + supported guardrail params


```shell
curl -X GET 'http://0.0.0.0:4000/guardrails/list'
```

Expected response

```json
{
    "guardrails": [
        {
        "guardrail_name": "aporia-post-guard",
        "guardrail_info": {
            "params": [
            {
                "name": "toxicity_score",
                "type": "float",
                "description": "Score between 0-1 indicating content toxicity level"
            },
            {
                "name": "pii_detection",
                "type": "boolean"
            }
            ]
        }
        }
    ]
}
```


### ✨ Guardrails with Mock LLM 


Send `mock_response` to test guardrails without making an LLM call. More info on `mock_response` [here](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-npnwjPQciVRok5yNZgKmFQ" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi my email is ishaan@berri.ai"}
    ],
    "mock_response": "This is a mock response",
    "guardrails": ["aporia-pre-guard", "aporia-post-guard"]
  }'
```



### Assign Keys to Users

You can now assign keys to users via Proxy UI


<Image img={require('../../img/ui_key.png')} />

## New Models

- `openrouter/openai/o1`
- `vertex_ai/mistral-large@2411`

## Fixes 

- Fix `vertex_ai/` mistral model pricing: https://github.com/BerriAI/litellm/pull/7345
- Missing model_group field in logs for aspeech call types https://github.com/BerriAI/litellm/pull/7392