---
title: v1.56.1
slug: v1.56.1
date: 2024-12-27T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGiM7ZrUwqu_Q/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1675971026692?e=1741824000&v=beta&t=eQnRdXPJo4eiINWTZARoYTfqh064pgZ-E21pQTSy8jc
tags: [key management, budgets/rate limits, logging, guardrails]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

# v1.56.1

`key management`, `budgets/rate limits`, `logging`, `guardrails`

:::info

Get a 7 day free trial for LiteLLM Enterprise [here](https://litellm.ai/#trial).

**no call needed**

:::

## ✨ Budget / Rate Limit Tiers

Define tiers with rate limits. Assign them to keys. 

Use this to control access and budgets across a lot of keys.

**[Start here](https://docs.litellm.ai/docs/proxy/rate_limit_tiers)**

```bash
curl -L -X POST 'http://0.0.0.0:4000/budget/new' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "budget_id": "high-usage-tier",
    "model_max_budget": {
        "gpt-4o": {"rpm_limit": 1000000}
    }
}'
```


## OTEL Bug Fix

LiteLLM was double logging litellm_request span. This is now fixed.

[Relevant PR](https://github.com/BerriAI/litellm/pull/7435)

## Logging for Finetuning Endpoints 

Logs for finetuning requests are now available on all logging providers (e.g. Datadog). 

What's logged per request:

- file_id
- finetuning_job_id
- any key/team metadata


**Start Here:**
- [Setup Finetuning](https://docs.litellm.ai/docs/fine_tuning)
- [Setup Logging](https://docs.litellm.ai/docs/proxy/logging#datadog)

## Dynamic Params for Guardrails 

You can now set custom parameters (like success threshold) for your guardrails in each request.

[See guardrails spec for more details](https://docs.litellm.ai/docs/proxy/guardrails/custom_guardrail#-pass-additional-parameters-to-guardrail)












