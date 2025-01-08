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












