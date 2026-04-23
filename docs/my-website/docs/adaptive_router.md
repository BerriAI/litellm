# [BETA] Adaptive Router

:::info

Beta feature. Share feedback on [Discord](https://discord.gg/wuPM9dRgDw) or [Slack](https://join.slack.com/t/litellmossslack/shared_invite/zt-3o7nkuyfr-p_kbNJj8taRfXGgQI1~YyA).

:::

**Requirements:** LiteLLM Proxy with a Postgres database. Quality estimates are stored in Postgres and loaded on startup — without a database the router works but forgets everything learned on restart.

You have a cheap model and an expensive one. You want to use the cheap one when it's good enough, and the expensive one when it actually matters — without hardcoding rules you'll spend months tuning.

The adaptive router does this automatically. It tracks which model performs best for each type of request (code, writing, analysis, etc.) and routes accordingly, balancing quality against cost based on weights you control.

## Quick start

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
    model_info:
      input_cost_per_token: 0.0000025
      adaptive_router_preferences:
        quality_tier: 3        # 1=budget, 2=mid, 3=frontier
        strengths: ["code_generation", "analytical_reasoning"]

  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
    model_info:
      input_cost_per_token: 0.00000015
      adaptive_router_preferences:
        quality_tier: 2
        strengths: ["factual_lookup"]

  - model_name: my-router
    litellm_params:
      model: auto_router/adaptive_router
      adaptive_router_config:
        available_models: ["gpt-4o", "gpt-4o-mini"]
        weights:
          quality: 0.7   # raise this if quality complaints; lower if bill too high
          cost: 0.3      # must sum to 1.0 with quality
```

Route to it by setting `model` to your adaptive router's name:

```bash
curl -X POST {{baseURL}}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "my-router",
    "messages": [
      {"role": "user", "content": "build me a python script that parses CSV"},
      {"role": "assistant", "content": "Here is a script using csv.DictReader..."},
      {"role": "user", "content": "now add error handling for missing files"},
      {"role": "assistant", "content": "Wrap the open() call in a try/except FileNotFoundError..."},
      {"role": "user", "content": "perfect, that worked. thanks!"}
    ]
  }'
```

The response includes a header telling you which model was actually picked:

```
x-litellm-adaptive-router-model: gpt-4o
```

The "thanks!" turn in the example above fires a satisfaction signal — that's what moves the bandit.

## Tuning cost vs. quality

The `weights` are your main lever:

| Goal | quality | cost |
|---|---|---|
| Minimize cost, quality is secondary | 0.3 | 0.7 |
| Balanced | 0.5 | 0.5 |
| Quality-first (default) | 0.7 | 0.3 |
| Quality non-negotiable | 0.9 | 0.1 |

The router learns over time. For the first ~10 requests per model, it relies on the tiers you declared. After that, real performance data takes over.

## Force a minimum quality tier per request

If a specific request needs a frontier model regardless of cost, pass this header:

```
x-litellm-min-quality-tier: 3
```

You can also pass `min_quality_tier` via request metadata instead of a header.

## What's being learned

The router classifies each request into one of 7 types and tracks how each model performs on each independently. A model that's great at factual lookup but poor at code will win factual requests and lose code requests — even if it's cheaper overall.

| Type | Example |
|---|---|
| `code_generation` | "write me a Python sort function" |
| `code_understanding` | "explain what this function does" |
| `technical_design` | "how should I design this API?" |
| `analytical_reasoning` | "calculate the probability that..." |
| `writing` | "draft an email to my team about..." |
| `factual_lookup` | "what is the capital of France?" |
| `general` | anything else |

[**See classifier code**](https://github.com/BerriAI/litellm/blob/litellm_adaptive_routing/litellm/router_strategy/adaptive_router/classifier.py)

Learning signals are inspired by [Signals: Trajectory Sampling and Triage for Agentic Interactions](https://arxiv.org/pdf/2604.00356).

## Inspect the current state

```
GET /adaptive_router/{router_name}/state
```

Returns current quality estimates per model per request type. Useful for understanding why a model is or isn't being picked.

```json
{
  "routers": [
    {
      "router_name": "smart-cheap-router",
      "available_models": ["fast", "smart"],
      "weights": { "quality": 0.7, "cost": 0.3 },
      "cells": [
        {
          "request_type": "analytical_reasoning",
          "model": "fast",
          "quality_mean": 0.5,
          "samples": 0
        },
        {
          "request_type": "analytical_reasoning",
          "model": "smart",
          "quality_mean": 0.95,
          "samples": 0
        }
      ]
    }
  ]
}
```

`quality_mean` is the key number — it's the router's current estimate of how well that model handles that request type. `samples` counts how many real observations have moved the prior (starts at 0; the cold-start prior mass is excluded).

## Known limitations

- Latency isn't scored — a slow model can still win on quality + cost
- Signals are regex-based and English-biased — no LLM judge
- Hard cap of 200 observations per cell; no decay yet
- Once a model is picked for a session, other models' turns in that session don't contribute to learning
