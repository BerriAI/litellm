---
title: "v1.84.1 - Gemini 3.5 Flash & Reliability Fixes"
slug: "v1-84-1"
date: 2026-05-20T00:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
  - name: Yuneng Jiang
    title: Senior Full Stack Engineer, LiteLLM
    url: https://www.linkedin.com/in/yuneng-david-jiang-455676139/
    image_url: https://avatars.githubusercontent.com/u/171294688?v=4
hide_table_of_contents: false
---

## Deploy this version

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="docker" label="Docker">

```bash
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
docker.litellm.ai/berriai/litellm:1.84.1
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.84.1
```

</TabItem>
</Tabs>

`v1.84.1` is a patch release on top of [`v1.84.0`](/release_notes/v1.84.0/v1-84-0). It adds day-0 support for Gemini 3.5 Flash and ships two reliability fixes — cross-pod spend accuracy and Vertex AI tool calling.

## New Models / Updated Models

#### New Model Support (1 new model)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| --- | --- | --- | --- | --- | --- |
| Gemini / Vertex AI | `gemini/gemini-3.5-flash`, `vertex_ai/gemini-3.5-flash` | 1M | $1.50 | $9.00 | Reasoning, vision, audio input, PDF input, prompt caching, web search, function calling, response schema |

#### Features

- **[Gemini](../../docs/providers/gemini)** / **[Vertex AI](../../docs/providers/vertex)**
    - Day-0 support for Gemini 3.5 Flash on both Google AI Studio and Vertex AI - [PR #28268](https://github.com/BerriAI/litellm/pull/28268)

### Bug Fixes

- **[Vertex AI](../../docs/providers/vertex)**
    - Omit the `function_call` / `function_response` `id` on Vertex Gemini 3.5+ tool turns, fixing HTTP 400 `Unknown name "id"` errors. Google AI Studio (`gemini` provider) still forwards the `id` on Gemini 3.5+ for strict tool-call matching - [PR #28324](https://github.com/BerriAI/litellm/pull/28324)

## Spend Tracking, Budgets and Rate Limiting

- Seed the Redis spend counter via `SET NX` instead of `INCRBYFLOAT` to prevent cross-pod double-seeding. On multi-pod deployments this previously caused team `spend` to jump to ~Nx the pod count after a Redis cache miss / TTL expiry, triggering false "Budget Crossed" alerts - [PR #27854](https://github.com/BerriAI/litellm/pull/27854)

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.84.0...v1.84.1
