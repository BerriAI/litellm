---
title: "v1.77.2-stable - Bedrock Batches API"
slug: "v1-77-2"
date: 2025-09-13T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg

hide_table_of_contents: false
---

import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:main-v1.77.2-stable
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.77.2.post1
```

</TabItem>
</Tabs>

---

## Key Highlights

- **Bedrock Batches API** - Support for creating Batch Inference Jobs on Bedrock using LiteLLM's unified batch API (OpenAI compatible)
- **Qwen API Tiered Pricing** - Cost tracking support for Dashscope (Qwen) models with multiple pricing tiers

## New Models / Updated Models

#### New Model Support

| Provider    | Model                           | Context Window | Pricing ($/1M tokens) | Features |
| ----------- | ------------------------------- | -------------- | --------------------- | -------- |
| DeepInfra   | `deepinfra/deepseek-ai/DeepSeek-R1` | 164K | **Input:** $0.70<br/>**Output:** $2.40 | Chat completions, tool calling |
| Heroku      | `heroku/claude-4-sonnet`        | 8K | Contact provider for pricing | Function calling, tool choice |
| Heroku      | `heroku/claude-3-7-sonnet`      | 8K | Contact provider for pricing | Function calling, tool choice |
| Heroku      | `heroku/claude-3-5-sonnet-latest` | 8K | Contact provider for pricing | Function calling, tool choice |
| Heroku      | `heroku/claude-3-5-haiku`       | 4K | Contact provider for pricing | Function calling, tool choice |
| Dashscope   | `dashscope/qwen-plus-latest`    | 1M | **Tiered Pricing:**<br/>• 0-256K tokens: $0.40 / $1.20<br/>• 256K-1M tokens: $1.20 / $3.60 | Function calling, reasoning |
| Dashscope   | `dashscope/qwen3-max-preview`   | 262K | **Tiered Pricing:**<br/>• 0-32K tokens: $1.20 / $6.00<br/>• 32K-128K tokens: $2.40 / $12.00<br/>• 128K-252K tokens: $3.00 / $15.00 | Function calling, reasoning |
| Dashscope   | `dashscope/qwen-flash`          | 1M | **Tiered Pricing:**<br/>• 0-256K tokens: $0.05 / $0.40<br/>• 256K-1M tokens: $0.25 / $2.00 | Function calling, reasoning |
| Dashscope   | `dashscope/qwen3-coder-plus`    | 1M | **Tiered Pricing:**<br/>• 0-32K tokens: $1.00 / $5.00<br/>• 32K-128K tokens: $1.80 / $9.00<br/>• 128K-256K tokens: $3.00 / $15.00<br/>• 256K-1M tokens: $6.00 / $60.00 | Function calling, reasoning, caching |
| Dashscope   | `dashscope/qwen3-coder-flash`   | 1M | **Tiered Pricing:**<br/>• 0-32K tokens: $0.30 / $1.50<br/>• 32K-128K tokens: $0.50 / $2.50<br/>• 128K-256K tokens: $0.80 / $4.00<br/>• 256K-1M tokens: $1.60 / $9.60 | Function calling, reasoning, caching |

---

#### Features

- **[Bedrock](../../docs/providers/bedrock_batches)**
    - Bedrock Batches API - batch processing support with file upload and request transformation - [PR #14518](https://github.com/BerriAI/litellm/pull/14518), [PR #14522](https://github.com/BerriAI/litellm/pull/14522)
- **[VLLM](../../docs/providers/vllm)**
    - Added transcription endpoint support - [PR #14523](https://github.com/BerriAI/litellm/pull/14523)
- **[Ollama](../../docs/providers/ollama)**
    - `ollama_chat/` - images, thinking, and content as list handling - [PR #14523](https://github.com/BerriAI/litellm/pull/14523)
- **General**
    - New debug flag for detailed request/response logging [PR #14482](https://github.com/BerriAI/litellm/pull/14482)

#### Bug Fixes

- **[Azure OpenAI](../../docs/providers/azure)**
    - Fixed extra_body injection causing payload rejection in image generation - [PR #14475](https://github.com/BerriAI/litellm/pull/14475)
- **[LM Studio](../../docs/providers/lm-studio)**
    - Resolved illegal Bearer header value issue - [PR #14512](https://github.com/BerriAI/litellm/pull/14512)

---

## LLM API Endpoints

#### Bug Fixes

- **[/messages](../../docs/anthropic_unified)**
    - Don't send content block after message w/ finish reason + usage block - [PR #14477](https://github.com/BerriAI/litellm/pull/14477)
- **[/generateContent](../../docs/generateContent)**
    - Gemini CLI Integration - Fixed token count errors - [PR #14451](https://github.com/BerriAI/litellm/pull/14451), [PR #14417](https://github.com/BerriAI/litellm/pull/14417)

---

## Spend Tracking, Budgets and Rate Limiting

#### Features

- **[Qwen API Tiered Pricing](../../docs/providers/dashscope)** - Added comprehensive tiered cost tracking for Dashscope/Qwen models - [PR #14471](https://github.com/BerriAI/litellm/pull/14471), [PR #14479](https://github.com/BerriAI/litellm/pull/14479)

#### Bug Fixes

- **Provider Budgets** - Fixed provider budget calculations - [PR #14459](https://github.com/BerriAI/litellm/pull/14459)

---

## Management Endpoints / UI

#### Features

- **User Headers Mapping** - New X-LiteLLM Users mapping feature for enhanced user tracking - [PR #14485](https://github.com/BerriAI/litellm/pull/14485)
- **Key Unblocking** - Support for hashed tokens in `/key/unblock` endpoint - [PR #14477](https://github.com/BerriAI/litellm/pull/14477)
- **Model Group Header Forwarding** - Enhanced wildcard model support with documentation - [PR #14528](https://github.com/BerriAI/litellm/pull/14528)

#### Bug Fixes

- **Log Tab Key Alias** - Fixed filtering inaccuracies for failed logs - [PR #14469](https://github.com/BerriAI/litellm/pull/14469), [PR #14529](https://github.com/BerriAI/litellm/pull/14529)

---

## Logging / Guardrail Integrations

#### Features

- **Noma Integration** - Added non-blocking monitor mode with anonymize input support - [PR #14401](https://github.com/BerriAI/litellm/pull/14401)

---

## Performance / Loadbalancing / Reliability improvements

#### Performance
- Removed dynamic creation of static values - [PR #14538](https://github.com/BerriAI/litellm/pull/14538)
- Using `_PROXY_MaxParallelRequestsHandler_v3` by default for optimal throughput - [PR #14450](https://github.com/BerriAI/litellm/pull/14450)
- Improved execution context propagation into logging tasks - [PR #14455](https://github.com/BerriAI/litellm/pull/14455)

---



## New Contributors
* @Sameerlite made their first contribution in [PR #14460](https://github.com/BerriAI/litellm/pull/14460)
* @holzman made their first contribution in [PR #14459](https://github.com/BerriAI/litellm/pull/14459)
* @sashank5644 made their first contribution in [PR #14469](https://github.com/BerriAI/litellm/pull/14469)
* @TomAlon made their first contribution in [PR #14401](https://github.com/BerriAI/litellm/pull/14401)
* @AlexsanderHamir made their first contribution in [PR #14538](https://github.com/BerriAI/litellm/pull/14538)

---

## **[Full Changelog](https://github.com/BerriAI/litellm/compare/v1.77.1.dev.2...v1.77.2.dev)**
