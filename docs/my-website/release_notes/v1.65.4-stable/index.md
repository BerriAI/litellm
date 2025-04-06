---
title: v1.65.4-stable
slug: v1.65.4-stable
date: 2025-04-05T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg

tags: []
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run
-e STORE_MODEL_IN_DB=True
-p 4000:4000
ghcr.io/berriai/litellm:main-v1.65.4-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.65.4.post1
```
</TabItem>
</Tabs>

## New Models / Updated Models
1. Databricks - claude-3-7-sonnet cost tracking [PR](https://github.com/BerriAI/litellm/blob/52b35cd8093b9ad833987b24f494586a1e923209/model_prices_and_context_window.json#L10350)
2. VertexAI - `gemini-2.5-pro-exp-03-25` cost tracking [PR](https://github.com/BerriAI/litellm/blob/52b35cd8093b9ad833987b24f494586a1e923209/model_prices_and_context_window.json#L4492)
3. VertexAI - `gemini-2.0-flash` cost tracking [PR](https://github.com/BerriAI/litellm/blob/52b35cd8093b9ad833987b24f494586a1e923209/model_prices_and_context_window.json#L4689)
4. Groq - add whisper ASR models to model cost map [PR](https://github.com/BerriAI/litellm/blob/52b35cd8093b9ad833987b24f494586a1e923209/model_prices_and_context_window.json#L3324)
5. IBM - Add watsonx/ibm/granite-3-8b-instruct to model cost map [PR](https://github.com/BerriAI/litellm/blob/52b35cd8093b9ad833987b24f494586a1e923209/model_prices_and_context_window.json#L91)
6. Google AI Studio - add gemini/gemini-2.5-pro-preview-03-25 to model cost map [PR](https://github.com/BerriAI/litellm/blob/52b35cd8093b9ad833987b24f494586a1e923209/model_prices_and_context_window.json#L4850)

## LLM Translation
1. Vertex AI - Support anyOf param for OpenAI json schema translation [Get Started](https://docs.litellm.ai/docs/providers/vertex#json-schema)
2. Anthropic- response_format + thinking param support  (works across Anthropic API, Bedrock, Vertex) [Get Started](https://docs.litellm.ai/docs/reasoning_content)
3. Anthropic - if thinking token is specified and max tokens is not - ensure max token to anthropic is higher than thinking tokens (works across Anthropic API, Bedrock, Vertex) [PR](https://github.com/BerriAI/litellm/pull/9594)
4. Bedrock - latency optimized inference support [Get Started](https://docs.litellm.ai/docs/providers/bedrock#usage---latency-optimized-inference)
5. Sagemaker - handle special tokens + multibyte character code in response [Get Started](https://docs.litellm.ai/docs/providers/aws_sagemaker)
6. MCP - add support for using SSE MCP servers [Get Started](https://docs.litellm.ai/docs/mcp#usage)
8. Anthropic - new `litellm.messages.create` interface for calling Anthropic `/v1/messages` via passthrough [Get Started](https://docs.litellm.ai/docs/anthropic_unified#usage)
11. Anthropic - support ‘file’ content type in message param (works across Anthropic API, Bedrock, Vertex) [Get Started](https://docs.litellm.ai/docs/providers/anthropic#usage---pdf)
12. Anthropic - map openai 'reasoning_effort' to anthropic 'thinking' param (works across Anthropic API, Bedrock, Vertex) [Get Started](https://docs.litellm.ai/docs/providers/anthropic#usage---thinking--reasoning_content)
13. Google AI Studio (Gemini) - [BETA] `/v1/files` upload support [NEEDS DOCS] 
14. Azure - fix o-series tool calling [NEEDS DOCS]
15.  Unified file id - [BETA] allow calling multiple providers with same file id [PR](https://github.com/BerriAI/litellm/pull/9718) [NEEDS DOCS]
16. Gemini - return logprobs if enabled [NEEDS DOCS]
17. Anthropic - Support prompt caching for Anthropic tool calls [NEEDS DOCS]
18. OpenRouter - unwrap extra body on open router calls [PR](https://github.com/BerriAI/litellm/pull/9747)
19. VertexAI - fix credential caching issue [PR](https://github.com/BerriAI/litellm/pull/9756)
20. XAI - filter out 'name' param for XAI [PR](https://github.com/BerriAI/litellm/pull/9761)
21. Gemini - image generation output support [NEEDS DOCS]
22. Databricks - support claude-3-7-sonnet w/ thinking + response_format [NEEDS DOCS on dbrx page]

## Spend Tracking Improvements
1. Reliability fix  - Check sent and received model for cost calculation [PR](https://github.com/BerriAI/litellm/pull/9669)
2. Vertex AI - Multimodal embedding cost tracking [Get Started](https://docs.litellm.ai/docs/providers/vertex#multi-modal-embeddings), [PR](https://github.com/BerriAI/litellm/pull/9623)

## Management Endpoints / UI
1. Key Create
    1. remove 'default key' on user signup
    2. fix showing user models available for personal key creation
2. Test Key
    1. Allow testing image generation models
3. New Usage 
    1. Report 'total_tokens' + report success/failure calls
    2. Remove double bars on scroll
    3. Ensure ‘daily spend’ chart ordered from earliest to latest date
    4. showing spend per model per day
    5. show key alias on usage tab
    6. Allow non-admins to view their activity
    7. Add date picker to new usage tab
4. Models
    1. Fix bulk adding models 
    2. support reusable credentials for passthrough endpoints
    3. Allow team members to see team models
5. Teams
    1. Fix json serialization error on update team metadata
6. Request Logs
    1. Add reasoning_content token tracking across all providers on streaming
7. API 
    1. return key alias on /user/daily/activity
8. SSO
    1. Allow assigning SSO users to teams on MSFT SSO 

## Logging / Guardrail Integrations
1. add json formatting for uncaught exceptions
2. AIM Guardrails - Support virtual key based policies
3. Logging - fix completion start time tracking
4. Prometheus - Allow adding authentication on Prometheus /metrics endpoints
5. Prometheus - Distinguish LLM Provider Exception vs. LiteLLM Exception in metric naming

## Performance / Loadbalancing / Reliability improvements
1. Preventing Deadlocks
    1. Reduce DB Deadlocks by storing spend updates in Redis and then committing to DB
    2. Ensure no deadlocks occur when updating DailyUserSpendTransaction
    3. High Traffic fix - ensure new DB + Redis architecture accurately tracks spend
    4. Use Redis for PodLock Manager instead of PG (ensures no deadlocks occur)
    5. v2 DB Deadlock Reduction Architecture – Add Max Size for In-Memory Queue + Backpressure Mechanism
    6. Emit operational metrics for new DB Transaction architecture
2. Prisma Migrations
    1. connects litellm proxy to litellm's prisma migration files
    2. Handle db schema updates from new `litellm-proxy-extras` sdk
3. Redis - support password for sync sentinel clients 
4. Fix "Circular reference detected" error when max_parallel_requests = 0 
5. Code QA - Ban hardcoded numbers

## Helm
1. fix: wrong indentation of ttlSecondsAfterFinished in chart

## General Proxy Improvements
1. Fix - only apply service_account_settings.enforced_params on service accounts
2. Fix - handle metadata null on `/chat/completion` 
3. Fix - Move daily user transaction logging outside of 'disable_spend_logs' flag, as they’re unrelated
