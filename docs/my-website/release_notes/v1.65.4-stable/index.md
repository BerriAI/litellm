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
docker.litellm.ai/berriai/litellm:main-v1.65.4-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.65.4.post1
```
</TabItem>
</Tabs>

v1.65.4-stable is live. Here are the improvements since v1.65.0-stable.

## Key Highlights
- **Preventing DB Deadlocks**: Fixes a high-traffic issue when multiple instances were writing to the DB at the same time. 
- **New Usage Tab**: Enables viewing spend by model and customizing date range

Let's dive in. 

### Preventing DB Deadlocks

<Image img={require('../../img/prevent_deadlocks.jpg')} />

This release fixes the DB deadlocking issue that users faced in high traffic (10K+ RPS). This is great because it enables user/key/team spend tracking works at that scale.

Read more about the new architecture [here](https://docs.litellm.ai/docs/proxy/db_deadlocks)


### New Usage Tab

<Image img={require('../../img/release_notes/spend_by_model.jpg')} />

The new Usage tab now brings the ability to track daily spend by model. This makes it easier to catch any spend tracking or token counting errors, when combined with the ability to view successful requests, and token usage.

To test this out, just go to Experimental > New Usage > Activity.


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
13. Google AI Studio (Gemini) - [BETA] `/v1/files` upload support [Get Started](../../docs/providers/google_ai_studio/files) 
14. Azure - fix o-series tool calling [Get Started](../../docs/providers/azure#tool-calling--function-calling)
15. Unified file id - [ALPHA] allow calling multiple providers with same file id [PR](https://github.com/BerriAI/litellm/pull/9718)
    - This is experimental, and not recommended for production use.
    - We plan to have a production-ready implementation by next week.
16. Google AI Studio (Gemini) - return logprobs [PR](https://github.com/BerriAI/litellm/pull/9713)
17. Anthropic - Support prompt caching for Anthropic tool calls [Get Started](https://docs.litellm.ai/docs/completion/prompt_caching)
18. OpenRouter - unwrap extra body on open router calls [PR](https://github.com/BerriAI/litellm/pull/9747)
19. VertexAI - fix credential caching issue [PR](https://github.com/BerriAI/litellm/pull/9756)
20. XAI - filter out 'name' param for XAI [PR](https://github.com/BerriAI/litellm/pull/9761)
21. Gemini - image generation output support [Get Started](../../docs/providers/gemini#image-generation)
22. Databricks - support claude-3-7-sonnet w/ thinking + response_format [Get Started](../../docs/providers/databricks#usage---thinking--reasoning_content)

## Spend Tracking Improvements
1. Reliability fix  - Check sent and received model for cost calculation [PR](https://github.com/BerriAI/litellm/pull/9669)
2. Vertex AI - Multimodal embedding cost tracking [Get Started](https://docs.litellm.ai/docs/providers/vertex#multi-modal-embeddings), [PR](https://github.com/BerriAI/litellm/pull/9623)

## Management Endpoints / UI

<Image img={require('../../img/release_notes/new_activity_tab.png')} />

1. New Usage Tab
    - Report 'total_tokens' + report success/failure calls
    - Remove double bars on scroll
    - Ensure ‘daily spend’ chart ordered from earliest to latest date
    - showing spend per model per day
    - show key alias on usage tab
    - Allow non-admins to view their activity
    - Add date picker to new usage tab
2. Virtual Keys Tab
    - remove 'default key' on user signup
    - fix showing user models available for personal key creation
3. Test Key Tab
    - Allow testing image generation models
4. Models Tab
    - Fix bulk adding models 
    - support reusable credentials for passthrough endpoints
    - Allow team members to see team models
5. Teams Tab
    - Fix json serialization error on update team metadata
6. Request Logs Tab
    - Add reasoning_content token tracking across all providers on streaming
7. API 
    - return key alias on /user/daily/activity [Get Started](../../docs/proxy/cost_tracking#daily-spend-breakdown-api)
8. SSO
    - Allow assigning SSO users to teams on MSFT SSO [PR](https://github.com/BerriAI/litellm/pull/9745)

## Logging / Guardrail Integrations

1. Console Logs - Add json formatting for uncaught exceptions [PR](https://github.com/BerriAI/litellm/pull/9619)
2. Guardrails - AIM Guardrails support for virtual key based policies [Get Started](../../docs/proxy/guardrails/aim_security)
3. Logging - fix completion start time tracking [PR](https://github.com/BerriAI/litellm/pull/9688)
4. Prometheus
    - Allow adding authentication on Prometheus /metrics endpoints [PR](https://github.com/BerriAI/litellm/pull/9766)
    - Distinguish LLM Provider Exception vs. LiteLLM Exception in metric naming [PR](https://github.com/BerriAI/litellm/pull/9760)
    - Emit operational metrics for new DB Transaction architecture [PR](https://github.com/BerriAI/litellm/pull/9719)

## Performance / Loadbalancing / Reliability improvements
1. Preventing Deadlocks
    - Reduce DB Deadlocks by storing spend updates in Redis and then committing to DB [PR](https://github.com/BerriAI/litellm/pull/9608)
    - Ensure no deadlocks occur when updating DailyUserSpendTransaction [PR](https://github.com/BerriAI/litellm/pull/9690)
    - High Traffic fix - ensure new DB + Redis architecture accurately tracks spend [PR](https://github.com/BerriAI/litellm/pull/9673)
    - Use Redis for PodLock Manager instead of PG (ensures no deadlocks occur) [PR](https://github.com/BerriAI/litellm/pull/9715)
    - v2 DB Deadlock Reduction Architecture – Add Max Size for In-Memory Queue + Backpressure Mechanism [PR](https://github.com/BerriAI/litellm/pull/9759)
    
2. Prisma Migrations [Get Started](../../docs/proxy/prod#9-use-prisma-migrate-deploy)
    - connects litellm proxy to litellm's prisma migration files
    - Handle db schema updates from new `litellm-proxy-extras` sdk
3. Redis - support password for sync sentinel clients [PR](https://github.com/BerriAI/litellm/pull/9622)
4. Fix "Circular reference detected" error when max_parallel_requests = 0 [PR](https://github.com/BerriAI/litellm/pull/9671)
5. Code QA - Ban hardcoded numbers [PR](https://github.com/BerriAI/litellm/pull/9709)

## Helm
1. fix: wrong indentation of ttlSecondsAfterFinished in chart [PR](https://github.com/BerriAI/litellm/pull/9611)

## General Proxy Improvements
1. Fix - only apply service_account_settings.enforced_params on service accounts [PR](https://github.com/BerriAI/litellm/pull/9683)
2. Fix - handle metadata null on `/chat/completion` [PR](https://github.com/BerriAI/litellm/issues/9717)
3. Fix - Move daily user transaction logging outside of 'disable_spend_logs' flag, as they’re unrelated [PR](https://github.com/BerriAI/litellm/pull/9772)

## Demo

Try this on the demo instance [today](https://docs.litellm.ai/docs/proxy/demo)

## Complete Git Diff

See the complete git diff since v1.65.0-stable, [here](https://github.com/BerriAI/litellm/releases/tag/v1.65.4-stable)

