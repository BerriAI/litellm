---
title: v1.63.14-stable
slug: v1.63.14-stable
date: 2025-03-22T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1743638400&v=beta&t=39KOXMUFedvukiWWVPHf3qI45fuQD7lNglICwN31DrI
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg

tags: [credential management, thinking content, responses api, snowflake]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

These are the changes since `v1.63.11-stable`.

This release brings:
- LLM Translation Improvements (MCP Support and Bedrock Application Profiles)
- Perf improvements for Usage-based Routing
- Streaming guardrail support via websockets

## Docker Run LiteLLM Proxy

```
docker run
-e STORE_MODEL_IN_DB=True
-p 4000:4000
ghcr.io/berriai/litellm:main-v1.63.14-stable
```

## Demo Instance

Here's a Demo Instance to test changes:
- Instance: https://demo.litellm.ai/
- Login Credentials:
    - Username: admin
    - Password: sk-1234



## New Models / Updated Models

- Azure gpt-4o - fixed pricing to latest global pricing - [PR](https://github.com/BerriAI/litellm/pull/9361)
- O1-Pro - add pricing + model information - [PR](https://github.com/BerriAI/litellm/pull/9397)
- Azure AI - mistral 3.1 small pricing added - [PR](https://github.com/BerriAI/litellm/pull/9453)
- Azure - gpt-4.5-preview pricing added - [PR](https://github.com/BerriAI/litellm/pull/9453)



## LLM Translation

1. **New LLM Features**

- Bedrock: Support bedrock application inference profiles [Docs](https://docs.litellm.ai/docs/providers/bedrock#bedrock-application-inference-profile)
   - Infer aws region from bedrock application profile id - (`arn:aws:bedrock:us-east-1:...`)
- Ollama - support calling via `/v1/completions` - [NEEDS DOCS]https://github.com/BerriAI/litellm/pull/9333
- Bedrock - support `us.deepseek.r1-v1:0` model name [Docs](../../docs/providers/bedrock#supported-aws-bedrock-models)

https://github.com/BerriAI/litellm/pull/9363
- OpenRouter - `OPENROUTER_API_BASE` env var support [Docs](../../docs/providers/openrouter.md)
- Azure - add audio model parameter support - [Docs](../../docs/providers/azure#azure-audio-model)
- OpenAI - PDF File support [Docs](../../docs/completion/document_understanding#openai-file-message-type)
- OpenAI - o1-pro Responses API streaming support [Docs](../../docs/response_api.md#streaming)
- [BETA] MCP - Use MCP Tools with LiteLLM SDK [Docs](../../docs/mcp)

2. **Bug Fixes**

- Voyage: prompt token on embedding tracking fix - [PR](https://github.com/BerriAI/litellm/commit/56d3e75b330c3c3862dc6e1c51c1210e48f1068e)
- Streaming - Prevents final chunk w/ usage from being ignored (impacted bedrock streaming + cost tracking) - [PR](https://github.com/BerriAI/litellm/commit/dd2c980d5bb9e1a3b125e364c5d841751e67c96d)
- Sagemaker - Fix ‘Too little data for declared Content-Length’ error - [PR](https://github.com/BerriAI/litellm/pull/9326)
- OpenAI-compatible models - fix issue when calling openai-compatible models w/ custom_llm_provider set - [PR](https://github.com/BerriAI/litellm/pull/9355)
- VertexAI - Embedding ‘outputDimensionality’ support - [PR](https://github.com/BerriAI/litellm/commit/437dbe724620675295f298164a076cbd8019d304)
- Anthropic - return consistent json response format on streaming/non-streaming - [PR](https://github.com/BerriAI/litellm/pull/9437)

## Spend Tracking Improvements

- `litellm_proxy/` - support reading litellm response cost header from proxy, when using client sdk 
- Reset Budget Job - fix budget reset error on keys/teams/users - [PR](https://github.com/BerriAI/litellm/pull/9329)

## UI

1. Users Page
   - Feature: Control default internal user settings [PR](https://github.com/BerriAI/litellm/pull/9374)
2. Icons:
   - Feature: Replace external "artificialanalysis.ai" icons by local svg [PR](https://github.com/BerriAI/litellm/pull/9374)
3. Sign In/Sign Out
   - Fix: Default login when `default_user_id` user does not exist in DB [PR](https://github.com/BerriAI/litellm/pull/9395)


## Logging Integrations

- Support post-call guardrails for streaming responses - https://github.com/BerriAI/litellm/commit/4a31b32a88b7729a032e58ab046079d17000087f [NEEDS DOCS] 
- Arize [Get Started](../../docs/observability/arize_integration)
   - fix invalid package import - [PR](https://github.com/BerriAI/litellm/pull/9338)
   - migrate to using standardloggingpayload for metadata, ensures spans land successfully - [PR](https://github.com/BerriAI/litellm/pull/9338)
   - fix logging to just log the LLM I/O - [PR](https://github.com/BerriAI/litellm/pull/9353)
   - Dynamic API Key/Space param support - [Get Started](../../docs/observability/arize_integration#pass-arize-spacekey-per-request)
- StandardLoggingPayload - Log litellm_model_name in payload. Allows knowing what the model sent to API provider was - [Get Started](../../docs/proxy/logging_spec#standardlogginghiddenparams)
- Prompt Management - Allow building custom prompt management integration - https://github.com/BerriAI/litellm/pull/9384

## Performance / Reliability improvements

- Redis Caching - add 5s default timeout, prevents hanging redis connection from impacting llm calls - https://github.com/BerriAI/litellm/commit/db92956ae33ed4c4e3233d7e1b0c7229817159bf
- Allow disabling all spend updates / writes to DB - patch to allow disabling all spend updates to DB with a flag - https://github.com/BerriAI/litellm/pull/9331
- Azure OpenAI - correctly re-use azure openai client - https://github.com/BerriAI/litellm/commit/f2026ef907c06d94440930917add71314b901413
- Azure OpenAI - uses litellm.ssl_verify on Azure/OpenAI clients 
- Usage-based routing - Wildcard model support https://github.com/BerriAI/litellm/pull/9355
- Usage-based routing - Support batch writing increments to redis - reduces latency to same as ‘simple-shuffle’ - https://github.com/BerriAI/litellm/pull/9357
- Router - show reason for model cooldown on ‘no healthy deployments available error’ - https://github.com/BerriAI/litellm/pull/9438
- Caching - add max value limit to an item in in-memory cache (1MB) - prevents OOM errors on large image url’s being sent through proxy - https://github.com/BerriAI/litellm/pull/9448


## General Improvements

- Multiple OIDC Provider support - https://github.com/BerriAI/litellm/commit/324864b7750747ae40345def796c1578263f5896
- Passthrough Endpoints - support returning api-base on pass-through endpoints Response Headers [Docs](../../docs/proxy/response_headers#litellm-specific-headers)
- SSL - support reading ssl security level from env var - Allows user to specify lower security settings - https://github.com/BerriAI/litellm/pull/9330
- Credentials - only poll Credentials table when `STORE_MODEL_IN_DB` is True - https://github.com/BerriAI/litellm/pull/9376
- Image URL Handling - new architecture doc on image url handling - https://docs.litellm.ai/docs/proxy/image_handling
- OpenAI - bump to pip install "openai==1.68.2" - https://github.com/BerriAI/litellm/commit/e85e3bc52a9de86ad85c3dbb12d87664ee567a5a
- Gunicorn - security fix - bump gunicorn==23.0.0 # server dep



## Complete Git Diff

[Here's the complete git diff](https://github.com/BerriAI/litellm/compare/v1.63.11-stable...v1.63.14.rc)