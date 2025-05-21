---
title: v1.63.2-stable
slug: v1.63.2-stable
date: 2025-03-08T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGiM7ZrUwqu_Q/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1675971026692?e=1741824000&v=beta&t=eQnRdXPJo4eiINWTZARoYTfqh064pgZ-E21pQTSy8jc
tags: [llm translation, thinking, reasoning_content, claude-3-7-sonnet]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';


These are the changes since `v1.61.20-stable`.

This release is primarily focused on:
- LLM Translation improvements (more `thinking` content improvements)
- UI improvements (Error logs now shown on UI)


:::info

This release will be live on 03/09/2025

::: 

<Image img={require('../../img/release_notes/v1632_release.jpg')} />


## Demo Instance

Here's a Demo Instance to test changes:
- Instance: https://demo.litellm.ai/
- Login Credentials:
    - Username: admin
    - Password: sk-1234


## New Models / Updated Models

1. Add `supports_pdf_input` for specific Bedrock Claude models [PR](https://github.com/BerriAI/litellm/commit/f63cf0030679fe1a43d03fb196e815a0f28dae92)
2. Add pricing for amazon `eu` models [PR](https://github.com/BerriAI/litellm/commits/main/model_prices_and_context_window.json)
3. Fix Azure O1 mini pricing [PR](https://github.com/BerriAI/litellm/commit/52de1949ef2f76b8572df751f9c868a016d4832c)

## LLM Translation

<Image img={require('../../img/release_notes/anthropic_thinking.jpg')}/>

1. Support `/openai/` passthrough for Assistant endpoints. [Get Started](https://docs.litellm.ai/docs/pass_through/openai_passthrough)
2. Bedrock Claude - fix tool calling transformation on invoke route. [Get Started](../../docs/providers/bedrock#usage---function-calling--tool-calling)
3. Bedrock Claude - response_format support for claude on invoke route. [Get Started](../../docs/providers/bedrock#usage---structured-output--json-mode)
4. Bedrock - pass `description` if set in response_format. [Get Started](../../docs/providers/bedrock#usage---structured-output--json-mode)
5. Bedrock - Fix passing response_format: {"type": "text"}. [PR](https://github.com/BerriAI/litellm/commit/c84b489d5897755139aa7d4e9e54727ebe0fa540)
6. OpenAI - Handle sending image_url as str to openai. [Get Started](https://docs.litellm.ai/docs/completion/vision)
7. Deepseek - return 'reasoning_content' missing on streaming. [Get Started](https://docs.litellm.ai/docs/reasoning_content)
8. Caching - Support caching on reasoning content. [Get Started](https://docs.litellm.ai/docs/proxy/caching)
9. Bedrock - handle thinking blocks in assistant message. [Get Started](https://docs.litellm.ai/docs/providers/bedrock#usage---thinking--reasoning-content)
10. Anthropic - Return `signature` on streaming. [Get Started](https://docs.litellm.ai/docs/providers/bedrock#usage---thinking--reasoning-content)
- Note: We've also migrated from `signature_delta` to `signature`. [Read more](https://docs.litellm.ai/release_notes/v1.63.0)
11. Support format param for specifying image type. [Get Started](../../docs/completion/vision.md#explicitly-specify-image-type)
12. Anthropic - `/v1/messages` endpoint - `thinking` param support. [Get Started](../../docs/anthropic_unified.md)
- Note: this refactors the [BETA] unified `/v1/messages` endpoint, to just work for the Anthropic API. 
13. Vertex AI - handle $id in response schema when calling vertex ai. [Get Started](https://docs.litellm.ai/docs/providers/vertex#json-schema)

## Spend Tracking Improvements

1. Batches API - Fix cost calculation to run on retrieve_batch. [Get Started](https://docs.litellm.ai/docs/batches)
2. Batches API - Log batch models in spend logs / standard logging payload. [Get Started](../../docs/proxy/logging_spec.md#standardlogginghiddenparams)

## Management Endpoints / UI

<Image img={require('../../img/release_notes/error_logs.jpg')} />

1. Virtual Keys Page
    - Allow team/org filters to be searchable on the Create Key Page
    - Add created_by and updated_by fields to Keys table
    - Show 'user_email' on key table
    - Show 100 Keys Per Page, Use full height, increase width of key alias
2. Logs Page
    - Show Error Logs on LiteLLM UI
    - Allow Internal Users to View their own logs
3. Internal Users Page 
    - Allow admin to control default model access for internal users
7. Fix session handling with cookies

## Logging / Guardrail Integrations

1. Fix prometheus metrics w/ custom metrics, when keys containing team_id make requests. [PR](https://github.com/BerriAI/litellm/pull/8935)

## Performance / Loadbalancing / Reliability improvements

1. Cooldowns - Support cooldowns on models called with client side credentials. [Get Started](https://docs.litellm.ai/docs/proxy/clientside_auth#pass-user-llm-api-keys--api-base)
2. Tag-based Routing - ensures tag-based routing across all endpoints (`/embeddings`, `/image_generation`, etc.). [Get Started](https://docs.litellm.ai/docs/proxy/tag_routing)

## General Proxy Improvements

1. Raise BadRequestError when unknown model passed in request
2. Enforce model access restrictions on Azure OpenAI proxy route
3. Reliability fix - Handle emoji’s in text - fix orjson error
4. Model Access Patch - don't overwrite litellm.anthropic_models when running auth checks
5. Enable setting timezone information in docker image 

## Complete Git Diff

[Here's the complete git diff](https://github.com/BerriAI/litellm/compare/v1.61.20-stable...v1.63.2-stable)