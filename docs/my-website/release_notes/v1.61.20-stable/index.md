---
title: v1.61.20-stable
slug: v1.61.20-stable
date: 2025-03-01T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGiM7ZrUwqu_Q/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1675971026692?e=1741824000&v=beta&t=eQnRdXPJo4eiINWTZARoYTfqh064pgZ-E21pQTSy8jc
tags: [llm translation, rerank, ui, thinking, reasoning_content, claude-3-7-sonnet]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

# v1.61.20-stable


These are the changes since `v1.61.13-stable`.

This release is primarily focused on:
- LLM Translation improvements (claude-3-7-sonnet + 'thinking'/'reasoning_content' support)
- UI improvements (add model flow, user management, etc)

## Demo Instance

Here's a Demo Instance to test changes:
- Instance: https://demo.litellm.ai/
- Login Credentials:
    - Username: admin
    - Password: sk-1234

## New Models / Updated Models

1. Anthropic 3-7 sonnet support + cost tracking (Anthropic API + Bedrock + Vertex AI + OpenRouter) 
    1. Anthropic API [Start here](https://docs.litellm.ai/docs/providers/anthropic#usage---thinking--reasoning_content)
    2. Bedrock API [Start here](https://docs.litellm.ai/docs/providers/bedrock#usage---thinking--reasoning-content)
    3. Vertex AI API [See here](../../docs/providers/vertex#usage---thinking--reasoning_content)
    4. OpenRouter [See here](https://github.com/BerriAI/litellm/blob/ba5bdce50a0b9bc822de58c03940354f19a733ed/model_prices_and_context_window.json#L5626)
2. Gpt-4.5-preview support + cost tracking [See here](https://github.com/BerriAI/litellm/blob/ba5bdce50a0b9bc822de58c03940354f19a733ed/model_prices_and_context_window.json#L79)
3. Azure AI - Phi-4 cost tracking [See here](https://github.com/BerriAI/litellm/blob/ba5bdce50a0b9bc822de58c03940354f19a733ed/model_prices_and_context_window.json#L1773)
4. Claude-3.5-sonnet - vision support updated on Anthropic API [See here](https://github.com/BerriAI/litellm/blob/ba5bdce50a0b9bc822de58c03940354f19a733ed/model_prices_and_context_window.json#L2888)
5. Bedrock llama vision support [See here](https://github.com/BerriAI/litellm/blob/ba5bdce50a0b9bc822de58c03940354f19a733ed/model_prices_and_context_window.json#L7714)
6. Cerebras llama3.3-70b pricing [See here](https://github.com/BerriAI/litellm/blob/ba5bdce50a0b9bc822de58c03940354f19a733ed/model_prices_and_context_window.json#L2697)

## LLM Translation

1. Infinity Rerank - support returning documents when return_documents=True [Start here](../../docs/providers/infinity#usage---returning-documents)
2. Amazon Deepseek - `<think>` param extraction into ‘reasoning_content’ [Start here](https://docs.litellm.ai/docs/providers/bedrock#bedrock-imported-models-deepseek-deepseek-r1)
3. Amazon Titan Embeddings - filter out ‘aws_’ params from request body [Start here](https://docs.litellm.ai/docs/providers/bedrock#bedrock-embedding)
4. Anthropic ‘thinking’ + ‘reasoning_content’ translation support (Anthropic API, Bedrock, Vertex AI)  [Start here](https://docs.litellm.ai/docs/reasoning_content)
5. VLLM - support ‘video_url’ [Start here](../../docs/providers/vllm#send-video-url-to-vllm)
6. Call proxy via litellm SDK: Support `litellm_proxy/` for embedding, image_generation, transcription, speech, rerank [Start here](https://docs.litellm.ai/docs/providers/litellm_proxy)
7. OpenAI Pass-through - allow using Assistants GET, DELETE on /openai pass through routes [Start here](https://docs.litellm.ai/docs/pass_through/openai_passthrough)
8. Message Translation - fix openai message for assistant msg if role is missing - openai allows this
9. O1/O3 - support ‘drop_params’ for o3-mini and o1 parallel_tool_calls param (not supported currently) [See here](https://docs.litellm.ai/docs/completion/drop_params)

## Spend Tracking Improvements

1. Cost tracking for rerank via Bedrock [See PR](https://github.com/BerriAI/litellm/commit/b682dc4ec8fd07acf2f4c981d2721e36ae2a49c5)
2. Anthropic pass-through - fix race condition causing cost to not be tracked [See PR](https://github.com/BerriAI/litellm/pull/8874)
3. Anthropic pass-through: Ensure accurate token counting [See PR](https://github.com/BerriAI/litellm/pull/8880)

## Management Endpoints / UI

1. Models Page - Allow sorting models by ‘created at’
2. Models Page - Edit Model Flow Improvements
3. Models Page - Fix Adding Azure, Azure AI Studio models on UI 
4. Internal Users Page - Allow Bulk Adding Internal Users on UI 
5. Internal Users Page - Allow sorting users by ‘created at’ 
6. Virtual Keys Page - Allow searching for UserIDs on the dropdown when assigning a user to a team [See PR](https://github.com/BerriAI/litellm/pull/8844)
7. Virtual Keys Page - allow creating a user when assigning keys to users [See PR](https://github.com/BerriAI/litellm/pull/8844)
8. Model Hub Page  - fix text overflow issue [See PR](https://github.com/BerriAI/litellm/pull/8749)
9. Admin Settings Page - Allow adding MSFT SSO on UI 
10. Backend - don't allow creating duplicate internal users in DB

## Helm

1. support ttlSecondsAfterFinished on the migration job - [See PR](https://github.com/BerriAI/litellm/pull/8593)
2. enhance migrations job with additional configurable properties - [See PR](https://github.com/BerriAI/litellm/pull/8636)

## Logging / Guardrail Integrations

1. Arize Phoenix support 
2. ‘No-log’ - fix ‘no-log’ param support on embedding calls 

## Performance / Loadbalancing / Reliability improvements

1. Single Deployment Cooldown logic - Use allowed_fails or allowed_fail_policy if set [Start here](https://docs.litellm.ai/docs/routing#advanced-custom-retries-cooldowns-based-on-error-type)

## General Proxy Improvements

1. Hypercorn - fix reading / parsing request body 
2. Windows - fix running proxy in windows 
3. DD-Trace - fix dd-trace enablement on proxy

## Complete Git Diff

View the complete git diff [here](https://github.com/BerriAI/litellm/compare/v1.61.13-stable...v1.61.20-stable).