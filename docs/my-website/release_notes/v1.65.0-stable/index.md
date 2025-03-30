---
title: v1.65.0-stable - Model Context Protocol (MCP)
slug: v1.65.0-stable
date: 2025-03-30T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1743638400&v=beta&t=39KOXMUFedvukiWWVPHf3qI45fuQD7lNglICwN31DrI
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
tags: [mcp]
hide_table_of_contents: false
---

LiteLLM v1.65.0 introduces significant enhancements including Model Context Protocol (MCP) tools, new models, and various performance improvements.

## New Models / Updated Models
- Support for Vertex AI gemini-2.0-flash-lite & Google AI Studio gemini-2.0-flash-lite [PR](https://github.com/BerriAI/litellm/pull/9523)
- Support for Vertex AI Fine-Tuned LLMs [PR](https://github.com/BerriAI/litellm/pull/9542)
- Nova Canvas image generation support [PR](https://github.com/BerriAI/litellm/pull/9525)
- OpenAI gpt-4o-transcribe support [PR](https://github.com/BerriAI/litellm/pull/9517)
- Added new Vertex AI text embedding model [PR](https://github.com/BerriAI/litellm/pull/9476)
- Updated model prices and context windows [PR](https://github.com/BerriAI/litellm/pull/9459)

## LLM Translation
- OpenAI Web Search Tool Call Support [PR](https://github.com/BerriAI/litellm/pull/9465)
- Vertex AI topLogprobs support [PR](https://github.com/BerriAI/litellm/pull/9518) 
- Fixed Vertex AI multimodal embedding translation [PR](https://github.com/BerriAI/litellm/pull/9471)
- Support litellm.api_base for Vertex AI + Gemini across completion, embedding, image_generation [PR](https://github.com/BerriAI/litellm/pull/9516)
- Fixed Mistral chat transformation [PR](https://github.com/BerriAI/litellm/pull/9606)

## Spend Tracking Improvements
- Log 'api_base' on spend logs [PR](https://github.com/BerriAI/litellm/pull/9509)
- Support for Gemini audio token cost tracking [PR](https://github.com/BerriAI/litellm/pull/9535)
- Fixed OpenAI audio input token cost tracking [PR](https://github.com/BerriAI/litellm/pull/9535)
- Added Daily User Spend Aggregate view - allows UI Usage tab to work > 1m rows [PR](https://github.com/BerriAI/litellm/pull/9538)
- Connected UI to "LiteLLM_DailyUserSpend" spend table [PR](https://github.com/BerriAI/litellm/pull/9603)

## UI
- Allowed team admins to add/update/delete models on UI [PR](https://github.com/BerriAI/litellm/pull/9572)
- Show API base and model ID on request logs [PR](https://github.com/BerriAI/litellm/pull/9572)
- Allow viewing keyinfo on request logs [PR](https://github.com/BerriAI/litellm/pull/9568)
- Enabled viewing all wildcard models on /model/info [PR](https://github.com/BerriAI/litellm/pull/9473)
- Added render supports_web_search on model hub [PR](https://github.com/BerriAI/litellm/pull/9469)

## Logging Integrations
- Fixed StandardLoggingPayload for GCS Pub Sub Logging Integration [PR](https://github.com/BerriAI/litellm/pull/9508)

## Performance / Reliability Improvements
- LiteLLM Redis semantic caching implementation [PR](https://github.com/BerriAI/litellm/pull/9356)
- Gracefully handle exceptions when DB is having an outage [PR](https://github.com/BerriAI/litellm/pull/9533)
- Allow Pods to startup + passing /health/readiness when allow_requests_on_db_unavailable: True and DB is down [PR](https://github.com/BerriAI/litellm/pull/9569)
- Removed hard coded final usage chunk on Bedrock streaming usage [PR](https://github.com/BerriAI/litellm/pull/9512)
- Refactored Vertex AI passthrough routes - fixes unpredictable behaviour with auto-setting default_vertex_region on router model add [PR](https://github.com/BerriAI/litellm/pull/9467)

## General Improvements
- Support for exposing MCP tools on litellm proxy [PR](https://github.com/BerriAI/litellm/pull/9426)
- Support discovering Gemini, Anthropic, xAI models by calling their /v1/model endpoint [PR](https://github.com/BerriAI/litellm/pull/9530)
- Fixed route check for non-proxy admins on JWT auth [PR](https://github.com/BerriAI/litellm/pull/9454)
- Added baseline Prisma database migrations [PR](https://github.com/BerriAI/litellm/pull/9565)
- Get master key from environment, if not set [PR](https://github.com/BerriAI/litellm/pull/9617)

## Documentation
- Fixed Predibase typo [PR](https://github.com/BerriAI/litellm/pull/9464)
- Updated README.md [PR](https://github.com/BerriAI/litellm/pull/9616)

## Security
- Bumped next from 14.2.21 to 14.2.25 in UI dashboard [PR](https://github.com/BerriAI/litellm/pull/9458)



## Complete Git Diff

[Here's the complete git diff](https://github.com/BerriAI/litellm/compare/v1.63.14-stable.patch1...v1.65.0-stable)
