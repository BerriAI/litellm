---
title: "[PRE-RELEASE] v1.74.9-stable"
slug: "v1-74-9"
date: 2025-07-27T10:00:00
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

:::info

This release is not live yet. 

:::

---

## Key Highlights 


---

## New Models / Updated Models

#### Pricing / Context Window Updates

| Provider    | Model                                  | Context Window | Input ($/1M tokens) | Output ($/1M tokens) |
| ----------- | -------------------------------------- | -------------- | ------------------- | -------------------- |
| Fireworks AI | `fireworks/models/kimi-k2-instruct | 131k | $0.6 | $2.5 | 
| OpenRouter | `openrouter/qwen/qwen-vl-plus` | 8192 | $0.21 | $0.63 | 
| OpenRouter | `openrouter/qwen/qwen3-coder` | 8192 | $1 | $5 | 
| OpenRouter | `openrouter/bytedance/ui-tars-1.5-7b` | 128k | $0.10 | $0.20 | 
| Groq | `groq/qwen/qwen3-32b` | 131k | $0.29 | $0.59 | 
| VertexAI | `vertex_ai/meta/llama-3.1-8b-instruct-maas` | 128k | $0.00 | $0.00 | 
| VertexAI | `vertex_ai/meta/llama-3.1-405b-instruct-maas` | 128k | $5 | $16 | 
| VertexAI | `vertex_ai/meta/llama-3.2-90b-vision-instruct-maas` | 128k | $0.00 | $0.00 | 
| Google AI Studio | `gemini/gemini-2.0-flash-live-001` | 1,048,576 | $0.35 | $1.5 | 
| Google AI Studio | `gemini/gemini-2.5-flash-lite` | 1,048,576 | $0.1 | $0.4 | 
| VertexAI | `vertex_ai/gemini-2.0-flash-lite-001` | 1,048,576 | $0.35 | $1.5 | 
| OpenAI | `gpt-4o-realtime-preview-2025-06-03` | 128k | $5 | $20 |

#### Features

- **[Lambda AI](../../docs/providers/lambda_ai)**
    - New LLM API provider - [PR #12817](https://github.com/BerriAI/litellm/pull/12817)
- **[Github Copilot](../../docs/providers/github_copilot)**
    - Dynamic endpoint support - [PR #12827](https://github.com/BerriAI/litellm/pull/12827)
- **[Morph](../../docs/providers/morph)**
    - New LLM API provider - [PR #12821](https://github.com/BerriAI/litellm/pull/12821)
- **[Groq](../../docs/providers/groq)**
    - Remove deprecated groq/qwen-qwq-32b - [PR #12832](https://github.com/BerriAI/litellm/pull/12831)
- **[Recraft](../../docs/providers/recraft)**
    - New image generation API - [PR #12832](https://github.com/BerriAI/litellm/pull/12832)
    - New image edits api - [PR #12874](https://github.com/BerriAI/litellm/pull/12874)
- **[Azure OpenAI](../../docs/providers/azure/azure)**
    - Support DefaultAzureCredential without hard-coded environment variables - [PR #12841](https://github.com/BerriAI/litellm/pull/12841)
- **[Hyperbolic](../../docs/providers/hyperbolic)**
    - New LLM API provider - [PR #12826](https://github.com/BerriAI/litellm/pull/12826)
- **[OpenAI](../../docs/providers/openai)**
    - `/realtime` API - pass through intent query param - [PR #12838](https://github.com/BerriAI/litellm/pull/12838)
- **[Bedrock](../../docs/providers/bedrock)**
    - Add inpainting support for Amazon Nova Canvas - [PR #12949](https://github.com/BerriAI/litellm/pull/12949) s/o @[SantoshDhaladhuli](https://github.com/SantoshDhaladhuli)

#### Bugs
- **Gemini ([Google AI Studio](../../docs/providers/gemini) + [VertexAI](../../docs/providers/vertex))**
    - Fix leaking file descriptor error on sync calls - [PR #12824](https://github.com/BerriAI/litellm/pull/12824)
- **IBM Watsonx**
    - use correct parameter name for tool choice - [PR #9980](https://github.com/BerriAI/litellm/pull/9980)
- **[Anthropic](../../docs/providers/anthropic)**
    - Only show ‘reasoning_effort’ for supported models - [PR #12847](https://github.com/BerriAI/litellm/pull/12847)
    - Handle $id and $schema in tool call requests (Anthropic API stopped accepting them) - [PR #12959](https://github.com/BerriAI/litellm/pull/12959)
- **[Openrouter](../../docs/providers/openrouter)**
    - filter out cache_control flag for non-anthropic models (allows usage with claude code) https://github.com/BerriAI/litellm/pull/12850
- **[Gemini](../../docs/providers/gemini)**
    - Shorten Gemini tool_call_id for Open AI compatibility - [PR #12941](https://github.com/BerriAI/litellm/pull/12941) s/o @[tonga54](https://github.com/tonga54)

---

## LLM API Endpoints

#### Features

- **[Passthrough endpoints](../../docs/pass_through/)**
    - Make key/user/team cost tracking OSS - [PR #12847](https://github.com/BerriAI/litellm/pull/12847)
- **[/v1/models](../../docs/providers/passthrough)**
    - Return fallback models as part of api response - [PR #12811](https://github.com/BerriAI/litellm/pull/12811) s/o @[murad-khafizov](https://github.com/murad-khafizov)
- **[/vector_stores](../../docs/providers/passthrough)**
    - Make permission management OSS - [PR #12990](https://github.com/BerriAI/litellm/pull/12990)

#### Bugs
1. `/batches`
    1. Skip invalid batch during cost tracking check (prev. Would stop all checks) - [PR #12782](https://github.com/BerriAI/litellm/pull/12782)
2. `/chat/completions`
    1. Fix async retryer on .acompletion() - [PR #12886](https://github.com/BerriAI/litellm/pull/12886)

---

## [MCP Gateway](../../docs/mcp)

#### Features
- **[Permission Management](../../docs/mcp#grouping-mcps-access-groups)**
    - Make permission management by key/team OSS - [PR #12988](https://github.com/BerriAI/litellm/pull/12988)
- **[MCP Alias](../../docs/mcp#mcp-aliases)**
    - Support mcp server aliases (useful for calling long mcp server names on Cursor) - [PR #12994](https://github.com/BerriAI/litellm/pull/12994)
- **Header Propagation**
    - Support propagating headers from client to backend MCP (useful for sending personal access tokens to backend MCP) - [PR #13003](https://github.com/BerriAI/litellm/pull/13003)

---

## Management Endpoints / UI

#### Features
- **Keys**
    - Regenerate Key State Management improvements - [PR #12729](https://github.com/BerriAI/litellm/pull/12729)
- **Models**
    - Wildcard model filter support - [PR #12597](https://github.com/BerriAI/litellm/pull/12597)
    - Fixes for handling team only models on UI - [PR #12632](https://github.com/BerriAI/litellm/pull/12632)
- **Usage Page**
    - Fix Y-axis labels overlap on Spend per Tag chart - [PR #12754](https://github.com/BerriAI/litellm/pull/12754)
- **Teams**
    - Allow setting custom key duration + show key creation stats - [PR #12722](https://github.com/BerriAI/litellm/pull/12722)
    - Enable team admins to update member roles - [PR #12629](https://github.com/BerriAI/litellm/pull/12629)
- **Users**
    - New `/user/bulk_update` endpoint - [PR #12720](https://github.com/BerriAI/litellm/pull/12720)
- **Logs Page**
    - Add `end_user` filter on UI Logs Page - [PR #12663](https://github.com/BerriAI/litellm/pull/12663)
- **MCP Servers**
    - Copy MCP Server name functionality - [PR #12760](https://github.com/BerriAI/litellm/pull/12760)
- **Vector Stores**
    - UI support for clicking into Vector Stores - [PR #12741](https://github.com/BerriAI/litellm/pull/12741)
    - Allow adding Vertex RAG Engine, OpenAI, Azure through UI - [PR #12752](https://github.com/BerriAI/litellm/pull/12752)
- **General**
    - Add Copy-on-Click for all IDs (Key, Team, Organization, MCP Server) - [PR #12615](https://github.com/BerriAI/litellm/pull/12615)
- **[SCIM](../../docs/proxy/scim)**
    - Add GET /ServiceProviderConfig endpoint - [PR #12664](https://github.com/BerriAI/litellm/pull/12664)

#### Bugs
- **Teams**
    - Ensure user id correctly added when creating new teams - [PR #12719](https://github.com/BerriAI/litellm/pull/12719)
    - Fixes for handling team-only models on UI - [PR #12632](https://github.com/BerriAI/litellm/pull/12632)

---

## Logging / Guardrail Integrations

#### Features
- **[Google Cloud Model Armor](../../docs/proxy/guardrails/google_cloud_model_armor)**
    - New guardrails integration - [PR #12492](https://github.com/BerriAI/litellm/pull/12492)
- **[Bedrock Guardrails](../../docs/proxy/guardrails/bedrock)**
    - Allow disabling exception on 'BLOCKED' action - [PR #12693](https://github.com/BerriAI/litellm/pull/12693)
- **[Guardrails AI](../../docs/proxy/guardrails/guardrails_ai)**
    - Support `llmOutput` based guardrails as pre-call hooks - [PR #12674](https://github.com/BerriAI/litellm/pull/12674)
- **[DataDog LLM Observability](../../docs/proxy/logging#datadog)**
    - Add support for tracking the correct span type based on LLM Endpoint used - [PR #12652](https://github.com/BerriAI/litellm/pull/12652)
- **[Custom Logging](../../docs/proxy/logging)**
    - Allow reading custom logger python scripts from S3 or GCS Bucket - [PR #12623](https://github.com/BerriAI/litellm/pull/12623)

#### Bugs
- **[General Logging](../../docs/proxy/logging)**
    - StandardLoggingPayload on cache_hits should track custom llm provider - [PR #12652](https://github.com/BerriAI/litellm/pull/12652)
- **[S3 Buckets](../../docs/proxy/logging#s3-buckets)**
    - S3 v2 log uploader crashes when using with guardrails - [PR #12733](https://github.com/BerriAI/litellm/pull/12733)

---

## Performance / Loadbalancing / Reliability improvements

#### Features
- **Health Checks**
    - Separate health app for liveness probes - [PR #12669](https://github.com/BerriAI/litellm/pull/12669)
    - Health check app on separate port - [PR #12718](https://github.com/BerriAI/litellm/pull/12718)
- **Caching**
    - Add Azure Blob cache support - [PR #12587](https://github.com/BerriAI/litellm/pull/12587)
- **Router**
    - Handle ZeroDivisionError with zero completion tokens in lowest_latency strategy - [PR #12734](https://github.com/BerriAI/litellm/pull/12734)

#### Bugs
- **Database**
    - Use upsert for managed object table to avoid UniqueViolationError - [PR #11795](https://github.com/BerriAI/litellm/pull/11795)
    - Refactor to support use_prisma_migrate for helm hook - [PR #12600](https://github.com/BerriAI/litellm/pull/12600)
- **Cache**
    - Fix: redis caching for embedding response models - [PR #12750](https://github.com/BerriAI/litellm/pull/12750)

---

## Helm Chart

- DB Migration Hook: refactor to support use_prisma_migrate - for helm hook [PR](https://github.com/BerriAI/litellm/pull/12600)
- Add envVars and extraEnvVars support to Helm migrations job - [PR #12591](https://github.com/BerriAI/litellm/pull/12591)

## General Proxy Improvements

#### Features
- **Control Plane + Data Plane Architecture**
    - Control Plane + Data Plane support - [PR #12601](https://github.com/BerriAI/litellm/pull/12601)
- **Proxy CLI**
    - Add "keys import" command to CLI - [PR #12620](https://github.com/BerriAI/litellm/pull/12620)
- **Swagger Documentation**
    - Add swagger docs for LiteLLM /chat/completions, /embeddings, /responses - [PR #12618](https://github.com/BerriAI/litellm/pull/12618)
- **Dependencies**
    - Loosen rich version from ==13.7.1 to >=13.7.1 - [PR #12704](https://github.com/BerriAI/litellm/pull/12704)


#### Bugs

- Verbose log is enabled by default fix - [PR #12596](https://github.com/BerriAI/litellm/pull/12596)

- Add support for disabling callbacks in request body - [PR #12762](https://github.com/BerriAI/litellm/pull/12762)
- Handle circular references in spend tracking metadata JSON serialization - [PR #12643](https://github.com/BerriAI/litellm/pull/12643)

---

## New Contributors
* @AntonioKL made their first contribution in https://github.com/BerriAI/litellm/pull/12591
* @marcelodiaz558 made their first contribution in https://github.com/BerriAI/litellm/pull/12541
* @dmcaulay made their first contribution in https://github.com/BerriAI/litellm/pull/12463
* @demoray made their first contribution in https://github.com/BerriAI/litellm/pull/12587
* @staeiou made their first contribution in https://github.com/BerriAI/litellm/pull/12631
* @stefanc-ai2 made their first contribution in https://github.com/BerriAI/litellm/pull/12622
* @RichardoC made their first contribution in https://github.com/BerriAI/litellm/pull/12607
* @yeahyung made their first contribution in https://github.com/BerriAI/litellm/pull/11795
* @mnguyen96 made their first contribution in https://github.com/BerriAI/litellm/pull/12619
* @rgambee made their first contribution in https://github.com/BerriAI/litellm/pull/11517
* @jvanmelckebeke made their first contribution in https://github.com/BerriAI/litellm/pull/12725
* @jlaurendi made their first contribution in https://github.com/BerriAI/litellm/pull/12704
* @doublerr made their first contribution in https://github.com/BerriAI/litellm/pull/12661

## **[Full Changelog](https://github.com/BerriAI/litellm/compare/v1.74.3-stable...v1.74.7-stable)**
