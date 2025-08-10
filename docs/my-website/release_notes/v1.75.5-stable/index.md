---
title: "v1.74.15-stable"
slug: "v1-74-15"
date: 2025-08-02T10:00:00
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

This release is not out yet. 

:::

---

## New Models / Updated Models

#### New Model Support

| Provider    | Model                                  | Context Window | Input ($/1M tokens) | Output ($/1M tokens) |
| ----------- | -------------------------------------- | -------------- | ------------------- | -------------------- | 
| Bedrock | `bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0` | 200k | $15 | $75 |
| Bedrock | `bedrock/openai.gpt-oss-20b-1:0` | 200k | 0.07 | 0.3 |
| Bedrock | `bedrock/openai.gpt-oss-120b-1:0` | 200k | 0.15 | 0.6 |
| Fireworks AI | `fireworks_ai/accounts/fireworks/models/glm-4p5` | 128k | 0.55 | 2.19 |
| Fireworks AI | `fireworks_ai/accounts/fireworks/models/glm-4p5-air` | 128k | 0.22 | 0.88 |
| Fireworks AI | `fireworks_ai/accounts/fireworks/models/gpt-oss-120b` | 131072 | 0.15 | 0.6 |
| Fireworks AI | `fireworks_ai/accounts/fireworks/models/gpt-oss-20b` | 131072 | 0.05 | 0.2 |
| Groq | `groq/openai/gpt-oss-20b` | 131072 | 0.1 | 0.5 |
| Groq | `groq/openai/gpt-oss-120b` | 131072 | 0.15 | 0.75 |
| OpenAI | `openai/gpt-5` | 400k | 1.25 | 10 | 
| OpenAI | `openai/gpt-5-2025-08-07` | 400k | 1.25 | 10 | 
| OpenAI | `openai/gpt-5-mini` | 400k | 0.25 | 2 |
| OpenAI | `openai/gpt-5-mini-2025-08-07` | 400k | 0.25 | 2 | 
| OpenAI | `openai/gpt-5-nano` | 400k | 0.05 | 0.4 | 
| OpenAI | `openai/gpt-5-nano-2025-08-07` | 400k | 0.05 | 0.4 | 
| OpenAI | `openai/gpt-5-chat` | 400k | 1.25 | 10 | 
| OpenAI | `openai/gpt-5-chat-latest` | 400k | 1.25 | 10 | 
| Azure | `azure/gpt-5` | 400k | 1.25 | 10 | 
| Azure | `azure/gpt-5-2025-08-07` | 400k | 1.25 | 10 | 
| Azure | `azure/gpt-5-mini` | 400k | 0.25 | 2 | 
| Azure | `azure/gpt-5-mini-2025-08-07` | 400k | 0.25 | 2 | 
| Azure | `azure/gpt-5-nano-2025-08-07` | 400k | 0.05 | 0.4 | 
| Azure | `azure/gpt-5-nano` | 400k | 0.05 | 0.4 | 
| Azure | `azure/gpt-5-chat` | 400k | 1.25 | 10 | 
| Azure | `azure/gpt-5-chat-latest` | 400k | 1.25 | 10 | 

#### Features

- **[OCI](../../docs/providers/oci)**
    - New LLM provider - [PR #13206](https://github.com/BerriAI/litellm/pull/13206)
- **[JinaAI](../../docs/providers/jina_ai)**
    - support multimodal embedding models - [PR #13181](https://github.com/BerriAI/litellm/pull/13181)
- **GPT-5 ([OpenAI](../../docs/providers/openai)/[Azure](../../docs/providers/azure))**
    - Support drop_params for temperature - [PR #13390](https://github.com/BerriAI/litellm/pull/13390)
    - Map max_tokens to max_completion_tokens - [PR #13390](https://github.com/BerriAI/litellm/pull/13390)
- **[Anthropic](../../docs/providers/anthropic)**
    - Add claude-opus-4-1 on model cost map - [PR #13384](https://github.com/BerriAI/litellm/pull/13384)
- **[OpenRouter](../../docs/providers/openrouter)**
    - Add gpt-oss to model cost map - [PR #13442](https://github.com/BerriAI/litellm/pull/13442)
- **[Cerebras](../../docs/providers/cerebras)**
    - Add gpt-oss to model cost map - [PR #13442](https://github.com/BerriAI/litellm/pull/13442)
- **[Azure](../../docs/providers/azure)**
    - Support drop params for ‘temperature’ on o-series models - [PR #13353](https://github.com/BerriAI/litellm/pull/13353)
- **[GradientAI](../../docs/providers/gradient_ai)**
    - New LLM Provider - [PR #12169](https://github.com/BerriAI/litellm/pull/12169)

#### Bugs

- **[OpenAI](../../docs/providers/openai)**
    - Add ‘service_tier’ and ‘safety_identifier’ as supported responses api params - [PR #13258](https://github.com/BerriAI/litellm/pull/13258)
    - Correct pricing for web search on 4o-mini - [PR #13269](https://github.com/BerriAI/litellm/pull/13269)
- **[Mistral](../../docs/providers/mistral)**
    - Handle $id and $schema fields when calling mistral - [PR #13389](https://github.com/BerriAI/litellm/pull/13389)
---

## LLM API Endpoints

#### Features

- `/responses` 
    - Responses API Session Handling w/ support for images - [PR #13347](https://github.com/BerriAI/litellm/pull/13347)
    - failed if input containing ResponseReasoningItem - [PR #13465](https://github.com/BerriAI/litellm/pull/13465)
    - Support custom tools - [PR #13418](https://github.com/BerriAI/litellm/pull/13418)

#### Bugs

- `/chat/completions` 
    - Fix completion_token_details usage object missing ‘text’ tokens - [PR #13234](https://github.com/BerriAI/litellm/pull/13234)
    - (SDK) handle tool being a pydantic object - [PR #13274](https://github.com/BerriAI/litellm/pull/13274)
    - include cost in streaming usage object - [PR #13418](https://github.com/BerriAI/litellm/pull/13418)
    - Exclude none fields on /chat/completion - allows usage with n8n - [PR #13320](https://github.com/BerriAI/litellm/pull/13320)
- `/responses` 
    - Transform function call in response for non-openai models (gemini/anthropic) - [PR #13260](https://github.com/BerriAI/litellm/pull/13260)
    - Fix unsupported operand error with model groups - [PR #13293](https://github.com/BerriAI/litellm/pull/13293)
    - Responses api session management for streaming responses - [PR #13396](https://github.com/BerriAI/litellm/pull/13396)
- `/v1/messages`
    - Added litellm claude code count tokens - [PR #13261](https://github.com/BerriAI/litellm/pull/13261)
- `/vector_stores`
    - Fix create/search vector store errors - [PR #13285](https://github.com/BerriAI/litellm/pull/13285)
---

## [MCP Gateway](../../docs/mcp)

#### Features

- Add route check for internal users - [PR #13350](https://github.com/BerriAI/litellm/pull/13350)
- MCP Guardrails - docs - [PR #13392](https://github.com/BerriAI/litellm/pull/13392)


#### Bugs

- Fix auth on UI for bearer token servers - [PR #13312](https://github.com/BerriAI/litellm/pull/13312)
- allow access group on mcp tool retrieval - [PR #13425](https://github.com/BerriAI/litellm/pull/13425)


---

## Management Endpoints / UI

#### Features

- **Usage Analytics**
    - New tab for user agent activity tracking - [PR #13146](https://github.com/BerriAI/litellm/pull/13146)
    - Daily usage per user analytics - [PR #13147](https://github.com/BerriAI/litellm/pull/13147)
    - Default usage chart date range set to last 7 days - [PR #12917](https://github.com/BerriAI/litellm/pull/12917)
    - New advanced date range picker component - [PR #13141](https://github.com/BerriAI/litellm/pull/13141), [PR #13221](https://github.com/BerriAI/litellm/pull/13221)
    - Show loader on usage cost charts after date selection - [PR #13113](https://github.com/BerriAI/litellm/pull/13113)
- **Models**
    - Added Voyage, Jinai, Deepinfra and VolcEngine providers on UI - [PR #13131](https://github.com/BerriAI/litellm/pull/13131)
    - Added Sagemaker on UI - [PR #13117](https://github.com/BerriAI/litellm/pull/13117)
    - Preserve model order in `/v1/models` and `/model_group/info` endpoints - [PR #13178](https://github.com/BerriAI/litellm/pull/13178)

- **Key Management**
    - Properly parse JSON options for key generation in UI - [PR #12989](https://github.com/BerriAI/litellm/pull/12989)
- **Authentication**
    - **JWT Fields**  
        - Add dot notation support for all JWT fields - [PR #13013](https://github.com/BerriAI/litellm/pull/13013)

#### Bugs

- **Permissions**
    - Fix object permission for organizations - [PR #13142](https://github.com/BerriAI/litellm/pull/13142)
    - Fix list team v2 security check - [PR #13094](https://github.com/BerriAI/litellm/pull/13094)
- **Models**
    - Fix model reload on model update - [PR #13216](https://github.com/BerriAI/litellm/pull/13216)
- **Router Settings**
    - Fix displaying models for fallbacks in UI - [PR #13191](https://github.com/BerriAI/litellm/pull/13191)
    - Fix wildcard model name handling with custom values - [PR #13116](https://github.com/BerriAI/litellm/pull/13116)
    - Fix fallback delete functionality - [PR #12606](https://github.com/BerriAI/litellm/pull/12606)

---

## Logging / Guardrail Integrations

#### Features

- **[MLFlow](../../docs/proxy/logging#mlflow)**
    - Allow adding tags for MLFlow logging requests - [PR #13108](https://github.com/BerriAI/litellm/pull/13108)
- **[Langfuse OTEL](../../docs/proxy/logging#langfuse)**
    - Add comprehensive metadata support to Langfuse OpenTelemetry integration - [PR #12956](https://github.com/BerriAI/litellm/pull/12956)
- **[Datadog LLM Observability](../../docs/proxy/logging#datadog)**
    - Allow redacting message/response content for specific logging integrations - [PR #13158](https://github.com/BerriAI/litellm/pull/13158)

#### Bugs

- **API Key Logging**
    - Fix API Key being logged inappropriately - [PR #12978](https://github.com/BerriAI/litellm/pull/12978)
- **MCP Spend Tracking**
    - Set default value for MCP namespace tool name in spend table - [PR #12894](https://github.com/BerriAI/litellm/pull/12894)

---

## Performance / Loadbalancing / Reliability improvements

#### Features

- **Background Health Checks**
    - Allow disabling background health checks for specific deployments - [PR #13186](https://github.com/BerriAI/litellm/pull/13186)
- **Database Connection Management**
    - Ensure stale Prisma clients disconnect DB connections properly - [PR #13140](https://github.com/BerriAI/litellm/pull/13140)
- **Jitter Improvements**
    - Fix jitter calculation (should be added not multiplied) - [PR #12901](https://github.com/BerriAI/litellm/pull/12901)

#### Bugs

- **Anthropic Streaming**
    - Always use choice index=0 for Anthropic streaming responses - [PR #12666](https://github.com/BerriAI/litellm/pull/12666)
- **Custom Auth**
    - Bubble up custom exceptions properly - [PR #13093](https://github.com/BerriAI/litellm/pull/13093)
- **OTEL with Managed Files**
    - Fix using managed files with OTEL integration - [PR #13171](https://github.com/BerriAI/litellm/pull/13171)

---

## General Proxy Improvements

#### Features

- **Database Migration**
    - Move to use_prisma_migrate by default - [PR #13117](https://github.com/BerriAI/litellm/pull/13117)
    - Resolve team-only models on auth checks - [PR #13117](https://github.com/BerriAI/litellm/pull/13117)
- **Infrastructure**
    - Loosened MCP Python version restrictions - [PR #13102](https://github.com/BerriAI/litellm/pull/13102)
    - Migrate build_and_test to CI/CD Postgres DB - [PR #13166](https://github.com/BerriAI/litellm/pull/13166)
- **Helm Charts**
    - Allow Helm hooks for migration jobs - [PR #13174](https://github.com/BerriAI/litellm/pull/13174)
    - Fix Helm migration job schema updates - [PR #12809](https://github.com/BerriAI/litellm/pull/12809)

#### Bugs

- **Docker**
    - Remove obsolete `version` attribute in docker-compose - [PR #13172](https://github.com/BerriAI/litellm/pull/13172)
    - Add openssl in runtime stage for non-root Dockerfile - [PR #13168](https://github.com/BerriAI/litellm/pull/13168)
- **Database Configuration**
    - Fix DB config through environment variables - [PR #13111](https://github.com/BerriAI/litellm/pull/13111)
- **Logging**
    - Suppress httpx logging - [PR #13217](https://github.com/BerriAI/litellm/pull/13217)
- **Token Counting**
    - Ignore unsupported keys like prefix in token counter - [PR #11954](https://github.com/BerriAI/litellm/pull/11954)
---

## New Contributors
* @5731la made their first contribution in https://github.com/BerriAI/litellm/pull/12989
* @restato made their first contribution in https://github.com/BerriAI/litellm/pull/12980
* @strickvl made their first contribution in https://github.com/BerriAI/litellm/pull/12956
* @Ne0-1 made their first contribution in https://github.com/BerriAI/litellm/pull/12995
* @maxrabin made their first contribution in https://github.com/BerriAI/litellm/pull/13079
* @lvuna made their first contribution in https://github.com/BerriAI/litellm/pull/12894
* @Maximgitman made their first contribution in https://github.com/BerriAI/litellm/pull/12666
* @pathikrit made their first contribution in https://github.com/BerriAI/litellm/pull/12901
* @huetterma made their first contribution in https://github.com/BerriAI/litellm/pull/12809
* @betterthanbreakfast made their first contribution in https://github.com/BerriAI/litellm/pull/13029
* @phosae made their first contribution in https://github.com/BerriAI/litellm/pull/12606
* @sahusiddharth made their first contribution in https://github.com/BerriAI/litellm/pull/12507
* @Amit-kr26 made their first contribution in https://github.com/BerriAI/litellm/pull/11954
* @kowyo made their first contribution in https://github.com/BerriAI/litellm/pull/13172
* @AnandKhinvasara made their first contribution in https://github.com/BerriAI/litellm/pull/13187
* @unique-jakub made their first contribution in https://github.com/BerriAI/litellm/pull/13174
* @tyumentsev4 made their first contribution in https://github.com/BerriAI/litellm/pull/13134
* @aayush-malviya-acquia made their first contribution in https://github.com/BerriAI/litellm/pull/12978
* @kankute-sameer made their first contribution in https://github.com/BerriAI/litellm/pull/13225
* @AlexanderYastrebov made their first contribution in https://github.com/BerriAI/litellm/pull/13178

## **[Full Changelog](https://github.com/BerriAI/litellm/compare/v1.74.9-stable...v1.74.15.rc)**