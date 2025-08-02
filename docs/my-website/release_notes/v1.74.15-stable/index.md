---
title: "[Pre-Release] v1.74.15-stable"
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

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:v1.74.15-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.74.15.post1
```

</TabItem>
</Tabs>

---

## Key Highlights

- **User Agent Activity Tracking** - Track how much usage each coding tool gets.
- **Prompt Management** - Use Git-Ops style prompt management with prompt templates.
- **MCP Gateway: Guardrails** - Support for using Guardrails with MCP servers.
- **Google AI Studio Imagen4** - Support for using Imagen4 models on Google AI Studio.

---

## User Agent Activity Tracking


[Read More](../../docs/proxy/user_analytics)

---

## Prompt Management

<br/>



[Read More](../../docs/proxy/prompt_management)

---

## New Models / Updated Models

#### New Model Support

| Provider    | Model                                  | Context Window | Type |
| ----------- | -------------------------------------- | -------------- | ---- |
| Google AI Studio | `gemini/imagen4` | N/A | Image Generation | 
| OpenRouter | `openrouter/grok4` | Various | Text Generation |

#### Features

- **[Google AI Studio](../../docs/providers/gemini)**
    - Added Google AI Studio Imagen4 model family support - [PR #13065](https://github.com/BerriAI/litellm/pull/13065)
    - Support for query_params in generateContent routes for API Key setting - [PR #13100](https://github.com/BerriAI/litellm/pull/13100)
- **[Azure OpenAI](../../docs/providers/azure/azure)**
    - Azure `api_version="preview"` support - [PR #13072](https://github.com/BerriAI/litellm/pull/13072)
    - Password protected certificate files support - [PR #12995](https://github.com/BerriAI/litellm/pull/12995)
- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Cost tracking via Anthropic `/v1/messages` - [PR #13072](https://github.com/BerriAI/litellm/pull/13072)
    - Computer use support (with fixes) - [PR #13150](https://github.com/BerriAI/litellm/pull/13150)
    - Framework name in UserAgent header - [PR #13159](https://github.com/BerriAI/litellm/pull/13159)
- **[OpenRouter](../../docs/providers/openrouter)**
    - Added Grok4 model support - [PR #13018](https://github.com/BerriAI/litellm/pull/13018)
- **[Anthropic](../../docs/providers/anthropic)**
    - Working mid-stream fallbacks with token usage tracking - [PR #13149](https://github.com/BerriAI/litellm/pull/13149), [PR #13170](https://github.com/BerriAI/litellm/pull/13170)
    - Improved cache_control_injection_points with negative index support - [PR #13187](https://github.com/BerriAI/litellm/pull/13187)
- **[Perplexity](../../docs/providers/perplexity)**
    - Citation annotations support - [PR #13225](https://github.com/BerriAI/litellm/pull/13225)

#### Bugs

- **[Gemini](../../docs/providers/gemini)**
    - Fix merge_reasoning_content_in_choices parameter issue - [PR #13066](https://github.com/BerriAI/litellm/pull/13066)
    - Fix CLI authorization format issues - [PR #13098](https://github.com/BerriAI/litellm/pull/13098), [PR #13189](https://github.com/BerriAI/litellm/pull/13189)
    - Environment variable support fixes - [PR #12507](https://github.com/BerriAI/litellm/pull/12507)
- **[VertexAI](../../docs/providers/vertex)**
    - Ensure multimodal embedding responses are logged properly - [PR #13050](https://github.com/BerriAI/litellm/pull/13050)
- **[vLLM/OpenAI-like](../../docs/providers/vllm)**
    - Fix missing extra_headers support for embeddings - [PR #13198](https://github.com/BerriAI/litellm/pull/13198)

---

## LLM API Endpoints

#### Features

- **[/v1/models](../../docs/providers/passthrough)**
    - Preserve model order in `/v1/models` and `/model_group/info` endpoints - [PR #13178](https://github.com/BerriAI/litellm/pull/13178)
- **[Passthrough endpoints](../../docs/pass_through/)**
    - Allow redefining LLM base API URL in passthrough endpoints - [PR #13134](https://github.com/BerriAI/litellm/pull/13134)

#### Bugs

- **Token Counting**
    - Ignore unsupported keys like prefix in token counter - [PR #11954](https://github.com/BerriAI/litellm/pull/11954)
- **Fallbacks**
    - Fix fallback delete functionality - [PR #12606](https://github.com/BerriAI/litellm/pull/12606)
- **JWT Fields**  
    - Add dot notation support for all JWT fields - [PR #13013](https://github.com/BerriAI/litellm/pull/13013)

---

## [MCP Gateway](../../docs/mcp)

#### Features

- **Health Check Improvements**
    - Add health check endpoints for MCP servers - [PR #13106](https://github.com/BerriAI/litellm/pull/13106)
    - Ensure MCPs load + don't run health check every time MCPs load on UI - [PR #13228](https://github.com/BerriAI/litellm/pull/13228)
- **Guardrails Integration**
    - Add pre and during call hooks initialization - [PR #13067](https://github.com/BerriAI/litellm/pull/13067)
    - Move pre and during hooks to ProxyLogging - [PR #13109](https://github.com/BerriAI/litellm/pull/13109)
    - MCP pre and during guardrails implementation - [PR #13188](https://github.com/BerriAI/litellm/pull/13188)
- **Protocol & Header Support**
    - Add protocol headers support - [PR #13062](https://github.com/BerriAI/litellm/pull/13062)
- **URL & Namespacing**
    - Improve MCP server URL validation for internal/Kubernetes URLs - [PR #13099](https://github.com/BerriAI/litellm/pull/13099)


#### Bugs

- **UI**
    - Fix scrolling issue with MCP tools - [PR #13015](https://github.com/BerriAI/litellm/pull/13015)
    - Fix MCP client list failure - [PR #13114](https://github.com/BerriAI/litellm/pull/13114)


[Read More](../../docs/mcp)


---

## Management Endpoints / UI

#### Features

- **Usage Analytics**
    - Default usage chart date range set to last 7 days - [PR #12917](https://github.com/BerriAI/litellm/pull/12917)
    - New advanced date range picker component - [PR #13141](https://github.com/BerriAI/litellm/pull/13141), [PR #13221](https://github.com/BerriAI/litellm/pull/13221)
    - Show loader on usage cost charts after date selection - [PR #13113](https://github.com/BerriAI/litellm/pull/13113)
- **User Agent Activity**
    - New tab for user agent activity tracking - [PR #13146](https://github.com/BerriAI/litellm/pull/13146)
    - Daily usage per user analytics - [PR #13147](https://github.com/BerriAI/litellm/pull/13147)
- **Provider Support**
    - Added Voyage, Jinai, Deepinfra and VolcEngine providers on UI - [PR #13131](https://github.com/BerriAI/litellm/pull/13131)
    - Added Sagemaker on UI - [PR #13117](https://github.com/BerriAI/litellm/pull/13117)
- **Key Management**
    - Properly parse JSON options for key generation in UI - [PR #12989](https://github.com/BerriAI/litellm/pull/12989)

#### Bugs

- **Permissions**
    - Fix object permission for organizations - [PR #13142](https://github.com/BerriAI/litellm/pull/13142)
    - Fix list team v2 security check - [PR #13094](https://github.com/BerriAI/litellm/pull/13094)
- **Model Management**
    - Fix model reload on model update - [PR #13216](https://github.com/BerriAI/litellm/pull/13216)
    - Fix fallbacks UI display - [PR #13191](https://github.com/BerriAI/litellm/pull/13191)
- **Configuration**
    - Fix wildcard model name handling with custom values - [PR #13116](https://github.com/BerriAI/litellm/pull/13116)

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