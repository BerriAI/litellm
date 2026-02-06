---
title: "v1.79.1-stable - Guardrail Playground"
slug: "v1-79-1"
date: 2025-11-01T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
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
docker.litellm.ai/berriai/litellm:v1.79.1-stable
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.80.0
```

</TabItem>
</Tabs>

---

## Key Highlights

- **Container API Support** - End-to-end OpenAI Container API support with proxy integration, logging, and cost tracking
- **FAL AI Image Generation** - Native support for FAL AI image generation models with cost tracking
- **UI Enhancements** - Guardrail Playground, Cache Settings, Tag Routing, SSO Settings
- **Batch API Rate Limiting** - Input-based rate limits support for Batch API requests
- **Vector Store Expansion** - Milvus vector store support and Azure AI virtual indexes
- **Memory Leak Fixes** - Resolved issues accounting for 90% of memory leaks on Python SDK & AI Gateway

---

## Dependency Upgrades

- **Dependencies**
    - Build(deps): bump starlette from 0.47.2 to 0.49.1 - [PR #16027](https://github.com/BerriAI/litellm/pull/16027)
    - Build(deps): bump fastapi from 0.116.1 to 0.120.1 - [PR #16054](https://github.com/BerriAI/litellm/pull/16054)
    - Build(deps): bump hono from 4.9.7 to 4.10.3 in /litellm-js/spend-logs - [PR #15915](https://github.com/BerriAI/litellm/pull/15915)


## New Models / Updated Models

#### New Model Support

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| Mistral | `mistral/codestral-embed` | 8K | $0.15 | - | Embeddings |
| Mistral | `mistral/codestral-embed-2505` | 8K | $0.15 | - | Embeddings |
| Gemini | `gemini/gemini-embedding-001` | 2K | $0.15 | - | Embeddings |
| FAL AI | `fal_ai/fal-ai/flux-pro/v1.1-ultra` | - | - | - | Image generation - $0.0398/image |
| FAL AI | `fal_ai/fal-ai/imagen4/preview` | - | - | - | Image generation - $0.0398/image |
| FAL AI | `fal_ai/fal-ai/recraft/v3/text-to-image` | - | - | - | Image generation - $0.0398/image |
| FAL AI | `fal_ai/fal-ai/stable-diffusion-v35-medium` | - | - | - | Image generation - $0.0398/image |
| FAL AI | `fal_ai/bria/text-to-image/3.2` | - | - | - | Image generation - $0.0398/image |
| OpenAI | `openai/sora-2-pro` | - | - | - | Video generation - $0.30/video/second |

#### Features

- **[Anthropic](../../docs/providers/anthropic)**
    - Extended Claude 3-7 Sonnet deprecation date from 2026-02-01 to 2026-02-19 - [PR #15976](https://github.com/BerriAI/litellm/pull/15976)
    - Extended Claude Opus 4-0 deprecation date from 2025-03-01 to 2026-05-01 - [PR #15976](https://github.com/BerriAI/litellm/pull/15976)
    - Removed Claude Haiku 3-5 deprecation date (previously 2025-03-01) - [PR #15976](https://github.com/BerriAI/litellm/pull/15976)
    - Added Claude Opus 4-1, Claude Opus 4-0 20250513, Claude Sonnet 4 20250514 deprecation dates - [PR #15976](https://github.com/BerriAI/litellm/pull/15976)
    - Added web search support for Claude Opus 4-1 - [PR #15976](https://github.com/BerriAI/litellm/pull/15976)

- **[Bedrock](../../docs/providers/bedrock)**
    - Fix empty assistant message handling in AWS Bedrock Converse API to prevent 400 Bad Request errors - [PR #15850](https://github.com/BerriAI/litellm/pull/15850)
    - Allow using ARNs when generating images via Bedrock - [PR #15789](https://github.com/BerriAI/litellm/pull/15789)
    - Add per model group header forwarding for Bedrock Invoke API - [PR #16042](https://github.com/BerriAI/litellm/pull/16042)
    - Preserve Bedrock inference profile IDs in health checks - [PR #15947](https://github.com/BerriAI/litellm/pull/15947)
    - Added fallback logic for detecting file content-type when S3 returns generic type - When using Bedrock with S3-hosted files, if the S3 object's Content-Type is not correctly set (e.g., binary/octet-stream instead of image/png), Bedrock can now handle it correctly - [PR #15635](https://github.com/BerriAI/litellm/pull/15635)

- **[Azure](../../docs/providers/azure)**
    - Add deprecation dates for Azure OpenAI models (gpt-4o-2024-08-06, gpt-4o-2024-11-20, gpt-4.1 series, o3-2025-04-16, text-embedding-3-small) - [PR #15976](https://github.com/BerriAI/litellm/pull/15976)
    - Fix Azure OpenAI ContextWindowExceededError mapping from Azure errors - [PR #15981](https://github.com/BerriAI/litellm/pull/15981)
    - Add handling for `v1` under Azure API versions - [PR #15984](https://github.com/BerriAI/litellm/pull/15984)
    - Fix azure doesn't accept extra body param - [PR #16116](https://github.com/BerriAI/litellm/pull/16116)

- **[OpenAI](../../docs/providers/openai)**
    - Add deprecation dates for gpt-3.5-turbo-1106, gpt-4-0125-preview, gpt-4-1106-preview, o1-mini-2024-09-12 - [PR #15976](https://github.com/BerriAI/litellm/pull/15976)
    - Add extended Sora-2 modality support (text + image inputs) - [PR #15976](https://github.com/BerriAI/litellm/pull/15976)
    - Updated OpenAI Sora-2-Pro pricing to $0.30/video/second - [PR #15976](https://github.com/BerriAI/litellm/pull/15976)

- **[OpenRouter](../../docs/providers/openrouter)**
    - Add Claude Haiku 4.5 pricing for OpenRouter - [PR #15909](https://github.com/BerriAI/litellm/pull/15909)
    - Add base_url config with environment variables documentation - [PR #15946](https://github.com/BerriAI/litellm/pull/15946)

- **[Mistral](../../docs/providers/mistral)**
    - Add codestral-embed-2505 embedding model - [PR #16071](https://github.com/BerriAI/litellm/pull/16071)

- **[Gemini (Google AI Studio + Vertex AI)](../../docs/providers/gemini)**
    - Fix gemini request mutation for tool use - [PR #16002](https://github.com/BerriAI/litellm/pull/16002)
    - Add gemini-embedding-001 pricing entry for Google GenAI API - [PR #16078](https://github.com/BerriAI/litellm/pull/16078)
    - Changes to fix frequency_penalty and presence_penalty issue for gemini-2.5-pro model - [PR #16041](https://github.com/BerriAI/litellm/pull/16041)

- **[DeepInfra](../../docs/providers/deepinfra)**
    - Add vision support for Qwen/Qwen3-chat-32b model - [PR #15976](https://github.com/BerriAI/litellm/pull/15976)

- **[Vercel AI Gateway](../../docs/providers/vercel_ai_gateway)**
    - Fix vercel_ai_gateway entry for glm-4.6 (moved from vercel_ai_gateway/glm-4.6 to vercel_ai_gateway/zai/glm-4.6) - [PR #16084](https://github.com/BerriAI/litellm/pull/16084)

- **[Fireworks](../../docs/providers/fireworks_ai)**
    - Don't add "accounts/fireworks/models" prefix for Fireworks Provider - [PR #15938](https://github.com/BerriAI/litellm/pull/15938)

- **[Cohere](../../docs/providers/cohere)**
    - Add OpenAI-compatible annotations support for Cohere v2 citations - [PR #16038](https://github.com/BerriAI/litellm/pull/16038)

- **[Deepgram](../../docs/providers/deepgram)**
    - Handle Deepgram detected language when available - [PR #16093](https://github.com/BerriAI/litellm/pull/16093)

### Bug Fixes

- **[Xai](../../docs/providers/xai)**
    - Add Xai websearch cost tracking - [PR #16001](https://github.com/BerriAI/litellm/pull/16001)

#### New Provider Support

- **[FAL AI](../../docs/image_generation)**
    - Add FAL AI Image Generation support - [PR #16067](https://github.com/BerriAI/litellm/pull/16067)

- **[OCI (Oracle Cloud Infrastructure)](../../docs/providers/oci)**
    - Add OCI Signer Authentication support - [PR #16064](https://github.com/BerriAI/litellm/pull/16064)

---

## LLM API Endpoints

#### Features

- **[Container API](../../docs/containers)**
    - Add end-to-end OpenAI Container API support to LiteLLM SDK - [PR #16136](https://github.com/BerriAI/litellm/pull/16136)
    - Add proxy support for container APIs - [PR #16049](https://github.com/BerriAI/litellm/pull/16049)
    - Add logging support for Container API - [PR #16049](https://github.com/BerriAI/litellm/pull/16049)
    - Add cost tracking support for containers with documentation - [PR #16117](https://github.com/BerriAI/litellm/pull/16117)

- **[Responses API](../../docs/response_api)**
    - Respect `LiteLLM-Disable-Message-Redaction` header for Responses API - [PR #15966](https://github.com/BerriAI/litellm/pull/15966)
    - Add /openai routes for responses API (Azure OpenAI SDK Compatibility) - [PR #15988](https://github.com/BerriAI/litellm/pull/15988)
    - Redact reasoning summaries in ResponsesAPI output when message logging is disabled - [PR #15965](https://github.com/BerriAI/litellm/pull/15965)
    - Support text.format parameter in Responses API for providers without native ResponsesAPIConfig - [PR #16023](https://github.com/BerriAI/litellm/pull/16023)
    - Add LLM provider response headers to Responses API - [PR #16091](https://github.com/BerriAI/litellm/pull/16091)

- **[Video Generation API](../../docs/video_generation)**
    - Add `custom_llm_provider` support for video endpoints (non-generation) - [PR #16121](https://github.com/BerriAI/litellm/pull/16121)
    - Fix documentation for videos - [PR #15937](https://github.com/BerriAI/litellm/pull/15937)
    - Add OpenAI client usage documentation for videos and fix navigation visibility - [PR #15996](https://github.com/BerriAI/litellm/pull/15996)

- **[Moderations API](../../docs/moderations)**
    - Moderations endpoint now respects `api_base` configuration parameter - [PR #16087](https://github.com/BerriAI/litellm/pull/16087)

- **[Vector Stores](../../docs/vector_stores)**
    - Milvus - search vector store support - [PR #16035](https://github.com/BerriAI/litellm/pull/16035)
    - Azure AI Vector Stores - support "virtual" indexes + create vector store on passthrough API - [PR #16160](https://github.com/BerriAI/litellm/pull/16160)

- **[Passthrough Endpoints](../../docs/pass_through/vertex_ai)**
    - Support multi-part form data on passthrough - [PR #16035](https://github.com/BerriAI/litellm/pull/16035)


---

## Management Endpoints / UI

#### Features

- **Virtual Keys**
    - Validation for Proxy Base URL in SSO Settings - [PR #16082](https://github.com/BerriAI/litellm/pull/16082)
    - Test Key UI Embeddings support - [PR #16065](https://github.com/BerriAI/litellm/pull/16065)
    - Add Key Type Select in Key Settings - [PR #16034](https://github.com/BerriAI/litellm/pull/16034)
    - Key Already Exist Error Notification - [PR #15993](https://github.com/BerriAI/litellm/pull/15993)

- **Models + Endpoints**
    - Changed API Base from Select to Input in New LLM Credentials - [PR #15987](https://github.com/BerriAI/litellm/pull/15987)
    - Remove limit from admin UI numerical input - [PR #15991](https://github.com/BerriAI/litellm/pull/15991)
    - Config Models should not be editable - [PR #16020](https://github.com/BerriAI/litellm/pull/16020)
    - Add tags in model creation - [PR #16138](https://github.com/BerriAI/litellm/pull/16138)
    - Add Tags to update model - [PR #16140](https://github.com/BerriAI/litellm/pull/16140)

- **Guardrails**
    - Add Apply Guardrail Testing Playground - [PR #16030](https://github.com/BerriAI/litellm/pull/16030)
    - Config Guardrails should not be editable and guardrail info fix - [PR #16142](https://github.com/BerriAI/litellm/pull/16142)

- **Cache Settings**
    - Allow setting cache settings on UI - [PR #16143](https://github.com/BerriAI/litellm/pull/16143)

- **Routing**
    - Allow setting all routing strategies, tag filtering on UI - [PR #16139](https://github.com/BerriAI/litellm/pull/16139)

- **Admin Settings**
    - Add license metadata to health/readiness endpoint - [PR #15997](https://github.com/BerriAI/litellm/pull/15997)
    - Litellm Backend SSO Changes - [PR #16029](https://github.com/BerriAI/litellm/pull/16029)



---

## Logging / Guardrail / Prompt Management Integrations

#### Features

- **[OpenTelemetry](../../docs/proxy/logging#opentelemetry)**
    - Enable OpenTelemetry context propagation by external tracers - [PR #15940](https://github.com/BerriAI/litellm/pull/15940)
    - Ensure error information is logged on OTEL - [PR #15978](https://github.com/BerriAI/litellm/pull/15978)

- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Fix duplicate trace in langfuse_otel - [PR #15931](https://github.com/BerriAI/litellm/pull/15931)
    - Support tool usage messages with Langfuse OTEL integration - [PR #15932](https://github.com/BerriAI/litellm/pull/15932)

- **[DataDog](../../docs/proxy/logging#datadog)**
    - Ensure key's metadata + guardrail is logged on DD - [PR #15980](https://github.com/BerriAI/litellm/pull/15980)

- **[Opik](../../docs/proxy/logging#opik)**
    - Enhance requester metadata retrieval from API key auth - [PR #15897](https://github.com/BerriAI/litellm/pull/15897)
    - User auth key metadata Documentation - [PR #16004](https://github.com/BerriAI/litellm/pull/16004)

- **[SQS](../../docs/proxy/logging#sqs)**
    - Add Base64 handling for SQS Logger - [PR #16028](https://github.com/BerriAI/litellm/pull/16028)

- **General**
    - Fix: User API key and team id and user id missing from custom callback is not misfiring - [PR #15982](https://github.com/BerriAI/litellm/pull/15982)

#### Guardrails

- **[IBM Guardrails](../../docs/proxy/guardrails)**
    - Update IBM Guardrails to correctly use SSL Verify argument - [PR #15975](https://github.com/BerriAI/litellm/pull/15975)
    - Add additional detail to ibm_guardrails.md documentation - [PR #15971](https://github.com/BerriAI/litellm/pull/15971)

- **[Model Armor](../../docs/proxy/guardrails)**
    - Support during_call for model armor guardrails - [PR #15970](https://github.com/BerriAI/litellm/pull/15970)

- **[Lasso Security](../../docs/proxy/guardrails)**
    - Upgrade to Lasso API v3 and fix ULID generation - [PR #15941](https://github.com/BerriAI/litellm/pull/15941)

- **[PANW Prisma AIRS](../../docs/proxy/guardrails)**
    - Add per-request profile overrides to PANW Prisma AIRS - [PR #16069](https://github.com/BerriAI/litellm/pull/16069)

- **[Grayswan](../../docs/proxy/guardrails)**
    - Improve Grayswan guardrail documentation - [PR #15875](https://github.com/BerriAI/litellm/pull/15875)

- **[Pillar AI](../../docs/proxy/guardrails)**
    - Graceful degradation for pillar service when using litellm - [PR #15857](https://github.com/BerriAI/litellm/pull/15857)

- **General**
    - Ensure Key Guardrails are applied - [PR #16025](https://github.com/BerriAI/litellm/pull/16025)

#### Prompt Management

- **[GitLab](../../docs/prompt_management)**
    - Add GitlabPromptCache and enable subfolder access - [PR #15712](https://github.com/BerriAI/litellm/pull/15712)

---

## Spend Tracking, Budgets and Rate Limiting

- **Cost Tracking**
    - Fix spend tracking for OCR/aOCR requests (log `pages_processed` + recognize `OCRResponse`) - [PR #16070](https://github.com/BerriAI/litellm/pull/16070)

- **Rate Limiting**
    - Add support for Batch API Rate limiting - PR1 adds support for input based rate limits - [PR #16075](https://github.com/BerriAI/litellm/pull/16075)
    - Handle multiple rate limit types per descriptor and prevent IndexError - [PR #16039](https://github.com/BerriAI/litellm/pull/16039)

---

## MCP Gateway

- **OAuth**
    - Add support for dynamic client registration - [PR #15921](https://github.com/BerriAI/litellm/pull/15921)
    - Respect X-Forwarded- headers in OAuth endpoints - [PR #16036](https://github.com/BerriAI/litellm/pull/16036)

---

## Performance / Loadbalancing / Reliability improvements

- **Memory Leak Fixes**
    - Fix: prevent httpx DeprecationWarning memory leak in AsyncHTTPHandler - [PR #16024](https://github.com/BerriAI/litellm/pull/16024)
    - Fix: resolve memory accumulation caused by Pydantic 2.11+ deprecation warnings - [PR #16110](https://github.com/BerriAI/litellm/pull/16110)
    - Fix(apscheduler): prevent memory leaks from jitter and frequent job intervals - [PR #15846](https://github.com/BerriAI/litellm/pull/15846)

- **Configuration**
    - Remove minimum validation for cache control injection index - [PR #16149](https://github.com/BerriAI/litellm/pull/16149)
    - Fix prompt_caching.md: wrong prompt_tokens definition - [PR #16044](https://github.com/BerriAI/litellm/pull/16044)


---

## Documentation Updates

- **Provider Documentation**
    - Use custom-llm-provider header in examples - [PR #16055](https://github.com/BerriAI/litellm/pull/16055)
    - Litellm docs readme fixes - [PR #16107](https://github.com/BerriAI/litellm/pull/16107)
    - Readme fixes add supported providers - [PR #16109](https://github.com/BerriAI/litellm/pull/16109)

- **Model References**
    - Add supports vision field to qwen-vl models in model_prices_and_context_window.json - [PR #16106](https://github.com/BerriAI/litellm/pull/16106)

- **General Documentation**
    - 1-79-0 docs - [PR #15936](https://github.com/BerriAI/litellm/pull/15936)
    - Add minimum resource requirement for production - [PR #16146](https://github.com/BerriAI/litellm/pull/16146)

---

## New Contributors

* @RobGeada made their first contribution in [PR #15975](https://github.com/BerriAI/litellm/pull/15975)
* @shanto12 made their first contribution in [PR #15946](https://github.com/BerriAI/litellm/pull/15946)
* @dima-hx430 made their first contribution in [PR #15976](https://github.com/BerriAI/litellm/pull/15976)
* @m-misiura made their first contribution in [PR #15971](https://github.com/BerriAI/litellm/pull/15971)
* @ylgibby made their first contribution in [PR #15947](https://github.com/BerriAI/litellm/pull/15947)
* @Somtom made their first contribution in [PR #15909](https://github.com/BerriAI/litellm/pull/15909)
* @rodolfo-nobrega made their first contribution in [PR #16023](https://github.com/BerriAI/litellm/pull/16023)
* @bernata made their first contribution in [PR #15997](https://github.com/BerriAI/litellm/pull/15997)
* @AlbertDeFusco made their first contribution in [PR #15881](https://github.com/BerriAI/litellm/pull/15881)
* @komarovd95 made their first contribution in [PR #15789](https://github.com/BerriAI/litellm/pull/15789)
* @langpingxue made their first contribution in [PR #15635](https://github.com/BerriAI/litellm/pull/15635)
* @OrionCodeDev made their first contribution in [PR #16070](https://github.com/BerriAI/litellm/pull/16070)
* @sbinnee made their first contribution in [PR #16078](https://github.com/BerriAI/litellm/pull/16078)
* @JetoPistola made their first contribution in [PR #16106](https://github.com/BerriAI/litellm/pull/16106)
* @gvioss made their first contribution in [PR #16093](https://github.com/BerriAI/litellm/pull/16093)
* @pale-aura made their first contribution in [PR #16084](https://github.com/BerriAI/litellm/pull/16084)
* @tanvithakur94 made their first contribution in [PR #16041](https://github.com/BerriAI/litellm/pull/16041)
* @li-boxuan made their first contribution in [PR #16044](https://github.com/BerriAI/litellm/pull/16044)
* @1stprinciple made their first contribution in [PR #15938](https://github.com/BerriAI/litellm/pull/15938)
* @raghav-stripe made their first contribution in [PR #16137](https://github.com/BerriAI/litellm/pull/16137)
* @steve-gore-snapdocs made their first contribution in [PR #16149](https://github.com/BerriAI/litellm/pull/16149)

---

## Full Changelog

**[View complete changelog on GitHub](https://github.com/BerriAI/litellm/compare/v1.79.0-stable...v1.80.0-stable)**

