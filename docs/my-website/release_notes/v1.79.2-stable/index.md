---
title: "v1.79.2-stable - OCR API Support, XAI Responses API, Secret Manager Integrations"
slug: "v1-79-2"
date: 2025-11-08T10:00:00
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
ghcr.io/berriai/litellm:v1.79.2-stable
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.79.2
```

</TabItem>
</Tabs>

---

## Key Highlights

- **OCR API Support** - VertexAI and Azure AI Doc Intelligence OCR providers with cost tracking
- **Secret Manager Integrations** - CyberArk and Custom Secret Manager support with key rotation
- **Search API Expansion** - Firecrawl and SearXNG search providers with tiered pricing
- **Bedrock Agentcore Provider** - New provider support for AWS Bedrock Agentcore
- **LiteLLM Custom Guardrail** - Built-in guardrail with UI configuration support

---

## New Models / Updated Models

#### New Model Support

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| Azure | `azure/gpt-5-pro` | 272K | $15.00 | $120.00 | Responses API, reasoning, vision, PDF input |
| Azure | `azure/gpt-image-1-mini` | - | - | - | Image generation - per pixel pricing |
| Azure | `azure/container` | - | - | - | Container API - $0.03/session |
| OpenAI | `openai/container` | - | - | - | Container API - $0.03/session |
| Cohere | `cohere/embed-v4.0` | 128K | $0.12 | - | Embeddings with image input support |
| Gemini | `gemini/gemini-live-2.5-flash-preview-native-audio-09-2025` | 1M | $0.30 | $2.00 | Native audio, vision, web search |
| Vertex AI | `vertex_ai/minimaxai/minimax-m2-maas` | 196K | $0.30 | $1.20 | Function calling, tool choice |
| NVIDIA | `nvidia/nemotron-nano-9b-v2` | - | - | - | Chat completions |

#### OCR Models

| Provider | Model | Cost Per Page | Features |
| -------- | ----- | ------------- | -------- |
| Azure AI | `azure_ai/doc-intelligence/prebuilt-read` | $0.0015 | Document reading |
| Azure AI | `azure_ai/doc-intelligence/prebuilt-layout` | $0.01 | Layout analysis |
| Azure AI | `azure_ai/doc-intelligence/prebuilt-document` | $0.01 | Document processing |
| Vertex AI | `vertex_ai/mistral-ocr-2505` | $0.0005 | OCR processing |

#### Search Models

| Provider | Model | Pricing | Features |
| -------- | ----- | ------- | -------- |
| Firecrawl | `firecrawl/search` | Tiered: $0.00166-$0.0166/query | 10-100 results per query |
| SearXNG | `searxng/search` | Free | Open-source metasearch |

#### Features

- **[Azure](../../docs/providers/azure)**
    - Add Azure GPT-5-Pro Responses API support with reasoning capabilities - [PR #16235](https://github.com/BerriAI/litellm/pull/16235)
    - Add gpt-image-1-mini pricing for Azure with quality tiers (low/medium/high) - [PR #16182](https://github.com/BerriAI/litellm/pull/16182)
    - Fix Azure GPT-5 incorrectly routed to O-series config (temperature parameter unsupported) - [PR #16246](https://github.com/BerriAI/litellm/pull/16246)
    - Fix Azure doesn't accept extra body param - [PR #16116](https://github.com/BerriAI/litellm/pull/16116)
    - Fix Azure DALL-E-3 health check content policy violation by using safe default prompt - [PR #16329](https://github.com/BerriAI/litellm/pull/16329)

- **[Bedrock](../../docs/providers/bedrock)**
    - Fix empty assistant message handling in AWS Bedrock Converse API to prevent 400 Bad Request errors - [PR #15850](https://github.com/BerriAI/litellm/pull/15850)
    - Fix: Filter AWS authentication params from Bedrock InvokeModel request body - [PR #16315](https://github.com/BerriAI/litellm/pull/16315)
    - Fix Bedrock proxy adding name to file content, breaks when cache_control in use - [PR #16275](https://github.com/BerriAI/litellm/pull/16275)
    - Fix global.anthropic.claude-haiku-4-5-20251001-v1:0 supports_reasoning flag and update pricing - [PR #16263](https://github.com/BerriAI/litellm/pull/16263)

- **[Gemini (Google AI Studio + Vertex AI)](../../docs/providers/gemini)**
    - Add gemini live audio model cost in model map - [PR #16183](https://github.com/BerriAI/litellm/pull/16183)
    - Fix translation problem with Gemini parallel tool calls - [PR #16194](https://github.com/BerriAI/litellm/pull/16194)
    - Fix: Send Gemini API key via x-goog-api-key header with custom api_base - [PR #16085](https://github.com/BerriAI/litellm/pull/16085)
    - Fix image_config.aspect_ratio not working for gemini-2.5-flash-image - [PR #15999](https://github.com/BerriAI/litellm/pull/15999)
    - Fix Gemini minimal reasoning env overrides disabling thoughts - [PR #16347](https://github.com/BerriAI/litellm/pull/16347)
    - Fix cache_read_input_token_cost for gemini-2.5-flash - [PR #16354](https://github.com/BerriAI/litellm/pull/16354)

- **[Anthropic](../../docs/providers/anthropic)**
    - Fix Anthropic token counting for VertexAI - [PR #16171](https://github.com/BerriAI/litellm/pull/16171)
    - Fix anthropic-adapter: properly translate Anthropic image format to OpenAI - [PR #16202](https://github.com/BerriAI/litellm/pull/16202)
    - Enable automated prompt caching message format for Claude on Databricks - [PR #16200](https://github.com/BerriAI/litellm/pull/16200)
    - Add support for Anthropic Memory Tool - [PR #16115](https://github.com/BerriAI/litellm/pull/16115)
    - Propagate cache creation/read token costs for model info to fix Anthropic long context cost calculations - [PR #16376](https://github.com/BerriAI/litellm/pull/16376)

- **[Vertex AI](../../docs/providers/vertex_ai)**
    - Add Vertex MiniMAX m2 model support - [PR #16373](https://github.com/BerriAI/litellm/pull/16373)
    - Correctly map 429 Resource Exhausted to RateLimitError - [PR #16363](https://github.com/BerriAI/litellm/pull/16363)
    - Add `vertex_credentials` support to `litellm.rerank()` for Vertex AI - [PR #16266](https://github.com/BerriAI/litellm/pull/16266)

- **[Databricks](../../docs/providers/databricks)**
    - Fix databricks streaming - [PR #16368](https://github.com/BerriAI/litellm/pull/16368)

- **[Deepgram](../../docs/providers/deepgram)**
    - Return the diarized transcript when it's required in the request - [PR #16133](https://github.com/BerriAI/litellm/pull/16133)

- **[Fireworks](../../docs/providers/fireworks_ai)**
    - Update Fireworks audio endpoints to new `api.fireworks.ai` domains - [PR #16346](https://github.com/BerriAI/litellm/pull/16346)

- **[Cohere](../../docs/providers/cohere)**
    - Add cohere embed-v4.0 model support - [PR #16358](https://github.com/BerriAI/litellm/pull/16358)

- **[Watsonx](../../docs/providers/watsonx)**
    - Support `reasoning_effort` for watsonx chat models - [PR #16261](https://github.com/BerriAI/litellm/pull/16261)

- **[OpenAI](../../docs/providers/openai)**
    - Remove automatic summary from reasoning_effort transformation - [PR #16210](https://github.com/BerriAI/litellm/pull/16210)

- **[XAI](../../docs/providers/xai)**
    - Remove Grok 4 Models Reasoning Effort Parameter - [PR #16265](https://github.com/BerriAI/litellm/pull/16265)

- **[Hosted VLLM](../../docs/providers/vllm)**
    - Fix HostedVLLMRerankConfig will not be used - [PR #16352](https://github.com/BerriAI/litellm/pull/16352)

#### New Provider Support

- **[Bedrock Agentcore](../../docs/providers/bedrock)**
    - Add Bedrock Agentcore as a provider on LiteLLM Python SDK and LiteLLM AI Gateway - [PR #16252](https://github.com/BerriAI/litellm/pull/16252)

---

## LLM API Endpoints

#### Features

- **[OCR API](../../docs/ocr)**
    - Add VertexAI OCR provider support + cost tracking - [PR #16216](https://github.com/BerriAI/litellm/pull/16216)
    - Add Azure AI Doc Intelligence OCR support - [PR #16219](https://github.com/BerriAI/litellm/pull/16219)

- **[Search API](../../docs/search)**
    - Add firecrawl search API support with tiered pricing - [PR #16257](https://github.com/BerriAI/litellm/pull/16257)
    - Add searxng search API provider - [PR #16259](https://github.com/BerriAI/litellm/pull/16259)

- **[Responses API](../../docs/response_api)**
    - Add shared_session support to responses API - [PR #16260](https://github.com/BerriAI/litellm/pull/16260)
    - Support responses API streaming in langfuse otel - [PR #16153](https://github.com/BerriAI/litellm/pull/16153)
    - Pass extra_body parameters to provider in Responses API requests - [PR #16320](https://github.com/BerriAI/litellm/pull/16320)

- **[Container API](../../docs/container_api)**
    - Add E2E Container API Support - [PR #16136](https://github.com/BerriAI/litellm/pull/16136)
    - Update container documentation to be similar to others - [PR #16327](https://github.com/BerriAI/litellm/pull/16327)

- **[Video Generation API](../../docs/video_generation)**
    - Add `custom_llm_provider` support for video endpoints (non-generation) - [PR #16121](https://github.com/BerriAI/litellm/pull/16121)

- **[Vector Stores](../../docs/vector_stores)**
    - Milvus - search vector store support + support multi-part form data on passthrough - [PR #16035](https://github.com/BerriAI/litellm/pull/16035)
    - Azure AI Vector Stores - support "virtual" indexes + create vector store on passthrough API - [PR #16160](https://github.com/BerriAI/litellm/pull/16160)
    - Milvus - Passthrough API support - adds create + read vector store support via passthrough API's - [PR #16170](https://github.com/BerriAI/litellm/pull/16170)

- **[Embeddings API](../../docs/embedding/supported_embedding)**
    - Use valid CallTypes enum value in embeddings endpoint - [PR #16328](https://github.com/BerriAI/litellm/pull/16328)

- **[Rerank API](../../docs/rerank)**
    - Generalize tiered pricing in generic cost calculator - [PR #16150](https://github.com/BerriAI/litellm/pull/16150)

#### Bugs

- **General**
    - Fix index field not populated in streaming mode with n>1 and tool calls - [PR #15962](https://github.com/BerriAI/litellm/pull/15962)
    - Pass aws_region_name in litellm_params - [PR #16321](https://github.com/BerriAI/litellm/pull/16321)
    - Add `retry-after` header support for errors `502`, `503`, `504` - [PR #16288](https://github.com/BerriAI/litellm/pull/16288)

---

## Management Endpoints / UI

#### Features

- **Virtual Keys**
    - UI - Delete Team Member with friction - [PR #16167](https://github.com/BerriAI/litellm/pull/16167)
    - UI - Litellm test key audio support - [PR #16251](https://github.com/BerriAI/litellm/pull/16251)
    - UI - Test Key Page Revert Model To Single Select - [PR #16390](https://github.com/BerriAI/litellm/pull/16390)

- **Models + Endpoints**
    - UI - Add Model Existing Credentials Improvement - [PR #16166](https://github.com/BerriAI/litellm/pull/16166)
    - UI - Add Azure AD Token field and Azure API Key optional - [PR #16331](https://github.com/BerriAI/litellm/pull/16331)
    - UI - Fixed Label for vLLM in Model Create Flow - [PR #16285](https://github.com/BerriAI/litellm/pull/16285)
    - UI - Include Model Access Group Models on Team Models Table - [PR #16298](https://github.com/BerriAI/litellm/pull/16298)
    - Fix /model_group/info Returning Entire Model List for SSO Users - [PR #16296](https://github.com/BerriAI/litellm/pull/16296)
    - Litellm non root docker Model Hub Table fix - [PR #16282](https://github.com/BerriAI/litellm/pull/16282)

- **Guardrails**
    - UI - Fix regression where Guardrail Entity Could not be selected and entity was not displayed - [PR #16165](https://github.com/BerriAI/litellm/pull/16165)
    - UI - Guardrail Info Page Show PII Config - [PR #16164](https://github.com/BerriAI/litellm/pull/16164)
    - Change guardrail_information to list type - [PR #16127](https://github.com/BerriAI/litellm/pull/16127)
    - UI - LiteLLM Guardrail - ensure you can see UI Friendly name for PII Patterns - [PR #16382](https://github.com/BerriAI/litellm/pull/16382)
    - UI - Guardrails - LiteLLM Content Filter, Allow Viewing/Editing Content Filter Settings - [PR #16383](https://github.com/BerriAI/litellm/pull/16383)
    - UI - Guardrails - allow updating guardrails through UI. Ensure litellm_params actually get updated in memory - [PR #16384](https://github.com/BerriAI/litellm/pull/16384)

- **SSO Settings**
    - Support dot notation on ui sso - [PR #16135](https://github.com/BerriAI/litellm/pull/16135)
    - UI - Prevent trailing slash in sso proxy base url input - [PR #16244](https://github.com/BerriAI/litellm/pull/16244)
    - UI - SSO Proxy Base URL input validation and remove normalizing / - [PR #16332](https://github.com/BerriAI/litellm/pull/16332)
    - UI - Surface SSO Create errors on create flow - [PR #16369](https://github.com/BerriAI/litellm/pull/16369)

- **Usage & Analytics**
    - UI - Tag Usage Top Model Table View and Label Fix - [PR #16249](https://github.com/BerriAI/litellm/pull/16249)
    - UI - Litellm usage date picker - [PR #16264](https://github.com/BerriAI/litellm/pull/16264)
    - Handle None values in daily spend sort key - [PR #16245](https://github.com/BerriAI/litellm/pull/16245)

- **Cache Settings**
    - UI - Cache Settings Redis Add Semantic Cache Settings - [PR #16398](https://github.com/BerriAI/litellm/pull/16398)

- **Admin Settings**
    - UI - Initial changes for supporting prompts to multiple models - [PR #16223](https://github.com/BerriAI/litellm/pull/16223)
    - Building UI for sanity testing - [PR #16399](https://github.com/BerriAI/litellm/pull/16399)

#### Bugs

- **General**
    - UI - Remove encoding_format in request for embedding models - [PR #16367](https://github.com/BerriAI/litellm/pull/16367)
    - UI - Revert Changes for Test Key Multiple Model Select - [PR #16372](https://github.com/BerriAI/litellm/pull/16372)

---

## Logging / Guardrail / Prompt Management Integrations

#### Features

- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Fix langfuse input tokens logic for cached tokens - [PR #16203](https://github.com/BerriAI/litellm/pull/16203)

- **[Opik](../../docs/proxy/logging#opik)**
    - Fix the bug with not incorrect attachment to existing trace & refactor - [PR #15529](https://github.com/BerriAI/litellm/pull/15529)

- **[S3](../../docs/proxy/logging#s3)**
    - S3 logger, add support for ssl_verify when using minio logger - [PR #16211](https://github.com/BerriAI/litellm/pull/16211)
    - Strip base64 in s3 - [PR #16157](https://github.com/BerriAI/litellm/pull/16157)
    - Add allowing Key based prefix to s3 path - [PR #16237](https://github.com/BerriAI/litellm/pull/16237)
    - Add Prometheus metric to track callback logging failures in S3 - [PR #16209](https://github.com/BerriAI/litellm/pull/16209)

- **[OpenTelemetry](../../docs/proxy/logging#opentelemetry)**
    - OTEL - Log Cost Breakdown on OTEL Logger - [PR #16334](https://github.com/BerriAI/litellm/pull/16334)

- **[DataDog](../../docs/proxy/logging#datadog)**
    - Add DD Agent Host support for `datadog` callback - [PR #16379](https://github.com/BerriAI/litellm/pull/16379)

#### Guardrails

- **[Noma](../../docs/proxy/guardrails)**
    - Revert Noma Apply Guardrail implementation - [PR #16214](https://github.com/BerriAI/litellm/pull/16214)
    - Litellm noma guardrail support images - [PR #16199](https://github.com/BerriAI/litellm/pull/16199)

- **[PANW Prisma AIRS](../../docs/proxy/guardrails)**
    - PANW prisma airs guardrail deduplication and enhanced session tracking - [PR #16273](https://github.com/BerriAI/litellm/pull/16273)

- **[LiteLLM Custom Guardrail](../../docs/proxy/guardrails)**
    - Add LiteLLM Gateway built in guardrail - [PR #16338](https://github.com/BerriAI/litellm/pull/16338)
    - UI - Allow configuring LiteLLM Custom Guardrail - [PR #16339](https://github.com/BerriAI/litellm/pull/16339)

---

## Spend Tracking, Budgets and Rate Limiting

- **Cost Tracking**
    - Fix OpenAI Responses API streaming tests usage field names and cost calculation - [PR #16236](https://github.com/BerriAI/litellm/pull/16236)

---

## MCP Gateway

- **Configuration**
    - Configure static mcp header - [PR #16179](https://github.com/BerriAI/litellm/pull/16179)
    - Persist mcp credentials in db - [PR #16308](https://github.com/BerriAI/litellm/pull/16308)

---

## Secret Managers

- **[CyberArk](../../docs/secret_managers)**
    - Add CyberArk Secrets Manager Integration - [PR #16278](https://github.com/BerriAI/litellm/pull/16278)
    - Cyber Ark - Add Key Rotations support - [PR #16289](https://github.com/BerriAI/litellm/pull/16289)

- **[HashiCorp Vault](../../docs/secret_managers)**
    - Add configurable mount name and path prefix for HashiCorp Vault - [PR #16253](https://github.com/BerriAI/litellm/pull/16253)
    - Secret Manager - Hashicorp, add auth via approle - [PR #16374](https://github.com/BerriAI/litellm/pull/16374)

- **[AWS Secrets Manager](../../docs/secret_managers)**
    - Add tags and descriptions support to aws secrets manager - [PR #16224](https://github.com/BerriAI/litellm/pull/16224)

- **[Custom Secret Manager](../../docs/secret_managers)**
    - Add Custom Secret Manager - Allow users to define and write a custom secret manager - [PR #16297](https://github.com/BerriAI/litellm/pull/16297)

- **General**
    - Email Notifications - Ensure Users get Key Rotated Email - [PR #16292](https://github.com/BerriAI/litellm/pull/16292)
    - Fix verify ssl on sts boto3 - [PR #16313](https://github.com/BerriAI/litellm/pull/16313)

---

## Performance / Loadbalancing / Reliability improvements

- **Memory Leak Fixes**
    - Resolve memory accumulation caused by Pydantic 2.11+ deprecation warnings - [PR #16110](https://github.com/BerriAI/litellm/pull/16110)

- **Error Handling**
    - Gracefully handle connection closed errors during streaming - [PR #16294](https://github.com/BerriAI/litellm/pull/16294)

- **Configuration**
    - Remove minimum validation for cache control injection index - [PR #16149](https://github.com/BerriAI/litellm/pull/16149)
    - Improve clearing logic - only remove unvisited endpoints - [PR #16400](https://github.com/BerriAI/litellm/pull/16400)

- **Redis**
    - Handle float redis_version from AWS ElastiCache Valkey - [PR #16207](https://github.com/BerriAI/litellm/pull/16207)

- **Hooks**
    - Add parallel execution handling in during_call_hook - [PR #16279](https://github.com/BerriAI/litellm/pull/16279)

- **Type Safety**
    - Resolve MyPy type checking errors and CI linting - [PR #16277](https://github.com/BerriAI/litellm/pull/16277)
    - Fix MyPy errors for aembedding call_type - CI pass - [PR #16360](https://github.com/BerriAI/litellm/pull/16360)

---

## Documentation Updates

- **Provider Documentation**
    - Docs - v1.79.1 - [PR #16163](https://github.com/BerriAI/litellm/pull/16163)
    - Fix broken link on model_management.md - [PR #16217](https://github.com/BerriAI/litellm/pull/16217)
    - Fix image generation response format - use 'images' array instead of 'image' object - [PR #16378](https://github.com/BerriAI/litellm/pull/16378)

- **General Documentation**
    - Add minimum resource requirement for production - [PR #16146](https://github.com/BerriAI/litellm/pull/16146)
    - Add benchmark comparison with other AI gateways - [PR #16248](https://github.com/BerriAI/litellm/pull/16248)
    - Fix typo of the word orginal - [PR #16255](https://github.com/BerriAI/litellm/pull/16255)

- **Security**
    - Remove tornado test files (including test.key), fixes Python 3.13 security issues - [PR #16342](https://github.com/BerriAI/litellm/pull/16342)

---

## New Contributors

* @steve-gore-snapdocs made their first contribution in [PR #16149](https://github.com/BerriAI/litellm/pull/16149)
* @timbmg made their first contribution in [PR #16120](https://github.com/BerriAI/litellm/pull/16120)
* @Nivg made their first contribution in [PR #16202](https://github.com/BerriAI/litellm/pull/16202)
* @pablobgar made their first contribution in [PR #16194](https://github.com/BerriAI/litellm/pull/16194)
* @AlanPonnachan made their first contribution in [PR #16150](https://github.com/BerriAI/litellm/pull/16150)
* @Chesars made their first contribution in [PR #16236](https://github.com/BerriAI/litellm/pull/16236)
* @bowenliang123 made their first contribution in [PR #16255](https://github.com/BerriAI/litellm/pull/16255)
* @dean-zavad made their first contribution in [PR #16199](https://github.com/BerriAI/litellm/pull/16199)
* @alexkuzmik made their first contribution in [PR #15529](https://github.com/BerriAI/litellm/pull/15529)
* @Granine made their first contribution in [PR #16281](https://github.com/BerriAI/litellm/pull/16281)
* @Oodapow made their first contribution in [PR #16279](https://github.com/BerriAI/litellm/pull/16279)
* @jgoodyear made their first contribution in [PR #16275](https://github.com/BerriAI/litellm/pull/16275)
* @Qanpi made their first contribution in [PR #16321](https://github.com/BerriAI/litellm/pull/16321)
* @ShimonMimoun made their first contribution in [PR #16313](https://github.com/BerriAI/litellm/pull/16313)
* @andriykislitsyn made their first contribution in [PR #16288](https://github.com/BerriAI/litellm/pull/16288)
* @reckless-huang made their first contribution in [PR #16263](https://github.com/BerriAI/litellm/pull/16263)
* @chenmoneygithub made their first contribution in [PR #16368](https://github.com/BerriAI/litellm/pull/16368)
* @stembe-digitalex made their first contribution in [PR #16354](https://github.com/BerriAI/litellm/pull/16354)
* @jfcherng made their first contribution in [PR #16352](https://github.com/BerriAI/litellm/pull/16352)
* @xingyaoww made their first contribution in [PR #16246](https://github.com/BerriAI/litellm/pull/16246)
* @emerzon made their first contribution in [PR #16373](https://github.com/BerriAI/litellm/pull/16373)
* @wwwillchen made their first contribution in [PR #16376](https://github.com/BerriAI/litellm/pull/16376)
* @fabriciojoc made their first contribution in [PR #16203](https://github.com/BerriAI/litellm/pull/16203)
* @jroberts2600 made their first contribution in [PR #16273](https://github.com/BerriAI/litellm/pull/16273)

---

## Full Changelog

**[View complete changelog on GitHub](https://github.com/BerriAI/litellm/compare/v1.79.1-nightly...v1.79.2.rc.1)**


