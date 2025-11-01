---
title: "v1.79.0-stable - Search APIs"
slug: "v1-79-0"
date: 2025-10-26T10:00:00
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
ghcr.io/berriai/litellm:v1.79.0-stable
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.79.0
```

</TabItem>
</Tabs>

---

## Major Changes

- **Cohere models will now be routed to Cohere v2 API by default** - [PR #15722](https://github.com/BerriAI/litellm/pull/15722)

---

## Key Highlights

- **Search APIs** - Native `/v1/search` endpoint with support for Perplexity, Tavily, Parallel AI, Exa AI, DataforSEO, and Google PSE with cost tracking
- **Vector Stores** - Vertex AI Search API integration as vector store through LiteLLM with passthrough endpoint support
- **Guardrails Expansion** - Apply guardrails across Responses API, Image Gen, Text completions, Audio transcriptions, Audio Speech, Rerank, and Anthropic Messages API via unified `apply_guardrails` function
- **New Guardrail Providers** - Gray Swan, Dynamo AI, IBM Guardrails, Lasso Security v3, and Bedrock Guardrail apply_guardrail endpoint support
- **Video Generation API** - Native support for OpenAI Sora-2 and Azure Sora-2 (Pro, Pro-High-Res) with cost tracking and logging support
- **Azure AI Speech (TTS)** - Native Azure AI Speech integration with cost tracking for standard and HD voices

---

## New Models / Updated Models

#### New Model Support

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| Bedrock | `anthropic.claude-3-7-sonnet-20240620-v1:0` | 200K | $3.60 | $18.00 | Chat, reasoning, vision, function calling, prompt caching, computer use |
| Bedrock GovCloud | `us-gov-west-1/anthropic.claude-3-7-sonnet-20250219-v1:0` | 200K | $3.60 | $18.00 | Chat, reasoning, vision, function calling, prompt caching, computer use |
| Vertex AI | `mistral-medium-3` | 128K | $0.40 | $2.00 | Chat, function calling, tool choice |
| Vertex AI | `codestral-2` | 128K | $0.30 | $0.90 | Chat, function calling, tool choice |
| Bedrock | `amazon.titan-image-generator-v1` | - | - | - | Image generation - $0.008/image, $0.01/premium image |
| Bedrock | `amazon.titan-image-generator-v2` | - | - | - | Image generation - $0.008/image, $0.01/premium image |
| OpenAI | `sora-2` | - | - | - | Video generation - $0.10/video/second |
| Azure | `sora-2` | - | - | - | Video generation - $0.10/video/second |
| Azure | `sora-2-pro` | - | - | - | Video generation - $0.30/video/second |
| Azure | `sora-2-pro-high-res` | - | - | - | Video generation - $0.50/video/second |

#### Features

- **[Anthropic](../../docs/providers/anthropic)**
    - Fix cache_control incorrectly applied to all content items instead of last item only - [PR #15699](https://github.com/BerriAI/litellm/pull/15699)
    - Forward anthropic-beta headers to Bedrock, VertexAI - [PR #15700](https://github.com/BerriAI/litellm/pull/15700)
    - Change max_tokens value to match max_output_tokens for claude sonnet - [PR #15715](https://github.com/BerriAI/litellm/pull/15715)

- **[Bedrock](../../docs/providers/bedrock)**
    - Add AWS us-gov-west-1 Claude 3.7 Sonnet costs - [PR #15775](https://github.com/BerriAI/litellm/pull/15775)
    - Fix the date for sonnet 3.7 in govcloud - [PR #15800](https://github.com/BerriAI/litellm/pull/15800)
    - Use proper bedrock model name in health check - [PR #15808](https://github.com/BerriAI/litellm/pull/15808)
    - Support for embeddings_by_type Response Format in Bedrock Cohere Embed v1 - [PR #15707](https://github.com/BerriAI/litellm/pull/15707)
    - Add titan image generations with cost tracking - [PR #15916](https://github.com/BerriAI/litellm/pull/15916)

- **[Gemini](../../docs/providers/gemini)**
    - Add imageConfig parameter for gemini-2.5-flash-image - [PR #15530](https://github.com/BerriAI/litellm/pull/15530)
    - Replace deprecated gemini-1.5-pro-preview-0514 - [PR #15852](https://github.com/BerriAI/litellm/pull/15852)
    - Update vertex ai gemini costs - [PR #15911](https://github.com/BerriAI/litellm/pull/15911)

- **[Ollama](../../docs/providers/ollama)**
    - Set 'think' to False when reasoning effort is minimal/none/disable - [PR #15763](https://github.com/BerriAI/litellm/pull/15763)
    - Handle parsing ollama chunk error - [PR #15717](https://github.com/BerriAI/litellm/pull/15717)

- **[Vertex AI](../../docs/providers/vertex)**
    - Add mistral medium 3 and Codestral 2 on vertex - [PR #15887](https://github.com/BerriAI/litellm/pull/15887)

- **[Databricks](../../docs/providers/databricks)**
    - Allow prompt caching to be used for Anthropic Claude on Databricks - [PR #15801](https://github.com/BerriAI/litellm/pull/15801)

- **[Azure](../../docs/providers/azure)**
    - Add Azure AVA TTS integration - [PR #15749](https://github.com/BerriAI/litellm/pull/15749)
    - Add Azure AVA (Speech AI) Cost Tracking - [PR #15754](https://github.com/BerriAI/litellm/pull/15754)
    - Azure AI Speech - Ensure `voice` is mapped from request body to SSML body, allow sending `role` and `style` - [PR #15810](https://github.com/BerriAI/litellm/pull/15810)
    - Add Azure support for video generation functionality (Sora-2) - [PR #15901](https://github.com/BerriAI/litellm/pull/15901)

- **[OpenAI](../../docs/providers/openai)**
    - OpenAI videos refactoring - [PR #15900](https://github.com/BerriAI/litellm/pull/15900)

- **General**
    - Read from custom-llm-provider header - [PR #15528](https://github.com/BerriAI/litellm/pull/15528)

---

## LLM API Endpoints

#### Features

- **[Responses API](../../docs/response_api)**
    - Add gpt 4.1 pricing for response endpoint - [PR #15593](https://github.com/BerriAI/litellm/pull/15593)
    - Fix Incorrect status value in responses api with gemini - [PR #15753](https://github.com/BerriAI/litellm/pull/15753)
    - Simplify reasoning item handling for gpt-5-codex - [PR #15815](https://github.com/BerriAI/litellm/pull/15815)
    - ErrorEvent ValidationError when OpenAI Responses API returns nested error structure - [PR #15804](https://github.com/BerriAI/litellm/pull/15804)
    - Fix reasoning item ID auto-generation causing encrypted content verification errors - [PR #15782](https://github.com/BerriAI/litellm/pull/15782)
    - Support tags in metadata - [PR #15867](https://github.com/BerriAI/litellm/pull/15867)
    - Security: prevent User A from retrieving User B's response, if response.id is leaked - [PR #15757](https://github.com/BerriAI/litellm/pull/15757)

- **[Batch API](../../docs/batch_api)**
    - Add pre and post call for list batches - [PR #15673](https://github.com/BerriAI/litellm/pull/15673)
    - Add function responsible to call precall - [PR #15636](https://github.com/BerriAI/litellm/pull/15636)
    - Fix "User default_user_id does not have access to the object" when object not in db - [PR #15873](https://github.com/BerriAI/litellm/pull/15873)

- **[OCR API](../../docs/ocr)**
    - Add Azure AI - OCR to docs - [PR #15768](https://github.com/BerriAI/litellm/pull/15768)
    - Add mode + Health check support for OCR models - [PR #15767](https://github.com/BerriAI/litellm/pull/15767)

- **[Search API](../../docs/search_api)**
    - Add def search() APIs for Web Search - Perplexity API - [PR #15769](https://github.com/BerriAI/litellm/pull/15769)
    - Add Tavily Search API - [PR #15770](https://github.com/BerriAI/litellm/pull/15770)
    - Add Parallel AI - Search API - [PR #15772](https://github.com/BerriAI/litellm/pull/15772)
    - Add EXA AI Search API to LiteLLM - [PR #15774](https://github.com/BerriAI/litellm/pull/15774)
    - Add /search endpoint on LiteLLM Gateway - [PR #15780](https://github.com/BerriAI/litellm/pull/15780)
    - Add DataforSEO Search API - [PR #15817](https://github.com/BerriAI/litellm/pull/15817)
    - Add Google PSE Search Provider - [PR #15816](https://github.com/BerriAI/litellm/pull/15816)
    - Add cost tracking for Search API requests - Google PSE, Tavily, Parallel AI, Exa AI - [PR #15821](https://github.com/BerriAI/litellm/pull/15821)
    - Backend: Allow storing configured Search APIs in DB - [PR #15862](https://github.com/BerriAI/litellm/pull/15862)
    - Exa Search API - ensure request params are sent to Exa AI - [PR #15855](https://github.com/BerriAI/litellm/pull/15855)

- **[Vector Stores](../../docs/vector_stores)**
    - Support Vertex AI Search API as vector store through LiteLLM - [PR #15781](https://github.com/BerriAI/litellm/pull/15781)
    - Azure AI - Search Vector Stores - [PR #15873](https://github.com/BerriAI/litellm/pull/15873)
    - VertexAI Search Vector Store - Passthrough endpoint support + Vector store search Cost tracking support - [PR #15824](https://github.com/BerriAI/litellm/pull/15824)
    - Don't raise error if managed object is not found - [PR #15873](https://github.com/BerriAI/litellm/pull/15873)
    - Show config.yaml vector stores on UI - [PR #15873](https://github.com/BerriAI/litellm/pull/15873)
    - Cost tracking for search spend - [PR #15859](https://github.com/BerriAI/litellm/pull/15859)

- **[Images API](../../docs/image_generation)**
    - Pass user-defined headers and extra_headers to image-edit calls - [PR #15811](https://github.com/BerriAI/litellm/pull/15811)

- **[Video Generation API](../../docs/video_generation)**
    - Add Azure support for video generation functionality (Sora-2, Sora-2-Pro, Sora-2-Pro-High-Res) - [PR #15901](https://github.com/BerriAI/litellm/pull/15901)
    - OpenAI video generation refactoring (Sora-2) - [PR #15900](https://github.com/BerriAI/litellm/pull/15900)

- **[Bedrock /invoke](../../docs/bedrock_invoke)**
    - Fix: Hooks broken on /bedrock passthrough due to missing metadata - [PR #15849](https://github.com/BerriAI/litellm/pull/15849)

- **[Realtime API](../../docs/realtime_api)**
    - Fix: OpenAI Realtime API integration fails due to websockets.exceptions.PayloadTooBig error - [PR #15751](https://github.com/BerriAI/litellm/pull/15751)

---

## Management Endpoints / UI

#### Features

- **Passthrough**
    - Set auth on passthrough endpoints, on the UI - [PR #15778](https://github.com/BerriAI/litellm/pull/15778)
    - Fix pass-through endpoint budget enforcement bug - [PR #15805](https://github.com/BerriAI/litellm/pull/15805)

- **Organizations**
    - Allow org admins to create teams on UI - [PR #15924](https://github.com/BerriAI/litellm/pull/15924)

- **Search Tools**
    - UI - Search Tools, allow adding search tools on UI + testing search - [PR #15871](https://github.com/BerriAI/litellm/pull/15871)
    - UI - Add logos for search providers - [PR #15872](https://github.com/BerriAI/litellm/pull/15872)

- **General**
    - Fix routing for custom server root path - [PR #15701](https://github.com/BerriAI/litellm/pull/15701)

---

## Logging / Guardrail / Prompt Management Integrations

#### Features

- **[OpenTelemetry](../../docs/proxy/logging#opentelemetry)**
    - Fix OpenTelemetry Logging functionality - [PR #15645](https://github.com/BerriAI/litellm/pull/15645)
    - Fix issue where headers were not being split correctly - [PR #15916](https://github.com/BerriAI/litellm/pull/15916)

- **[Sentry](../../docs/proxy/logging#sentry)**
    - Add SENTRY_ENVIRONMENT configuration for Sentry integration - [PR #15760](https://github.com/BerriAI/litellm/pull/15760)

- **[Helicone](../../docs/proxy/logging#helicone)**
    - Fix JSON serialization error in Helicone logging by removing OpenTelemetry span from metadata - [PR #15728](https://github.com/BerriAI/litellm/pull/15728)

- **[MLFlow](../../docs/proxy/logging#mlflow)**
    - Fix MLFlow tags - split request_tags into (key, val) if request_tag has colon - [PR #15914](https://github.com/BerriAI/litellm/pull/15914)

- **General**
    - Rename configured_cold_storage_logger to cold_storage_custom_logger - [PR #15798](https://github.com/BerriAI/litellm/pull/15798)

#### Guardrails

- **[Gray Swan](../../docs/proxy/guardrails)**
    - Add GraySwan Guardrails support - [PR #15756](https://github.com/BerriAI/litellm/pull/15756)
    - Rename GraySwan to Gray Swan - [PR #15771](https://github.com/BerriAI/litellm/pull/15771)

- **[Dynamo AI](../../docs/proxy/guardrails)**
    - New Guardrail - Dynamo AI Guardrail - [PR #15920](https://github.com/BerriAI/litellm/pull/15920)

- **[IBM Guardrails](../../docs/proxy/guardrails)**
    - IBM Guardrails integration - [PR #15924](https://github.com/BerriAI/litellm/pull/15924)

- **[Lasso Security](../../docs/proxy/guardrails)**
    - Add v3 API Support - [PR #12452](https://github.com/BerriAI/litellm/pull/12452)
    - Fixed lasso import config, redis cluster hash tags for test keys - [PR #15917](https://github.com/BerriAI/litellm/pull/15917)

- **[Bedrock Guardrails](../../docs/proxy/guardrails)**
    - Implement Bedrock Guardrail apply_guardrail endpoint support - [PR #15892](https://github.com/BerriAI/litellm/pull/15892)

- **General**
    - Guardrails - Responses API, Image Gen, Text completions, Audio transcriptions, Audio Speech, Rerank, Anthropic Messages API support via the unified `apply_guardrails` function - [PR #15706](https://github.com/BerriAI/litellm/pull/15706)

---

## Spend Tracking, Budgets and Rate Limiting

- **Rate Limiting**
    - Support absolute RPM/TPM in priority_reservation - [PR #15813](https://github.com/BerriAI/litellm/pull/15813)
    - Org level tpm/rpm limits + Team tpm/rpm validation when assigned to org - [PR #15549](https://github.com/BerriAI/litellm/pull/15549)

---

## MCP Gateway

- **OAuth**
    - Auth Header Fix for MCP Tool Call - [PR #15736](https://github.com/BerriAI/litellm/pull/15736)
    - Add response_type + PKCE parameters to OAuth authorization endpoint - [PR #15720](https://github.com/BerriAI/litellm/pull/15720)

---

## Performance / Loadbalancing / Reliability improvements

- **Database**
    - Minimize the occurrence of deadlocks - [PR #15281](https://github.com/BerriAI/litellm/pull/15281)

- **Redis**
    - Apply max_connections configuration to Redis async client - [PR #15797](https://github.com/BerriAI/litellm/pull/15797)

- **Caching**
    - Add documentation for `enable_caching_on_provider_specific_optional_params` setting - [PR #15885](https://github.com/BerriAI/litellm/pull/15885)

---

## Documentation Updates

- **Provider Documentation**
    - Update worker recommendation - [PR #15702](https://github.com/BerriAI/litellm/pull/15702)
    - Fix the wrong request body in json mode doc - [PR #15729](https://github.com/BerriAI/litellm/pull/15729)
    - Add details in docs - [PR #15721](https://github.com/BerriAI/litellm/pull/15721)
    - Add responses api on openai docs - [PR #15866](https://github.com/BerriAI/litellm/pull/15866)
    - Add OpenAI responses api - [PR #15868](https://github.com/BerriAI/litellm/pull/15868)

---

## New Contributors

* @tlecomte made their first contribution in [PR #15528](https://github.com/BerriAI/litellm/pull/15528)
* @tomhaynes made their first contribution in [PR #15645](https://github.com/BerriAI/litellm/pull/15645)
* @talalryz made their first contribution in [PR #15720](https://github.com/BerriAI/litellm/pull/15720)
* @1vinodsingh1 made their first contribution in [PR #15736](https://github.com/BerriAI/litellm/pull/15736)
* @nuernber made their first contribution in [PR #15775](https://github.com/BerriAI/litellm/pull/15775)
* @Thomas-Mildner made their first contribution in [PR #15760](https://github.com/BerriAI/litellm/pull/15760)
* @javiergarciapleo made their first contribution in [PR #15721](https://github.com/BerriAI/litellm/pull/15721)
* @lshgdut made their first contribution in [PR #15717](https://github.com/BerriAI/litellm/pull/15717)
* @kk-wangjifeng made their first contribution in [PR #15530](https://github.com/BerriAI/litellm/pull/15530)
* @anthonyivn2 made their first contribution in [PR #15801](https://github.com/BerriAI/litellm/pull/15801)
* @romanglo made their first contribution in [PR #15707](https://github.com/BerriAI/litellm/pull/15707)
* @mythral made their first contribution in [PR #15859](https://github.com/BerriAI/litellm/pull/15859)
* @mubashirosmani made their first contribution in [PR #15866](https://github.com/BerriAI/litellm/pull/15866)
* @CAFxX made their first contribution in [PR #15281](https://github.com/BerriAI/litellm/pull/15281)
* @reflection made their first contribution in [PR #15914](https://github.com/BerriAI/litellm/pull/15914)
* @shadielfares made their first contribution in [PR #15917](https://github.com/BerriAI/litellm/pull/15917)

---

## PR Count Summary

### 10/26/2025
* New Models / Updated Models: 20
* LLM API Endpoints: 29
* Management Endpoints / UI: 5
* Logging / Guardrail / Prompt Management Integrations: 10
* Spend Tracking, Budgets and Rate Limiting: 2
* MCP Gateway: 2
* Performance / Loadbalancing / Reliability improvements: 3
* Documentation Updates: 5

---

## Full Changelog

**[View complete changelog on GitHub](https://github.com/BerriAI/litellm/compare/v1.78.5-stable...v1.79.0-stable)**

