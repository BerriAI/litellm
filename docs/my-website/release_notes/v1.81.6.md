---
title: "[Preview] v1.81.6 - Logs v2 with Tool Call Tracing"
slug: "v1-81-6"
date: 2026-01-31T00:00:00
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

:::danger Known Issue - CPU Usage

This release had known issues with CPU usage. This has been fixed in [v1.81.9-stable](./v1-81-9).

**We recommend using v1.81.9-stable instead.**

:::

## Deploy this version

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

<Tabs>
<TabItem value="docker" label="Docker">

```bash
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
docker.litellm.ai/berriai/litellm:main-v1.81.6
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.81.6
```

</TabItem>
</Tabs>

## Key Highlights

Logs View v2 with Tool Call Tracing - Redesigned logs interface with side panel, structured tool visualization, and error message search for faster debugging.

Let's dive in.

### Logs View v2 with Tool Call Tracing

This release introduces comprehensive tool call tracing through LiteLLM's redesigned Logs View v2, enabling developers to debug and monitor AI agent workflows in production environments seamlessly.

This means you can now onboard use cases like tracing complex multi-step agent interactions, debugging tool execution failures, and monitoring MCP server calls while maintaining full visibility into request/response payloads with syntax highlighting.

Developers can access the new Logs View through LiteLLM's UI to inspect tool calls in structured format, search logs by error messages or request patterns, and correlate agent activities across sessions with collapsible side panel views.

{/* TODO: Add image from Slack (group_7219.png) - save as logs_v2_tool_tracing.png */}
{/* <Image img={require('../../img/release_notes/logs_v2_tool_tracing.png')} style={{ maxWidth: '800px', width: '100%' }} /> */}

[Get Started](../../docs/proxy/ui_logs)

## New Models / Updated Models

#### New Model Support

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| AWS Bedrock | `amazon.nova-2-pro-preview-20251202-v1:0` | 1M | $2.19 | $17.50 | Chat completions, vision, video, PDF, function calling, prompt caching, reasoning |
| Google Vertex AI | `gemini-robotics-er-1.5-preview` | 1M | $0.30 | $2.50 | Chat completions, multimodal (text, image, video, audio), function calling, reasoning |
| OpenRouter | `openrouter/xiaomi/mimo-v2-flash` | 262K | $0.09 | $0.29 | Chat completions, function calling, reasoning |
| OpenRouter | `openrouter/moonshotai/kimi-k2.5` | - | - | - | Chat completions |
| OpenRouter | `openrouter/z-ai/glm-4.7` | 202K | $0.40 | $1.50 | Chat completions, vision, function calling, reasoning |

#### Features

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Messages API Bedrock Converse caching and PDF support - [PR #19785](https://github.com/BerriAI/litellm/pull/19785)
    - Translate advanced-tool-use to Bedrock-specific headers for Claude Opus 4.5 - [PR #19841](https://github.com/BerriAI/litellm/pull/19841)
    - Support tool search header translation for Sonnet 4.5 - [PR #19871](https://github.com/BerriAI/litellm/pull/19871)
    - Filter unsupported beta headers for AWS Bedrock Invoke API - [PR #19877](https://github.com/BerriAI/litellm/pull/19877)
    - Nova grounding improvements - [PR #19598](https://github.com/BerriAI/litellm/pull/19598), [PR #20159](https://github.com/BerriAI/litellm/pull/20159)

- **[Anthropic](../../docs/providers/anthropic)**
    - Remove explicit cache_control null in tool_result content - [PR #19919](https://github.com/BerriAI/litellm/pull/19919)
    - Fix tool handling - [PR #19805](https://github.com/BerriAI/litellm/pull/19805)

- **[Google Gemini / Vertex AI](../../docs/providers/gemini)**
    - Add Gemini Robotics-ER 1.5 preview support - [PR #19845](https://github.com/BerriAI/litellm/pull/19845)
    - Support file retrieval in GoogleAIStudioFilesHandle - [PR #20018](https://github.com/BerriAI/litellm/pull/20018)
    - Add /delete endpoint support - [PR #20055](https://github.com/BerriAI/litellm/pull/20055)
    - Add custom_llm_provider as gemini translation - [PR #19988](https://github.com/BerriAI/litellm/pull/19988)
    - Subtract implicit cached tokens from text_tokens for correct cost calculation - [PR #19775](https://github.com/BerriAI/litellm/pull/19775)
    - Remove unsupported prompt-caching-scope-2026-01-05 header for vertex ai - [PR #20058](https://github.com/BerriAI/litellm/pull/20058)
    - Add disable flag for anthropic gemini cache translation - [PR #20052](https://github.com/BerriAI/litellm/pull/20052)
    - Convert image URLs to base64 in tool messages for Anthropic on Vertex AI - [PR #19896](https://github.com/BerriAI/litellm/pull/19896)

- **[xAI](../../docs/providers/xai)**
    - Add grok reasoning content support - [PR #19850](https://github.com/BerriAI/litellm/pull/19850)
    - Add websearch params support for Responses API - [PR #19915](https://github.com/BerriAI/litellm/pull/19915)
    - Add routing of xai chat completions to responses when web search options is present - [PR #20051](https://github.com/BerriAI/litellm/pull/20051)
    - Correct cached token cost calculation - [PR #19772](https://github.com/BerriAI/litellm/pull/19772)

- **[Azure OpenAI](../../docs/providers/azure)**
    - Use generic cost calculator for audio token pricing - [PR #19771](https://github.com/BerriAI/litellm/pull/19771)
    - Allow tool_choice for Azure GPT-5 chat models - [PR #19813](https://github.com/BerriAI/litellm/pull/19813)
    - Set gpt-5.2-codex mode to responses for Azure and OpenRouter - [PR #19770](https://github.com/BerriAI/litellm/pull/19770)

- **[OpenAI](../../docs/providers/openai)**
    - Fix max_input_tokens for gpt-5.2-codex - [PR #20009](https://github.com/BerriAI/litellm/pull/20009)
    - Fix gpt-image-1.5 cost calculation not including output image tokens - [PR #19515](https://github.com/BerriAI/litellm/pull/19515)

- **[Hosted VLLM](../../docs/providers/vllm)**
    - Support thinking parameter in anthropic_messages() and .completion() - [PR #19787](https://github.com/BerriAI/litellm/pull/19787)
    - Route through base_llm_http_handler to support ssl_verify - [PR #19893](https://github.com/BerriAI/litellm/pull/19893)
    - Fix vllm embedding format - [PR #20056](https://github.com/BerriAI/litellm/pull/20056)

- **[OCI GenAI](../../docs/providers/oci)**
    - Serialize imageUrl as object for OCI GenAI API - [PR #19661](https://github.com/BerriAI/litellm/pull/19661)

- **[Volcengine](../../docs/providers/volcano)**
    - Add context for volcengine models (deepseek-v3-2, glm-4-7, kimi-k2-thinking) - [PR #19335](https://github.com/BerriAI/litellm/pull/19335)

- **[Chinese Providers](../../docs/providers/)**
    - Add prompt caching and reasoning support for MiniMax, GLM, Xiaomi - [PR #19924](https://github.com/BerriAI/litellm/pull/19924)

- **[Vercel AI Gateway](../../docs/providers/vercel_ai_gateway)**
    - Add embeddings support - [PR #19660](https://github.com/BerriAI/litellm/pull/19660)

### Bug Fixes

- **[Google](../../docs/providers/gemini)**
    - Fix gemini-robotics-er-1.5-preview entry - [PR #19974](https://github.com/BerriAI/litellm/pull/19974)

- **General**
    - Fix output_tokens_details.reasoning_tokens None - [PR #19914](https://github.com/BerriAI/litellm/pull/19914)
    - Fix stream_chunk_builder to preserve images from streaming chunks - [PR #19654](https://github.com/BerriAI/litellm/pull/19654)
    - Fix aspectRatio mapping in image edit - [PR #20053](https://github.com/BerriAI/litellm/pull/20053)
    - Handle unknown models in Azure AI cost calculator - [PR #20150](https://github.com/BerriAI/litellm/pull/20150)

- **[GigaChat](../../docs/providers/gigachat)**
    - Ensure function content is valid JSON - [PR #19232](https://github.com/BerriAI/litellm/pull/19232)

## LLM API Endpoints

#### Features

- **[Messages API (/messages)](../../docs/mcp)**
    - Add LiteLLM x Claude Agent SDK Integration - [PR #20035](https://github.com/BerriAI/litellm/pull/20035)

- **[A2A / MCP Gateway API (/a2a, /mcp)](../../docs/mcp)**
    - Add A2A agent header-based context propagation support - [PR #19504](https://github.com/BerriAI/litellm/pull/19504)
    - Enable progress notifications for MCP tool calls - [PR #19809](https://github.com/BerriAI/litellm/pull/19809)
    - Fix support for non-standard MCP URL patterns - [PR #19738](https://github.com/BerriAI/litellm/pull/19738)
    - Add backward compatibility for legacy A2A card formats (/.well-known/agent.json) - [PR #19949](https://github.com/BerriAI/litellm/pull/19949)
    - Add support for agent parameter in /interactions endpoint - [PR #19866](https://github.com/BerriAI/litellm/pull/19866)

- **[Responses API (/responses)](../../docs/response_api)**
    - Fix custom_llm_provider for provider-specific params - [PR #19798](https://github.com/BerriAI/litellm/pull/19798)
    - Extract input tokens details as dict in ResponseAPILoggingUtils - [PR #20046](https://github.com/BerriAI/litellm/pull/20046)

- **[Batch API (/batches)](../../docs/batches)**
    - Fix /batches to return encoded ids (from managed objects table) - [PR #19040](https://github.com/BerriAI/litellm/pull/19040)
    - Fix Batch and File user level permissions - [PR #19981](https://github.com/BerriAI/litellm/pull/19981)
    - Add cost tracking and usage object in retrieve_batch call type - [PR #19986](https://github.com/BerriAI/litellm/pull/19986)

- **[Embeddings API (/embeddings)](../../docs/embedding/supported_embedding)**
    - Add supported input formats documentation - [PR #20073](https://github.com/BerriAI/litellm/pull/20073)

- **[RAG API (/rag/ingest, /vector_store)](../../docs/rag_ingest)**
    - Add UI for /rag/ingest API - Upload docs, pdfs etc to create vector stores - [PR #19822](https://github.com/BerriAI/litellm/pull/19822)
    - Add support for using S3 Vectors as Vector Store Provider - [PR #19888](https://github.com/BerriAI/litellm/pull/19888)
    - Add s3_vectors as provider on /vector_store/search API + UI for creating + PDF support - [PR #19895](https://github.com/BerriAI/litellm/pull/19895)
    - Add permission management for users and teams on Vector Stores - [PR #19972](https://github.com/BerriAI/litellm/pull/19972)
    - Enable router support for completions in RAG query pipeline - [PR #19550](https://github.com/BerriAI/litellm/pull/19550)

- **[Search API (/search)](../../docs/search)**
    - Add /list endpoint to list what search tools exist in router - [PR #19969](https://github.com/BerriAI/litellm/pull/19969)
    - Fix router search tools v2 integration - [PR #19840](https://github.com/BerriAI/litellm/pull/19840)

- **[Passthrough Endpoints (/\{provider\}_passthrough)](../../docs/pass_through/intro)**
    - Add /openai_passthrough route for OpenAI passthrough requests - [PR #19989](https://github.com/BerriAI/litellm/pull/19989)
    - Add support for configuring role_mappings via environment variables - [PR #19498](https://github.com/BerriAI/litellm/pull/19498)
    - Add Vertex AI LLM credentials sensitive keyword "vertex_credentials" for masking - [PR #19551](https://github.com/BerriAI/litellm/pull/19551)
    - Fix prevention of provider-prefixed model name leaks in responses - [PR #19943](https://github.com/BerriAI/litellm/pull/19943)
    - Fix proxy support for slashes in Google Vertex generateContent model names - [PR #19737](https://github.com/BerriAI/litellm/pull/19737), [PR #19753](https://github.com/BerriAI/litellm/pull/19753)
    - Support model names with slashes in Vertex AI passthrough URLs - [PR #19944](https://github.com/BerriAI/litellm/pull/19944)
    - Fix regression in Vertex AI passthroughs for router models - [PR #19967](https://github.com/BerriAI/litellm/pull/19967)
    - Add regression tests for Vertex AI passthrough model names - [PR #19855](https://github.com/BerriAI/litellm/pull/19855)

#### Bugs

- **General**
    - Fix token calculations and refactor - [PR #19696](https://github.com/BerriAI/litellm/pull/19696)

## Management Endpoints / UI

#### Features

- **Proxy CLI Auth**
    - Add configurable CLI JWT expiration via environment variable - [PR #19780](https://github.com/BerriAI/litellm/pull/19780)
    - Fix team cli auth flow - [PR #19666](https://github.com/BerriAI/litellm/pull/19666)

- **Virtual Keys**
    - UI: Auto Truncation of Table Values - [PR #19718](https://github.com/BerriAI/litellm/pull/19718)
    - Fix Create Key: Expire Key Input Duration - [PR #19807](https://github.com/BerriAI/litellm/pull/19807)
    - Bulk Update Keys Endpoint - [PR #19886](https://github.com/BerriAI/litellm/pull/19886)

- **Logs View**
    - **v2 Logs view with side panel and improved UX** - [PR #20091](https://github.com/BerriAI/litellm/pull/20091)
    - New View to render "Tools" on Logs View - [PR #20093](https://github.com/BerriAI/litellm/pull/20093)
    - Add Pretty print view of request/response - [PR #20096](https://github.com/BerriAI/litellm/pull/20096)
    - Add error_message search in Spend Logs Endpoint - [PR #19960](https://github.com/BerriAI/litellm/pull/19960)
    - UI: Adding Error message search to ui spend logs - [PR #19963](https://github.com/BerriAI/litellm/pull/19963)
    - Spend Logs: Settings Modal - [PR #19918](https://github.com/BerriAI/litellm/pull/19918)
    - Fix error_code in Spend Logs metadata - [PR #20015](https://github.com/BerriAI/litellm/pull/20015)
    - Spend Logs: Show Current Store and Retention Status - [PR #20017](https://github.com/BerriAI/litellm/pull/20017)
    - Allow Dynamic Setting of store_prompts_in_spend_logs - [PR #19913](https://github.com/BerriAI/litellm/pull/19913)
    - [Docs: UI Spend Logs Settings](../../docs/proxy/ui_spend_log_settings) - [PR #20197](https://github.com/BerriAI/litellm/pull/20197)

- **Models + Endpoints**
    - Add sortBy and sortOrder params for /v2/model/info - [PR #19903](https://github.com/BerriAI/litellm/pull/19903)
    - Fix Sorting for /v2/model/info - [PR #19971](https://github.com/BerriAI/litellm/pull/19971)
    - UI: Model Page Server Sort - [PR #19908](https://github.com/BerriAI/litellm/pull/19908)

- **Usage & Analytics**
    - UI: Usage Export: Breakdown by Teams and Keys - [PR #19953](https://github.com/BerriAI/litellm/pull/19953)
    - UI: Usage: Model Breakdown Per Key - [PR #20039](https://github.com/BerriAI/litellm/pull/20039)

- **UI Improvements**
    - UI: Allow Admins to control what pages are visible on LeftNav - [PR #19907](https://github.com/BerriAI/litellm/pull/19907)
    - UI: Add Light/Dark Mode Switch for Development - [PR #19804](https://github.com/BerriAI/litellm/pull/19804)
    - UI: Dark Mode: Delete Resource Modal - [PR #20098](https://github.com/BerriAI/litellm/pull/20098)
    - UI: Tables: Reusable Table Sort Component - [PR #19970](https://github.com/BerriAI/litellm/pull/19970)
    - UI: New Badge Dot Render - [PR #20024](https://github.com/BerriAI/litellm/pull/20024)
    - UI: Feedback Prompts: Option To Hide Prompts - [PR #19831](https://github.com/BerriAI/litellm/pull/19831)
    - UI: Navbar: Fixed Default Logo + Bound Logo Box - [PR #20092](https://github.com/BerriAI/litellm/pull/20092)
    - UI: Navbar: User Dropdown - [PR #20095](https://github.com/BerriAI/litellm/pull/20095)
    - Change default key type from 'Default' to 'LLM API' - [PR #19516](https://github.com/BerriAI/litellm/pull/19516)

- **Team & User Management**
    - Fix /team/member_add User Email and ID Verifications - [PR #19814](https://github.com/BerriAI/litellm/pull/19814)
    - Fix SSO Email Case Sensitivity - [PR #19799](https://github.com/BerriAI/litellm/pull/19799)
    - UI: Internal User: Bulk Add - [PR #19721](https://github.com/BerriAI/litellm/pull/19721)

- **AI Gateway Features**
    - Add support for making silent LLM calls without logging - [PR #19544](https://github.com/BerriAI/litellm/pull/19544)
    - UI: Fix MCP tools instructions to display comma-separated strings - [PR #20101](https://github.com/BerriAI/litellm/pull/20101)

#### Bugs

- Fix Model Name During Fallback - [PR #20177](https://github.com/BerriAI/litellm/pull/20177)
- Fix Health Endpoints when Callback Objects Defined - [PR #20182](https://github.com/BerriAI/litellm/pull/20182)
- Fix Unable to reset user max budget to unlimited - [PR #19796](https://github.com/BerriAI/litellm/pull/19796)
- Fix Password comparison with non-ASCII characters - [PR #19568](https://github.com/BerriAI/litellm/pull/19568)
- Correct error message for DISABLE_ADMIN_ENDPOINTS - [PR #19861](https://github.com/BerriAI/litellm/pull/19861)
- Prevent clearing content filter patterns when editing guardrail - [PR #19671](https://github.com/BerriAI/litellm/pull/19671)
- Fix Prompt Studio history to load tools and system messages - [PR #19920](https://github.com/BerriAI/litellm/pull/19920)
- Add WATSONX_ZENAPIKEY to WatsonX credentials - [PR #20086](https://github.com/BerriAI/litellm/pull/20086)
- UI: Vector Store: Allow Config Defined Models to Be Selected - [PR #20031](https://github.com/BerriAI/litellm/pull/20031)

## Logging / Guardrail / Prompt Management Integrations

#### Features

- **[DataDog](../../docs/proxy/logging#datadog)**
    - Add agent support for LLM Observability - [PR #19574](https://github.com/BerriAI/litellm/pull/19574)
    - Add datadog cost management support and fix startup callback issue - [PR #19584](https://github.com/BerriAI/litellm/pull/19584)
    - Add datadog_llm_observability to /health/services allowed list - [PR #19952](https://github.com/BerriAI/litellm/pull/19952)
    - Check for agent mode before requiring DD_API_KEY/DD_SITE - [PR #20156](https://github.com/BerriAI/litellm/pull/20156)

- **[OpenTelemetry](../../docs/observability/opentelemetry_integration)**
    - Propagate JWT auth metadata to OTEL spans - [PR #19627](https://github.com/BerriAI/litellm/pull/19627)
    - Fix thread leak in dynamic header path - [PR #19946](https://github.com/BerriAI/litellm/pull/19946)

- **[Prometheus](../../docs/proxy/logging#prometheus)**
    - Add callbacks and labels - [PR #19708](https://github.com/BerriAI/litellm/pull/19708)
    - Add clientip and user agent in metrics - [PR #19717](https://github.com/BerriAI/litellm/pull/19717)
    - Add tpm-rpm limit metrics - [PR #19725](https://github.com/BerriAI/litellm/pull/19725)
    - Add model_id label to metrics - [PR #19678](https://github.com/BerriAI/litellm/pull/19678)
    - Safely handle None metadata in logging - [PR #19691](https://github.com/BerriAI/litellm/pull/19691)
    - Resolve high CPU when router_settings in DB by avoiding REGISTRY.collect() - [PR #20087](https://github.com/BerriAI/litellm/pull/20087)

- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Add litellm_callback_logging_failures_metric for Langfuse, Langfuse Otel and other Otel providers - [PR #19636](https://github.com/BerriAI/litellm/pull/19636)

- **General Logging**
    - Use return value from CustomLogger.async_post_call_success_hook - [PR #19670](https://github.com/BerriAI/litellm/pull/19670)
    - Add async_post_call_response_headers_hook to CustomLogger - [PR #20083](https://github.com/BerriAI/litellm/pull/20083)
    - Add mock client factory pattern and mock support for PostHog, Helicone, and Braintrust integrations - [PR #19707](https://github.com/BerriAI/litellm/pull/19707)

#### Guardrails

- **[Presidio](../../docs/proxy/guardrails/pii_masking_v2)**
    - Reuse HTTP connections to prevent performance degradation - [PR #19964](https://github.com/BerriAI/litellm/pull/19964)

- **Onyx**
    - Add timeout to onyx guardrail - [PR #19731](https://github.com/BerriAI/litellm/pull/19731)

- **General**
    - Add guardrail model argument feature - [PR #19619](https://github.com/BerriAI/litellm/pull/19619)
    - Fix guardrails issues with streaming-response regex - [PR #19901](https://github.com/BerriAI/litellm/pull/19901)
    - Remove enterprise requirement for guardrail monitoring (docs) - [PR #19833](https://github.com/BerriAI/litellm/pull/19833)

## Spend Tracking, Budgets and Rate Limiting

- Add event-driven coordination for global spend query to prevent cache stampede - [PR #20030](https://github.com/BerriAI/litellm/pull/20030)

## Performance / Loadbalancing / Reliability improvements

- **Resolve high CPU when router_settings in DB** - by avoiding REGISTRY.collect() in PrometheusServicesLogger - [PR #20087](https://github.com/BerriAI/litellm/pull/20087)
- **Reuse HTTP connections in Presidio** - to prevent performance degradation - [PR #19964](https://github.com/BerriAI/litellm/pull/19964)
- **Event-driven coordination for global spend query** - prevent cache stampede - [PR #20030](https://github.com/BerriAI/litellm/pull/20030)
- Fix recursive Pydantic validation issue - [PR #19531](https://github.com/BerriAI/litellm/pull/19531)
- Refactor argument handling into helper function to reduce code bloat - [PR #19720](https://github.com/BerriAI/litellm/pull/19720)
- Optimize logo fetching and resolve MCP import blockers - [PR #19719](https://github.com/BerriAI/litellm/pull/19719)
- Improve logo download performance using async HTTP client - [PR #20155](https://github.com/BerriAI/litellm/pull/20155)
- Fix server root path configuration - [PR #19790](https://github.com/BerriAI/litellm/pull/19790)
- Refactor: Extract transport context creation into separate method - [PR #19794](https://github.com/BerriAI/litellm/pull/19794)
- Add native_background_mode configuration to override polling_via_cache for specific models - [PR #19899](https://github.com/BerriAI/litellm/pull/19899)
- Initialize tiktoken environment at import time to enable offline usage - [PR #19882](https://github.com/BerriAI/litellm/pull/19882)
- Improve tiktoken performance using local cache in lazy loading - [PR #19774](https://github.com/BerriAI/litellm/pull/19774)
- Fix timeout errors in chat completion calls to be correctly reported in failure callbacks - [PR #19842](https://github.com/BerriAI/litellm/pull/19842)
- Fix environment variable type handling for NUM_RETRIES - [PR #19507](https://github.com/BerriAI/litellm/pull/19507)
- Use safe_deep_copy in silent experiment kwargs to prevent mutation - [PR #20170](https://github.com/BerriAI/litellm/pull/20170)
- Improve error handling by inspecting BadRequestError after all other policy types - [PR #19878](https://github.com/BerriAI/litellm/pull/19878)

## Database Changes

### Schema Updates

| Table | Change Type | Description | PR | Migration |
| ----- | ----------- | ----------- | -- | --------- |
| `LiteLLM_ManagedVectorStoresTable` | New Columns | Added `team_id` and `user_id` fields for permission management | [PR #19972](https://github.com/BerriAI/litellm/pull/19972) | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260131150814_add_team_user_to_vector_stores/migration.sql) |

### Migration Improvements

- Fix Docker: Use correct schema path for Prisma generation - [PR #19631](https://github.com/BerriAI/litellm/pull/19631)
- Resolve 'relation does not exist' migration errors in setup_database - [PR #19281](https://github.com/BerriAI/litellm/pull/19281)
- Fix migration issue and improve Docker image stability - [PR #19843](https://github.com/BerriAI/litellm/pull/19843)
- Run Prisma generate as nobody user in non-root Docker container for security - [PR #20000](https://github.com/BerriAI/litellm/pull/20000)
- Bump litellm-proxy-extras version to 0.4.28 - [PR #20166](https://github.com/BerriAI/litellm/pull/20166)

## Documentation Updates

- **[Add Claude Agents SDK x LiteLLM Guide](../../docs/mcp)** - [PR #20036](https://github.com/BerriAI/litellm/pull/20036)
- **[Add Cookbook: Using Claude Agent SDK + MCPs with LiteLLM](https://github.com/BerriAI/litellm/tree/main/cookbook)** - [PR #20081](https://github.com/BerriAI/litellm/pull/20081)
- Fix A2A Python SDK URL in documentation - [PR #19832](https://github.com/BerriAI/litellm/pull/19832)
- **[Add Sarvam usage documentation](../../docs/providers/sarvam)** - [PR #19844](https://github.com/BerriAI/litellm/pull/19844)
- **[Add supported input formats for embeddings](../../docs/embedding/supported_embedding)** - [PR #20073](https://github.com/BerriAI/litellm/pull/20073)
- **[UI Spend Logs Settings Docs](../../docs/proxy/ui_spend_log_settings)** - [PR #20197](https://github.com/BerriAI/litellm/pull/20197)
- Add OpenAI Agents SDK to OSS Adopters list in README - [PR #19820](https://github.com/BerriAI/litellm/pull/19820)
- Update docs: Remove enterprise requirement for guardrail monitoring - [PR #19833](https://github.com/BerriAI/litellm/pull/19833)
- Add missing environment variable documentation - [PR #20138](https://github.com/BerriAI/litellm/pull/20138)
- Improve documentation blog index page - [PR #20188](https://github.com/BerriAI/litellm/pull/20188)

## Infrastructure / Testing Improvements

- Add test coverage for Router.get_valid_args and improve code coverage reporting - [PR #19797](https://github.com/BerriAI/litellm/pull/19797)
- Add validation of model cost map as CI job - [PR #19993](https://github.com/BerriAI/litellm/pull/19993)
- Add Realtime API benchmarks - [PR #20074](https://github.com/BerriAI/litellm/pull/20074)
- Add Init Containers support in community helm chart - [PR #19816](https://github.com/BerriAI/litellm/pull/19816)
- Add libsndfile to main Dockerfile for ARM64 audio processing support - [PR #19776](https://github.com/BerriAI/litellm/pull/19776)

## New Contributors

* @ruanjf made their first contribution in https://github.com/BerriAI/litellm/pull/19551
* @moh-dev-stack made their first contribution in https://github.com/BerriAI/litellm/pull/19507
* @formorter made their first contribution in https://github.com/BerriAI/litellm/pull/19498
* @priyam-that made their first contribution in https://github.com/BerriAI/litellm/pull/19516
* @marcosgriselli made their first contribution in https://github.com/BerriAI/litellm/pull/19550
* @natimofeev made their first contribution in https://github.com/BerriAI/litellm/pull/19232
* @zifeo made their first contribution in https://github.com/BerriAI/litellm/pull/19805
* @pragyasardana made their first contribution in https://github.com/BerriAI/litellm/pull/19816
* @ryewilson made their first contribution in https://github.com/BerriAI/litellm/pull/19833
* @lizhen921 made their first contribution in https://github.com/BerriAI/litellm/pull/19919
* @boarder7395 made their first contribution in https://github.com/BerriAI/litellm/pull/19666
* @rushilchugh01 made their first contribution in https://github.com/BerriAI/litellm/pull/19938
* @cfchase made their first contribution in https://github.com/BerriAI/litellm/pull/19893
* @ayim made their first contribution in https://github.com/BerriAI/litellm/pull/19872
* @varunsripad123 made their first contribution in https://github.com/BerriAI/litellm/pull/20018
* @nht1206 made their first contribution in https://github.com/BerriAI/litellm/pull/20046
* @genga6 made their first contribution in https://github.com/BerriAI/litellm/pull/20009

**Full Changelog**: https://github.com/BerriAI/litellm/compare/v1.81.3.rc...v1.81.6
