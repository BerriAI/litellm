---
title: "v1.81.0"
slug: "v1-81-0"
date: 2026-01-18T10:00:00
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
docker.litellm.ai/berriai/litellm:v1.81.0
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.81.0
```

</TabItem>
</Tabs>

---

## Major change - /chat/completions Image URL Download Size Limit

To improve reliability and prevent memory issues, LiteLLM now includes a configurable **50MB limit** on image URL downloads by default. Previously, there was no limit on image downloads, which could occasionally cause memory issues with very large images.

### How It Works

Requests with image URLs exceeding 50MB will receive a helpful error message:

```bash
curl -X POST 'https://your-litellm-proxy.com/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/very-large-image.jpg"
            }
          }
        ]
      }
    ]
  }'
```

**Error Response:**

```json
{
  "error": {
    "message": "Error: Image size (75.50MB) exceeds maximum allowed size (50.0MB). url=https://example.com/very-large-image.jpg",
    "type": "ImageFetchError"
  }
}
```

### Configuring the Limit

The default 50MB limit works well for most use cases, but you can easily adjust it if needed:

**Increase the limit (e.g., to 100MB):**

```bash
export MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=100
```

**Disable image URL downloads (for security):**

```bash
export MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0
```

**Docker Configuration:**

```bash
docker run \
  -e MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=100 \
  -p 4000:4000 \
  docker.litellm.ai/berriai/litellm:v1.81.0
```

**Proxy Config (config.yaml):**

```yaml
general_settings:
  master_key: sk-1234
  
# Set via environment variable
environment_variables:
  MAX_IMAGE_URL_DOWNLOAD_SIZE_MB: "100"
```

### Why Add This?

This feature improves reliability by:
- Preventing memory issues from very large images
- Aligning with OpenAI's 50MB payload limit
- Validating image sizes early (when Content-Length header is available)

---

## Key Highlights

- **🛡️ Image URL Download Limits** - Configurable 50MB default limit to prevent OOM crashes
- **🔧 Guardrails Improvements** - Fail-open option for Grayswan, Pangea default_on support, better error handling
- **💰 Cost Tracking** - Fixed Gemini image token costs, improved cache token tracking
- **🎯 Claude Code Support** - Tool search, end-user tracking, web search integration, prompt caching
- **📊 UI Enhancements** - Deleted keys/teams tables, model hub health checks, usage filters
- **🔐 Security Fixes** - Privilege escalation fix, better error message sanitization
- **⚡ Performance** - Removed bottlenecks causing high CPU usage under heavy load
- **🆕 New Models** - GPT-5.2-codex, Azure Grok pricing, Cerebras GLM-4.7, and more

---

## Guardrails & Security

### Guardrails Improvements

- **Grayswan Guardrail** - Implement fail-open option (default: True) - [PR #18266](https://github.com/BerriAI/litellm/pull/18266)
- **Pangea Guardrail** - Respect `default_on` during initialization - [PR #18912](https://github.com/BerriAI/litellm/pull/18912)
- **Guardrail Error Handling** - Fix SerializationIterator error and pass tools to guardrail - [PR #18932](https://github.com/BerriAI/litellm/pull/18932)
- **Custom Guardrails** - Properly handle custom guardrails parameters - [PR #18978](https://github.com/BerriAI/litellm/pull/18978)
- **Clean Error Messages** - Use clean error messages for blocked requests - [PR #19023](https://github.com/BerriAI/litellm/pull/19023)
- **Responses API Support** - Guardrail moderation support with responses API - [PR #18957](https://github.com/BerriAI/litellm/pull/18957)
- **Model-Level Guardrails** - Fix model-level guardrails not taking effect - [PR #18895](https://github.com/BerriAI/litellm/pull/18895)
- **Panw Prisma AIRS** - Add custom violation message support - [PR #19272](https://github.com/BerriAI/litellm/pull/19272)

### Security Fixes

- **Privilege Escalation** - Fix `/user/new` privilege escalation vulnerability - [PR #19116](https://github.com/BerriAI/litellm/pull/19116)
- **Budget Validation** - Correct budget limit validation operator (>=) for team members - [PR #19207](https://github.com/BerriAI/litellm/pull/19207)
- **Custom CA Certificates** - Add Custom CA certificates to boto3 clients - [PR #18942](https://github.com/BerriAI/litellm/pull/18942)

---

## Claude Code Features

LiteLLM now provides comprehensive support for Claude Code (Anthropic's `/messages` API) with several new features:

### Tool Search Support

Add support for Tool Search on `/messages` API across Azure, Bedrock, and Anthropic API - [PR #19165](https://github.com/BerriAI/litellm/pull/19165)

### End-User Tracking

Track end-users with Claude Code for better analytics and monitoring - [PR #19171](https://github.com/BerriAI/litellm/pull/19171)

[**Documentation**](../../docs/providers/anthropic)

### Web Search Integration

Add web search support using LiteLLM `/search` endpoint with web search interception hook - [PR #19263](https://github.com/BerriAI/litellm/pull/19263), [PR #19294](https://github.com/BerriAI/litellm/pull/19294)

### Bedrock Improvements

- **Converse API Usage** - Ensure budget tokens are passed to converse API correctly - [PR #19107](https://github.com/BerriAI/litellm/pull/19107)
- **Invoke API Usage** - Fix Claude Code Bedrock Invoke usage and request signing - [PR #19111](https://github.com/BerriAI/litellm/pull/19111)
- **Prompt Caching** - Add support for Prompt Caching with Bedrock Converse - [PR #19123](https://github.com/BerriAI/litellm/pull/19123)

---

## Cost Tracking & Pricing

### Cost Calculation Fixes

- **Gemini Image Tokens** - Include IMAGE token count in cost calculation for Gemini models - [PR #18876](https://github.com/BerriAI/litellm/pull/18876)
- **Gemini Cache Tokens** - Fix negative text_tokens when using cache with images - [PR #18768](https://github.com/BerriAI/litellm/pull/18768)
- **Image Generation Tokens** - Fix image tokens spend logging for `/images/generations` - [PR #19009](https://github.com/BerriAI/litellm/pull/19009)
- **Gemini Image Generation** - Fix incorrect `prompt_tokens_details` in Gemini Image Generation - [PR #19070](https://github.com/BerriAI/litellm/pull/19070)
- **Zero Cost Models** - Add support for 0 cost models - [PR #19027](https://github.com/BerriAI/litellm/pull/19027)

### Pricing Updates

- **OpenRouter GPT-OSS-20B** - Correct pricing for `openrouter/openai/gpt-oss-20b` - [PR #18899](https://github.com/BerriAI/litellm/pull/18899)
- **Azure Claude Opus 4.5** - Add pricing for `azure_ai/claude-opus-4-5` - [PR #19003](https://github.com/BerriAI/litellm/pull/19003)
- **Novita Models** - Update Novita models prices - [PR #19005](https://github.com/BerriAI/litellm/pull/19005)
- **Azure Grok** - Fix Azure Grok prices - [PR #19102](https://github.com/BerriAI/litellm/pull/19102)
- **GCP GLM-4.7** - Fix GCP GLM-4.7 pricing - [PR #19172](https://github.com/BerriAI/litellm/pull/19172)
- **DeepSeek** - Sync DeepSeek chat/reasoner to V3.2 pricing - [PR #18884](https://github.com/BerriAI/litellm/pull/18884)
- **Gemini Cache Read** - Correct cache_read pricing for gemini-2.5-pro models - [PR #18157](https://github.com/BerriAI/litellm/pull/18157)
- **Case-Insensitive Lookup** - Fix case-insensitive model cost map lookup - [PR #18208](https://github.com/BerriAI/litellm/pull/18208)

---

## New Models & Providers

### New Model Support

| Provider | Model | Features |
| -------- | ----- | -------- |
| OpenAI | `gpt-5.2-codex` | Code generation |
| Azure | `azure/gpt-5.2-codex` | Code generation |
| Cerebras | `cerebras/zai-glm-4.7` | Reasoning, function calling |
| Replicate | All chat models | Full support for all Replicate chat models |

### Provider Updates

- **Anthropic**
  - Prevent dropping thinking when any message has thinking_blocks - [PR #18929](https://github.com/BerriAI/litellm/pull/18929)
  - Add missing anthropic tool results in response - [PR #18945](https://github.com/BerriAI/litellm/pull/18945)
  - Preserve web_fetch_tool_result in multi-turn conversations - [PR #18142](https://github.com/BerriAI/litellm/pull/18142)
  - Fix anthropic token counter with thinking - [PR #19067](https://github.com/BerriAI/litellm/pull/19067)
  - Add better error handling for Anthropic - [PR #18955](https://github.com/BerriAI/litellm/pull/18955)
  - Fix Anthropic during call error - [PR #19060](https://github.com/BerriAI/litellm/pull/19060)

- **Gemini**
  - Fix missing `completion_tokens_details` in Gemini 3 Flash when reasoning_effort is not used - [PR #18898](https://github.com/BerriAI/litellm/pull/18898)
  - Add presence_penalty support for Google AI Studio - [PR #18154](https://github.com/BerriAI/litellm/pull/18154)
  - Forward extra_headers in generateContent adapter - [PR #18935](https://github.com/BerriAI/litellm/pull/18935)
  - Add medium value support for detail param - [PR #19187](https://github.com/BerriAI/litellm/pull/19187)
  - Fix Gemini Image Generation imageConfig parameters - [PR #18948](https://github.com/BerriAI/litellm/pull/18948)
  - Dereference $defs/$ref in tool response content - [PR #19062](https://github.com/BerriAI/litellm/pull/19062)

- **Vertex AI**
  - Fix Vertex AI 400 Error with CachedContent model mismatch - [PR #19193](https://github.com/BerriAI/litellm/pull/19193)
  - Improve passthrough endpoint URL parsing and construction - [PR #17526](https://github.com/BerriAI/litellm/pull/17526)
  - Add type object to tool schemas missing type field - [PR #19103](https://github.com/BerriAI/litellm/pull/19103)
  - Keep type field in Gemini schema when properties is empty - [PR #18979](https://github.com/BerriAI/litellm/pull/18979)

- **Bedrock**
  - Add OpenAI-compatible service_tier parameter translation - [PR #18091](https://github.com/BerriAI/litellm/pull/18091)
  - Fix model ID encoding for Bedrock passthrough - [PR #18944](https://github.com/BerriAI/litellm/pull/18944)
  - Respect max_completion_tokens in thinking feature - [PR #18946](https://github.com/BerriAI/litellm/pull/18946)
  - Add user auth in standard logging object for Bedrock passthrough - [PR #19140](https://github.com/BerriAI/litellm/pull/19140)
  - Fix header forwarding in Bedrock passthrough - [PR #19007](https://github.com/BerriAI/litellm/pull/19007)
  - Strip throughput tier suffixes from model names - [PR #19147](https://github.com/BerriAI/litellm/pull/19147)
  - Fix Bedrock stability model usage issues - [PR #19199](https://github.com/BerriAI/litellm/pull/19199)

- **OCI**
  - Handle OpenAI-style image_url object in multimodal messages - [PR #18272](https://github.com/BerriAI/litellm/pull/18272)

- **Text Completion**
  - Support token IDs (list of integers) as prompt - [PR #18011](https://github.com/BerriAI/litellm/pull/18011)

- **Ollama**
  - Set finish_reason to tool_calls and remove broken capability check - [PR #18924](https://github.com/BerriAI/litellm/pull/18924)

- **Watsonx**
  - Allow passing scope ID for Watsonx inferencing - [PR #18959](https://github.com/BerriAI/litellm/pull/18959)

- **Replicate**
  - Add all chat Replicate models support - [PR #18954](https://github.com/BerriAI/litellm/pull/18954)

- **OpenRouter**
  - Add OpenRouter support for image/generation endpoints - [PR #19059](https://github.com/BerriAI/litellm/pull/19059)

- **Azure Model Router**
  - New Model - Azure Model Router on LiteLLM AI Gateway - [PR #19054](https://github.com/BerriAI/litellm/pull/19054)

- **GPT-5 Models**
  - Correct context window sizes for GPT-5 model variants - [PR #18928](https://github.com/BerriAI/litellm/pull/18928)
  - Correct max_input_tokens for GPT-5 models - [PR #19056](https://github.com/BerriAI/litellm/pull/19056)

- **Volcengine**
  - Add max_tokens settings for Volcengine models (deepseek-v3-2, glm-4-7, kimi-k2-thinking) - [PR #19076](https://github.com/BerriAI/litellm/pull/19076)

---

## UI & Management Endpoints

### New Features

- **Deleted Keys and Teams Table** - View deleted keys and teams for audit purposes - [PR #18228](https://github.com/BerriAI/litellm/pull/18228), [PR #19268](https://github.com/BerriAI/litellm/pull/19268)
- **Organization Table Filters** - Add filters to organization table - [PR #18916](https://github.com/BerriAI/litellm/pull/18916)
- **Organization List Query Params** - Add query parameters to `/organization/list` - [PR #18910](https://github.com/BerriAI/litellm/pull/18910)
- **Model Hub Health Information** - Display health information in public model hub - [PR #19256](https://github.com/BerriAI/litellm/pull/19256), [PR #19258](https://github.com/BerriAI/litellm/pull/19258)
- **Status Query for Keys and Teams** - Add status query parameter for keys and teams list - [PR #19260](https://github.com/BerriAI/litellm/pull/19260)
- **Team Daily Activity** - Show internal users their spend only - [PR #19227](https://github.com/BerriAI/litellm/pull/19227)
- **Usage Filters** - Allow top virtual keys and models to show more entries - [PR #19050](https://github.com/BerriAI/litellm/pull/19050)
- **Usage Model Activity Chart** - Fix Y axis on model activity chart - [PR #19055](https://github.com/BerriAI/litellm/pull/19055)
- **Usage Export Report** - Add Team ID and Team Name in export report - [PR #19047](https://github.com/BerriAI/litellm/pull/19047)
- **Key Generate Permission Error** - Simplify key generate permission error - [PR #18997](https://github.com/BerriAI/litellm/pull/18997)
- **Refetch Keys After Create** - Refetch keys after key creation - [PR #18994](https://github.com/BerriAI/litellm/pull/18994)
- **Keys Table Refresh** - Refresh keys list on delete - [PR #19262](https://github.com/BerriAI/litellm/pull/19262)
- **Anthropic Models QOL** - Quality of life improvements for Anthropic models - [PR #19058](https://github.com/BerriAI/litellm/pull/19058)
- **Edit Key Team Dropdown** - Add search to key edit team dropdown - [PR #19119](https://github.com/BerriAI/litellm/pull/19119)
- **Reusable Model Select** - Create reusable model select component - [PR #19164](https://github.com/BerriAI/litellm/pull/19164)
- **Team Settings Model Dropdown** - Edit settings model dropdown - [PR #19186](https://github.com/BerriAI/litellm/pull/19186)
- **Team Member Icon Buttons** - Refactor team member icon buttons - [PR #19192](https://github.com/BerriAI/litellm/pull/19192)
- **Prevent Team Admin Deletions** - Allow preventing team admins from deleting members from teams - [PR #19128](https://github.com/BerriAI/litellm/pull/19128)
- **Community Engagement Buttons** - Add community engagement buttons - [PR #19114](https://github.com/BerriAI/litellm/pull/19114)
- **Dropdown Clear Button** - Add allowClear to dropdown components for better UX - [PR #18778](https://github.com/BerriAI/litellm/pull/18778)
- **Model Hub Client Exception** - Fix model hub client side exception - [PR #19045](https://github.com/BerriAI/litellm/pull/19045)
- **Feedback Form** - UI Feedback Form - why LiteLLM - [PR #18999](https://github.com/BerriAI/litellm/pull/18999)
- **User Metrics for Prometheus** - Add user metrics for Prometheus - [PR #18785](https://github.com/BerriAI/litellm/pull/18785)
- **Reusable Table Filters** - Refactor user and team table filters to reusable component - [PR #19010](https://github.com/BerriAI/litellm/pull/19010)
- **New Badges** - Adjusting new badges - [PR #19278](https://github.com/BerriAI/litellm/pull/19278)

### API Endpoints

- **MSFT SSO** - Allow setting custom MSFT Base URLs - [PR #18977](https://github.com/BerriAI/litellm/pull/18977)
- **MSFT SSO Attributes** - Allow overriding env var attribute names - [PR #18998](https://github.com/BerriAI/litellm/pull/18998)
- **Containers API** - Container API routes return 401 for non-admin users - routes missing from openai_routes - [PR #19115](https://github.com/BerriAI/litellm/pull/19115)
- **Containers API Regional Endpoints** - Allow routing to regional endpoints - [PR #19118](https://github.com/BerriAI/litellm/pull/19118)
- **Azure Storage Circular Reference** - Fix Azure Storage circular reference error - [PR #19120](https://github.com/BerriAI/litellm/pull/19120)
- **Batch Deletion and Retrieve** - Fix batch deletion and retrieve - [PR #18340](https://github.com/BerriAI/litellm/pull/18340)
- **Prompt Deletion** - Fix prompt deletion fails with Prisma FieldNotFoundError - [PR #18966](https://github.com/BerriAI/litellm/pull/18966)
- **SCIM Compliance** - Fix SCIM GET /Users error and enforce SCIM 2.0 compliance - [PR #17420](https://github.com/BerriAI/litellm/pull/17420)
- **Feature Flag for SCIM** - Feature flag for SCIM compliance fix - [PR #18878](https://github.com/BerriAI/litellm/pull/18878)

---

## Performance & Reliability

### Performance Improvements

- **Remove CPU Bottleneck** - Remove bottleneck causing high CPU usage & overhead under heavy load - [PR #19049](https://github.com/BerriAI/litellm/pull/19049)
- **O(1) Model Cost Key** - Add CI enforcement for O(1) operations in `_get_model_cost_key` to prevent performance regressions - [PR #19052](https://github.com/BerriAI/litellm/pull/19052)
- **Azure Embeddings Connection Leaks** - Fix Azure embeddings JSON parsing to prevent connection leaks and ensure proper router cooldown - [PR #19167](https://github.com/BerriAI/litellm/pull/19167)
- **Token Counter** - Do not fallback to token counter if `disable_token_counter` is enabled - [PR #19041](https://github.com/BerriAI/litellm/pull/19041)

### Rate Limiting

- **Dynamic Rate Limiter** - Fix TPM 25% limiting by ensuring priority queue logic - [PR #19092](https://github.com/BerriAI/litellm/pull/19092)

### Reliability

- **Fallback Endpoints** - Add fallback endpoints support - [PR #19185](https://github.com/BerriAI/litellm/pull/19185)
- **Stream Timeout** - Fix stream_timeout parameter functionality - [PR #19191](https://github.com/BerriAI/litellm/pull/19191)
- **Mid-Stream Fallbacks** - Add handling for user-disabled mid-stream fallbacks - [PR #19078](https://github.com/BerriAI/litellm/pull/19078)
- **Model Matching Priority** - Fix model matching priority in configuration - [PR #19012](https://github.com/BerriAI/litellm/pull/19012)
- **Num Retries** - Fix num_retries in litellm_params as per config - [PR #18975](https://github.com/BerriAI/litellm/pull/18975)
- **Exception Mapping** - Handle exceptions without response parameter - [PR #18919](https://github.com/BerriAI/litellm/pull/18919)

---

## Observability & Logging

### Logging Improvements

- **OpenTelemetry** - Update semantic conventions to 1.38 (gen_ai attributes) - [PR #18793](https://github.com/BerriAI/litellm/pull/18793)
- **LangSmith** - Hoist thread grouping metadata (session_id, thread) - [PR #18982](https://github.com/BerriAI/litellm/pull/18982)
- **Langfuse JSON Logging** - Include Langfuse logger in JSON logging when Langfuse callback is used - [PR #19162](https://github.com/BerriAI/litellm/pull/19162)
- **JSON Logging** - Enable JSON logging via configuration and add regression test - [PR #19037](https://github.com/BerriAI/litellm/pull/19037)
- **Spend Logs Cleanup** - Cleanup spend logs cron verification, fix, and docs - [PR #19085](https://github.com/BerriAI/litellm/pull/19085)
- **Header Forwarding** - Fix header forwarding for embeddings endpoint - [PR #18960](https://github.com/BerriAI/litellm/pull/18960)
- **LLM Provider Headers** - Preserve llm_provider-* headers in error responses - [PR #19020](https://github.com/BerriAI/litellm/pull/19020)
- **Turn Off Message Logging** - Fix turn_off_message_logging not redacting request messages in proxy_server_request field - [PR #18897](https://github.com/BerriAI/litellm/pull/18897)
- **Logfire Base URL** - Add ability to customize Logfire base URL through env var - [PR #19148](https://github.com/BerriAI/litellm/pull/19148)

---

## MCP (Model Context Protocol)

### MCP Improvements

- **Prevent Duplicate Reload** - Prevent duplicate MCP reload scheduler registration - [PR #18934](https://github.com/BerriAI/litellm/pull/18934)
- **Forward Extra Headers** - Forward MCP extra headers case-insensitively - [PR #18940](https://github.com/BerriAI/litellm/pull/18940)
- **REST Auth Checks** - Fix MCP REST auth checks - [PR #19051](https://github.com/BerriAI/litellm/pull/19051)
- **Responses Telemetry** - Fix generating two telemetry events in responses - [PR #18938](https://github.com/BerriAI/litellm/pull/18938)
- **Chat Completions** - Fix MCP chat completions - [PR #19129](https://github.com/BerriAI/litellm/pull/19129)
- **Troubleshooting Guide** - Add MCP troubleshooting guide - [PR #19122](https://github.com/BerriAI/litellm/pull/19122)
- **Auth Message UI** - Add auth message UI documentation - [PR #19063](https://github.com/BerriAI/litellm/pull/19063)

---

## Responses API

- **Caching Support** - Add support for caching for responses API - [PR #19068](https://github.com/BerriAI/litellm/pull/19068)
- **Retry Policy** - Add retry policy support to responses API - [PR #19074](https://github.com/BerriAI/litellm/pull/19074)
- **Content Validation** - Fix responses content can't be none - [PR #19064](https://github.com/BerriAI/litellm/pull/19064)

---

## Realtime API

- **Model Name from Query Param** - Fix model name from query param in realtime request - [PR #19135](https://github.com/BerriAI/litellm/pull/19135)
- **A2A Message Send** - Use non-streaming method for endpoint v1/a2a/message/send - [PR #19025](https://github.com/BerriAI/litellm/pull/19025)

---

## Infrastructure & Deployment

### Helm Chart

- **Config Mount** - Fix mount config.yaml as single file in Helm chart - [PR #19146](https://github.com/BerriAI/litellm/pull/19146)
- **Helm Chart Versioning** - Sync Helm chart versioning with production standards and Docker versions - [PR #18868](https://github.com/BerriAI/litellm/pull/18868)
- **Helm Chart Testing** - Add Helm chart testing - [PR #18983](https://github.com/BerriAI/litellm/pull/18983)
- **Custom Callbacks Mounting** - Add guide for mounting custom callbacks in Helm/K8s - [PR #19136](https://github.com/BerriAI/litellm/pull/19136)

### Docker

- **Keepalive Timeout** - Make keepalive_timeout parameter work for Gunicorn - [PR #19087](https://github.com/BerriAI/litellm/pull/19087)

### Database

- **Prisma Migration** - Include proxy/prisma_migration.py in non-root - [PR #18971](https://github.com/BerriAI/litellm/pull/18971)
- **Migration Update** - Update prisma_migration.py - [PR #19083](https://github.com/BerriAI/litellm/pull/19083)
- **DB Migration Test** - Stabilize db_migration_disable_update_check test log check - [PR #18882](https://github.com/BerriAI/litellm/pull/18882)
- **Created/Updated Fields** - Add created_at/updated_at fields to LiteLLM_ProxyModelTable - [PR #18937](https://github.com/BerriAI/litellm/pull/18937)

### Dependencies

- **Boto3 Update** - Update boto3 to 1.40.15 and aioboto3 to 15.5.0 - [PR #19090](https://github.com/BerriAI/litellm/pull/19090)
- **LiteLLM Version** - Bump litellm version to 0.1.28 - [PR #19127](https://github.com/BerriAI/litellm/pull/19127)

---

## Testing & CI

- **UI E2E Tests** - Neon E2E DB Script - [PR #18985](https://github.com/BerriAI/litellm/pull/18985)
- **Flaky Test Fixes** - Remove flaky Azure OIDC embedding test - [PR #18993](https://github.com/BerriAI/litellm/pull/18993)
- **Security Test** - Fix security test - [PR #18987](https://github.com/BerriAI/litellm/pull/18987)
- **Responses ID Security** - Temporarily disable flaky responses_id_security tests - [PR #19013](https://github.com/BerriAI/litellm/pull/19013)
- **Mock Tests** - Stabilize mock tests - [PR #19141](https://github.com/BerriAI/litellm/pull/19141)
- **Stream Chunk Builder** - Fix test_stream_chunk_builder_litellm_mixed_calls - [PR #19179](https://github.com/BerriAI/litellm/pull/19179)
- **Azure SDK Init** - Skip Azure SDK init check for acreate_skill - [PR #19178](https://github.com/BerriAI/litellm/pull/19178)
- **Route Validation** - Handle wildcard routes in route validation test - [PR #19182](https://github.com/BerriAI/litellm/pull/19182)
- **Duplicate Issue Checker** - Add automated duplicate issue checker and template safeguards - [PR #19218](https://github.com/BerriAI/litellm/pull/19218)
- **Label Component Workflow** - Extend label-component workflow to auto-label 'claude code' issues - [PR #19242](https://github.com/BerriAI/litellm/pull/19242)
- **License Check** - Add jaraco liccheck - [PR #19188](https://github.com/BerriAI/litellm/pull/19188)
- **CVE Documentation** - Document temporary grype ignore for CVE-2026-22184 - [PR #19181](https://github.com/BerriAI/litellm/pull/19181)
- **Allowed CVEs** - Add ALLOWED_CVES - [PR #19200](https://github.com/BerriAI/litellm/pull/19200)

---

## Documentation

- **Architecture** - Add LiteLLM architecture md doc - [PR #19057](https://github.com/BerriAI/litellm/pull/19057), [PR #19252](https://github.com/BerriAI/litellm/pull/19252)
- **Troubleshooting Guide** - Add troubleshooting guide - [PR #19096](https://github.com/BerriAI/litellm/pull/19096), [PR #19097](https://github.com/BerriAI/litellm/pull/19097), [PR #19099](https://github.com/BerriAI/litellm/pull/19099)
- **CPU and Memory Issues** - Add structured issue reporting guides for CPU and memory issues - [PR #19117](https://github.com/BerriAI/litellm/pull/19117)
- **Redis Requirement** - Add Redis requirement warning for high-traffic deployments - [PR #18892](https://github.com/BerriAI/litellm/pull/18892)
- **Load Balancing** - Update load balancing and routing with enable_pre_call_checks - [PR #18888](https://github.com/BerriAI/litellm/pull/18888)
- **Pass Through** - Updated pass_through with guided param - [PR #18886](https://github.com/BerriAI/litellm/pull/18886)
- **Message Content Types** - Update message content types link and add content types table - [PR #18209](https://github.com/BerriAI/litellm/pull/18209)
- **Redis Initialization** - Add Redis initialization with kwargs - [PR #19183](https://github.com/BerriAI/litellm/pull/19183)
- **SAP Gen AI Hub** - Improve documentation for routing LLM calls via SAP Gen AI Hub - [PR #19166](https://github.com/BerriAI/litellm/pull/19166)
- **Deleted Keys and Teams** - Deleted Keys and Teams docs - [PR #19291](https://github.com/BerriAI/litellm/pull/19291)
- **Claude Code End User Tracking** - Claude Code end user tracking guide - [PR #19176](https://github.com/BerriAI/litellm/pull/19176)

---

## Bug Fixes

### General Fixes

- **Swagger UI** - Fix Swagger UI path execute error with server_root_path in OpenAPI schema - [PR #18947](https://github.com/BerriAI/litellm/pull/18947)
- **Pydantic Serializer Warnings** - Normalize OpenAI SDK BaseModel choices/messages to avoid Pydantic serializer warnings - [PR #18972](https://github.com/BerriAI/litellm/pull/18972)
- **Video Status/Content** - Fix video status/content credential injection for wildcard models - [PR #18854](https://github.com/BerriAI/litellm/pull/18854)
- **Contextual Gap Checks** - Add contextual gap checks and word-form digits - [PR #18301](https://github.com/BerriAI/litellm/pull/18301)
- **Orphaned Files** - Clean up orphaned files from repository root - [PR #19150](https://github.com/BerriAI/litellm/pull/19150)

---

## New Contributors

* @yogeshwaran10 made their first contribution in [PR #18898](https://github.com/BerriAI/litellm/pull/18898)
* @theonlypal made their first contribution in [PR #18937](https://github.com/BerriAI/litellm/pull/18937)
* @jonmagic made their first contribution in [PR #18935](https://github.com/BerriAI/litellm/pull/18935)
* @houdataali made their first contribution in [PR #19025](https://github.com/BerriAI/litellm/pull/19025)
* @hummat made their first contribution in [PR #18972](https://github.com/BerriAI/litellm/pull/18972)
* @berkeyalciin made their first contribution in [PR #18966](https://github.com/BerriAI/litellm/pull/18966)
* @MateuszOssGit made their first contribution in [PR #18959](https://github.com/BerriAI/litellm/pull/18959)
* @xfan001 made their first contribution in [PR #18947](https://github.com/BerriAI/litellm/pull/18947)
* @nulone made their first contribution in [PR #18884](https://github.com/BerriAI/litellm/pull/18884)
* @debnil-mercor made their first contribution in [PR #18919](https://github.com/BerriAI/litellm/pull/18919)
* @hakhundov made their first contribution in [PR #17420](https://github.com/BerriAI/litellm/pull/17420)
* @rohanwinsor made their first contribution in [PR #19078](https://github.com/BerriAI/litellm/pull/19078)
* @pgolm made their first contribution in [PR #19020](https://github.com/BerriAI/litellm/pull/19020)
* @vikigenius made their first contribution in [PR #19148](https://github.com/BerriAI/litellm/pull/19148)
* @burnerburnerburnerman made their first contribution in [PR #19090](https://github.com/BerriAI/litellm/pull/19090)
* @yfge made their first contribution in [PR #19076](https://github.com/BerriAI/litellm/pull/19076)
* @danielnyari-seon made their first contribution in [PR #19083](https://github.com/BerriAI/litellm/pull/19083)
* @guilherme-segantini made their first contribution in [PR #19166](https://github.com/BerriAI/litellm/pull/19166)
* @jgreek made their first contribution in [PR #19147](https://github.com/BerriAI/litellm/pull/19147)
* @anand-kamble made their first contribution in [PR #19193](https://github.com/BerriAI/litellm/pull/19193)

---

## Full Changelog

**[View complete changelog on GitHub](https://github.com/BerriAI/litellm/compare/v1.80.15.rc.1...v1.81.0.rc.1)**
