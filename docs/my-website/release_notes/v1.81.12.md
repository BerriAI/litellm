---
title: "[Preview] v1.81.12 - Guardrail Policy Templates & Action Builder"
slug: "v1-81-12"
date: 2026-02-14T00:00:00
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

## Deploy this version

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:main-v1.81.12.rc.1
```

</TabItem>
<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.81.12.rc1
```

</TabItem>
</Tabs>

## Key Highlights

- **Policy Templates** - [Pre-configured guardrail policy templates for common safety and compliance use-cases (including NSFW, toxic content, and child safety)](../../docs/proxy/guardrails/policy_templates)
- **Guardrail Action Builder** - [Build and customize guardrail policy flows with the new action-builder UI and conditional execution support](../../docs/proxy/guardrails/policy_templates)
- **MCP OAuth2 M2M + Tracing** - [Add machine-to-machine OAuth2 support for MCP servers and OpenTelemetry tracing for MCP calls through AI Gateway](../../docs/mcp)
- **Responses API `shell` Tool & `context_management` support** - [Server-side context management (compaction) and Shell tool support for the OpenAI Responses API](../../docs/response_api)
- **Access Groups** - [Create access groups to manage model, MCP server, and agent access across teams and keys](../../docs/proxy/access_groups)
- **50+ New Bedrock Regional Model Entries** - DeepSeek V3.2, MiniMax M2.1, Kimi K2.5, Qwen3 Coder Next, and NVIDIA Nemotron Nano across multiple regions
- **Add Semgrep & fix OOMs** - [Static analysis rules and out-of-memory fixes](#add-semgrep--fix-ooms) - [PR #20912](https://github.com/BerriAI/litellm/pull/20912)

---

## Add Semgrep & fix OOMs

This release fixes out-of-memory (OOM) risks from unbounded `asyncio.Queue()` usage. Log queues (e.g. GCS bucket) and DB spend-update queues were previously unbounded and could grow without limit under load. They now use a configurable max size (`LITELLM_ASYNCIO_QUEUE_MAXSIZE`, default 1000); when full, queues flush immediately to make room instead of growing memory. A Semgrep rule (`.semgrep/rules/python/unbounded-memory.yml`) was added to flag similar unbounded-memory patterns in future code. [PR #20912](https://github.com/BerriAI/litellm/pull/20912)

---

## Guardrail Action Builder

This release adds a visual action builder for guardrail policies with conditional execution support. You can now chain guardrails into multi-step pipelines — if a simple guardrail fails, route to an advanced one instead of immediately blocking. Each step has configurable ON PASS and ON FAIL actions (Next Step, Block, or Allow), and you can test the full pipeline with a sample message before saving.

![Guardrail Action Builder](../img/release_notes/guard_actions.png)

### Access Groups

Access Groups simplify defining resource access across your organization. One group can grant access to models, MCP servers, and agents—simply attach it to a key or team. Create groups in the Admin UI, define which resources each group includes, then assign the group when creating keys or teams. Updates to a group apply automatically to all attached keys and teams.

<Image img={require('../img/ui_access_groups.png')} />

## New Providers and Endpoints

### New Providers (2 new providers)

| Provider | Supported LiteLLM Endpoints | Description |
| -------- | --------------------------- | ----------- |
| [Scaleway](../../docs/providers/scaleway) | `/chat/completions` | Scaleway Generative APIs for chat completions |
| [Sarvam AI](../../docs/providers/sarvam) | `/chat/completions`, `/audio/transcriptions`, `/audio/speech` | Sarvam AI STT and TTS support for Indian languages |

---

## New Models / Updated Models

#### New Model Support (19 highlighted models)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) |
| -------- | ----- | -------------- | ------------------- | -------------------- |
| AWS Bedrock | `deepseek.v3.2` | 164K | $0.62 | $1.85 |
| AWS Bedrock | `minimax.minimax-m2.1` | 196K | $0.30 | $1.20 |
| AWS Bedrock | `moonshotai.kimi-k2.5` | 262K | $0.60 | $3.00 |
| AWS Bedrock | `moonshotai.kimi-k2-thinking` | 262K | $0.73 | $3.03 |
| AWS Bedrock | `qwen.qwen3-coder-next` | 262K | $0.50 | $1.20 |
| AWS Bedrock | `nvidia.nemotron-nano-3-30b` | 262K | $0.06 | $0.24 |
| Azure AI | `azure_ai/kimi-k2.5` | 262K | $0.60 | $3.00 |
| Vertex AI | `vertex_ai/zai-org/glm-5-maas` | 200K | $1.00 | $3.20 |
| MiniMax | `minimax/MiniMax-M2.5` | 1M | $0.30 | $1.20 |
| MiniMax | `minimax/MiniMax-M2.5-lightning` | 1M | $0.30 | $2.40 |
| Dashscope | `dashscope/qwen3-max` | 258K | Tiered pricing | Tiered pricing |
| Perplexity | `perplexity/preset/pro-search` | - | Per-request | Per-request |
| Perplexity | `perplexity/openai/gpt-4o` | - | Per-request | Per-request |
| Perplexity | `perplexity/openai/gpt-5.2` | - | Per-request | Per-request |
| Vercel AI Gateway | `vercel_ai_gateway/anthropic/claude-opus-4.6` | 200K | $5.00 | $25.00 |
| Vercel AI Gateway | `vercel_ai_gateway/anthropic/claude-sonnet-4` | 200K | $3.00 | $15.00 |
| Vercel AI Gateway | `vercel_ai_gateway/anthropic/claude-haiku-4.5` | 200K | $1.00 | $5.00 |
| Sarvam AI | `sarvam/sarvam-m` | 8K | Free tier | Free tier |
| Anthropic | `fast/claude-opus-4-6` | 1M | $30.00 | $150.00 |

*Note: AWS Bedrock models are available across multiple regions (us-east-1, us-east-2, us-west-2, eu-central-1, eu-north-1, ap-northeast-1, ap-south-1, ap-southeast-3, sa-east-1). 54 regional model entries were added in total.*

#### Features

- **[Anthropic](../../docs/providers/anthropic)**
    - Enable non-tool structured outputs on Claude Opus 4.5 and 4.6 using `output_format` param - [PR #20548](https://github.com/BerriAI/litellm/pull/20548)
    - Add support for `anthropic_messages` call type in prompt caching - [PR #19233](https://github.com/BerriAI/litellm/pull/19233)
    - Managing Anthropic Beta Headers with remote URL fetching - [PR #20935](https://github.com/BerriAI/litellm/pull/20935), [PR #21110](https://github.com/BerriAI/litellm/pull/21110)
    - Remove `x-anthropic-billing` block - [PR #20951](https://github.com/BerriAI/litellm/pull/20951)
    - Use Authorization Bearer for OAuth tokens instead of x-api-key - [PR #21039](https://github.com/BerriAI/litellm/pull/21039)
    - Filter unsupported JSON schema constraints for structured outputs - [PR #20813](https://github.com/BerriAI/litellm/pull/20813)
    - New Claude Opus 4.6 features for `/v1/messages` - [PR #20733](https://github.com/BerriAI/litellm/pull/20733)
    - Fix `reasoning_effort=None` and `"none"` should return None for Opus 4.6 - [PR #20800](https://github.com/BerriAI/litellm/pull/20800)

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Extend model support with 4 new beta models - [PR #21035](https://github.com/BerriAI/litellm/pull/21035)
    - Add Claude Opus 4.6 to `_supports_tool_search_on_bedrock` - [PR #21017](https://github.com/BerriAI/litellm/pull/21017)
    - Correct Bedrock Claude Opus 4.6 model IDs (remove `:0` suffix) - [PR #20564](https://github.com/BerriAI/litellm/pull/20564), [PR #20671](https://github.com/BerriAI/litellm/pull/20671)
    - Add `output_config` as supported param - [PR #20748](https://github.com/BerriAI/litellm/pull/20748)

- **[Vertex AI](../../docs/providers/vertex)**
    - Add Vertex GLM-5 model support - [PR #21053](https://github.com/BerriAI/litellm/pull/21053)
    - Propagate `extra_headers` anthropic-beta to request body - [PR #20666](https://github.com/BerriAI/litellm/pull/20666)
    - Preserve `usageMetadata` in `_hidden_params` - [PR #20559](https://github.com/BerriAI/litellm/pull/20559)
    - Map `IMAGE_PROHIBITED_CONTENT` to `content_filter` - [PR #20524](https://github.com/BerriAI/litellm/pull/20524)
    - Add RAG ingest for Vertex AI - [PR #21120](https://github.com/BerriAI/litellm/pull/21120)

- **[OCI / Cohere](../../docs/providers/cohere)**
    - OCI Cohere responseFormat/Pydantic support - [PR #20663](https://github.com/BerriAI/litellm/pull/20663)
    - Fix OCI Cohere system messages by populating `preambleOverride` - [PR #20958](https://github.com/BerriAI/litellm/pull/20958)

- **[Perplexity](../../docs/providers/perplexity)**
    - Perplexity Research API support with preset search - [PR #20860](https://github.com/BerriAI/litellm/pull/20860)

- **[MiniMax](../../docs/providers/minimax)**
    - Add MiniMax-M2.5 and MiniMax-M2.5-lightning models - [PR #21054](https://github.com/BerriAI/litellm/pull/21054)

- **[Kimi / Moonshot](../../docs/providers/moonshot)**
    - Add Kimi model pricing by region - [PR #20855](https://github.com/BerriAI/litellm/pull/20855)
    - Add `moonshotai.kimi-k2.5` - [PR #20863](https://github.com/BerriAI/litellm/pull/20863)

- **[Dashscope](../../docs/providers/dashscope)**
    - Add `dashscope/qwen3-max` model with tiered pricing - [PR #20919](https://github.com/BerriAI/litellm/pull/20919)

- **[Vercel AI Gateway](../../docs/providers/vercel_ai_gateway)**
    - Add new Vercel AI Anthropic models - [PR #20745](https://github.com/BerriAI/litellm/pull/20745)

- **[Azure AI](../../docs/providers/azure_ai)**
    - Add `azure_ai/kimi-k2.5` to Azure model DB - [PR #20896](https://github.com/BerriAI/litellm/pull/20896)
    - Support Azure AD token auth for non-Claude azure_ai models - [PR #20981](https://github.com/BerriAI/litellm/pull/20981)
    - Fix Azure batches issues - [PR #21092](https://github.com/BerriAI/litellm/pull/21092)

- **[DeepSeek](../../docs/providers/deepseek)**
    - Sync DeepSeek model metadata and add bare-name fallback - [PR #20938](https://github.com/BerriAI/litellm/pull/20938)

- **[Gemini](../../docs/providers/gemini)**
    - Handle image in assistant message for Gemini - [PR #20845](https://github.com/BerriAI/litellm/pull/20845)
    - Add missing tpm/rpm for Gemini models - [PR #21175](https://github.com/BerriAI/litellm/pull/21175)

- **General**
    - Add 30 missing models to pricing JSON - [PR #20797](https://github.com/BerriAI/litellm/pull/20797)
    - Cleanup 39 deprecated OpenRouter models - [PR #20786](https://github.com/BerriAI/litellm/pull/20786)
    - Standardize endpoint `display_name` naming convention - [PR #20791](https://github.com/BerriAI/litellm/pull/20791)
    - Fix and stabilize model cost map formatting - [PR #20895](https://github.com/BerriAI/litellm/pull/20895)
    - Export `PermissionDeniedError` from `litellm.__init__` - [PR #20960](https://github.com/BerriAI/litellm/pull/20960)

### Bug Fixes

- **[Anthropic](../../docs/providers/anthropic)**
    - Fix `get_supported_anthropic_messages_params` - [PR #20752](https://github.com/BerriAI/litellm/pull/20752)
    - Fix `base_model` name for body and deployment name in URL - [PR #20747](https://github.com/BerriAI/litellm/pull/20747)

- **[Azure](../../docs/providers/azure/azure)**
    - Preserve `content_policy_violation` error details from Azure OpenAI - [PR #20883](https://github.com/BerriAI/litellm/pull/20883)

- **[Vertex AI](../../docs/providers/vertex)**
    - Fix Gemini multi-turn tool calling message formatting (added and reverted) - [PR #20569](https://github.com/BerriAI/litellm/pull/20569), [PR #21051](https://github.com/BerriAI/litellm/pull/21051)

---

## LLM API Endpoints

#### Features

- **[Responses API](../../docs/response_api)**
    - Add server-side context management (compaction) support - [PR #21058](https://github.com/BerriAI/litellm/pull/21058)
    - Add Shell tool support for OpenAI Responses API - [PR #21063](https://github.com/BerriAI/litellm/pull/21063)
    - Preserve tool call argument deltas when streaming id is omitted - [PR #20712](https://github.com/BerriAI/litellm/pull/20712)
    - Preserve interleaved thinking/redacted_thinking blocks during streaming - [PR #20702](https://github.com/BerriAI/litellm/pull/20702)

- **[Chat Completions](../../docs/completion/input)**
    - Add Web Search support using LiteLLM `/search` (web search interception hook) - [PR #20483](https://github.com/BerriAI/litellm/pull/20483)
    - Preserved nullable object fields by carrying schema properties - [PR #19132](https://github.com/BerriAI/litellm/pull/19132)
    - Support `prompt_cache_key` for OpenAI and Azure chat completions - [PR #20989](https://github.com/BerriAI/litellm/pull/20989)

- **[Pass-Through Endpoints](../../docs/pass_through/bedrock)**
    - Add support for `langchain_aws` via LiteLLM passthrough - [PR #20843](https://github.com/BerriAI/litellm/pull/20843)
    - Add `custom_body` parameter to `endpoint_func` in `create_pass_through_route` - [PR #20849](https://github.com/BerriAI/litellm/pull/20849)

- **[Vector Stores](../../docs/providers/openai)**
    - Add `target_model_names` for vector store endpoints - [PR #21089](https://github.com/BerriAI/litellm/pull/21089)

- **General**
    - Add `output_config` as supported param - [PR #20748](https://github.com/BerriAI/litellm/pull/20748)
    - Add managed error file support - [PR #20838](https://github.com/BerriAI/litellm/pull/20838)

#### Bugs

- **General**
    - Stop leaking Python tracebacks in streaming SSE error responses - [PR #20850](https://github.com/BerriAI/litellm/pull/20850)
    - Fix video list pagination cursors not encoded with provider metadata - [PR #20710](https://github.com/BerriAI/litellm/pull/20710)
    - Handle `metadata=None` in SDK path retry/error logic - [PR #20873](https://github.com/BerriAI/litellm/pull/20873)
    - Fix Spend logs pickle error with Pydantic models and redaction - [PR #20685](https://github.com/BerriAI/litellm/pull/20685)
    - Remove duplicate `PerplexityResponsesConfig` from `LLM_CONFIG_NAMES` - [PR #21105](https://github.com/BerriAI/litellm/pull/21105)

---

## Management Endpoints / UI

#### Features

- **Access Groups**
    - New Access Groups feature for managing model, MCP server, and agent access - [PR #21022](https://github.com/BerriAI/litellm/pull/21022)
    - Access Groups table and details page UI - [PR #21165](https://github.com/BerriAI/litellm/pull/21165)
    - Refactor `model_ids` to `model_names` for backwards compatibility - [PR #21166](https://github.com/BerriAI/litellm/pull/21166)

- **Policies**
    - Allow connecting Policies to Tags, simulating Policies, viewing key/team counts - [PR #20904](https://github.com/BerriAI/litellm/pull/20904)
    - Guardrail pipeline support for conditional sequential execution - [PR #21177](https://github.com/BerriAI/litellm/pull/21177)
    - Pipeline flow builder UI for guardrail policies - [PR #21188](https://github.com/BerriAI/litellm/pull/21188)

- **SSO / Auth**
    - New Login With SSO Button - [PR #20908](https://github.com/BerriAI/litellm/pull/20908)
    - M2M OAuth2 UI Flow - [PR #20794](https://github.com/BerriAI/litellm/pull/20794)
    - Allow Organization and Team Admins to call `/invitation/new` - [PR #20987](https://github.com/BerriAI/litellm/pull/20987)
    - Invite User: Email Integration Alert - [PR #20790](https://github.com/BerriAI/litellm/pull/20790)
    - Populate identity fields in proxy admin JWT early-return path - [PR #21169](https://github.com/BerriAI/litellm/pull/21169)

- **Spend Logs**
    - Show predefined error codes in filter with user definable fallback - [PR #20773](https://github.com/BerriAI/litellm/pull/20773)
    - Paginated searchable model select - [PR #20892](https://github.com/BerriAI/litellm/pull/20892)
    - Sorting columns support - [PR #21143](https://github.com/BerriAI/litellm/pull/21143)
    - Allow sorting on `/spend/logs/ui` - [PR #20991](https://github.com/BerriAI/litellm/pull/20991)

- **UI Improvements**
    - Navbar: Option to hide Usage Popup - [PR #20910](https://github.com/BerriAI/litellm/pull/20910)
    - Model Page: Improve Credentials Messaging - [PR #21076](https://github.com/BerriAI/litellm/pull/21076)
    - Fallbacks: Default configurable to 10 models - [PR #21144](https://github.com/BerriAI/litellm/pull/21144)
    - Fallback display with arrows and card structure - [PR #20922](https://github.com/BerriAI/litellm/pull/20922)
    - Team Info: Migrate to AntD Tabs + Table - [PR #20785](https://github.com/BerriAI/litellm/pull/20785)
    - AntD refactoring and 0 cost models fix - [PR #20687](https://github.com/BerriAI/litellm/pull/20687)
    - Zscaler AI Guard UI - [PR #21077](https://github.com/BerriAI/litellm/pull/21077)
    - Include Config Defined Pass Through Endpoints - [PR #20898](https://github.com/BerriAI/litellm/pull/20898)
    - Rename "HTTP" to "Streamable HTTP (Recommended)" in MCP server page - [PR #21000](https://github.com/BerriAI/litellm/pull/21000)
    - MCP server discovery UI - [PR #21079](https://github.com/BerriAI/litellm/pull/21079)

- **Virtual Keys**
    - Allow Management keys to access `user/daily/activity` and team - [PR #20124](https://github.com/BerriAI/litellm/pull/20124)
    - Skip premium check for empty metadata fields on team/key update - [PR #20598](https://github.com/BerriAI/litellm/pull/20598)

#### Bugs

- Logs: Fix Input and Output Copying - [PR #20657](https://github.com/BerriAI/litellm/pull/20657)
- Teams: Fix Available Teams - [PR #20682](https://github.com/BerriAI/litellm/pull/20682)
- Spend Logs: Reset Filters Resets Custom Date Range - [PR #21149](https://github.com/BerriAI/litellm/pull/21149)
- Usage: Request Chart stack variant fix - [PR #20894](https://github.com/BerriAI/litellm/pull/20894)
- Add Auto Router: Description Text Input Focus - [PR #21004](https://github.com/BerriAI/litellm/pull/21004)
- Guardrail Edit: LiteLLM Content Filter Categories - [PR #21002](https://github.com/BerriAI/litellm/pull/21002)
- Add null guard for models in API keys table - [PR #20655](https://github.com/BerriAI/litellm/pull/20655)
- Show error details instead of 'Data Not Available' for failed requests - [PR #20656](https://github.com/BerriAI/litellm/pull/20656)
- Fix Spend Management Tests - [PR #21088](https://github.com/BerriAI/litellm/pull/21088)
- Fix JWT email domain validation error message - [PR #21212](https://github.com/BerriAI/litellm/pull/21212)

---

## AI Integrations

### Logging

- **[PostHog](../../docs/observability/posthog_integration)**
    - Fix JSON serialization error for non-serializable objects - [PR #20668](https://github.com/BerriAI/litellm/pull/20668)

- **[Prometheus](../../docs/proxy/logging#prometheus)**
    - Sanitize label values to prevent metric scrape failures - [PR #20600](https://github.com/BerriAI/litellm/pull/20600)

- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Prevent empty proxy request spans from being sent to Langfuse - [PR #19935](https://github.com/BerriAI/litellm/pull/19935)

- **[OpenTelemetry](../../docs/proxy/logging#otel)**
    - Auto-infer `otlp_http` exporter when endpoint is configured - [PR #20438](https://github.com/BerriAI/litellm/pull/20438)

- **[CloudZero](../../docs/proxy/logging)**
    - Update CBF field mappings per LIT-1907 - [PR #20906](https://github.com/BerriAI/litellm/pull/20906)

- **General**
    - Allow `MAX_CALLBACKS` override via env var - [PR #20781](https://github.com/BerriAI/litellm/pull/20781)
    - Add `standard_logging_payload_excluded_fields` config option - [PR #20831](https://github.com/BerriAI/litellm/pull/20831)
    - Enable `verbose_logger` when `LITELLM_LOG=DEBUG` - [PR #20496](https://github.com/BerriAI/litellm/pull/20496)
    - Guard against None `litellm_metadata` in batch logging path - [PR #20832](https://github.com/BerriAI/litellm/pull/20832)
    - Propagate model-level tags from config to SpendLogs - [PR #20769](https://github.com/BerriAI/litellm/pull/20769)

### Guardrails

- **Policy Templates**
    - New Policy Templates: pre-configured guardrail combinations for specific use-cases - [PR #21025](https://github.com/BerriAI/litellm/pull/21025)
    - Add NSFW policy template, toxic keywords in multiple languages, child safety content filter, JSON content viewer - [PR #21205](https://github.com/BerriAI/litellm/pull/21205)
    - Add toxic/abusive content filter guardrails - [PR #20934](https://github.com/BerriAI/litellm/pull/20934)

- **Pipeline Execution**
    - Add guardrail pipeline support for conditional sequential execution - [PR #21177](https://github.com/BerriAI/litellm/pull/21177)
    - Agent Guardrails on streaming output - [PR #21206](https://github.com/BerriAI/litellm/pull/21206)
    - Pipeline flow builder UI - [PR #21188](https://github.com/BerriAI/litellm/pull/21188)

- **[Zscaler AI Guard](../../docs/apply_guardrail)**
    - Zscaler AI Guard bug fixes and support during post-call - [PR #20801](https://github.com/BerriAI/litellm/pull/20801)
    - Zscaler AI Guard UI - [PR #21077](https://github.com/BerriAI/litellm/pull/21077)

- **[ZGuard](../../docs/apply_guardrail)**
    - Add team policy mapping for ZGuard - [PR #20608](https://github.com/BerriAI/litellm/pull/20608)

- **General**
    - Add logging to all unified guardrails + link to custom code guardrail templates - [PR #20900](https://github.com/BerriAI/litellm/pull/20900)
    - Forward request headers + `litellm_version` to generic guardrails - [PR #20729](https://github.com/BerriAI/litellm/pull/20729)
    - Empty `guardrails`/`policies` arrays should not trigger enterprise license check - [PR #20567](https://github.com/BerriAI/litellm/pull/20567)
    - Fix OpenAI moderation guardrails - [PR #20718](https://github.com/BerriAI/litellm/pull/20718)
    - Fix `/v2/guardrails/list` returning sensitive values - [PR #20796](https://github.com/BerriAI/litellm/pull/20796)
    - Fix guardrail status error - [PR #20972](https://github.com/BerriAI/litellm/pull/20972)
    - Reuse `get_instance_fn` in `initialize_custom_guardrail` - [PR #20917](https://github.com/BerriAI/litellm/pull/20917)

---

## Spend Tracking, Budgets and Rate Limiting

- **Prevent shared backend model key from being polluted** by per-deployment custom pricing - [PR #20679](https://github.com/BerriAI/litellm/pull/20679)
- **Avoid in-place mutation** in SpendUpdateQueue aggregation - [PR #20876](https://github.com/BerriAI/litellm/pull/20876)

---

## MCP Gateway (12 updates)

- **MCP M2M OAuth2 Support** - Add support for machine-to-machine OAuth2 for MCP servers - [PR #20788](https://github.com/BerriAI/litellm/pull/20788)
- **MCP Server Discovery UI** - Browse and discover available MCP servers from the UI - [PR #21079](https://github.com/BerriAI/litellm/pull/21079)
- **MCP Tracing** - Add OpenTelemetry tracing for MCP calls running through AI Gateway - [PR #21018](https://github.com/BerriAI/litellm/pull/21018)
- **MCP OAuth2 Debug Headers** - Client-side debug headers for OAuth2 troubleshooting - [PR #21151](https://github.com/BerriAI/litellm/pull/21151)
- **Fix MCP "Session not found" errors** - Resolve session persistence issues - [PR #21040](https://github.com/BerriAI/litellm/pull/21040)
- **Fix MCP OAuth2 root endpoints** returning "MCP server not found" - [PR #20784](https://github.com/BerriAI/litellm/pull/20784)
- **Fix MCP OAuth2 query param merging** when `authorization_url` already contains params - [PR #20968](https://github.com/BerriAI/litellm/pull/20968)
- **Fix MCP SCOPES on Atlassian** issue - [PR #21150](https://github.com/BerriAI/litellm/pull/21150)
- **Fix MCP StreamableHTTP backend** - Use `anyio.fail_after` instead of `asyncio.wait_for` - [PR #20891](https://github.com/BerriAI/litellm/pull/20891)
- **Inject `NPM_CONFIG_CACHE`** into STDIO MCP subprocess env - [PR #21069](https://github.com/BerriAI/litellm/pull/21069)
- **Block spaces and hyphens** in MCP server names and aliases - [PR #21074](https://github.com/BerriAI/litellm/pull/21074)

---

## Performance / Loadbalancing / Reliability improvements (8 improvements)

- **Remove orphan entries from queue** - Fix memory leak in scheduler queue - [PR #20866](https://github.com/BerriAI/litellm/pull/20866)
- **Remove repeated provider parsing** in budget limiter hot path - [PR #21043](https://github.com/BerriAI/litellm/pull/21043)
- **Use current retry exception** for retry backoff instead of stale exception - [PR #20725](https://github.com/BerriAI/litellm/pull/20725)
- **Add Semgrep & fix OOMs** - Static analysis rules and out-of-memory fixes - [PR #20912](https://github.com/BerriAI/litellm/pull/20912)
- **Add Pyroscope** for continuous profiling and observability - [PR #21167](https://github.com/BerriAI/litellm/pull/21167)
- **Respect `ssl_verify`** with shared aiohttp sessions - [PR #20349](https://github.com/BerriAI/litellm/pull/20349)
- **Fix shared health check serialization** - [PR #21119](https://github.com/BerriAI/litellm/pull/21119)
- **Change model mismatch logs** from WARNING to DEBUG - [PR #20994](https://github.com/BerriAI/litellm/pull/20994)

---

## Database Changes

### Schema Updates

| Table | Change Type | Description | PR | Migration |
| ----- | ----------- | ----------- | -- | --------- |
| `LiteLLM_VerificationToken` | New Indexes | Added indexes on `user_id`+`team_id`, `team_id`, and `budget_reset_at`+`expires` | [PR #20736](https://github.com/BerriAI/litellm/pull/20736) | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260209085821_add_verificationtoken_indexes/migration.sql) |
| `LiteLLM_PolicyAttachmentTable` | New Column | Added `tags` text array for policy-to-tag connections | [PR #21061](https://github.com/BerriAI/litellm/pull/21061) | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260212103349_adjust_tags_policy_table/migration.sql) |
| `LiteLLM_AccessGroupTable` | New Table | Access groups for managing model, MCP server, and agent access | [PR #21022](https://github.com/BerriAI/litellm/pull/21022) | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260212143306_add_access_group_table/migration.sql) |
| `LiteLLM_AccessGroupTable` | Column Change | Renamed `access_model_ids` to `access_model_names` | [PR #21166](https://github.com/BerriAI/litellm/pull/21166) | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260213170952_access_group_change_to_model_name/migration.sql) |
| `LiteLLM_ManagedVectorStoreTable` | New Table | Managed vector store tracking with model mappings | - | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260213105436_add_managed_vector_store_table/migration.sql) |
| `LiteLLM_TeamTable`, `LiteLLM_VerificationToken` | New Column | Added `access_group_ids` text array | [PR #21022](https://github.com/BerriAI/litellm/pull/21022) | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260212143306_add_access_group_table/migration.sql) |
| `LiteLLM_GuardrailsTable` | New Column | Added `team_id` text column | - | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260214094754_schema_sync/migration.sql) |

---

## Documentation Updates (14 updates)

- LiteLLM Observatory section added to v1.81.9 release notes - [PR #20675](https://github.com/BerriAI/litellm/pull/20675)
- Callback registration optimization added to release notes - [PR #20681](https://github.com/BerriAI/litellm/pull/20681)
- Middleware performance blog post - [PR #20677](https://github.com/BerriAI/litellm/pull/20677)
- UI Team Soft Budget documentation - [PR #20669](https://github.com/BerriAI/litellm/pull/20669)
- UI Contributing and Troubleshooting guide - [PR #20674](https://github.com/BerriAI/litellm/pull/20674)
- Reorganize Admin UI subsection - [PR #20676](https://github.com/BerriAI/litellm/pull/20676)
- SDK proxy authentication (OAuth2/JWT auto-refresh) - [PR #20680](https://github.com/BerriAI/litellm/pull/20680)
- Forward client headers to LLM API documentation fix - [PR #20768](https://github.com/BerriAI/litellm/pull/20768)
- Add docs guide for using policies - [PR #20914](https://github.com/BerriAI/litellm/pull/20914)
- Add native thinking param examples for Claude Opus 4.6 - [PR #20799](https://github.com/BerriAI/litellm/pull/20799)
- Fix Claude Code MCP tutorial - [PR #21145](https://github.com/BerriAI/litellm/pull/21145)
- Add API base URLs for Dashscope (International and China/Beijing) - [PR #21083](https://github.com/BerriAI/litellm/pull/21083)
- Fix `DEFAULT_NUM_WORKERS_LITELLM_PROXY` default (1, not 4) - [PR #21127](https://github.com/BerriAI/litellm/pull/21127)
- Correct ElevenLabs support status in README - [PR #20643](https://github.com/BerriAI/litellm/pull/20643)

---

## New Contributors
* @iver56 made their first contribution in [PR #20643](https://github.com/BerriAI/litellm/pull/20643)
* @eliasaronson made their first contribution in [PR #20666](https://github.com/BerriAI/litellm/pull/20666)
* @NirantK made their first contribution in [PR #19656](https://github.com/BerriAI/litellm/pull/19656)
* @looksgood made their first contribution in [PR #20919](https://github.com/BerriAI/litellm/pull/20919)
* @kelvin-tran made their first contribution in [PR #20548](https://github.com/BerriAI/litellm/pull/20548)
* @bluet made their first contribution in [PR #20873](https://github.com/BerriAI/litellm/pull/20873)
* @itayov made their first contribution in [PR #20729](https://github.com/BerriAI/litellm/pull/20729)
* @CSteigstra made their first contribution in [PR #20960](https://github.com/BerriAI/litellm/pull/20960)
* @rahulrd25 made their first contribution in [PR #20569](https://github.com/BerriAI/litellm/pull/20569)
* @muraliavarma made their first contribution in [PR #20598](https://github.com/BerriAI/litellm/pull/20598)
* @joaokopernico made their first contribution in [PR #21039](https://github.com/BerriAI/litellm/pull/21039)
* @datzscaler made their first contribution in [PR #21077](https://github.com/BerriAI/litellm/pull/21077)
* @atapia27 made their first contribution in [PR #20922](https://github.com/BerriAI/litellm/pull/20922)
* @fpagny made their first contribution in [PR #21121](https://github.com/BerriAI/litellm/pull/21121)
* @aidankovacic-8451 made their first contribution in [PR #21119](https://github.com/BerriAI/litellm/pull/21119)
* @luisgallego-aily made their first contribution in [PR #19935](https://github.com/BerriAI/litellm/pull/19935)

---

## Full Changelog
[v1.81.9.rc.1...v1.81.12.rc.1](https://github.com/BerriAI/litellm/compare/v1.81.9.rc.1...v1.81.12.rc.1)
