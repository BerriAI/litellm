---
title: "v1.81.3-stable - Performance - 25% CPU Usage Reduction"
slug: "v1-81-3"
date: 2026-01-26T10:00:00
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
docker.litellm.ai/berriai/litellm:v1.81.3-stable
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.81.3.rc.2
```

</TabItem>
</Tabs>

---

## New Models / Updated Models

### New Model Support

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Deprecation Date |
| -------- | ----- | -------------- | ------------------- | -------------------- | ---------------- |
| OpenAI | `gpt-audio`, `gpt-audio-2025-08-28` | 128K | $32/1M audio tokens, $2.5/1M text tokens | $64/1M audio tokens, $10/1M text tokens | - | 
| OpenAI | `gpt-audio-mini`, `gpt-audio-mini-2025-08-28` | 128K | $10/1M audio tokens, $0.6/1M text tokens | $20/1M audio tokens, $2.4/1M text tokens | - |
| Deepinfra, Vertex AI, Google AI Studio, OpenRouter, Vercel AI Gateway | `gemini-2.0-flash-001`, `gemini-2.0-flash` |  - | - | - | 2026-03-31 |
| Groq | `openai/gpt-oss-120b` | 131K | 0.075/1M cache read | 0.6/1M output tokens | - |
| Groq | `groq/openai/gpt-oss-20b` | 131K | 0.0375/1M cache read, $0.075/1M text tokens | 0.3/1M output tokens | - |
| Vertex AI | `gemini-2.5-computer-use-preview-10-2025` | 128K |  $1.25 | $10 | - |
| Azure AI | `claude-haiku-4-5` | $1.25/1M cache read, $2/1M cache read above 1 hr, $0.1/1M text tokens | $5/1M output tokens | - |
| Azure AI | `claude-sonnet-4-5` | $3.75/1M cache read, $6/1M cache read above 1 hr, $3/1M text tokens | $15/1M output tokens | - |
| Azure AI | `claude-opus-4-5` | $6.25/1M cache read, $10/1M cache read above 1 hr, $0.5/1M text tokens | $25/1M output tokens | - |
| Azure AI | `claude-opus-4-1` | $18.75/1M cache read, $30/1M cache read above 1 hr, $1.5/1M text tokens | $75/1M output tokens | - |

### Features

- **[OpenAI](../../docs/providers/openai)**
    - Add gpt-audio and gpt-audio-mini models to pricing - [PR #19509](https://github.com/BerriAI/litellm/pull/19509)
    - correct audio token costs for gpt-4o-audio-preview models - [PR #19500](https://github.com/BerriAI/litellm/pull/19500)
    - Limit stop sequence as per openai spec (ensures JetBrains IDE compatibility) - [PR #19562](https://github.com/BerriAI/litellm/pull/19562)

- **[VertexAI](../../docs/providers/vertex)**
    - Docs - Google Workload Identity Federation (WIF) support - [PR #19320](https://github.com/BerriAI/litellm/pull/19320)

- **[Agentcore](../../docs/providers/bedrock_agentcore)**
    - Fixes streaming issues with AWS Bedrock AgentCore where responses would stop after the first chunk, particularly affecting OAuth-enabled agents - [PR #17141](https://github.com/BerriAI/litellm/pull/17141)

- **[Chatgpt](../../docs/providers/chatgpt)**
    - Adds support for calling chatgpt subscription via LiteLLM - [PR #19030](https://github.com/BerriAI/litellm/pull/19030)
    - Adds responses API bridge support for chatgpt subscription provider - [PR #19030](https://github.com/BerriAI/litellm/pull/19030)

- **[Bedrock](../../docs/providers/bedrock)**
    - support for output format for bedrock invoke via v1/messages - [PR #19560](https://github.com/BerriAI/litellm/pull/19560)
    
- **[Azure](../../docs/providers/azure/azure)**
    - Add support for Azure OpenAI v1 API - [PR #19313](https://github.com/BerriAI/litellm/pull/19313)
    - preserve content_policy_violation details for images (#19328) - [PR #19372](https://github.com/BerriAI/litellm/pull/19372)
    - Support OpenAI-format nested tool definitions for Responses API - [PR #19526](https://github.com/BerriAI/litellm/pull/19526)

- **Gemini([Vertex AI](../../docs/providers/vertex), [Google AI Studio](../../docs/providers/gemini))**
    - use responseJsonSchema for Gemini 2.0+ models - [PR #19314](https://github.com/BerriAI/litellm/pull/19314)

- **[Volcengine](../../docs/providers/volcano)**
    - Support Volcengine responses api - [PR #18508](https://github.com/BerriAI/litellm/pull/18508)

- **[Anthropic](../../docs/providers/anthropic)**
    - Add Support for calling Claude Code Max subscriptions via LiteLLM - [PR #19453](https://github.com/BerriAI/litellm/pull/19453)
    - Add Structured output for /v1/messages with Anthropic API, Azure Anthropic API, Bedrock Converse - [PR #19545](https://github.com/BerriAI/litellm/pull/19545)

- **[Brave Search](../../docs/search/brave)**
    - New Search provider - [PR #19433](https://github.com/BerriAI/litellm/pull/19433)

- **Sarvam ai**
    - Add support for new sarvam models  - [PR #19479](https://github.com/BerriAI/litellm/pull/19479)

- **[GMI](../../docs/providers/gmi)**
    - add GMI Cloud provider support - [PR #19376](https://github.com/BerriAI/litellm/pull/19376)


### Bug Fixes

- **[Anthropic](../../docs/providers/anthropic)**
    - Fix anthropic-beta sent client side being overridden instead of appended to - [PR #19343](https://github.com/BerriAI/litellm/pull/19343)
    - Filter out unsupported fields from JSON schema for Anthropic's output_format API - [PR #19482](https://github.com/BerriAI/litellm/pull/19482)

- **[Bedrock](../../docs/providers/bedrock)**
    - Expose stability models via /image_edits endpoint and ensure proper request transformation - [PR #19323](https://github.com/BerriAI/litellm/pull/19323)
    - Claude Code x Bedrock Invoke fails with advanced-tool-use-2025-11-20 - [PR #19373](https://github.com/BerriAI/litellm/pull/19373)
    - deduplicate tool calls in assistant history - [PR #19324](https://github.com/BerriAI/litellm/pull/19324)
    - fix: correct us.anthropic.claude-opus-4-5 In-region pricing - [PR #19310](https://github.com/BerriAI/litellm/pull/19310)
    - Fix request validation errors when using Claude 4 via bedrock invoke - [PR #19381](https://github.com/BerriAI/litellm/pull/19381)
    - Handle thinking with tool calls for Claude 4 models - [PR #19506](https://github.com/BerriAI/litellm/pull/19506)
    - correct streaming choice index for tool calls - [PR #19506](https://github.com/BerriAI/litellm/pull/19506)

- **[Ollama](../../docs/providers/ollama)**
    - Fix tool call errors due with improved message extraction - [PR #19369](https://github.com/BerriAI/litellm/pull/19369)

- **[VertexAI](../../docs/providers/vertex)**
    - Removed optional vertex_count_tokens_location param before request is sent to vertex - [PR #19359](https://github.com/BerriAI/litellm/pull/19359)

- **Gemini([Vertex AI](../../docs/providers/vertex), [Google AI Studio](../../docs/providers/gemini))**
    - Supports setting media_resolution and fps parameters on each video file, when using Gemini video understanding - [PR #19273](https://github.com/BerriAI/litellm/pull/19273)
    - handle reasoning_effort as dict from OpenAI Agents SDK - [PR #19419](https://github.com/BerriAI/litellm/pull/19419)
    - add file content support in tool results - [PR #19416](https://github.com/BerriAI/litellm/pull/19416)

- **[Azure](../../docs/providers/azure_ai)**
    - Fix Azure AI costs for Anthropic models - [PR #19530](https://github.com/BerriAI/litellm/pull/19530)

- **[Giga Chat](../../docs/providers/gigachat)**
    - Add tool choice mapping - [PR #19645](https://github.com/BerriAI/litellm/pull/19645)
---

## AI API Endpoints (LLMs, MCP, Agents)

### Features

- **[Files API](../../docs/files_endpoints)**
    - Add managed files support when load_balancing is True - [PR #19338](https://github.com/BerriAI/litellm/pull/19338)

- **[Claude Plugin Marketplace](../../docs/tutorials/claude_code_plugin_marketplace)**
    - Add self hosted Claude Code Plugin Marketplace - [PR #19378](https://github.com/BerriAI/litellm/pull/19378)

- **[MCP](../../docs/mcp)**
    - Add MCP Protocol version 2025-11-25 support - [PR #19379](https://github.com/BerriAI/litellm/pull/19379)
    - Log MCP tool calls and list tools in the LiteLLM Spend Logs table for easier debugging - [PR #19469](https://github.com/BerriAI/litellm/pull/19469)

- **[Vertex AI](../../docs/providers/vertex)**
    - Ensure only anthropic betas are forwarded down to LLM API (by default) - [PR #19542](https://github.com/BerriAI/litellm/pull/19542)
    - Allow overriding to support forwarding incoming headers are forwarded down to target - [PR #19524](https://github.com/BerriAI/litellm/pull/19524)

- **[Chat/Completions](../../docs/completion/input)**
    - Add MCP tools response to chat completions - [PR #19552](https://github.com/BerriAI/litellm/pull/19552)
    - Add custom vertex ai finish reasons to the output - [PR #19558](https://github.com/BerriAI/litellm/pull/19558)
    - Return MCP execution in /chat/completions before model output during streaming - [PR #19623](https://github.com/BerriAI/litellm/pull/19623)

### Bugs

- **[Responses API](../../docs/response_api)**
    - Fix duplicate messages during MCP streaming tool execution - [PR #19317](https://github.com/BerriAI/litellm/pull/19317)
    - Fix pickle error when using OpenAI's Responses API with stream=True and tool_choice of type allowed_tools (an OpenAI-native parameter) - [PR #17205](https://github.com/BerriAI/litellm/pull/17205)
    - stream tool call events for non-openai models - [PR #19368](https://github.com/BerriAI/litellm/pull/19368)
    - preserve tool output ordering for gemini in responses bridge - [PR #19360](https://github.com/BerriAI/litellm/pull/19360)
    - Add ID caching to prevent ID mismatch text-start and text-delta - [PR #19390](https://github.com/BerriAI/litellm/pull/19390)
    - Include output_item, reasoning_summary_Text_done and reasoning_summary_part_done events for non-openai models - [PR #19472](https://github.com/BerriAI/litellm/pull/19472) 

- **[Chat/Completions](../../docs/completion/input)**
    - fix: drop_params not dropping prompt_cache_key for non-OpenAI providers - [PR #19346](https://github.com/BerriAI/litellm/pull/19346)

- **[Realtime API](../../docs/realtime)**
    - disable SSL for ws:// WebSocket connections - [PR #19345](https://github.com/BerriAI/litellm/pull/19345)

- **[Generate Content](../../docs/generateContent)**
    - Log actual user input when google genai/vertex endpoints are called client-side - [PR #19156](https://github.com/BerriAI/litellm/pull/19156)

- **[/messages/count_tokens Anthropic Token Counting](../../docs/anthropic_count_tokens)**
    - ensure it works for Anthropic, Azure AI Anthropic on AI Gateway - [PR #19432](https://github.com/BerriAI/litellm/pull/19432)

- **[MCP](../../docs/mcp)**
    - forward static_headers to MCP servers - [PR #19366](https://github.com/BerriAI/litellm/pull/19366)

- **[Batch API](../../docs/batches)**
    - Fix: generation config empty for batch - [PR #19556](https://github.com/BerriAI/litellm/pull/19556)

- **[Pass Through Endpoints](../../docs/proxy/pass_through)**
    - Always reupdate registry - [PR #19420](https://github.com/BerriAI/litellm/pull/19420)
---

## Management Endpoints / UI

### Features

- **Cost Estimator**
    - Fix model dropdown - [PR #19529](https://github.com/BerriAI/litellm/pull/19529)

- **Claude Code Plugins**
    - Allow Adding Claude Code Plugins via UI - [PR #19387](https://github.com/BerriAI/litellm/pull/19387)

- **Guardrails**
    - New Policy management UI - [PR #19668](https://github.com/BerriAI/litellm/pull/19668)
    - Allow adding policies on Keys/Teams + Viewing on Info panels - [PR #19688](https://github.com/BerriAI/litellm/pull/19688)

- **General**
    - respects custom authentication header override - [PR #19276](https://github.com/BerriAI/litellm/pull/19276)

- **Playground**
    - Button to Fill Custom API Base - [PR #19440](https://github.com/BerriAI/litellm/pull/19440)
    - display mcp output on the play ground - [PR #19553](https://github.com/BerriAI/litellm/pull/19553)

- **Models**
    - Paginate /v2/models/info - [PR #19521](https://github.com/BerriAI/litellm/pull/19521)
    - All Model Tab Pagination - [PR #19525](https://github.com/BerriAI/litellm/pull/19525)
    - Adding Optional scope Param to /models - [PR #19539](https://github.com/BerriAI/litellm/pull/19539)
    - Model Search - [PR #19622](https://github.com/BerriAI/litellm/pull/19622)
    - Filter by Model ID and Team ID - [PR #19713](https://github.com/BerriAI/litellm/pull/19713)

- **MCP Servers**
    - MCP Tools Tab Resetting to Overview - [PR #19468](https://github.com/BerriAI/litellm/pull/19468)

- **Organizations**
    - Prevent org admin from creating a new user with proxy_admin permissions - [PR #19296](https://github.com/BerriAI/litellm/pull/19296)
    - Edit Page: Reusable Model Select - [PR #19601](https://github.com/BerriAI/litellm/pull/19601)

- **Teams**
    - Reusable Model Select - [PR #19543](https://github.com/BerriAI/litellm/pull/19543)
    - [Fix] Team Update with Organization having All Proxy Models - [PR #19604](https://github.com/BerriAI/litellm/pull/19604)

- **Logs**
    - Include tool arguments in spend logs table - [PR #19640](https://github.com/BerriAI/litellm/pull/19640)

- **Fallbacks / Loadbalancing**
    - New fallbacks modal - [PR #19673](https://github.com/BerriAI/litellm/pull/19673)
    - Set fallbacks/loadbalancing by team/key - [PR #19686](https://github.com/BerriAI/litellm/pull/19686)

### Bugs

- **Playground**
    - increase model selector width in playground Compare view - [PR #19423](https://github.com/BerriAI/litellm/pull/19423)

- **Virtual Keys**
    - Sorting Shows Incorrect Entries - [PR #19534](https://github.com/BerriAI/litellm/pull/19534)

- **General**
    - UI 404 error when SERVER_ROOT_PATH is set - [PR #19467](https://github.com/BerriAI/litellm/pull/19467)
    - Redirect to ui/login on expired JWT - [PR #19687](https://github.com/BerriAI/litellm/pull/19687)

- **SSO**
    - Fix SSO user roles not updating for existing users - [PR #19621](https://github.com/BerriAI/litellm/pull/19621)

- **Guardrails**
    - ensure guardrail patterns persist on edit and mode toggle - [PR #19265](https://github.com/BerriAI/litellm/pull/19265)
---

## AI Integrations

### Logging

- **General Logging**
    - prevent printing duplicate StandardLoggingPayload logs - [PR #19325](https://github.com/BerriAI/litellm/pull/19325)
    - Fix: log duplication when json_logs is enabled - [PR #19705](https://github.com/BerriAI/litellm/pull/19705)
- **Langfuse OTEL**
    - ignore service logs and fix callback shadowing - [PR #19298](https://github.com/BerriAI/litellm/pull/19298)
- **Langfuse**
    - Send litellm_trace_id - [PR #19528](https://github.com/BerriAI/litellm/pull/19528)
    - Add Langfuse mock mode for testing without API calls - [PR #19676](https://github.com/BerriAI/litellm/pull/19676)
- **GCS Bucket**
    - prevent unbounded queue growth due to slow API calls - [PR #19297](https://github.com/BerriAI/litellm/pull/19297)
    - Add GCS mock mode for testing without API calls - [PR #19683](https://github.com/BerriAI/litellm/pull/19683)
- **Responses API Logging**
    - Fix pydantic serialization error - [PR #19486](https://github.com/BerriAI/litellm/pull/19486)
- **Arize Phoenix**
    - add openinference span kinds to arize phoenix - [PR #19267](https://github.com/BerriAI/litellm/pull/19267)
- **Prometheus**
    - Added new prometheus metrics for user count and team count - [PR #19520](https://github.com/BerriAI/litellm/pull/19520)

### Guardrails

- **Bedrock Guardrails**
    - Ensure post_call guardrail checks input+output - [PR #19151](https://github.com/BerriAI/litellm/pull/19151)
- **Prompt Security**
    - fixing prompt-security's guardrail implementation - [PR #19374](https://github.com/BerriAI/litellm/pull/19374)
- **Presidio**
    - Fixes crash in Presidio Guardrail when running in background threads (logging_hook) - [PR #19714](https://github.com/BerriAI/litellm/pull/19714)
- **Pillar Security**
    - Migrate Pillar Security to Generic Guardrail API - [PR #19364](https://github.com/BerriAI/litellm/pull/19364)
- **Policy Engine**
    - New LiteLLM Policy engine - create policies to manage guardrails, conditions - permissions per Key, Team - [PR #19612](https://github.com/BerriAI/litellm/pull/19612)
- **General**
    - add case-insensitive support for guardrail mode and actions - [PR #19480](https://github.com/BerriAI/litellm/pull/19480)

### Prompt Management

- **General**
    - fix prompt info lookup and delete using correct IDs - [PR #19358](https://github.com/BerriAI/litellm/pull/19358)

### Secret Manager

- **AWS Secret Manager**
    - ensure auto-rotation updates existing AWS secret instead of creating new one - [PR #19455](https://github.com/BerriAI/litellm/pull/19455)
- **Hashicorp Vault**
    - Ensure key rotations work with Vault - [PR #19634](https://github.com/BerriAI/litellm/pull/19634)

---

## Spend Tracking, Budgets and Rate Limiting

- **Pricing Updates**
    - Add openai/dall-e base pricing entries - [PR #19133](https://github.com/BerriAI/litellm/pull/19133)
    - Add `input_cost_per_video_per_second` in ModelInfoBase - [PR #19398](https://github.com/BerriAI/litellm/pull/19398)

---

## Performance / Loadbalancing / Reliability improvements


- **General**
    - Fix date overflow/division by zero in proxy utils - [PR #19527](https://github.com/BerriAI/litellm/pull/19527)
    - Fix in-flight request termination on SIGTERM when health-check runs in a separate process - [PR #19427](https://github.com/BerriAI/litellm/pull/19427)
    - Fix Pass through routes to work with server root path - [PR #19383](https://github.com/BerriAI/litellm/pull/19383)
    - Fix logging error for stop iteration - [PR #19649](https://github.com/BerriAI/litellm/pull/19649)
    - prevent retrying 4xx client errors - [PR #19275](https://github.com/BerriAI/litellm/pull/19275)
    - add better error handling for misconfig on health check - [PR #19441](https://github.com/BerriAI/litellm/pull/19441)

- **Router**
    - Fix Azure RPM calculation formula - [PR #19513](https://github.com/BerriAI/litellm/pull/19513)
    - Persist scheduler request queue to redis - [PR #19304](https://github.com/BerriAI/litellm/pull/19304)
    - pass search_tools to Router during DB-triggered initialization - [PR #19388](https://github.com/BerriAI/litellm/pull/19388)
    - Fixed PromptCachingCache to correctly handle messages where cache_control is a sibling key of string content - [PR #19266](https://github.com/BerriAI/litellm/pull/19266)

- **Memory Leaks/OOM**
    - prevent OOM with nested $defs in tool schemas - [PR #19112](https://github.com/BerriAI/litellm/pull/19112)
    - fix: HTTP client memory leaks in Presidio, OpenAI, and Gemini - [PR #19190](https://github.com/BerriAI/litellm/pull/19190)

- **Non root**
    - fix logfile and pidfile of supervisor for non root environment - [PR #17267](https://github.com/BerriAI/litellm/pull/17267)
    - resolve Read-only file system error in non-root images - [PR #19449](https://github.com/BerriAI/litellm/pull/19449)

- **Dockerfile**
    - Redis Semantic Caching - add missing redisvl dependency to requirements.txt - [PR #19417](https://github.com/BerriAI/litellm/pull/19417)
    - Bump OTEL versions to support a2a dependency - resolves modulenotfounderror for Microsoft Agents by @Harshit28j in #18991

- **DB**
    - Handle PostgreSQL cached plan errors during rolling deployments - [PR #19424](https://github.com/BerriAI/litellm/pull/19424)

- **Timeouts**
    - Fix: total timeout is not respected - [PR #19389](https://github.com/BerriAI/litellm/pull/19389)

- **SDK**
    - Field-Existence Checks to Type Classes to Prevent Attribute Errors - [PR #18321](https://github.com/BerriAI/litellm/pull/18321)
    - add google-cloud-aiplatform as optional dependency with clear error message - [PR #19437](https://github.com/BerriAI/litellm/pull/19437)
    - Make grpc dependency optional - [PR #19447](https://github.com/BerriAI/litellm/pull/19447)
    - Add support for retry policies - [PR #19645](https://github.com/BerriAI/litellm/pull/19645)

- **Performance**
    - Cut chat_completion latency by ~21% by reducing pre-call processing time - [PR #19535](https://github.com/BerriAI/litellm/pull/19535)
    - Optimize strip_trailing_slash with O(1) index check - [PR #19679](https://github.com/BerriAI/litellm/pull/19679)
    - Optimize use_custom_pricing_for_model with set intersection - [PR #19677](https://github.com/BerriAI/litellm/pull/19677)
    - perf: skip pattern_router.route() for non-wildcard models - [PR #19664](https://github.com/BerriAI/litellm/pull/19664)
    - perf: Add LRU caching to get_model_info for faster cost lookups - [PR #19606](https://github.com/BerriAI/litellm/pull/19606)

---

## General Proxy Improvements

### Doc Improvements
    - new tutorial for adding MCPs to Cursor via LiteLLM - [PR #19317](https://github.com/BerriAI/litellm/pull/19317)
    - fix vertex_region to vertex_location in Vertex AI pass-through docs - [PR #19380](https://github.com/BerriAI/litellm/pull/19380)
    - clarify Gemini and Vertex AI model prefix in json file - [PR #19443](https://github.com/BerriAI/litellm/pull/19443)
    - update Claude Code integration guides - [PR #19415](https://github.com/BerriAI/litellm/pull/19415)
    - adjust opencode tutorial - [PR #19605](https://github.com/BerriAI/litellm/pull/19605)
    - add spend-queue-troubleshooting docs - [PR #19659](https://github.com/BerriAI/litellm/pull/19659)
    - docs: add litellm-enterprise requirement for managed files - [PR #19689](https://github.com/BerriAI/litellm/pull/19689)

### Helm
    - Add support for keda in helm chart - [PR #19337](https://github.com/BerriAI/litellm/pull/19337)
    - sync Helm chart version with LiteLLM release version - [PR #19438](https://github.com/BerriAI/litellm/pull/19438)
    - Enable PreStop hook configuration in values.yaml - [PR #19613](https://github.com/BerriAI/litellm/pull/19613)

### General
    - Add health check scripts and parallel execution support - [PR #19295](https://github.com/BerriAI/litellm/pull/19295)


---

## New Contributors


* @dushyantzz made their first contribution in [PR #19158](https://github.com/BerriAI/litellm/pull/19158)
* @obod-mpw made their first contribution in [PR #19133](https://github.com/BerriAI/litellm/pull/19133)
* @msexxeta made their first contribution in [PR #19030](https://github.com/BerriAI/litellm/pull/19030)
* @rsicart made their first contribution in [PR #19337](https://github.com/BerriAI/litellm/pull/19337)
* @cluebbehusen made their first contribution in [PR #19311](https://github.com/BerriAI/litellm/pull/19311)
* @Lucky-Lodhi2004 made their first contribution in [PR #19315](https://github.com/BerriAI/litellm/pull/19315)
* @binbandit made their first contribution in [PR #19324](https://github.com/BerriAI/litellm/pull/19324)
* @flex-myeonghyeon made their first contribution in [PR #19381](https://github.com/BerriAI/litellm/pull/19381)
* @Lrakotoson made their first contribution in [PR #18321](https://github.com/BerriAI/litellm/pull/18321)
* @bensi94 made their first contribution in [PR #18787](https://github.com/BerriAI/litellm/pull/18787)
* @victorigualada made their first contribution in [PR #19368](https://github.com/BerriAI/litellm/pull/19368)
* @VedantMadane made their first contribution in #19266
* @stiyyagura0901 made their first contribution in #19276
* @kamilio made their first contribution in [PR #19447](https://github.com/BerriAI/litellm/pull/19447)
* @jonathansampson made their first contribution in [PR #19433](https://github.com/BerriAI/litellm/pull/19433)
* @rynecarbone made their first contribution in [PR #19416](https://github.com/BerriAI/litellm/pull/19416)
* @jayy-77 made their first contribution in #19366
* @davida-ps made their first contribution in [PR #19374](https://github.com/BerriAI/litellm/pull/19374)
* @joaodinissf made their first contribution in [PR #19506](https://github.com/BerriAI/litellm/pull/19506)
* @ecao310 made their first contribution in [PR #19520](https://github.com/BerriAI/litellm/pull/19520)
* @mpcusack-altos made their first contribution in [PR #19577](https://github.com/BerriAI/litellm/pull/19577)
* @milan-berri made their first contribution in [PR #19602](https://github.com/BerriAI/litellm/pull/19602)
* @xqe2011 made their first contribution in #19621

---

## Full Changelog

**[View complete changelog on GitHub](https://github.com/BerriAI/litellm/releases/tag/v1.81.3.rc)**
