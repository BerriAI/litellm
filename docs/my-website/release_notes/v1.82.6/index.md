---
title: "v1.82.6 - gpt-5.4-mini, gpt-5.4-nano, Volcengine Doubao Seed 2.0, Multi-Proxy Control Plane"
slug: "v1-82-6"
date: 2026-03-23T00:00:00
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

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:main-1.82.6.rc.1
```

</TabItem>
<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.82.6.rc.1
```

</TabItem>
</Tabs>

## Key Highlights

- **gpt-5.4-mini and gpt-5.4-nano — day 0** — Full pricing and routing support for `gpt-5.4-mini` (272K context, $0.75/$4.50) and `gpt-5.4-nano` (1.05M context, $0.20/$1.25) on OpenAI and Azure - [PR #23958](https://github.com/BerriAI/litellm/pull/23958)
- **Volcengine Doubao Seed 2.0** — `doubao-seed-2-0-pro`, `doubao-seed-2-0-lite`, `doubao-seed-2-0-mini`, and `doubao-seed-2-0-code-preview` added with tiered pricing support
- **Multi-proxy worker control plane** — New control plane for coordinating multiple proxy worker processes — centralized config, routing, and health management across workers - [PR #24217](https://github.com/BerriAI/litellm/pull/24217)
- **Security: privilege escalation fix** — Fixed privilege escalation on `/key/block`, `/key/unblock`, and `/key/update` `max_budget` — non-admin users could previously modify keys they didn't own - [PR #23781](https://github.com/BerriAI/litellm/pull/23781)
- **Anthropic reasoning summary opt-out** — New `anthropic_reasoning_summary` flag to disable automatic injection of the default reasoning summary in Anthropic API responses - [PR #22904](https://github.com/BerriAI/litellm/pull/22904)
- **Prompt management for Responses API** — Prompt templates and versioning now work with the OpenAI Responses API - [PR #23999](https://github.com/BerriAI/litellm/pull/23999)
- **Per-model-group deployment affinity** — Router now supports sticky deployment routing per model group, reducing cold-start variance in production - [PR #24110](https://github.com/BerriAI/litellm/pull/24110)

---

## New Models / Updated Models

#### New Model Support (12 new models)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| OpenAI | `gpt-5.4-mini` | 272K | $0.75 | $4.50 | chat, vision, tools, reasoning, prompt caching |
| OpenAI | `gpt-5.4-nano` | 1.05M | $0.20 | $1.25 | chat, vision, tools, reasoning, prompt caching |
| Azure OpenAI | `azure/gpt-5.4-mini` | 1.05M | $0.75 | $4.50 | chat, vision, tools, reasoning |
| Azure OpenAI | `azure/gpt-5.4-nano` | 1.05M | $0.20 | $1.25 | chat, vision, tools, reasoning |
| OpenAI | `gpt-4-0314` | 8K | $30.00 | $60.00 | chat (restored; deprecation 2026-03-26) |
| xAI | `xai/grok-4.20-beta-0309-reasoning` | 2M | $2.00 | $6.00 | chat, vision, tools, web search, reasoning |
| xAI | `xai/grok-4.20-beta-0309-non-reasoning` | 2M | $2.00 | $6.00 | chat, vision, tools, web search |
| xAI | `xai/grok-4.20-multi-agent-beta-0309` | 2M | - | - | chat, vision, tools, web search |
| Volcengine | `volcengine/doubao-seed-2-0-pro-260215` | 256K | tiered | tiered | chat, vision, reasoning |
| Volcengine | `volcengine/doubao-seed-2-0-lite-260215` | 256K | tiered | tiered | chat, vision, reasoning |
| Volcengine | `volcengine/doubao-seed-2-0-mini-260215` | 256K | tiered | tiered | chat, vision, reasoning |
| Volcengine | `volcengine/doubao-seed-2-0-code-preview-260215` | 256K | tiered | tiered | chat, vision, reasoning |

#### Updated Models

- **[OpenAI](../../docs/providers/openai)**
    - Add `supports_minimal_reasoning_effort` to entire `gpt-5.x` model series (gpt-5.1 through gpt-5.4, including codex, pro, nano, and mini variants) and `azure/gpt-5.1-2025-11-13`
    - Add `supports_minimal_reasoning_effort` to `xai/grok-beta`

- **[Azure AI](../../docs/providers/azure_ai)**
    - Add Cohere Rerank 4.0 models (`azure_ai/cohere-rerank-v4`, `azure_ai/cohere-rerank-v4-multilingual`) to model cost map
    - Add DeepSeek V3.2 models (`azure_ai/DeepSeek-V3-2`, `azure_ai/DeepSeek-V3-2-speciale`) to model cost map

- **[Google Vertex AI](../../docs/providers/vertex)**
    - Correct `supported_regions` for Vertex AI DeepSeek models - [PR #23864](https://github.com/BerriAI/litellm/pull/23864)

#### Features

- **[OpenAI](../../docs/providers/openai)**
    - Day 0 support for `gpt-5.4-mini` and `gpt-5.4-nano` on OpenAI and Azure - [PR #23958](https://github.com/BerriAI/litellm/pull/23958)
    - Auto-route `gpt-5.4+` calls using both tools and reasoning to the Responses API on Azure - [PR #23926](https://github.com/BerriAI/litellm/pull/23926)

- **[Anthropic](../../docs/providers/anthropic)**
    - Opt-out flag for default reasoning summary injection (`anthropic_reasoning_summary: false`) - [PR #22904](https://github.com/BerriAI/litellm/pull/22904)
    - Support `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_BASE_URL` environment variables as alternative to `ANTHROPIC_API_KEY` - [PR #24140](https://github.com/BerriAI/litellm/pull/24140)

- **[Google Gemini](../../docs/providers/gemini)**
    - Context circulation support for server-side tool combination (Gemini native feature) - [PR #24073](https://github.com/BerriAI/litellm/pull/24073)

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Support `cache_control_injection_points` for `tool_config` location in Bedrock requests - [PR #24076](https://github.com/BerriAI/litellm/pull/24076)
    - Support batch cancel via Vertex AI / Bedrock batch API - [PR #23957](https://github.com/BerriAI/litellm/pull/23957)

#### Bugs

- **[Anthropic](../../docs/providers/anthropic)**
    - Align `translate_thinking_for_model` with default reasoning summary injection — fixes cases where summary was injected inconsistently - [PR #22909](https://github.com/BerriAI/litellm/pull/22909)
    - Preserve cache directive on file-type content blocks — cache headers were dropped on file messages - [PR #23906](https://github.com/BerriAI/litellm/pull/23906)
    - Fix `cache_control` directive dropped on document/file message blocks - [PR #23911](https://github.com/BerriAI/litellm/pull/23911)
    - Filter beta header after transformation (not before) to prevent invalid header injection - [PR #23715](https://github.com/BerriAI/litellm/pull/23715)
    - Add `additionalProperties: false` for OpenAI strict mode in Anthropic adapter - [PR #24072](https://github.com/BerriAI/litellm/pull/24072)
    - Fix thinking blocks dropped when `thinking` field is null - [PR #24070](https://github.com/BerriAI/litellm/pull/24070)

- **[Google Vertex AI](../../docs/providers/vertex)**
    - Fix streaming `finish_reason='stop'` instead of `'tool_calls'` for `gemini-3.1-flash-lite-preview` - [PR #23895](https://github.com/BerriAI/litellm/pull/23895)
    - Respect `vertex_count_tokens_location` for Claude `count_tokens` calls on Vertex - [PR #23907](https://github.com/BerriAI/litellm/pull/23907)
    - Pass model to context caching URL builder for custom `api_base` - [PR #23928](https://github.com/BerriAI/litellm/pull/23928)
    - Fix Vertex AI Batch output file download failing with 500 - [PR #23718](https://github.com/BerriAI/litellm/pull/23718)

- **[Azure AI](../../docs/providers/azure_ai)**
    - Preserve annotations in Bing Search grounding responses from Azure AI Agents - [PR #23939](https://github.com/BerriAI/litellm/pull/23939)
    - Auto-route Azure `gpt-5.4+` tools+reasoning calls to Responses API - [PR #23926](https://github.com/BerriAI/litellm/pull/23926)

- **[Mistral](../../docs/providers/mistral)**
    - Preserve diarization segments in transcription response — `segments` field was being dropped - [PR #23925](https://github.com/BerriAI/litellm/pull/23925)

- **[Fireworks AI](../../docs/providers/fireworks_ai)**
    - Skip `#transform=inline` for base64 data URLs — avoids double-encoding of inline image data - [PR #23818](https://github.com/BerriAI/litellm/pull/23818)

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Respect `api_base` and `aws_bedrock_runtime_endpoint` in the `count_tokens` endpoint - [PR #24199](https://github.com/BerriAI/litellm/pull/24199)

- **Responses API**
    - Map Chat Completion file type to Responses API `input_file` correctly - [PR #23618](https://github.com/BerriAI/litellm/pull/23618)
    - Surface Anthropic code execution results as `code_interpreter_call` in Responses API output - [PR #23784](https://github.com/BerriAI/litellm/pull/23784)
    - Capture incomplete terminal errors in background streaming — previously only `response.completed` triggered flush - [PR #23881](https://github.com/BerriAI/litellm/pull/23881)
    - Align emulated file search Responses behavior with native output format - [PR #23969](https://github.com/BerriAI/litellm/pull/23969)

- **General**
    - Map Anthropic `refusal` finish reason to `content_filter` for OpenAI compatibility - [PR #23899](https://github.com/BerriAI/litellm/pull/23899)
    - Preserve custom attributes on final stream chunk — were being dropped on the last SSE event - [PR #23530](https://github.com/BerriAI/litellm/pull/23530)
    - Fix ensure alternating roles in message arrays - [PR #24015](https://github.com/BerriAI/litellm/pull/24015)
    - Short-circuit web search interception for `github_copilot` provider - [PR #24143](https://github.com/BerriAI/litellm/pull/24143)
    - Fix proxy-only failure call type not being set correctly - [PR #24050](https://github.com/BerriAI/litellm/pull/24050)

---

## LLM API Endpoints

#### Features

- **[Video Generation API](../../docs/video_generation)**
    - Add create character endpoints and new video generation endpoints - [PR #23737](https://github.com/BerriAI/litellm/pull/23737)

- **[Responses API](../../docs/response_api)**
    - Prompt management support for Responses API — use prompt templates and versioning with `/v1/responses` - [PR #23999](https://github.com/BerriAI/litellm/pull/23999)

- **[Azure](../../docs/providers/azure)**
    - Use `AZURE_DEFAULT_API_VERSION` env var as default for proxy `--api_version` flag - [PR #24120](https://github.com/BerriAI/litellm/pull/24120)

#### Bugs

- **General**
    - Fix logging for incomplete streaming responses and custom pricing on `/v1/messages` and `/v1/responses` - [PR #24080](https://github.com/BerriAI/litellm/pull/24080)

---

## Management Endpoints / UI

#### Features

- **Multi-Proxy Control Plane**
    - New control plane for managing multiple proxy worker processes — centralized routing, config sync, and health tracking across workers - [PR #24217](https://github.com/BerriAI/litellm/pull/24217)

- **Audit Logs**
    - Export audit logs to external callback systems (S3, custom callbacks) - [PR #23167](https://github.com/BerriAI/litellm/pull/23167)

- **Teams**
    - `/v2/team/list` — new endpoint with org admin access control, `members_count`, and DB indexes for performance - [PR #23938](https://github.com/BerriAI/litellm/pull/23938)
    - Modernize Teams Table in UI — antd-based redesign with table refresh, infinite scroll dropdown, and leftnav migration - [PR #24189](https://github.com/BerriAI/litellm/pull/24189), [PR #24342](https://github.com/BerriAI/litellm/pull/24342)

- **Virtual Keys**
    - Disable custom virtual key values via UI setting — prevent users from specifying their own key strings - [PR #23812](https://github.com/BerriAI/litellm/pull/23812)

- **Setup Wizard**
    - Interactive `litellm --setup` wizard for configuring providers, API keys, and proxy settings from the CLI - [PR #23644](https://github.com/BerriAI/litellm/pull/23644)

#### Bugs

- Fix empty filter results showing stale data in UI Logs view - [PR #23792](https://github.com/BerriAI/litellm/pull/23792)
- Fix internal users being able to create invalid keys - [PR #23795](https://github.com/BerriAI/litellm/pull/23795)
- Fix key alias re-validation on update blocking legacy aliases - [PR #23798](https://github.com/BerriAI/litellm/pull/23798)
- Fix per-entity breakdown missing from aggregated daily activity endpoint - [PR #23471](https://github.com/BerriAI/litellm/pull/23471)
- Fix `team_member_budget_duration` missing from `NewTeamRequest` - [PR #23484](https://github.com/BerriAI/litellm/pull/23484)
- Fix CSV export empty on Global Usage page - [PR #23819](https://github.com/BerriAI/litellm/pull/23819)
- Fix DefaultInternalUserParams Pydantic default not matching runtime fallback - [PR #23666](https://github.com/BerriAI/litellm/pull/23666)
- Fix key update endpoint returning 401 instead of 404 for nonexistent keys - [PR #24063](https://github.com/BerriAI/litellm/pull/24063)
- Fix `/key/block` and `/key/unblock` returning 404 (not 401) for non-existent keys - [PR #23977](https://github.com/BerriAI/litellm/pull/23977)
- Fix pass-through subpath auth for non-admin users - [PR #24079](https://github.com/BerriAI/litellm/pull/24079)
- Fix duplicate callback logs for pass-through endpoint failures - [PR #23509](https://github.com/BerriAI/litellm/pull/23509)
- Fix Default Team Settings missing permission options in UI - [PR #24039](https://github.com/BerriAI/litellm/pull/24039)
- Fix guardrail mode type crash on non-string values in Logs UI - [PR #24035](https://github.com/BerriAI/litellm/pull/24035)
- Fix create key tags dropdown - [PR #24273](https://github.com/BerriAI/litellm/pull/24273)

---

## AI Integrations

### Logging

- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Fix OpenTelemetry traceparent propagation — `traceparent` header was not being forwarded correctly to Langfuse spans - [PR #24048](https://github.com/BerriAI/litellm/pull/24048)

- **[LangSmith](../../docs/proxy/logging#langsmith)**
    - Populate `usage_metadata` in outputs for Cost column tracking - [PR #24043](https://github.com/BerriAI/litellm/pull/24043)

- **[Audit Log Export](../../docs/proxy/logging)**
    - Export audit logs to external callback systems (S3, custom destinations) - [PR #23167](https://github.com/BerriAI/litellm/pull/23167)

- **General**
    - Preserve `router_model_group` in generic API log entries - [PR #24044](https://github.com/BerriAI/litellm/pull/24044)
    - Merge `hidden_params` into metadata for streaming requests — previously only non-streaming requests had full metadata - [PR #24220](https://github.com/BerriAI/litellm/pull/24220)

### Guardrails

- **[Akto](../../docs/proxy/guardrails/akto)** — New Akto guardrail integration for API security testing and threat detection - [PR #23250](https://github.com/BerriAI/litellm/pull/23250)
- **MCP JWT Signer** — Built-in guardrail for zero-trust MCP authentication — automatically signs outbound MCP requests with JWT tokens - [PR #23897](https://github.com/BerriAI/litellm/pull/23897)
- **`pre_mcp_call` header mutation** — `pre_mcp_call` guardrail hooks can now mutate outbound MCP request headers - [PR #23889](https://github.com/BerriAI/litellm/pull/23889)
- **Fix model-level guardrails not executing for non-streaming post_call** — guardrails configured at the model level were silently skipped on synchronous (non-streaming) responses - [PR #23774](https://github.com/BerriAI/litellm/pull/23774)
- **Defer logging until post-call guardrails complete** — logging callbacks were firing before guardrail post_call hooks finished, causing incomplete log entries - [PR #24135](https://github.com/BerriAI/litellm/pull/24135)

### Prompt Management

- **[Responses API](../../docs/response_api)**
    - Prompt management (templates, versioning) now supported for `/v1/responses` - [PR #23999](https://github.com/BerriAI/litellm/pull/23999)

### Secret Managers

No major secret manager changes in this release.

---

## MCP Gateway

#### Bugs

- Fix `oauth2_flow` not being set when building `MCPServer` in `_execute_with_mcp_client` — caused MCP server auth failures for OAuth2-protected servers - [PR #23468](https://github.com/BerriAI/litellm/pull/23468)
- Upgrade `mcp` SDK to 1.26.0 - [PR #24179](https://github.com/BerriAI/litellm/pull/24179)

---

## Spend Tracking, Budgets and Rate Limiting

- **Proxy-wide default API key TPM/RPM limits** — Set global default rate limits applied to all API keys that don't have explicit limits configured - [PR #24088](https://github.com/BerriAI/litellm/pull/24088)
- **Fix rate limit check before creating polling ID** — polling IDs were being created before the rate limit check, consuming slots even for rejected requests - [PR #24106](https://github.com/BerriAI/litellm/pull/24106)

---

## Performance / Loadbalancing / Reliability improvements

- **Per-model-group deployment affinity** — Router can now pin requests to specific deployments within a model group, reducing cold-start latency and improving cache hit rates for stateful workloads - [PR #24110](https://github.com/BerriAI/litellm/pull/24110)
- **Auto-recover shared aiohttp session when closed** — proxy was crashing with `RuntimeError: Session is closed` after idle periods; session now auto-recovers - [PR #23808](https://github.com/BerriAI/litellm/pull/23808)
- **Kill orphaned Prisma engine subprocess on failed disconnect** — zombie Prisma engine processes were accumulating on DB reconnect failures, exhausting file descriptors - [PR #24149](https://github.com/BerriAI/litellm/pull/24149)
- **Add `IF NOT EXISTS` to index creation in migration** — migration was failing on re-runs if indexes already existed - [PR #24105](https://github.com/BerriAI/litellm/pull/24105)

---

## Security

- **Fix privilege escalation on key management endpoints** — non-admin users could call `/key/block`, `/key/unblock`, and `/key/update` with `max_budget` to modify keys they don't own. Now enforces ownership checks - [PR #23781](https://github.com/BerriAI/litellm/pull/23781)
- **Fix global secret redaction** — secrets were not being redacted from all log paths; now uses root logger filter + key-name-based pattern matching to ensure full coverage - [PR #24305](https://github.com/BerriAI/litellm/pull/24305)

---

## Documentation Updates

- No major documentation-only changes in this release.

---

## New Contributors

* @Chesars made their first contribution in [PR #21441](https://github.com/BerriAI/litellm/pull/21441)
* @michelligabriele made their first contribution in [PR #23471](https://github.com/BerriAI/litellm/pull/23471)
* @voidborne-d made their first contribution in [PR #23808](https://github.com/BerriAI/litellm/pull/23808)
* @andrzej-pomirski-yohana made their first contribution in [PR #23784](https://github.com/BerriAI/litellm/pull/23784)
* @kelvin-tran made their first contribution in [PR #23911](https://github.com/BerriAI/litellm/pull/23911)
* @themavik made their first contribution in [PR #24043](https://github.com/BerriAI/litellm/pull/24043)
* @emerzon made their first contribution in [PR #24044](https://github.com/BerriAI/litellm/pull/24044)
* @jyeros made their first contribution in [PR #24048](https://github.com/BerriAI/litellm/pull/24048)
* @alilxxey made their first contribution in [PR #24050](https://github.com/BerriAI/litellm/pull/24050)
* @xr843 made their first contribution in [PR #24070](https://github.com/BerriAI/litellm/pull/24070)
* @ephrimstanley (Point72) made their first contribution in [PR #24088](https://github.com/BerriAI/litellm/pull/24088)
* @superpoussin22 made their first contribution in [PR #24105](https://github.com/BerriAI/litellm/pull/24105)
* @devin-petersohn made their first contribution in [PR #24140](https://github.com/BerriAI/litellm/pull/24140)
* @johnib made their first contribution in [PR #24143](https://github.com/BerriAI/litellm/pull/24143)
* @stias made their first contribution in [PR #24199](https://github.com/BerriAI/litellm/pull/24199)
* @milan-berri made their first contribution in [PR #24220](https://github.com/BerriAI/litellm/pull/24220)

---

## Diff Summary

## 03/23/2026
* New Models / Updated Models: 12 new
* LLM API Endpoints: 6
* Management Endpoints / UI: 17
* Logging / Guardrail / Prompt Management Integrations: 9
* MCP Gateway: 2
* Spend Tracking, Budgets and Rate Limiting: 2
* Performance / Loadbalancing / Reliability improvements: 4
* Security: 2

---

## Full Changelog
[v1.82.3-stable...v1.82.6.rc.1](https://github.com/BerriAI/litellm/compare/v1.82.3-stable...v1.82.6.rc.1)
