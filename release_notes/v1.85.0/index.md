---
title: "v1.85.0 - Realtime GA, MCP Gateway Expansion & Hardened Multi-Tenancy"
slug: "v1-85-0"
date: 2026-05-16T00:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
  - name: Yuneng Jiang
    title: Senior Full Stack Engineer, LiteLLM
    url: https://www.linkedin.com/in/yuneng-david-jiang-455676139/
    image_url: https://avatars.githubusercontent.com/u/171294688?v=4
hide_table_of_contents: false
---

## Deploy this version

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="docker" label="Docker">

```bash
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
docker.litellm.ai/berriai/litellm:1.85.0
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.85.0
```

</TabItem>
</Tabs>

## Key Highlights

- **OpenAI Realtime GA** — first-class support for the GA OpenAI Realtime API (plus beta compatibility), including `gpt-realtime-2` pricing and `/openai/v1/realtime` logging.
- **Hardened multi-tenancy** — a large sweep of per-tenant scoping fixes across keys, projects, batches, files, MCP servers, and analytics endpoints (project-hijack/key-org isolation, service-account resource isolation, per-entity team/agent activity scoping).
- **MCP Gateway expansion** — org-level MCP server/toolset permissions, OBO (on-behalf-of) MCP auth, `delegate_auth_to_upstream` PKCE passthrough, and MCP access-group name namespacing.
- **Observability overhaul** — broad Prometheus fixes (label-count correctness, end-user cardinality cap, PromQL escaping), OTEL handler isolation + GenAI message-content capture, and decoupled S3 audit-log config.
- **New models** — xAI `grok-4.3` / `grok-4.3-latest`, OpenAI `gpt-realtime-2`, OpenRouter `qwen/qwen3.6-plus`, SambaNova `MiniMax-M2.7`, and Bedrock Z.AI `GLM-5`.

---

## New Models / Updated Models

#### New Model Support (5 new models)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| --- | --- | --- | --- | --- | --- |
| OpenAI | `gpt-realtime-2` | 32K | $4.00 (audio in $32.00) | $16.00 (audio out $64.00) | Realtime (`/v1/realtime`), audio in/out, function calling, parallel tool calls |
| xAI | `xai/grok-4.3` | 1M | $1.25 (>200K: $2.50) | $2.50 (>200K: $5.00) | Reasoning, vision, prompt caching, response schema, web search, tool calling |
| xAI | `xai/grok-4.3-latest` | 1M | $1.25 (>200K: $2.50) | $2.50 (>200K: $5.00) | Reasoning, vision, prompt caching, response schema, web search, tool calling |
| OpenRouter | `openrouter/qwen/qwen3.6-plus` | 1M | $0.325 | $1.95 | Reasoning, vision, function calling, tool choice |
| SambaNova | `sambanova/MiniMax-M2.7` | 204.8K | $0.30 | $1.20 | Reasoning, function calling, tool choice |

Pricing/metadata also updated for existing entries: Gemini multimodal-embedding pricing repointed to the Vertex pricing source with image/audio/video per-unit costs, audio-token cost reductions on realtime/Gemini entries, and a `gemini-embedding-2-preview` cost alignment.

- xAI grok-4.3 / grok-4.3-latest metadata - [PR #27154](https://github.com/BerriAI/litellm/pull/27154), [PR #27396](https://github.com/BerriAI/litellm/pull/27396)
- OpenAI gpt-realtime-2 pricing - [PR #27653](https://github.com/BerriAI/litellm/pull/27653)
- OpenRouter Qwen 3.6 Plus metadata - [PR #27486](https://github.com/BerriAI/litellm/pull/27486)
- New chat model metadata + Bedrock Z.AI GLM-5 - [PR #27313](https://github.com/BerriAI/litellm/pull/27313), [PR #24338](https://github.com/BerriAI/litellm/pull/24338)
- GPT-4o-Transcribe pricing fix - [PR #27875](https://github.com/BerriAI/litellm/pull/27875)

#### Features

- **[Anthropic](../../docs/providers/anthropic)**
    - Forward `output_config.effort`, reject garbage `reasoning_effort` with 400, and omit thinking/output_config when `reasoning_effort="none"` - [PR #27074](https://github.com/BerriAI/litellm/pull/27074), [PR #27039](https://github.com/BerriAI/litellm/pull/27039)
    - Add Bedrock Claude Platform route - [PR #27678](https://github.com/BerriAI/litellm/pull/27678)
    - Inject dummy tool without `modify_params` - [PR #27620](https://github.com/BerriAI/litellm/pull/27620)
- **[Bedrock](../../docs/providers/bedrock)**
    - Add Z.AI GLM-5 model support - [PR #24338](https://github.com/BerriAI/litellm/pull/24338)
    - Handle document content blocks in Converse API message conversion - [PR #24644](https://github.com/BerriAI/litellm/pull/24644)
    - Refactor response stream shape handling - [PR #27257](https://github.com/BerriAI/litellm/pull/27257)
- **[Vertex AI](../../docs/providers/vertex)**
    - Model Garden OpenAPI support for publisher model IDs - [PR #26076](https://github.com/BerriAI/litellm/pull/26076)
    - Omit `system_instruction`/`tools`/`toolConfig` when `cachedContent` set - [PR #26077](https://github.com/BerriAI/litellm/pull/26077)
- **[Gemini](../../docs/providers/gemini)**
    - Follow provider defaults for Gemini 3 thinking - [PR #25764](https://github.com/BerriAI/litellm/pull/25764)
    - Handle Gemini Files API URIs without fetching - [PR #24922](https://github.com/BerriAI/litellm/pull/24922)
    - Normalize `response_schema` on native `generateContent` - [PR #27775](https://github.com/BerriAI/litellm/pull/27775)
- **[xAI](../../docs/providers/xai)**
    - Add `parallel_tool_calls` to supported params - [PR #25106](https://github.com/BerriAI/litellm/pull/25106)
- **[Azure](../../docs/providers/azure)**
    - Authenticate to Azure with a token - [PR #27556](https://github.com/BerriAI/litellm/pull/27556)
    - Azure Sentinel audit-log support - [PR #27280](https://github.com/BerriAI/litellm/pull/27280)
- **General**
    - `gpt-5.5` reasoning-effort capability flags + `supports_low_reasoning_effort` - [PR #26456](https://github.com/BerriAI/litellm/pull/26456)
    - Match `litellm.completion` supported params with proxy model info - [PR #27720](https://github.com/BerriAI/litellm/pull/27720)

#### Bug Fixes

- **[OpenRouter](../../docs/providers/openrouter)**
    - Strip `openrouter/` prefix from model names - [PR #24282](https://github.com/BerriAI/litellm/pull/24282)
- **[Azure](../../docs/providers/azure)**
    - Forward `api_version` to `aembedding()` for Azure AI Foundry v1 endpoints - [PR #24911](https://github.com/BerriAI/litellm/pull/24911)
    - Route Azure container file requests by decoded deployment - [PR #26402](https://github.com/BerriAI/litellm/pull/26402)
- **[Anthropic](../../docs/providers/anthropic)** / **[Vertex](../../docs/providers/vertex)**
    - Fix Vertex Anthropic streaming status-error hangs - [PR #27310](https://github.com/BerriAI/litellm/pull/27310)
    - Fix Anthropic streaming reasoning token usage - [PR #27319](https://github.com/BerriAI/litellm/pull/27319)
- **[Fireworks AI](../../docs/providers/fireworks_ai)**
    - Strip `thinking_blocks` from chat messages before the Fireworks API call - [PR #27881](https://github.com/BerriAI/litellm/pull/27881)
- **[hosted vLLM](../../docs/providers/vllm)**
    - Normalize custom tools for chat completions - [PR #25763](https://github.com/BerriAI/litellm/pull/25763)
- **General**
    - Decode unified `file_id` when `model_file_id_mapping` is unavailable - [PR #27406](https://github.com/BerriAI/litellm/pull/27406)
    - Pass `output_config` through to backends that accept it - [PR #26439](https://github.com/BerriAI/litellm/pull/26439)
    - Resolve provider from deployment for multi-provider default config - [PR #27517](https://github.com/BerriAI/litellm/pull/27517)
    - Return `503` from `/health` when the targeted model is unhealthy or DB is disconnected - [PR #27003](https://github.com/BerriAI/litellm/pull/27003)
    - Guard URL-valued model destinations and align resource-model auth checks - [PR #26915](https://github.com/BerriAI/litellm/pull/26915), [PR #26963](https://github.com/BerriAI/litellm/pull/26963)

## LLM API Endpoints

#### Features

- **[Realtime API](../../docs/realtime)**
    - OpenAI Realtime GA support and beta compatibility - [PR #27110](https://github.com/BerriAI/litellm/pull/27110)
    - Add `/openai/v1/realtime` to routes for logging - [PR #27323](https://github.com/BerriAI/litellm/pull/27323)
- **[Responses API](../../docs/response_api)**
    - Persist and replay streamed Responses API requests from cache - [PR #24580](https://github.com/BerriAI/litellm/pull/24580)
    - Route `gpt-5.4+` chat-without-tools to the Responses API - [PR #27618](https://github.com/BerriAI/litellm/pull/27618)
    - Preserve `cache_control` in Responses → Chat Completion transformation - [PR #27727](https://github.com/BerriAI/litellm/pull/27727)
    - Normalize chat `tool_choice` for the completions→responses bridge - [PR #27634](https://github.com/BerriAI/litellm/pull/27634)
- **[Batches](../../docs/batches)**
    - Bedrock batch model-invocation job retrieval - [PR #26834](https://github.com/BerriAI/litellm/pull/26834)
    - Transform Vertex AI batch prediction outputs to OpenAI format - [PR #25627](https://github.com/BerriAI/litellm/pull/25627)
    - Set `response=null` on batch error entries per OpenAI spec - [PR #27041](https://github.com/BerriAI/litellm/pull/27041)
- **[Embeddings](../../docs/embedding/supported_embedding)**
    - Default OpenAI-path `encoding_format` to `float` - [PR #26976](https://github.com/BerriAI/litellm/pull/26976)
    - Separate embeddings for multimodal inputs + combined multimodal embeddings via nested input - [PR #24337](https://github.com/BerriAI/litellm/pull/24337), [PR #24341](https://github.com/BerriAI/litellm/pull/24341)
- **[Audio Transcription](../../docs/audio_transcription)**
    - Add NVIDIA Riva STT provider - [PR #27185](https://github.com/BerriAI/litellm/pull/27185)
- **[Vector Stores](../../docs/vector_stores)**
    - Resolve embedding config at request time, never persist credentials - [PR #27082](https://github.com/BerriAI/litellm/pull/27082)
    - Tighten managed-store access - [PR #26930](https://github.com/BerriAI/litellm/pull/26930)

#### Bugs

- **General**
    - Preserve `compact_20260112` context management on Bedrock `/v1/messages` - [PR #27534](https://github.com/BerriAI/litellm/pull/27534)
    - Fix managed file `model_mappings` when the router resolves a single deployment dict - [PR #26950](https://github.com/BerriAI/litellm/pull/26950)
    - Omit `model` from Azure deployment image-gen / image-edit bodies - [PR #27103](https://github.com/BerriAI/litellm/pull/27103)
    - Fix Bedrock passthrough call-ID headers - [PR #27412](https://github.com/BerriAI/litellm/pull/27412)
    - Pin Responses API affinity to the Azure resource on model-group switch - [PR #27703](https://github.com/BerriAI/litellm/pull/27703)
    - Align `vertex_ai/gemini-embedding-2-preview` cost with Vertex multimodal pricing - [PR #27848](https://github.com/BerriAI/litellm/pull/27848)
    - Consolidate batch + dynamic limiter check/increment - [PR #26954](https://github.com/BerriAI/litellm/pull/26954)

## Management Endpoints / UI

#### Features

- **Virtual Keys**
    - Bulk key updates for a team - [PR #26468](https://github.com/BerriAI/litellm/pull/26468)
    - Rename "Default" key type to "Full Access" and reorder dropdown - [PR #27218](https://github.com/BerriAI/litellm/pull/27218)
    - Add `Expires` to key Overview header; merge User into one field - [PR #27696](https://github.com/BerriAI/litellm/pull/27696)
- **Teams & Models**
    - Search teams by team ID alongside name - [PR #27684](https://github.com/BerriAI/litellm/pull/27684)
    - Add a "Your Usage" view for admin users on the usage page - [PR #26746](https://github.com/BerriAI/litellm/pull/26746)
    - Add Vertex AI Search as a vector-store provider in the UI - [PR #27790](https://github.com/BerriAI/litellm/pull/27790)
    - "Last Minute" quick-select on the Logs time range - [PR #27446](https://github.com/BerriAI/litellm/pull/27446)
    - Add missing Z.AI (`zai`) provider to the Add-Model dropdown - [PR #26419](https://github.com/BerriAI/litellm/pull/26419)
- **SSO / Auth**
    - JWT scope + wildcard support for routing overrides, issuer verification with unscoped warning - [PR #26325](https://github.com/BerriAI/litellm/pull/26325), [PR #27008](https://github.com/BerriAI/litellm/pull/27008)
    - Grafana Cloud Pyroscope authentication - [PR #26902](https://github.com/BerriAI/litellm/pull/26902)
    - Show full IdP claims in `/sso/debug/callback` - [PR #27498](https://github.com/BerriAI/litellm/pull/27498)

#### Bugs — access scoping & correctness

- **Multi-tenancy isolation**
    - Scope project, key-org, team, and agent-activity lookups per entity; reject `user_id=None` on non-admin analytics; re-validate `user_id` after `/user/info` re-parses query - [PR #27011](https://github.com/BerriAI/litellm/pull/27011), [PR #27014](https://github.com/BerriAI/litellm/pull/27014), [PR #26929](https://github.com/BerriAI/litellm/pull/26929), [PR #27009](https://github.com/BerriAI/litellm/pull/27009)
    - Constrain cloud-storage file paths and batch-file model access - [PR #27019](https://github.com/BerriAI/litellm/pull/27019), [PR #27015](https://github.com/BerriAI/litellm/pull/27015)
    - Isolate managed resources for service-account API keys - [PR #27004](https://github.com/BerriAI/litellm/pull/27004)
    - Tighten resource-ownership checks and sensitive public-endpoint guards - [PR #26951](https://github.com/BerriAI/litellm/pull/26951), [PR #26912](https://github.com/BerriAI/litellm/pull/26912)
- **Authorization hardening**
    - Block missing write routes for proxy admin viewers; restore admin-viewer read parity on Logs + Settings - [PR #27007](https://github.com/BerriAI/litellm/pull/27007), [PR #26846](https://github.com/BerriAI/litellm/pull/26846)
    - Encode upstream URL path identifiers; require a trusted proxy for header-identity auth - [PR #26860](https://github.com/BerriAI/litellm/pull/26860), [PR #26825](https://github.com/BerriAI/litellm/pull/26825)
    - Bind generic SSO state to a session cookie; allow non-admin compliance-path reads - [PR #26944](https://github.com/BerriAI/litellm/pull/26944), [PR #27234](https://github.com/BerriAI/litellm/pull/27234)
- **Keys / Teams / SCIM**
    - Honor `key access_group_ids` when a team restricts models; resolve access-group names in team filtering and same-name deployment routing - [PR #26275](https://github.com/BerriAI/litellm/pull/26275), [PR #25224](https://github.com/BerriAI/litellm/pull/25224), [PR #26161](https://github.com/BerriAI/litellm/pull/26161)
    - Revoke virtual keys when SCIM deprovisions a user; fix SCIM user-lookup filters - [PR #26861](https://github.com/BerriAI/litellm/pull/26861), [PR #27308](https://github.com/BerriAI/litellm/pull/27308)
    - Key-rotation bug fix; honor `team_member_permissions` on `/key/list` - [PR #27756](https://github.com/BerriAI/litellm/pull/27756), [PR #27026](https://github.com/BerriAI/litellm/pull/27026)
    - `/config/update` targeted per-section writes (drop `store_model_in_db` gate) - [PR #26643](https://github.com/BerriAI/litellm/pull/26643)
    - Scope CLI stored token to `base_url`; redact Gemini API key from URL query params in error traces - [PR #26945](https://github.com/BerriAI/litellm/pull/26945), [PR #24943](https://github.com/BerriAI/litellm/pull/24943)
- **UI fixes**
    - Remove the insecure `?token=` URL handler from the login page; clear admin session cookies before establishing an invited user's session; URL-encode `team_id` in `teamInfoCall` - [PR #26924](https://github.com/BerriAI/litellm/pull/26924), [PR #27227](https://github.com/BerriAI/litellm/pull/27227), [PR #27466](https://github.com/BerriAI/litellm/pull/27466)
    - Project dropdown empty for internal users (3 bugs); remove blank leading entry from access-group model dropdown; omit `allowed_routes` from key edit save when unchanged - [PR #26664](https://github.com/BerriAI/litellm/pull/26664), [PR #27521](https://github.com/BerriAI/litellm/pull/27521), [PR #27553](https://github.com/BerriAI/litellm/pull/27553)
    - Member/team access-group fix; team model test-connection authorization - [PR #27317](https://github.com/BerriAI/litellm/pull/27317), [PR #27487](https://github.com/BerriAI/litellm/pull/27487)

## AI Integrations

### Logging

- **[Prometheus](../../docs/proxy/prometheus)**
    - Fix custom-metadata label counts, cap end-user metric cardinality, fix remaining-metric zero values, escape `api_key` for PromQL string literals, emit `litellm_remaining_tokens_metric` for Bedrock & Vertex - [PR #27268](https://github.com/BerriAI/litellm/pull/27268), [PR #27272](https://github.com/BerriAI/litellm/pull/27272), [PR #27348](https://github.com/BerriAI/litellm/pull/27348), [PR #27013](https://github.com/BerriAI/litellm/pull/27013), [PR #27705](https://github.com/BerriAI/litellm/pull/27705)
    - Fix `/metrics` hang when `require_auth_for_metrics_endpoint` is true and auth succeeds; point `/metrics` 401 at the opt-out flag; fix metric labels for litellm-side rejects - [PR #25980](https://github.com/BerriAI/litellm/pull/25980), [PR #27502](https://github.com/BerriAI/litellm/pull/27502), [PR #26947](https://github.com/BerriAI/litellm/pull/26947)
- **[OpenTelemetry](../../docs/observability/opentelemetry_integration)**
    - Isolate dual OTEL handlers; honor `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`; fix proxy-integration tracing bugs - [PR #27018](https://github.com/BerriAI/litellm/pull/27018), [PR #27403](https://github.com/BerriAI/litellm/pull/27403), [PR #27757](https://github.com/BerriAI/litellm/pull/27757)
- **[Arize](../../docs/observability/arize_integration)** / **LangSmith**
    - Arize `_set_usage_outputs` handles raw OpenAI Pydantic `CompletionUsage`; remove unwanted metadata info from LangSmith - [PR #26506](https://github.com/BerriAI/litellm/pull/26506), [PR #26894](https://github.com/BerriAI/litellm/pull/26894)
- **General**
    - Decouple S3 audit-log config via `s3_audit_callback_params` - [PR #27222](https://github.com/BerriAI/litellm/pull/27222)
    - Set `verbose_logger` level when `LITELLM_LOG=INFO`; require a team-management role on `/team/{id}/callback`; close callback-config and observability-credential side channels; guard dynamic integration hosts - [PR #26401](https://github.com/BerriAI/litellm/pull/26401), [PR #26819](https://github.com/BerriAI/litellm/pull/26819), [PR #27081](https://github.com/BerriAI/litellm/pull/27081), [PR #26921](https://github.com/BerriAI/litellm/pull/26921)

### Guardrails

- **General**
    - Add Qohash Nexus guardrail hook - [PR #24927](https://github.com/BerriAI/litellm/pull/24927)
    - Run model-level `post_call` guardrails on streaming requests; ensure post-call guardrail fires exactly once - [PR #26922](https://github.com/BerriAI/litellm/pull/26922), [PR #27012](https://github.com/BerriAI/litellm/pull/27012), [PR #26109](https://github.com/BerriAI/litellm/pull/26109)
    - Preserve Responses event streams in Presidio output masking - [PR #26878](https://github.com/BerriAI/litellm/pull/26878)
    - Cover multimodal + Responses-API content shapes; tighten tool-permission checks; optional skip of tool message in unified guardrail inputs - [PR #26957](https://github.com/BerriAI/litellm/pull/26957), [PR #26969](https://github.com/BerriAI/litellm/pull/26969), [PR #27441](https://github.com/BerriAI/litellm/pull/27441)
    - Handle legacy dict shape for `metadata.guardrails` in the Team UI - [PR #27224](https://github.com/BerriAI/litellm/pull/27224)

### Prompt Management

- **General**
    - Block path-traversal in BitBucket / Arize Phoenix / AssemblyAI clients; sandbox jinja2 in the GitLab/Arize/BitBucket prompt managers - [PR #26943](https://github.com/BerriAI/litellm/pull/26943), [PR #27043](https://github.com/BerriAI/litellm/pull/27043)

### Secret Managers

- **General**
    - Audit-log `/cache/settings` and `/config_overrides/hashicorp_vault` mutations - [PR #26953](https://github.com/BerriAI/litellm/pull/26953)

## Spend Tracking, Budgets and Rate Limiting

- **Rate Limiting**
    - Atomic TPM rate limit; include model name + configured TPM/RPM in priority rate-limit 429 errors - [PR #27001](https://github.com/BerriAI/litellm/pull/27001), [PR #27216](https://github.com/BerriAI/litellm/pull/27216)
    - Load team-member RPM/TPM from membership budget in the combined view - [PR #24925](https://github.com/BerriAI/litellm/pull/24925)
- **Budgets**
    - Skip the personal-budget hook when a reservation covers the counter - [PR #27021](https://github.com/BerriAI/litellm/pull/27021)
    - Treat `0` `team_member_budget` as no cap; enforce team-member budget without a user row; reset org/tag/proxy budgets correctly - [PR #27133](https://github.com/BerriAI/litellm/pull/27133), [PR #27273](https://github.com/BerriAI/litellm/pull/27273), [PR #27326](https://github.com/BerriAI/litellm/pull/27326), [PR #27488](https://github.com/BerriAI/litellm/pull/27488)
    - Flush virtual-key `model_max` budget spend to Redis after success logging; tighten budget spend admission - [PR #27334](https://github.com/BerriAI/litellm/pull/27334), [PR #26845](https://github.com/BerriAI/litellm/pull/26845)
- **Tag Budgets & Routing**
    - Enforce tag budgets on `x-litellm-tags` header requests; tag-budget reset drops stale management-cache entries; union `x-litellm-tags` with static team/key tags; fix internal tag-usage scoping; always merge caller-supplied tags into request metadata - [PR #27573](https://github.com/BerriAI/litellm/pull/27573), [PR #27568](https://github.com/BerriAI/litellm/pull/27568), [PR #27247](https://github.com/BerriAI/litellm/pull/27247), [PR #27315](https://github.com/BerriAI/litellm/pull/27315), [PR #27784](https://github.com/BerriAI/litellm/pull/27784)
    - Tag-routing test preventing header-regex bypass for strict plain-text tags - [PR #26805](https://github.com/BerriAI/litellm/pull/26805)
- **Spend Logs / Cost**
    - Pass `service_tier` through Azure and Azure AI cost calculation - [PR #24926](https://github.com/BerriAI/litellm/pull/24926)
    - Opt-in suppression of stack traces in spend-tracking error logs; keep spend-log cleanup running after batch failures; redact echoed prompts in `error_information`; prevent `secret_fields` from leaking into spend logs; drop client-supplied pricing fields from request bodies - [PR #26899](https://github.com/BerriAI/litellm/pull/26899), [PR #27303](https://github.com/BerriAI/litellm/pull/27303), [PR #27689](https://github.com/BerriAI/litellm/pull/27689), [PR #27143](https://github.com/BerriAI/litellm/pull/27143), [PR #27071](https://github.com/BerriAI/litellm/pull/27071)

## MCP Gateway

- **Features**
    - Org-level MCP server and toolset permissions - [PR #26960](https://github.com/BerriAI/litellm/pull/26960)
    - OBO (on-behalf-of) MCP auth - [PR #27421](https://github.com/BerriAI/litellm/pull/27421)
    - `delegate_auth_to_upstream` flag for PKCE passthrough - [PR #27834](https://github.com/BerriAI/litellm/pull/27834)
    - Support MCP access-group names in URL-based namespacing - [PR #27726](https://github.com/BerriAI/litellm/pull/27726)
- **Bugs**
    - Sanitize tool names to Anthropic's `[a-zA-Z0-9_-]{1,128}` pattern - [PR #26788](https://github.com/BerriAI/litellm/pull/26788)
    - Require a trusted-proxy gate before honoring `X-Forwarded-*` on OAuth discovery; preserve oauth2 m2m auth for tools routes; run `pre_call_tool_check` on the OpenAPI/local-registry path - [PR #26841](https://github.com/BerriAI/litellm/pull/26841), [PR #26871](https://github.com/BerriAI/litellm/pull/26871), [PR #27016](https://github.com/BerriAI/litellm/pull/27016)
    - Redact MCP server URL/headers for non-admin viewers; replace user-API-key auth with authorization-or-cookie for MCP server creation - [PR #27027](https://github.com/BerriAI/litellm/pull/27027), [PR #27190](https://github.com/BerriAI/litellm/pull/27190)
    - Fix MCP DB reload partial failures; surface upstream 401 for token-forwarding MCP servers - [PR #27314](https://github.com/BerriAI/litellm/pull/27314), [PR #27847](https://github.com/BerriAI/litellm/pull/27847)

## Performance / Loadbalancing / Reliability improvements

- **Routing & Reliability**
    - Trigger fallbacks on mid-stream `httpx.TimeoutException` - [PR #26998](https://github.com/BerriAI/litellm/pull/26998)
    - Register cooldowns on failure + fail fast on stale `encrypted_content` (Responses) - [PR #27820](https://github.com/BerriAI/litellm/pull/27820)
    - Register model info under the responses/-stripped variant - [PR #27531](https://github.com/BerriAI/litellm/pull/27531)
    - Fix Redis Sentinel client handling for authenticated Sentinel setups - [PR #26302](https://github.com/BerriAI/litellm/pull/26302)
- **Proxy hot path**
    - Token-verification query optimization - [PR #26202](https://github.com/BerriAI/litellm/pull/26202)
    - Run daily activity aggregation off the event loop - [PR #27264](https://github.com/BerriAI/litellm/pull/27264)
    - Shared IAM cache + static credentials in `BaseAWSLLM` - [PR #27125](https://github.com/BerriAI/litellm/pull/27125)
    - Isolate semantic cache entries; stable Redis key generation across working directories; remove a duplicate in-memory cache-size constant - [PR #26990](https://github.com/BerriAI/litellm/pull/26990), [PR #27025](https://github.com/BerriAI/litellm/pull/27025), [PR #26385](https://github.com/BerriAI/litellm/pull/26385)
    - Early proxy request-size enforcement; coerce non-str `x-litellm-*` header values to avoid an httpx `TypeError` - [PR #27311](https://github.com/BerriAI/litellm/pull/27311), [PR #27504](https://github.com/BerriAI/litellm/pull/27504)
    - Separate DB read and write endpoints - [PR #27493](https://github.com/BerriAI/litellm/pull/27493)
- **Health checks**
    - Shared health-check polling; `health_check_reasoning_effort` for model health checks; skip `disable_background_health_check` models on `GET /health`; scope `/health` response to the caller's models; remove the separate health app - [PR #26434](https://github.com/BerriAI/litellm/pull/26434), [PR #27115](https://github.com/BerriAI/litellm/pull/27115), [PR #27716](https://github.com/BerriAI/litellm/pull/27716), [PR #26935](https://github.com/BerriAI/litellm/pull/26935), [PR #27430](https://github.com/BerriAI/litellm/pull/27430)
- **Config / startup robustness**
    - Hot-reload config YAML when `--reload` is set; break the managed-resources import cycle on Python 3.13; reject bare-str file-input sinks (local-file read hardening) - [PR #27274](https://github.com/BerriAI/litellm/pull/27274), [PR #27160](https://github.com/BerriAI/litellm/pull/27160), [PR #27762](https://github.com/BerriAI/litellm/pull/27762)
- **Packaging / Docker / Helm / CI**
    - Pin Wolfi & uv to multi-arch index digests; remove the hardcoded Prisma binary target for multi-arch builds; clear flagged OS-package advisories on the Docker image; refresh dependency locks - [PR #27123](https://github.com/BerriAI/litellm/pull/27123), [PR #27170](https://github.com/BerriAI/litellm/pull/27170), [PR #27225](https://github.com/BerriAI/litellm/pull/27225), [PR #27126](https://github.com/BerriAI/litellm/pull/27126)
    - Helm: skip startup `prisma db push` when a migrations Job is enabled; increase default probe timeouts, disable debug logging by default - [PR #27200](https://github.com/BerriAI/litellm/pull/27200), [PR #27237](https://github.com/BerriAI/litellm/pull/27237)
    - CI: Rerun Failed Tests for all pytest jobs, block PRs that drop coverage, Redis-backed VCR replay caches, reduce cassette bloat, mutation-testing workflow, dev-tag detection in the release workflow, Playwright apt-install skip - [PR #27155](https://github.com/BerriAI/litellm/pull/27155), [PR #27340](https://github.com/BerriAI/litellm/pull/27340), [PR #26838](https://github.com/BerriAI/litellm/pull/26838), [PR #27159](https://github.com/BerriAI/litellm/pull/27159), [PR #27409](https://github.com/BerriAI/litellm/pull/27409), [PR #27576](https://github.com/BerriAI/litellm/pull/27576), [PR #26966](https://github.com/BerriAI/litellm/pull/26966), [PR #27169](https://github.com/BerriAI/litellm/pull/27169)
    - Remove legacy deployment artifacts and litellm-js packages; remove a redundant backup pricing file; misc test/import cleanup - [PR #27541](https://github.com/BerriAI/litellm/pull/27541), [PR #16590](https://github.com/BerriAI/litellm/pull/16590), [PR #27699](https://github.com/BerriAI/litellm/pull/27699), [PR #27633](https://github.com/BerriAI/litellm/pull/27633)
    - Tighten router-settings-override and mock-testing trust; drop blank-text fallback for empty Bedrock Converse thinking blocks - [PR #26968](https://github.com/BerriAI/litellm/pull/26968), [PR #27850](https://github.com/BerriAI/litellm/pull/27850)

## Documentation Updates

- Update the Greptile README logo to a higher-quality image - [PR #25385](https://github.com/BerriAI/litellm/pull/25385)
- Add a `BudgetManager.reset_cost` docstring - [PR #27867](https://github.com/BerriAI/litellm/pull/27867)
- Add a `_LoopWrapper` class docstring - [PR #27870](https://github.com/BerriAI/litellm/pull/27870)

## New Contributors

- @kimimgo made their first contribution in [#24282](https://github.com/BerriAI/litellm/pull/24282)
- @shubham-arora-clear made their first contribution in [#24644](https://github.com/BerriAI/litellm/pull/24644)
- @ohnoah made their first contribution in [#24580](https://github.com/BerriAI/litellm/pull/24580)
- @ushiromiya-lion made their first contribution in [#25106](https://github.com/BerriAI/litellm/pull/25106)
- @gowtham2809 made their first contribution in [#25224](https://github.com/BerriAI/litellm/pull/25224)
- @he-yufeng made their first contribution in [#26401](https://github.com/BerriAI/litellm/pull/26401)
- @MackDing made their first contribution in [#26419](https://github.com/BerriAI/litellm/pull/26419)
- @dgu1-godaddy made their first contribution in [#26834](https://github.com/BerriAI/litellm/pull/26834)
- @Vedanshu7 made their first contribution in [#24943](https://github.com/BerriAI/litellm/pull/24943)
- @dennishenry made their first contribution in [#27190](https://github.com/BerriAI/litellm/pull/27190)
- @SHARP155 made their first contribution in [#27466](https://github.com/BerriAI/litellm/pull/27466)
- @mats852 made their first contribution in [#24927](https://github.com/BerriAI/litellm/pull/24927)

**Full Changelog**: https://github.com/BerriAI/litellm/compare/v1.84.0...v1.85.0

---

## 05/16/2026 (`v1.85.0`)

> Counts cover PRs **new in `v1.85.0`** relative to `v1.84.0` **stable**. 14 PRs that were backported into `v1.84.0` stable (and documented in the v1.84.0 release notes) are excluded here to avoid double-counting.

* New Models / Updated Models: 43
* LLM API Endpoints: 24
* Management Endpoints / UI: 54
* AI Integrations (Logging / Guardrails / Prompt Mgmt / Secret Managers): 32
* Spend Tracking, Budgets and Rate Limiting: 23
* MCP Gateway: 12
* Performance / Loadbalancing / Reliability improvements: 41
* Documentation Updates: 3

Total: 232 PRs

---
