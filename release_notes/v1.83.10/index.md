---
title: "v1.83.10 - Claude Opus 4.7, Prompt Compression & Multi-Window Budgets"
slug: "v1-83-10"
date: 2026-04-27T00:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
  - name: Ryan Crabbe
    title: Full Stack Engineer, LiteLLM
    url: https://www.linkedin.com/in/ryan-crabbe-0b9687214
    image_url: https://github.com/ryan-crabbe.png
  - name: Yuneng Jiang
    title: Senior Full Stack Engineer, LiteLLM
    url: https://www.linkedin.com/in/yuneng-david-jiang-455676139/
    image_url: https://avatars.githubusercontent.com/u/171294688?v=4
  - name: Shivam Rawat
    title: Forward Deployed Engineer, LiteLLM
    url: https://linkedin.com/in/shivam-rawat-482937318
    image_url: https://github.com/shivamrawat1.png
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
docker.litellm.ai/berriai/litellm:main-v1.83.10-stable
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.83.10
```

</TabItem>
</Tabs>

## Key Highlights

- **Claude Opus 4.7 day-0 support** — Opus 4.7 across [Anthropic](../../docs/providers/anthropic), [Bedrock](../../docs/providers/bedrock), [Vertex AI](../../docs/providers/vertex), [Azure AI](../../docs/providers/azure_ai), and [Perplexity](../../docs/providers/perplexity), with reasoning, vision, prompt caching, computer use, and 1M-token context.
- **`litellm.compress()`** — [BM25-based prompt compression with a retrieval tool](../../docs/completion/prompt_compression) for trimming long context before it hits the model.
- **Multi-Threshold Budget Alerts** — [virtual keys can fire alerts at multiple configurable spend thresholds](../../docs/proxy/alerting) (e.g. 50% / 80% / 95%) instead of a single soft-budget level.
- **Concurrent Budget Windows** — [keys and teams can run multiple budget periods (daily + monthly) simultaneously](../../docs/proxy/users), each with its own reset cadence.
- **Per-Team Guardrail Opt-Out** — [teams can opt out of specific global guardrails from team settings](../../docs/proxy/guardrails/quick_start) without touching config files.
- **PromptGuard Guardrail Integration** — [first-class pre/post-call guardrail for prompt-injection detection](../../docs/proxy/guardrails/promptguard).
- **uv Packaging Migration** — [Poetry replaced by uv across packaging, CI, and Docker](../../docs/extras/code_quality) for faster, reproducible builds.

---

## Breaking Changes

#### Caller-supplied `tags` are stripped unless the key/team opts in

- **What changed:** Tags supplied by the caller — `metadata.tags`, `litellm_metadata.tags`, root-level `tags`, and the `x-litellm-tags` header — are stripped from the request before [tag-based routing](../../docs/proxy/tag_routing) and [tag-based spend attribution](../../docs/proxy/cost_tracking#custom-tags) run, unless the calling key or its parent team carries `metadata.allow_client_tags: true`. Tags configured on the model deployment, key metadata, or team metadata are unaffected. The proxy logs a `WARNING` line on each strip:
  ```
  Stripped caller-supplied tags from metadata, tags (root): this key/team does not have `allow_client_tags: true` in its metadata. Set it to opt into client-supplied routing/budget tags.
  ```
  — [PR #25905](https://github.com/BerriAI/litellm/pull/25905)

- **Who is affected:** Any deployment that relied on clients passing `tags` in the request body or `x-litellm-tags` header for tag-based cost tracking, tag budgets, or tag-based routing. After upgrade, those tags will silently fall through to the default bucket / default deployment, and per-tag spend reports will appear empty.

- **Restore prior behavior:** Set `allow_client_tags: true` in the metadata of the affected key (or the team owning it). Either flag is sufficient — if the key or its parent team carries the flag, caller-supplied tags pass through.
  ```bash
  # Per key
  curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
    -H 'Authorization: Bearer sk-1234' \
    -H 'Content-Type: application/json' \
    -d '{"metadata": {"allow_client_tags": true}}'

  # Per team
  curl -L -X POST 'http://0.0.0.0:4000/team/new' \
    -H 'Authorization: Bearer sk-1234' \
    -H 'Content-Type: application/json' \
    -d '{"metadata": {"allow_client_tags": true}}'
  ```

  Existing keys/teams can be patched with `/key/update` or `/team/update` carrying the same `metadata` payload.

#### `os.environ/…` values in the UI or API

- **What changed:** Values such as `os.environ/OPENAI_API_KEY` (and other `os.environ/…` patterns) are no longer expanded when they come from **request-supplied** fields—including the Admin UI and the same proxy APIs the UI calls. — [PR #25592](https://github.com/BerriAI/litellm/pull/25592)

- **Who is affected:** Anyone who entered literal `os.environ/SECRET_NAME` strings in the UI or API and expected the proxy to substitute the host environment at runtime.

- **What to use instead:** Provider API keys and similar secrets should be stored with [**Reusable Credentials**](../../docs/proxy/ui_credentials.md) and attached to models (for example via `litellm_credential_name`). For observability callbacks (Langfuse, LangSmith, etc.), set keys and endpoints in proxy `config.yaml` or in environment variables the process reads at startup—not as `os.environ/…` strings inside per-request metadata.

---

## New Models / Updated Models

#### New Model Support (10 new models)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| Anthropic | `claude-opus-4-7`, `claude-opus-4-7-20260416` | 1M | $5.00 | $25.00 | Chat, reasoning, vision, computer use, prompt caching, PDF input, xhigh reasoning effort |
| AWS Bedrock | `anthropic.claude-opus-4-7`, `us.anthropic.claude-opus-4-7`, `eu.anthropic.claude-opus-4-7`, `au.anthropic.claude-opus-4-7`, `global.anthropic.claude-opus-4-7` | 1M | $5.50 | $27.50 | Chat, reasoning, vision, computer use, prompt caching, PDF input, native structured output |
| Vertex AI | `vertex_ai/claude-opus-4-7`, `vertex_ai/claude-opus-4-7@default` | 1M | $5.00 | $25.00 | Chat, reasoning, vision, computer use, prompt caching, PDF input |
| Azure AI | `azure_ai/claude-opus-4-7` | 200K | $5.00 | $25.00 | Chat, reasoning, vision, computer use, prompt caching, PDF input |
| Perplexity | `perplexity/anthropic/claude-opus-4-7` | - | - | - | Web search, function calling (Responses mode) |
| Google Gemini | `gemini/veo-3.1-lite-generate-preview` | 1024 | - | $0.05 / sec | Video generation preview |
| OpenRouter | `openrouter/google/gemini-3.1-flash-lite-preview` | 1.05M | $0.25 | $1.50 | Chat, code execution, file search, function calling, prompt caching, reasoning, web search, vision, video/audio/PDF input |
| xAI | `xai/grok-4.20-0309-reasoning` | 2M | $2.00 | $6.00 | Function calling, reasoning, tool choice, vision, web search |
| W&B Inference | `wandb/MiniMaxAI/MiniMax-M2.5` | 197K | $0.30 | $1.20 | Function calling, reasoning, response schema |
| W&B Inference | `wandb/moonshotai/Kimi-K2.5` | 262K | $0.60 | $3.00 | Function calling, reasoning, response schema, vision |

#### Features

- **[Anthropic](../../docs/providers/anthropic)**
    - Day-0 support for Claude Opus 4.7 across Anthropic native, Bedrock, Vertex AI, Azure AI, and Perplexity - [PR #25867](https://github.com/BerriAI/litellm/pull/25867)
    - Hotfix follow-ups for Opus 4.7 routing/version-string handling - [PR #25875](https://github.com/BerriAI/litellm/pull/25875), [PR #25876](https://github.com/BerriAI/litellm/pull/25876)
    - Retry `/v1/messages` after invalid thinking signature errors - [PR #25674](https://github.com/BerriAI/litellm/pull/25674)

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Normalize custom tool JSON schema for both Invoke and Converse APIs - [PR #25396](https://github.com/BerriAI/litellm/pull/25396)
    - Bedrock API response null-type handling - [PR #25810](https://github.com/BerriAI/litellm/pull/25810), [PR #24147](https://github.com/BerriAI/litellm/pull/24147)
    - Prevent negative streaming costs for start-only cache usage - [PR #25846](https://github.com/BerriAI/litellm/pull/25846)
    - Accurate cache token cost breakdown in UI and SpendLogs - [PR #25735](https://github.com/BerriAI/litellm/pull/25735)
    - Remove unresolved merge conflict markers in Bedrock test file - [PR #25995](https://github.com/BerriAI/litellm/pull/25995)
    - Replace flaky Bedrock gpt-oss tool-call live test with request-body mock - [PR #25739](https://github.com/BerriAI/litellm/pull/25739)
    - Mock Bedrock Moonshot tests + fix `TogetherAIConfig` recursion - [PR #25920](https://github.com/BerriAI/litellm/pull/25920)
    - Remove dead Bedrock `clear_thinking` interleaved-thinking-beta assertion - [PR #25913](https://github.com/BerriAI/litellm/pull/25913)

- **[Google Vertex AI](../../docs/providers/vertex)**
    - Normalize Gemini `finish_reason` enum through `map_finish_reason` - [PR #25337](https://github.com/BerriAI/litellm/pull/25337)
    - Add `us-south1` region for `vertex_ai/qwen3-235b-a22b-instruct-2507-maas` - [PR #25382](https://github.com/BerriAI/litellm/pull/25382)
    - Add `vertex_ai/claude-opus-4-7` and `vertex_ai/claude-opus-4-7@default` cost map entries - cost map

- **[Google Gemini](../../docs/providers/gemini)**
    - Veo 3.1 Lite pricing, video resolution usage, and tiered cost tracking - [PR #25348](https://github.com/BerriAI/litellm/pull/25348)

- **[Azure AI](../../docs/providers/azure_ai)**
    - Add `azure_ai/claude-opus-4-7` cost map entry - cost map
    - Populate `standard_logging_object` for Azure passthrough via logging hook - [PR #25679](https://github.com/BerriAI/litellm/pull/25679)

- **[OpenAI](../../docs/providers/openai)**
    - Omit null `encoding_format` for OpenAI embedding requests - [PR #25395](https://github.com/BerriAI/litellm/pull/25395) (later reverted in [PR #25698](https://github.com/BerriAI/litellm/pull/25698) — see Bug Fixes)

- **[xAI](../../docs/providers/xai)**
    - Add `xai/grok-4.20-0309-reasoning` cost map entry - [PR #25930](https://github.com/BerriAI/litellm/pull/25930)

- **[Together AI](../../docs/providers/togetherai)**
    - Expose reasoning effort fields in `get_model_info` and add `together_ai/gpt-oss-120b` - [PR #25263](https://github.com/BerriAI/litellm/pull/25263)
    - Replace deprecated Mixtral with serverless Qwen3.5-9B in tests - [PR #25728](https://github.com/BerriAI/litellm/pull/25728)

- **[DashScope](../../docs/providers/dashscope)**
    - Preserve `cache_control` for explicit prompt caching - [PR #25331](https://github.com/BerriAI/litellm/pull/25331)

- **[GitHub Copilot](../../docs/providers/github_copilot)**
    - Allow overriding the default GitHub Copilot authentication endpoint - [PR #25915](https://github.com/BerriAI/litellm/pull/25915)

- **[W&B Inference](../../docs/providers/wandb_inference)**
    - Add Kimi-K2.5 and MiniMax-M2.5 cost map entries - [PR #25409](https://github.com/BerriAI/litellm/pull/25409)

### Bug Fixes

- **[Anthropic](../../docs/providers/anthropic)**
    - Return actual upstream status code from `/v1/messages/count_tokens` instead of always 200 - [PR #21352](https://github.com/BerriAI/litellm/pull/21352)

- **[Vertex AI](../../docs/providers/vertex)**
    - Gemini `finish_reason` enum normalization (see Features above) - [PR #25337](https://github.com/BerriAI/litellm/pull/25337)

- **[Embeddings API](../../docs/embedding/supported_embedding)**
    - Revert null-`encoding_format` omission after downstream regression - [PR #25698](https://github.com/BerriAI/litellm/pull/25698)

- **General**
    - Fix `version` shown in docs banner - [PR #25875](https://github.com/BerriAI/litellm/pull/25875)

## LLM API Endpoints

#### Features

- **[Responses API](../../docs/response_api)**
    - Add Responses API params to cache key allow-list - [PR #25673](https://github.com/BerriAI/litellm/pull/25673)

- **[OCR API](../../docs/ocr)**
    - Mistral-style `pages` param via Azure DI analyze query string - [PR #25929](https://github.com/BerriAI/litellm/pull/25929)
    - Add missing Mistral OCR params to allowlist - [PR #25858](https://github.com/BerriAI/litellm/pull/25858)

- **[Embeddings API](../../docs/embedding/supported_embedding)**
    - OpenAI `encoding_format` handling for null values (initial fix later reverted) - [PR #25395](https://github.com/BerriAI/litellm/pull/25395), [PR #25698](https://github.com/BerriAI/litellm/pull/25698)

- **[Anthropic Messages](../../docs/anthropic_unified/)**
    - Retry on invalid thinking signature - [PR #25674](https://github.com/BerriAI/litellm/pull/25674)
    - Return actual status code on `count_tokens` upstream errors - [PR #21352](https://github.com/BerriAI/litellm/pull/21352)

- **[Pass-Through Endpoints](../../docs/pass_through/intro)**
    - Populate `standard_logging_object` for Azure passthrough - [PR #25679](https://github.com/BerriAI/litellm/pull/25679)
    - Restrict `x-pass-` header forwarding for credential and protocol headers - [PR #25916](https://github.com/BerriAI/litellm/pull/25916)

- **[Video Generation](../../docs/proxy/veo_video_generation)**
    - Veo 3.1 Lite resolution-aware tiered cost tracking - [PR #25348](https://github.com/BerriAI/litellm/pull/25348)

- **General — `litellm.compress()`**
    - New BM25-based [prompt compression API](../../docs/completion/prompt_compression) with retrieval tool, exposed via `litellm.compress()` for trimming long prompts before model invocation - [PR #25637](https://github.com/BerriAI/litellm/pull/25637)

#### Bugs

- **General**
    - Tighten `api_key` value check in credential validation - [PR #25917](https://github.com/BerriAI/litellm/pull/25917)
    - Tighten environment-reference handling in request parameters - [PR #25592](https://github.com/BerriAI/litellm/pull/25592)
    - Harden request parameter handling - [PR #25827](https://github.com/BerriAI/litellm/pull/25827)
    - Add shared path utilities and prevent directory traversal - [PR #25834](https://github.com/BerriAI/litellm/pull/25834)
    - Add URL validation for user-supplied URLs - [PR #25906](https://github.com/BerriAI/litellm/pull/25906)
    - Read guardrail config from admin metadata; fix tag-routing consistency - [PR #25905](https://github.com/BerriAI/litellm/pull/25905)
    - Enforce organization boundaries in admin operations - [PR #25904](https://github.com/BerriAI/litellm/pull/25904)
    - Resolve `prometheus_helpers` file/package shadow breaking `/global/spend/logs` - [PR #26026](https://github.com/BerriAI/litellm/pull/26026)
    - Harden CORS credentials, `create_views` exception handling, and spend-log cleanup loop - [PR #25559](https://github.com/BerriAI/litellm/pull/25559)
    - Prevent API key leaks in error tracebacks, logs, and alerts - [PR #25117](https://github.com/BerriAI/litellm/pull/25117)
    - Remove leading space from license `public_key.pem` - [PR #25339](https://github.com/BerriAI/litellm/pull/25339)
    - Cache invalidation: stop double-hashing token in bulk update and key rotation - [PR #25552](https://github.com/BerriAI/litellm/pull/25552)
    - `model_max_budget` silently broken for routed models - [PR #25549](https://github.com/BerriAI/litellm/pull/25549)
    - Bump 22 of 25 vulnerable dependabot-reported dependencies - [PR #25442](https://github.com/BerriAI/litellm/pull/25442)
    - Fix `multiple values` `TypeError` in `get_cache_key` - [PR #20261](https://github.com/BerriAI/litellm/pull/20261)
    - S3v2: use prepared URL for SigV4-signed S3 requests - [PR #25074](https://github.com/BerriAI/litellm/pull/25074)
    - Health-check reasoning-token max-token precedence - [PR #25936](https://github.com/BerriAI/litellm/pull/25936)
    - `BACKGROUND_HEALTH_CHECK_MAX_TOKENS` env var - [PR #25344](https://github.com/BerriAI/litellm/pull/25344)
    - Batch-limit stale managed object cleanup to prevent 300K-row UPDATE - [PR #25227](https://github.com/BerriAI/litellm/pull/25227)
    - Preserve provider response headers in `StandardLoggingPayload` - [PR #25807](https://github.com/BerriAI/litellm/pull/25807)
    - Optimize DB query to prevent OOM during health checks - [PR #25732](https://github.com/BerriAI/litellm/pull/25732)
    - `PodLockManager.release_lock` atomic compare-and-delete (re-land #21226) - [PR #24466](https://github.com/BerriAI/litellm/pull/24466)
    - `routing_strategy_args` returns `None` when strategy is not latency-based - [PR #25882](https://github.com/BerriAI/litellm/pull/25882)
    - `is_tool_name_prefixed` validates against known MCP server prefixes - [PR #25085](https://github.com/BerriAI/litellm/pull/25085)
    - Persist default router end-budget across restarts - [PR #25991](https://github.com/BerriAI/litellm/pull/25991)
    - Enforce team membership in team-scoped key management checks - [PR #25686](https://github.com/BerriAI/litellm/pull/25686)
    - Agent endpoint and routing permission checks - [PR #25922](https://github.com/BerriAI/litellm/pull/25922)
    - JWT-auth `key_alias=user_id` for Prometheus metrics — initial fix and revert - [PR #25340](https://github.com/BerriAI/litellm/pull/25340), [PR #25438](https://github.com/BerriAI/litellm/pull/25438)
    - Gate post-custom-auth DB lookups behind opt-in flag - [PR #25634](https://github.com/BerriAI/litellm/pull/25634)
    - Align field-level checks in user and key update endpoints - [PR #25541](https://github.com/BerriAI/litellm/pull/25541)
    - `/spend/logs` filter handling aligned with user scoping - [PR #25594](https://github.com/BerriAI/litellm/pull/25594)
    - Replace `custom_code` guardrail sandbox with RestrictedPython - [PR #25818](https://github.com/BerriAI/litellm/pull/25818)
    - Presidio: use correct text positions in `anonymize_text` - [PR #24998](https://github.com/BerriAI/litellm/pull/24998)

## Management Endpoints / UI

#### Features

- **Virtual Keys**
    - Configurable multi-threshold budget alerts (e.g. 50% / 80% / 95%) - [PR #25989](https://github.com/BerriAI/litellm/pull/25989)
    - Multiple concurrent budget windows per API key and team (`#24883`) - [PR #25109](https://github.com/BerriAI/litellm/pull/25109)
    - Per-member model scope + team `default_team_member_models` - [PR #24950](https://github.com/BerriAI/litellm/pull/24950)
    - Migrate regenerate key modal to AntD - [PR #25406](https://github.com/BerriAI/litellm/pull/25406)
    - Strip empty premium fields from key update payload - [PR #26023](https://github.com/BerriAI/litellm/pull/26023)
    - Default invite-user modal global role to least privilege - [PR #25721](https://github.com/BerriAI/litellm/pull/25721)

- **Teams**
    - Allow editing router settings after team creation - [PR #25398](https://github.com/BerriAI/litellm/pull/25398)
    - Per-team opt-out for specific global guardrails - [PR #25575](https://github.com/BerriAI/litellm/pull/25575)
    - Enterprise notice banner on deleted Keys/Teams - [PR #25814](https://github.com/BerriAI/litellm/pull/25814)
    - Invalidate org queries after team mutations - [PR #25812](https://github.com/BerriAI/litellm/pull/25812)
    - E2E test for editing team model TPM/RPM limits - [PR #25658](https://github.com/BerriAI/litellm/pull/25658)

- **Models + Endpoints**
    - Claude Code BYOK support in UI Settings - [PR #25998](https://github.com/BerriAI/litellm/pull/25998)
    - E2E tests for Add Model flow - [PR #25590](https://github.com/BerriAI/litellm/pull/25590)
    - Pre-select backend default for boolean guardrail provider fields - [PR #25700](https://github.com/BerriAI/litellm/pull/25700)
    - Render guardrail `optional_params` bool defaults in `Select` - [PR #25806](https://github.com/BerriAI/litellm/pull/25806)
    - Use AntD `Select` for MCP `ToolTestPanel` boolean inputs - [PR #25809](https://github.com/BerriAI/litellm/pull/25809)
    - Persist `extra_headers` on MCP server edit - [PR #26003](https://github.com/BerriAI/litellm/pull/26003)
    - Migrate Guardrail Test Playground from `@tremor/react` to AntD - [PR #25749](https://github.com/BerriAI/litellm/pull/25749)
    - Migrate router_settings page from Tremor to AntD - [PR #25879](https://github.com/BerriAI/litellm/pull/25879)
    - Reduce Tremor usage in Guardrails Monitor layout - [PR #25803](https://github.com/BerriAI/litellm/pull/25803)
    - Remove Chat UI link from Swagger docs message - [PR #25727](https://github.com/BerriAI/litellm/pull/25727)
    - Delete policy attachments via controlled modal - [PR #25324](https://github.com/BerriAI/litellm/pull/25324)

- **Auth / SSO**
    - Resolve login redirect loop when reverse proxy adds `HttpOnly` to cookies - [PR #23532](https://github.com/BerriAI/litellm/pull/23532)
    - Gate post-custom-auth DB lookups behind opt-in flag - [PR #25634](https://github.com/BerriAI/litellm/pull/25634)

- **Logs / Activity**
    - Isolate logs team-filter dropdown from root teams state bleed - [PR #25716](https://github.com/BerriAI/litellm/pull/25716)
    - Align `/spend/logs` filter handling with user scoping - [PR #25594](https://github.com/BerriAI/litellm/pull/25594)

- **Helm**
    - Add `tpl` support to `extraContainers` and `extraInitContainers` - [PR #25494](https://github.com/BerriAI/litellm/pull/25494)

#### Bugs

- Strip empty premium fields from key update payload - [PR #26023](https://github.com/BerriAI/litellm/pull/26023)
- Tighten `api_key` value check in credential validation - [PR #25917](https://github.com/BerriAI/litellm/pull/25917)
- `extra_headers` not persisting on MCP server edit - [PR #26003](https://github.com/BerriAI/litellm/pull/26003)
- Logs team-filter dropdown leakage - [PR #25716](https://github.com/BerriAI/litellm/pull/25716)
- Add `getCookie` to `cookieUtils` mock in `user_dashboard` test - [PR #25719](https://github.com/BerriAI/litellm/pull/25719)
- Remove deprecated `tests/ui_e2e_tests/` suite - [PR #25657](https://github.com/BerriAI/litellm/pull/25657)
- Restrict `x-pass-` header forwarding - [PR #25916](https://github.com/BerriAI/litellm/pull/25916)
- Blog dark-mode text invisible on dark background - [PR #25620](https://github.com/BerriAI/litellm/pull/25620)
- Default invite-user role least-privilege - [PR #25721](https://github.com/BerriAI/litellm/pull/25721)

## AI Integrations

### Logging

- **[Prometheus](../../docs/proxy/logging#prometheus)**
    - Add 7m and 10m latency histogram buckets - [PR #25071](https://github.com/BerriAI/litellm/pull/25071)
    - Performance improvements for Prometheus exporter - [PR #25934](https://github.com/BerriAI/litellm/pull/25934)
    - Resolve `prometheus_helpers` file/package shadow breaking `/global/spend/logs` - [PR #26026](https://github.com/BerriAI/litellm/pull/26026)

- **[Azure Pass-Through](../../docs/pass_through/azure_passthrough)**
    - Populate `standard_logging_object` via logging hook - [PR #25679](https://github.com/BerriAI/litellm/pull/25679)

- **General**
    - Preserve provider response headers in `StandardLoggingPayload` - [PR #25807](https://github.com/BerriAI/litellm/pull/25807)

### Guardrails

- **[PromptGuard](../../docs/proxy/guardrails/promptguard)**
    - New PromptGuard guardrail integration for prompt-injection detection - [PR #24268](https://github.com/BerriAI/litellm/pull/24268)

- **[Custom Code Guardrails](../../docs/proxy/guardrails/custom_guardrail)**
    - Replace `custom_code` sandbox with RestrictedPython - [PR #25818](https://github.com/BerriAI/litellm/pull/25818)

- **[Presidio](../../docs/proxy/guardrails/pii_masking_v2)**
    - Use correct text positions in `anonymize_text` - [PR #24998](https://github.com/BerriAI/litellm/pull/24998)

- **General**
    - Per-team opt-out for specific global guardrails - [PR #25575](https://github.com/BerriAI/litellm/pull/25575)
    - UI: pre-select backend default for boolean guardrail provider fields - [PR #25700](https://github.com/BerriAI/litellm/pull/25700)
    - UI: render guardrail `optional_params` boolean defaults in `Select` - [PR #25806](https://github.com/BerriAI/litellm/pull/25806)
    - Read guardrail config from admin metadata and fix tag-routing consistency - [PR #25905](https://github.com/BerriAI/litellm/pull/25905)

### Caching

- Add Responses API params to cache key allow-list - [PR #25673](https://github.com/BerriAI/litellm/pull/25673)
- Prevent `multiple values` `TypeError` in `get_cache_key` - [PR #20261](https://github.com/BerriAI/litellm/pull/20261)
- S3v2: use prepared URL for SigV4-signed S3 requests - [PR #25074](https://github.com/BerriAI/litellm/pull/25074)

### Prompt Management / Compression

- New `litellm.compress()` BM25-based [prompt compression API](../../docs/completion/prompt_compression) with retrieval tool - [PR #25637](https://github.com/BerriAI/litellm/pull/25637)

### Secret Managers

- No new secret manager provider additions in this release.

## Spend Tracking, Budgets and Rate Limiting

- Configurable multi-threshold budget alerts for virtual keys (e.g. 50% / 80% / 95%) - [PR #25989](https://github.com/BerriAI/litellm/pull/25989)
- Multiple concurrent budget windows per API key and team (`#24883`) - [PR #25109](https://github.com/BerriAI/litellm/pull/25109)
- Bedrock/Anthropic accurate cache token cost breakdown in UI and SpendLogs - [PR #25735](https://github.com/BerriAI/litellm/pull/25735)
- Bedrock: prevent negative streaming costs for start-only cache usage - [PR #25846](https://github.com/BerriAI/litellm/pull/25846)
- Fix virtual-key projected-spend soft budget alerts - [PR #25838](https://github.com/BerriAI/litellm/pull/25838)
- Enforce project-level model-specific rate limits in parallel-request limiter - [PR #25994](https://github.com/BerriAI/litellm/pull/25994)
- Persist default router end-budget across restarts - [PR #25991](https://github.com/BerriAI/litellm/pull/25991)
- Align reset times for legacy entities (Team Members, End Users) with the standardized calendar - [PR #25440](https://github.com/BerriAI/litellm/pull/25440)
- Batch-limit stale managed-object cleanup to prevent 300K-row UPDATE - [PR #25227](https://github.com/BerriAI/litellm/pull/25227)
- Cache invalidation: stop double-hashing token in bulk update and key rotation - [PR #25552](https://github.com/BerriAI/litellm/pull/25552)
- `model_max_budget` silently broken for routed models - [PR #25549](https://github.com/BerriAI/litellm/pull/25549)
- Expose reasoning-effort fields in `get_model_info` (and add `together_ai/gpt-oss-120b` to cost map) - [PR #25263](https://github.com/BerriAI/litellm/pull/25263)
- Veo 3.1 Lite resolution-aware tiered cost tracking - [PR #25348](https://github.com/BerriAI/litellm/pull/25348)
- Add `us-south1` region for Vertex `qwen3-235b-a22b-instruct-2507-maas` cost map - [PR #25382](https://github.com/BerriAI/litellm/pull/25382)

## MCP Gateway

- Validate `is_tool_name_prefixed` against the set of known MCP server prefixes - [PR #25085](https://github.com/BerriAI/litellm/pull/25085)
- Restore PKCE-triggering 401 when no stored per-user token exists - [PR #26032](https://github.com/BerriAI/litellm/pull/26032)
- Expose per-server `InitializeResult.instructions` from the MCP gateway - [PR #25694](https://github.com/BerriAI/litellm/pull/25694)
- Extract shared PKCE helpers into `utils/pkce.ts` - [PR #25878](https://github.com/BerriAI/litellm/pull/25878)
- UI: AntD `Select` for MCP `ToolTestPanel` boolean inputs - [PR #25809](https://github.com/BerriAI/litellm/pull/25809)
- UI: persist `extra_headers` on MCP server edit - [PR #26003](https://github.com/BerriAI/litellm/pull/26003)

## Performance / Loadbalancing / Reliability improvements

- Prometheus exporter performance improvements - [PR #25934](https://github.com/BerriAI/litellm/pull/25934)
- Optimize DB query to prevent OOM during health checks - [PR #25732](https://github.com/BerriAI/litellm/pull/25732)
- `PodLockManager.release_lock` atomic compare-and-delete (re-land of #21226) - [PR #24466](https://github.com/BerriAI/litellm/pull/24466)
- Health-check reasoning-token max-token precedence - [PR #25936](https://github.com/BerriAI/litellm/pull/25936)
- New `BACKGROUND_HEALTH_CHECK_MAX_TOKENS` environment variable - [PR #25344](https://github.com/BerriAI/litellm/pull/25344)
- Return `None` for `routing_strategy_args` when strategy is not latency-based - [PR #25882](https://github.com/BerriAI/litellm/pull/25882)
- Bump proxy dependencies; raise minimum Python to 3.10 - [PR #26022](https://github.com/BerriAI/litellm/pull/26022)
- Bump 22 of 25 vulnerable dependabot-reported dependencies - [PR #25442](https://github.com/BerriAI/litellm/pull/25442)
- Migrate packaging, CI, and Docker from Poetry to uv - [PR #25007](https://github.com/BerriAI/litellm/pull/25007)
- `[Infra]` Bump `llm_translation_testing` resource class to `xlarge` and tolerate worker restarts - [PR #25887](https://github.com/BerriAI/litellm/pull/25887), [PR #25898](https://github.com/BerriAI/litellm/pull/25898)
- `[Infra]` Expand CI branch filters for non-`main` PR targets - [PR #25819](https://github.com/BerriAI/litellm/pull/25819)
- `[Infra]` Guard `main` to only accept PRs from staging and hotfix branches - [PR #25733](https://github.com/BerriAI/litellm/pull/25733)
- `[Infra]` Remove unused `publish_proxy_extras` and `prisma_schema_sync` jobs from CircleCI config - [PR #25821](https://github.com/BerriAI/litellm/pull/25821)
- `fix(ci)`: increase `test-server-root-path` timeout to 30m - [PR #25741](https://github.com/BerriAI/litellm/pull/25741)
- Remove non-existent `litellm_mcps_tests_coverage` from coverage combine - [PR #25737](https://github.com/BerriAI/litellm/pull/25737)
- Helm: add `tpl` support to `extraContainers` / `extraInitContainers` - [PR #25494](https://github.com/BerriAI/litellm/pull/25494)
- Advisor tool orchestration loop for non-Anthropic providers - [PR #25579](https://github.com/BerriAI/litellm/pull/25579)

## Documentation Updates

- Cost discrepancy debugging guide - [PR #25622](https://github.com/BerriAI/litellm/pull/25622)
- Week 2 onboarding checklist - [PR #25452](https://github.com/BerriAI/litellm/pull/25452)
- Add "Copy Page as Markdown" + `llms.txt` to docs site - [PR #25975](https://github.com/BerriAI/litellm/pull/25975)
- Docs announcement bar for Trivy compromise resolution - [PR #25870](https://github.com/BerriAI/litellm/pull/25870)
- Restyle docs.litellm.ai/blog to engineering blog aesthetic - [PR #25580](https://github.com/BerriAI/litellm/pull/25580)
- Ramp-style engineering blog restyle + Redis circuit breaker post - [PR #25583](https://github.com/BerriAI/litellm/pull/25583)
- Add back arrow to blog post pages - [PR #25587](https://github.com/BerriAI/litellm/pull/25587)
- Fallbacks image - [PR #25731](https://github.com/BerriAI/litellm/pull/25731)
- General docs update - [PR #25736](https://github.com/BerriAI/litellm/pull/25736)
- Backfill release notes for v1.83.3-stable and v1.83.7.rc.1 - [PR #25723](https://github.com/BerriAI/litellm/pull/25723), [PR #25726](https://github.com/BerriAI/litellm/pull/25726)
- Fix version shown in docs - [PR #25875](https://github.com/BerriAI/litellm/pull/25875)

## New Contributors

* @hunterchris made their first contribution in https://github.com/BerriAI/litellm/pull/20261
* @Dmitry-Kucher made their first contribution in https://github.com/BerriAI/litellm/pull/24998
* @kulia26 made their first contribution in https://github.com/BerriAI/litellm/pull/25071
* @jaxhend made their first contribution in https://github.com/BerriAI/litellm/pull/23532
* @abhyudayareddy made their first contribution in https://github.com/BerriAI/litellm/pull/25337
* @avarga1 made their first contribution in https://github.com/BerriAI/litellm/pull/25263
* @acebot712 made their first contribution in https://github.com/BerriAI/litellm/pull/24268
* @meutsabdahal made their first contribution in https://github.com/BerriAI/litellm/pull/25395
* @shreyescodes made their first contribution in https://github.com/BerriAI/litellm/pull/25559
* @Lucas-Song-Dev made their first contribution in https://github.com/BerriAI/litellm/pull/25324
* @steromano87 made their first contribution in https://github.com/BerriAI/litellm/pull/25915
* @jlav made their first contribution in https://github.com/BerriAI/litellm/pull/25494

**Full Changelog**: https://github.com/BerriAI/litellm/compare/v1.83.7-stable...v1.83.10-stable

---

## 04/27/2026

* New Models / Updated Models: 23
* LLM API Endpoints: 18
* Management Endpoints / UI: 22
* AI Integrations (Logging / Guardrails / Caching / Prompt): 16
* Spend Tracking, Budgets and Rate Limiting: 13
* MCP Gateway: 6
* Performance / Loadbalancing / Reliability improvements: 17
* Documentation Updates: 11
