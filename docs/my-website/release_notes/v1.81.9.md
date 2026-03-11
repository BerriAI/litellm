---
title: "v1.81.9 - Control which MCP Servers are exposed on the Internet"
slug: "v1-81-9"
date: 2026-02-07T00:00:00
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

:::info Stable Release Branch

For each stable release, we now maintain a dedicated branch with the format `litellm_stable_release_branch_x_xx_xx` for the version.

This allows easier patching for day 0 model launches.

**Branch for v1.81.9:** [litellm_stable_release_branch_1_81_9](https://github.com/BerriAI/litellm/tree/litellm_stable_release_branch_1_81_9)

:::

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
ghcr.io/berriai/litellm:main-v1.81.9-stable
```

</TabItem>
<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.81.9
```

</TabItem>
</Tabs>

## Key Highlights

- **Claude Opus 4.6** - [Full support across Anthropic, AWS Bedrock, Azure AI, and Vertex AI with adaptive thinking and 1M context window](../../blog/claude_opus_4_6)
- **A2A Agent Gateway** - [Call A2A (Agent-to-Agent) registered agents through the standard `/chat/completions` API](../../docs/a2a_invoking_agents)
- **Expose MCP servers on the public internet** - [Launch MCP servers with public/private visibility and IP-based access control for internet-facing deployments](../../docs/mcp_public_internet)
- **UI Team Soft Budget Alerts** - [Set soft budgets on teams and receive email alerts when spending crosses the threshold — without blocking requests](../../docs/proxy/ui_team_soft_budget_alerts)
- **Performance Optimizations** - Multiple performance improvements including ~40% Prometheus CPU reduction, LRU caching, and optimized logging paths
- **LiteLLM Observatory** - [Automated 24-hour load tests](../../blog/litellm-observatory)
- **30% Faster Request Processing for Callback-Heavy Deployments** - [Performance improvement for callback heavy deployments][PR #20354](https://github.com/BerriAI/litellm/pull/20354)

---

## 30% Faster Request Processing for Callback-Heavy Deployments

 If you use logging callbacks like Langfuse, Datadog, or Prometheus, every request was paying an unnecessary cost: three loops that re-sorted your callbacks on every single request, even though the callback list hadn't changed. The more callbacks you had configured, the more time was wasted. We moved this work to happen once at startup instead of on every request. For deployments with the default callback set, this is a ~30% speedup in request setup. For deployments with many callbacks configured, the improvement is even larger.

---

## LiteLLM Observatory

LiteLLM Observatory is a long-running release-validation system we built to catch regressions before they reach users. The system is built to be extensible—you can add new tests, configure models and failure thresholds, and queue runs against any deployment. Our goal is to achieve 100% coverage of LiteLLM functionality through these tests. We run 24-hour load tests against our production deployments before all releases, surfacing issues like resource lifecycle bugs, OOMs, and CPU regressions that only appear under sustained load.

---

## MCP Servers on the Public Internet

This release makes it safe to expose MCP servers on the public internet by adding public/private visibility and IP-based access control. You can now run internet-facing MCP services while restricting access to trusted networks and keeping internal tools private.

[Get started](../../docs/mcp_public_internet)

<Image
img={require('../img/release_notes/mcp_internet.png')}
style={{ maxWidth: '900px', width: '100%' }}
/>

## UI Team Soft Budget Alerts

Set a soft budget on any team to receive email alerts when spending crosses the threshold — without blocking any requests. Configure the threshold and alerting emails directly from the Admin UI, with no proxy restart needed.

[Get started](../../docs/proxy/ui_team_soft_budget_alerts)

<Image
img={require('../img/ui_team_soft_budget_alerts.png')}
style={{ maxWidth: '900px', width: '100%' }}
/>

Let's dive in.

---

## New Models / Updated Models

#### New Model Support (13 new models)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) |
| -------- | ----- | -------------- | ------------------- | -------------------- |
| Anthropic | `claude-opus-4-6` | 1M | $5.00 | $25.00 |
| AWS Bedrock | `anthropic.claude-opus-4-6-v1` | 1M | $5.00 | $25.00 |
| Azure AI | `azure_ai/claude-opus-4-6` | 200K | $5.00 | $25.00 |
| Vertex AI | `vertex_ai/claude-opus-4-6` | 1M | $5.00 | $25.00 |
| Google Gemini | `gemini/deep-research-pro-preview-12-2025` | 65K | $2.00 | $12.00 |
| Vertex AI | `vertex_ai/deep-research-pro-preview-12-2025` | 65K | $2.00 | $12.00 |
| Moonshot | `moonshot/kimi-k2.5` | 262K | $0.60 | $3.00 |
| OpenRouter | `openrouter/qwen/qwen3-235b-a22b-2507` | 262K | $0.07 | $0.10 |
| OpenRouter | `openrouter/qwen/qwen3-235b-a22b-thinking-2507` | 262K | $0.11 | $0.60 |
| Together AI | `together_ai/zai-org/GLM-4.7` | 200K | $0.45 | $2.00 |
| Together AI | `together_ai/moonshotai/Kimi-K2.5` | 256K | $0.50 | $2.80 |
| ElevenLabs | `elevenlabs/eleven_v3` | - | $0.18/1K chars | - |
| ElevenLabs | `elevenlabs/eleven_multilingual_v2` | - | $0.18/1K chars | - |

#### Features

- **[Anthropic](../../docs/providers/anthropic)**
    - Full Claude Opus 4.6 support with adaptive thinking across all regions (us, eu, apac, au) - [PR #20506](https://github.com/BerriAI/litellm/pull/20506), [PR #20508](https://github.com/BerriAI/litellm/pull/20508), [PR #20514](https://github.com/BerriAI/litellm/pull/20514), [PR #20551](https://github.com/BerriAI/litellm/pull/20551)
    - Map reasoning content to anthropic thinking block (streaming + non-streaming) - [PR #20254](https://github.com/BerriAI/litellm/pull/20254)

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Add 1hr tiered caching costs for long-context models - [PR #20214](https://github.com/BerriAI/litellm/pull/20214)
    - Support TTL (1h) field in prompt caching for Bedrock Claude 4.5 models - [PR #20338](https://github.com/BerriAI/litellm/pull/20338)
    - Add Nova Sonic speech-to-speech model support - [PR #20244](https://github.com/BerriAI/litellm/pull/20244)
    - Fix empty assistant message for Converse API - [PR #20390](https://github.com/BerriAI/litellm/pull/20390)
    - Fix content blocked handling - [PR #20606](https://github.com/BerriAI/litellm/pull/20606)

- **[Google Gemini / Vertex AI](../../docs/providers/gemini)**
    - Add Gemini Deep Research model support - [PR #20406](https://github.com/BerriAI/litellm/pull/20406)
    - Fix Vertex AI Gemini streaming content_filter handling - [PR #20105](https://github.com/BerriAI/litellm/pull/20105)
    - Allow using OpenAI-style tools for `web_search` with Vertex AI/Gemini models - [PR #20280](https://github.com/BerriAI/litellm/pull/20280)
    - Fix `supports_native_streaming` for Gemini and Vertex AI models - [PR #20408](https://github.com/BerriAI/litellm/pull/20408)
    - Add mapping for responses tools in file IDs - [PR #20402](https://github.com/BerriAI/litellm/pull/20402)

- **[Cohere](../../docs/providers/cohere)**
    - Support `dimensions` param for Cohere embed v4 - [PR #20235](https://github.com/BerriAI/litellm/pull/20235)

- **[Cerebras](../../docs/providers/cerebras)**
    - Add reasoning param support for GPT OSS Cerebras - [PR #20258](https://github.com/BerriAI/litellm/pull/20258)

- **[Moonshot](../../docs/providers/moonshot)**
    - Add Kimi K2.5 model entries - [PR #20273](https://github.com/BerriAI/litellm/pull/20273)

- **[OpenRouter](../../docs/providers/openrouter)**
    - Add Qwen3-235B models - [PR #20455](https://github.com/BerriAI/litellm/pull/20455)

- **[Together AI](../../docs/providers/togetherai)**
    - Add GLM-4.7 and Kimi-K2.5 models - [PR #20319](https://github.com/BerriAI/litellm/pull/20319)

- **[ElevenLabs](../../docs/providers/elevenlabs)**
    - Add `eleven_v3` and `eleven_multilingual_v2` TTS models - [PR #20522](https://github.com/BerriAI/litellm/pull/20522)

- **[Vercel AI Gateway](../../docs/providers/vercel_ai_gateway)**
    - Add missing capability flags to models - [PR #20276](https://github.com/BerriAI/litellm/pull/20276)

- **[GitHub Copilot](../../docs/providers/github_copilot)**
    - Fix system prompts being dropped and auto-add required Copilot headers - [PR #20113](https://github.com/BerriAI/litellm/pull/20113)

- **[GigaChat](../../docs/providers/gigachat)**
    - Fix incorrect merging of consecutive user messages for GigaChat provider - [PR #20341](https://github.com/BerriAI/litellm/pull/20341)

- **[xAI](../../docs/providers/xai_realtime)**
    - Add xAI `/realtime` API support - works with LiveKit SDK - [PR #20381](https://github.com/BerriAI/litellm/pull/20381)

- **[OpenAI](../../docs/providers/openai)**
    - Add `gpt-5-search-api` model and docs clarifications - [PR #20512](https://github.com/BerriAI/litellm/pull/20512)

### Bug Fixes

- **[Anthropic](../../docs/providers/anthropic)**
    - Fix extra inputs not permitted error for `provider_specific_fields` - [PR #20334](https://github.com/BerriAI/litellm/pull/20334)

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Fix: Managed Batches inconsistent state management for list and cancel batches - [PR #20331](https://github.com/BerriAI/litellm/pull/20331)

- **[OpenAI Embeddings](../../docs/providers/openai)**
    - Fix `open_ai_embedding_models` to have `custom_llm_provider` None - [PR #20253](https://github.com/BerriAI/litellm/pull/20253)

---

## LLM API Endpoints

#### Features

- **[Messages API](../../docs/providers/anthropic)**
    - Filter unsupported Claude Code beta headers for non-Anthropic providers - [PR #20578](https://github.com/BerriAI/litellm/pull/20578)
    - Fix inconsistent response format in `anthropic.messages.acreate()` when using non-Anthropic providers - [PR #20442](https://github.com/BerriAI/litellm/pull/20442)
    - Fix 404 on `/api/event_logging/batch` endpoint that caused Claude Code "route not found" errors - [PR #20504](https://github.com/BerriAI/litellm/pull/20504)

- **[A2A Agent Gateway](../../docs/a2a)**
    - Allow calling A2A agents through LiteLLM `/chat/completions` API - [PR #20358](https://github.com/BerriAI/litellm/pull/20358)
    - Use A2A registered agents with `/chat/completions` - [PR #20362](https://github.com/BerriAI/litellm/pull/20362)
    - Fix A2A agents deployed with localhost/internal URLs in their agent cards - [PR #20604](https://github.com/BerriAI/litellm/pull/20604)

- **[Files API](../../docs/providers/gemini)**
    - Add support for delete and GET via file_id for Gemini - [PR #20329](https://github.com/BerriAI/litellm/pull/20329)

- **General**
    - Add User-Agent customization support - [PR #19881](https://github.com/BerriAI/litellm/pull/19881)
    - Fix search tools not found when using per-request routers - [PR #19818](https://github.com/BerriAI/litellm/pull/19818)
    - Forward extra headers in chat - [PR #20386](https://github.com/BerriAI/litellm/pull/20386)

---

## Management Endpoints / UI

#### Features

- **SSO Configuration**
    - SSO Config Team Mappings - [PR #20111](https://github.com/BerriAI/litellm/pull/20111)
    - UI - SSO: Add Team Mappings - [PR #20299](https://github.com/BerriAI/litellm/pull/20299)
    - Extract user roles from JWT access token for Keycloak compatibility - [PR #20591](https://github.com/BerriAI/litellm/pull/20591)

- **Auth / SDK**
    - Add `proxy_auth` for auto OAuth2/JWT token management in SDK - [PR #20238](https://github.com/BerriAI/litellm/pull/20238)

- **Virtual Keys**
    - Key `reset_spend` endpoint - [PR #20305](https://github.com/BerriAI/litellm/pull/20305)
    - UI - Keys: Allowed Routes to Key Info and Edit Pages - [PR #20369](https://github.com/BerriAI/litellm/pull/20369)
    - Add Key info endpoint object permission data - [PR #20407](https://github.com/BerriAI/litellm/pull/20407)
    - Keys and Teams Router Setting + Allow Override of Router Settings - [PR #20205](https://github.com/BerriAI/litellm/pull/20205)

- **Teams & Budgets**
    - Add `soft_budget` to Team Table + Create/Update Endpoints - [PR #20530](https://github.com/BerriAI/litellm/pull/20530)
    - Team Soft Budget Email Alerts - [PR #20553](https://github.com/BerriAI/litellm/pull/20553)
    - UI - Team Settings: Soft Budget + Alerting Emails - [PR #20634](https://github.com/BerriAI/litellm/pull/20634)
    - UI - User Budget Page: Unlimited Budget Checkbox - [PR #20380](https://github.com/BerriAI/litellm/pull/20380)
    - `/user/update` allow for `max_budget` resets - [PR #20375](https://github.com/BerriAI/litellm/pull/20375)

- **UI Improvements**
    - Default Team Settings: Migrate to use Reusable Model Select - [PR #20310](https://github.com/BerriAI/litellm/pull/20310)
    - Navbar: Option to Hide Community Engagement Buttons - [PR #20308](https://github.com/BerriAI/litellm/pull/20308)
    - Show team alias on Models health page - [PR #20359](https://github.com/BerriAI/litellm/pull/20359)
    - Admin Settings: Add option for Authentication for public AI Hub - [PR #20444](https://github.com/BerriAI/litellm/pull/20444)
    - Adjust daily spend date filtering for user timezone - [PR #20472](https://github.com/BerriAI/litellm/pull/20472)

- **SCIM**
    - Add base `/scim/v2` endpoint for SCIM resource discovery - [PR #20301](https://github.com/BerriAI/litellm/pull/20301)

- **Proxy CLI**
    - CLI arguments for RDS IAM auth - [PR #20437](https://github.com/BerriAI/litellm/pull/20437)

#### Bugs

- Fix: Remove unnecessary key blocking on UI login that prevented access - [PR #20210](https://github.com/BerriAI/litellm/pull/20210)
- UI - Team Settings: Disable Global Guardrail Persistence - [PR #20307](https://github.com/BerriAI/litellm/pull/20307)
- UI - Model Info Page: Fix Input and Output Labels - [PR #20462](https://github.com/BerriAI/litellm/pull/20462)
- UI - Model Page: Column Resizing on Smaller Screens - [PR #20599](https://github.com/BerriAI/litellm/pull/20599)
- Fix `/key/list` `user_id` Empty String Edge Case - [PR #20623](https://github.com/BerriAI/litellm/pull/20623)
- Add array type checks for model, agent, and MCP hub data to prevent UI crashes - [PR #20469](https://github.com/BerriAI/litellm/pull/20469)
- Fix unique constraint on daily tables + logging when updates fail - [PR #20394](https://github.com/BerriAI/litellm/pull/20394)

---

## Logging / Guardrail / Prompt Management Integrations

#### Bug Fixes (3 fixes)

- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Fix Langfuse OTEL trace export failing when spans contain null attributes - [PR #20382](https://github.com/BerriAI/litellm/pull/20382)

- **[Prometheus](../../docs/proxy/logging#prometheus)**
    - Fix incorrect failure metrics labels causing miscounted error rates - [PR #20152](https://github.com/BerriAI/litellm/pull/20152)

- **[Slack Alerts](../../docs/proxy/alerting)**
    - Fix Slack alert delivery failing for certain budget threshold configurations - [PR #20257](https://github.com/BerriAI/litellm/pull/20257)

#### Guardrails (7 updates)

- **Custom Code Guardrails**
    - Add HTTP support to custom code guardrails + Unified guardrails for MCP + Agent guardrail support - [PR #20619](https://github.com/BerriAI/litellm/pull/20619)
    - Custom Code Guardrails UI Playground - [PR #20377](https://github.com/BerriAI/litellm/pull/20377)

- **Team-Based Guardrails**
    - Implement team-based isolation guardrails management - [PR #20318](https://github.com/BerriAI/litellm/pull/20318)

- **[OpenAI Moderations](../../docs/apply_guardrail)**
    - Ensure OpenAI Moderations Guard works with OpenAI Embeddings - [PR #20523](https://github.com/BerriAI/litellm/pull/20523)

- **[GraySwan / Cygnal](../../docs/apply_guardrail)**
    - Fix fail-open for GraySwan and pass metadata to Cygnal API endpoint - [PR #19837](https://github.com/BerriAI/litellm/pull/19837)

- **General**
    - Check for `model_response_choices` before guardrail input - [PR #19784](https://github.com/BerriAI/litellm/pull/19784)
    - Preserve streaming content on guardrail-sampled chunks - [PR #20027](https://github.com/BerriAI/litellm/pull/20027)

---

## Spend Tracking, Budgets and Rate Limiting

- **Support 0 cost models** - Allow zero-cost model entries for internal/free-tier models - [PR #20249](https://github.com/BerriAI/litellm/pull/20249)

---

## MCP Gateway (9 updates)

- **MCP Semantic Filtering** - Filter MCP tools using semantic similarity to reduce tool sprawl for LLM calls - [PR #20296](https://github.com/BerriAI/litellm/pull/20296), [PR #20316](https://github.com/BerriAI/litellm/pull/20316)
- **UI - MCP Semantic Filtering** - Add support for MCP Semantic Filtering configuration on UI - [PR #20454](https://github.com/BerriAI/litellm/pull/20454)
- **MCP IP-Based Access Control** - Set MCP servers as private/public available on internet with IP-based restrictions - [PR #20607](https://github.com/BerriAI/litellm/pull/20607), [PR #20620](https://github.com/BerriAI/litellm/pull/20620)
- **Fix MCP "Session not found" error** on VSCode reconnect - [PR #20298](https://github.com/BerriAI/litellm/pull/20298)
- **Fix OAuth2 'Capabilities: none' bug** for upstream MCP servers - [PR #20602](https://github.com/BerriAI/litellm/pull/20602)
- **Include Config Defined Search Tools** in `/search_tools/list` - [PR #20371](https://github.com/BerriAI/litellm/pull/20371)
- **UI - Search Tools**: Show Config Defined Search Tools - [PR #20436](https://github.com/BerriAI/litellm/pull/20436)
- **Ensure MCP permissions are enforced** when using JWT Auth - [PR #20383](https://github.com/BerriAI/litellm/pull/20383)
- **Fix `gcs_bucket_name` not being passed** correctly for MCP server storage configuration - [PR #20491](https://github.com/BerriAI/litellm/pull/20491)

---

## Performance / Loadbalancing / Reliability improvements (14 improvements)

- **Prometheus ~40% CPU reduction** - Parallelize budget metrics, fix caching bug, reduce CPU usage - [PR #20544](https://github.com/BerriAI/litellm/pull/20544)
- **Prevent closed client errors** by reverting httpx client caching - [PR #20025](https://github.com/BerriAI/litellm/pull/20025)
- **Avoid unnecessary Router creation** when no models or search tools are configured - [PR #20661](https://github.com/BerriAI/litellm/pull/20661)
- **Optimize `wrapper_async`** with `CallTypes` caching and reduced lookups - [PR #20204](https://github.com/BerriAI/litellm/pull/20204)
- **Cache `_get_relevant_args_to_use_for_logging()`** at module level - [PR #20077](https://github.com/BerriAI/litellm/pull/20077)
- **LRU cache for `normalize_request_route`** - [PR #19812](https://github.com/BerriAI/litellm/pull/19812)
- **Optimize `get_standard_logging_metadata`** with set intersection - [PR #19685](https://github.com/BerriAI/litellm/pull/19685)
- **Early-exit guards in `completion_cost`** for unused features - [PR #20020](https://github.com/BerriAI/litellm/pull/20020)
- **Optimize `get_litellm_params`** with sparse kwargs extraction - [PR #19884](https://github.com/BerriAI/litellm/pull/19884)
- **Guard debug log f-strings** and remove redundant dict copies - [PR #19961](https://github.com/BerriAI/litellm/pull/19961)
- **Replace enum construction with frozenset lookup** - [PR #20302](https://github.com/BerriAI/litellm/pull/20302)
- **Guard debug f-string in `update_environment_variables`** - [PR #20360](https://github.com/BerriAI/litellm/pull/20360)
- **Warn when budget lookup fails** to surface silent caching misses - [PR #20545](https://github.com/BerriAI/litellm/pull/20545)
- **Add INFO-level session reuse logging** per request for better observability - [PR #20597](https://github.com/BerriAI/litellm/pull/20597)

---

## Database Changes

### Schema Updates

| Table | Change Type | Description | PR | Migration |
| ----- | ----------- | ----------- | -- | --------- |
| `LiteLLM_TeamTable` | New Column | Added `allow_team_guardrail_config` boolean field for team-based guardrail isolation | [PR #20318](https://github.com/BerriAI/litellm/pull/20318) | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260205091235_allow_team_guardrail_config/migration.sql) |
| `LiteLLM_DeletedTeamTable` | New Column | Added `allow_team_guardrail_config` boolean field | [PR #20318](https://github.com/BerriAI/litellm/pull/20318) | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260205091235_allow_team_guardrail_config/migration.sql) |
| `LiteLLM_TeamTable` | New Column | Added `soft_budget` (double precision) for soft budget alerting | [PR #20530](https://github.com/BerriAI/litellm/pull/20530) | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260205144610_add_soft_budget_to_team_table/migration.sql) |
| `LiteLLM_DeletedTeamTable` | New Column | Added `soft_budget` (double precision) | [PR #20653](https://github.com/BerriAI/litellm/pull/20653) | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260207110613_add_soft_budget_to_deleted_teams_table/migration.sql) |
| `LiteLLM_MCPServerTable` | New Column | Added `available_on_public_internet` boolean for MCP IP-based access control | [PR #20607](https://github.com/BerriAI/litellm/pull/20607) | [Migration](https://github.com/BerriAI/litellm/blob/main/litellm-proxy-extras/litellm_proxy_extras/migrations/20260207093506_add_available_on_public_internet_to_mcp_servers/migration.sql) |

---

## Documentation Updates (14 updates)

- Add FAQ for setting up and verifying LITELLM_LICENSE - [PR #20284](https://github.com/BerriAI/litellm/pull/20284)
- Model request tags documentation - [PR #20290](https://github.com/BerriAI/litellm/pull/20290)
- Add Prisma migration troubleshooting guide - [PR #20300](https://github.com/BerriAI/litellm/pull/20300)
- MCP Semantic Filtering documentation - [PR #20316](https://github.com/BerriAI/litellm/pull/20316)
- Add CopilotKit SDK doc as supported agents SDK - [PR #20396](https://github.com/BerriAI/litellm/pull/20396)
- Add documentation for Nova Sonic - [PR #20320](https://github.com/BerriAI/litellm/pull/20320)
- Update Vertex AI Text to Speech doc to show use of audio - [PR #20255](https://github.com/BerriAI/litellm/pull/20255)
- Improve Okta SSO setup guide with step-by-step instructions - [PR #20353](https://github.com/BerriAI/litellm/pull/20353)
- Langfuse doc update - [PR #20443](https://github.com/BerriAI/litellm/pull/20443)
- Expose MCPs on public internet documentation - [PR #20626](https://github.com/BerriAI/litellm/pull/20626)
- Add blog post: Achieving Sub-Millisecond Proxy Overhead - [PR #20309](https://github.com/BerriAI/litellm/pull/20309)
- Add blog post about litellm-observatory - [PR #20622](https://github.com/BerriAI/litellm/pull/20622)
- Update Opus 4.6 blog with adaptive thinking - [PR #20637](https://github.com/BerriAI/litellm/pull/20637)
- `gpt-5-search-api` docs clarifications - [PR #20512](https://github.com/BerriAI/litellm/pull/20512)

---

## New Contributors
* @Quentin-M made their first contribution in [PR #19818](https://github.com/BerriAI/litellm/pull/19818)
* @amirzaushnizer made their first contribution in [PR #20235](https://github.com/BerriAI/litellm/pull/20235)
* @cscguochang made their first contribution in [PR #20214](https://github.com/BerriAI/litellm/pull/20214)
* @krauckbot made their first contribution in [PR #20273](https://github.com/BerriAI/litellm/pull/20273)
* @agrattan0820 made their first contribution in [PR #19784](https://github.com/BerriAI/litellm/pull/19784)
* @nina-hu made their first contribution in [PR #20472](https://github.com/BerriAI/litellm/pull/20472)
* @swayambhu94 made their first contribution in [PR #20469](https://github.com/BerriAI/litellm/pull/20469)
* @ssadedin made their first contribution in [PR #20566](https://github.com/BerriAI/litellm/pull/20566)

---

## Full Changelog
[v1.81.6-nightly...v1.81.9](https://github.com/BerriAI/litellm/compare/v1.81.6-nightly...v1.81.9)
