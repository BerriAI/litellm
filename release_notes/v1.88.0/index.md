---
title: "v1.88.0 - Claude Opus 4.8, MCP Access-Group Authorization & Typed OpenTelemetry"
slug: "v1-88-0"
date: 2026-06-04T18:45:10
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

import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="docker" label="Docker">

```bash
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
docker.litellm.ai/berriai/litellm:1.88.0
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.88.0
```

</TabItem>
</Tabs>

## Key Highlights

`v1.88.0` is the stable release, graduated from the `v1.88.0` release candidates.

- **Claude Opus 4.8** is supported across Anthropic, Bedrock (including `global` / `us` / `eu` / `au` regional routes), Azure AI, and Vertex, at 1M-token context with adaptive thinking and `output_config` goal mode.
- **MCP access-group authorization** was reworked end to end: key and team access groups now resolve to MCP servers, grants are additive with opt-in member assignment, and clients can route through stateful or stateless sessions by session id.
- **Typed OpenTelemetry instrumentation** lands a semconv-aligned span model that carries `team_metadata`, `http.route`, and model names on inference spans.
- **Streaming is ~30% cheaper per chunk** on the Anthropic and Bedrock hot path.
- **Agent-to-agent (A2A)** gains well-known agent-card discovery and a LangGraph Platform mode.

## New Models / Updated Models

#### New Model Support (Claude Opus 4.8 across 9 provider routes)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| --- | --- | --- | --- | --- | --- |
| Anthropic | `claude-opus-4-8` | 1,000,000 | $5.00 | $25.00 | Vision, function calling, prompt caching, reasoning (adaptive + max/xhigh effort), PDF input, computer use, response schema, tool choice, output_config |
| Vertex AI | `vertex_ai/claude-opus-4-8` | 1,000,000 | $5.00 | $25.00 | Same as Anthropic direct |
| Azure AI | `azure_ai/claude-opus-4-8` | 200,000 | $5.00 | $25.00 | Same as Anthropic direct |
| Bedrock | `anthropic.claude-opus-4-8` (+ `global.` / `us.` / `eu.` / `au.` routes) | 1,000,000 | $5.00 | $25.00 | Same, plus native structured output |

Plus a reasoning-effort flag cleanup across existing Claude catalog entries: `supports_minimal_reasoning_effort` removed where unsupported, `supports_max_reasoning_effort` normalized, and a new `bedrock_output_config_effort_ceiling` (`high` / `xhigh` / `max`) field on Bedrock entries - [PR #29238](https://github.com/BerriAI/litellm/pull/29238).

#### Features

- **[Anthropic](https://docs.litellm.ai/docs/providers/anthropic)**
    - Add Claude Opus 4.8 and prune stale reasoning-effort flags - [PR #29238](https://github.com/BerriAI/litellm/pull/29238)
- **[Bedrock](https://docs.litellm.ai/docs/providers/bedrock)**
    - Claude Code goal mode via `output_config` for Bedrock Opus - [PR #28898](https://github.com/BerriAI/litellm/pull/28898)
    - Support tool search results and chat annotations - [PR #29120](https://github.com/BerriAI/litellm/pull/29120)

#### Bug Fixes

- **[Anthropic](https://docs.litellm.ai/docs/providers/anthropic)**
    - Stop injecting unsupported `output_config.effort=xhigh` for Claude Code on Sonnet/Opus 4.6 - [PR #29304](https://github.com/BerriAI/litellm/pull/29304)
- **[Vertex AI](https://docs.litellm.ai/docs/providers/vertex)**
    - Strip `output_config.effort` for Vertex Claude models that reject it (Haiku 4.5) - [PR #29585](https://github.com/BerriAI/litellm/pull/29585)
- **[Bedrock](https://docs.litellm.ai/docs/providers/bedrock)**
    - Align `toolUse` / `toolSpec` names and allow hyphens - [PR #28874](https://github.com/BerriAI/litellm/pull/28874)
- **[Azure](https://docs.litellm.ai/docs/providers/azure)**
    - Preserve AD token refresh in the v1 OpenAI client path - [PR #28627](https://github.com/BerriAI/litellm/pull/28627)
- **[OpenAI](https://docs.litellm.ai/docs/providers/openai)**
    - Fix the double provider-prefix bug on model names - [PR #28661](https://github.com/BerriAI/litellm/pull/28661)
- **General**
    - Hydrate wildcard model-discovery credentials - [PR #28284](https://github.com/BerriAI/litellm/pull/28284)

## LLM API Endpoints

#### Features

- **[Realtime API](https://docs.litellm.ai/docs/realtime)**
    - Tool calling for the Gemini and Vertex AI live API - [PR #26590](https://github.com/BerriAI/litellm/pull/26590)
- **[A2A](https://docs.litellm.ai/docs/a2a)**
    - Well-known agent-card discovery and LangGraph Platform mode - [PR #28860](https://github.com/BerriAI/litellm/pull/28860)
- **Context Management**
    - `compact_20260112` polyfill so non-Anthropic providers get context compaction - [PR #28868](https://github.com/BerriAI/litellm/pull/28868)
- **Video**
    - Vertex Veo video edit, using DB credentials in the video handlers - [PR #29098](https://github.com/BerriAI/litellm/pull/29098)
- **Pass-through**
    - Extend `passthrough_managed_object_ids` to Azure - [PR #29160](https://github.com/BerriAI/litellm/pull/29160)

#### Bugs

- **[Realtime API](https://docs.litellm.ai/docs/realtime)**
    - Send TEXT frames and a valid guardrail `session.update` - [PR #28848](https://github.com/BerriAI/litellm/pull/28848)
- **[Moderations](https://docs.litellm.ai/docs/moderation)**
    - Wire streaming flags through to the unified dispatcher - [PR #27324](https://github.com/BerriAI/litellm/pull/27324)
- **[Batches](https://docs.litellm.ai/docs/batches)**
    - Strip LiteLLM policy tracking from OpenAI batch metadata - [PR #28425](https://github.com/BerriAI/litellm/pull/28425)
    - Map the stripped batch `body.model` back to the proxy alias for auth - [PR #29264](https://github.com/BerriAI/litellm/pull/29264)
- **Vector Stores**
    - Restrict vector store index create/delete to proxy admins - [PR #29202](https://github.com/BerriAI/litellm/pull/29202)
- **Video**
    - Resolve managed video model ids for auth - [PR #29545](https://github.com/BerriAI/litellm/pull/29545)
- **Pass-through**
    - Bedrock Knowledge Base pass-through: preserve SigV4 headers and the signed request body - [PR #27526](https://github.com/BerriAI/litellm/pull/27526)
    - Enforce `allowed_passthrough_routes` for `auth=true` pass-through - [PR #29256](https://github.com/BerriAI/litellm/pull/29256)
    - De-duplicate pass-through endpoint logs - [PR #29598](https://github.com/BerriAI/litellm/pull/29598)
    - Match pass-through registry routes bare-to-bare when `SERVER_ROOT_PATH` is set, fixing pass-through 404s - [PR #29658](https://github.com/BerriAI/litellm/pull/29658)

## Management Endpoints / UI

#### Features

- **Virtual Keys & Teams**
    - Expose `keys_count` on `/v2/team/list` and wire the UI Resources badge - [PR #28502](https://github.com/BerriAI/litellm/pull/28502)
    - Allow team members to create keys on org-scoped teams - [PR #29310](https://github.com/BerriAI/litellm/pull/29310)
    - Exempt UI and CLI session tokens from team-key budget ceilings, hardened so custom `default_key_generate_params` cannot re-impose them - [PR #29612](https://github.com/BerriAI/litellm/pull/29612), [PR #29639](https://github.com/BerriAI/litellm/pull/29639)
    - Record ownership for service-account keys, plus a Prisma JSON serialization fix - [PR #28990](https://github.com/BerriAI/litellm/pull/28990)
- **Deployment**
    - Helm: split per-component ServiceAccounts for gateway, backend, and UI - [PR #28712](https://github.com/BerriAI/litellm/pull/28712)
    - Enterprise: `RESEND_FROM_EMAIL` for self-hosted Resend sends - [PR #28830](https://github.com/BerriAI/litellm/pull/28830)

#### Bugs

- **Virtual Keys & Teams**
    - Refresh the team cache on `team_model_add` / `team_model_delete` - [PR #28683](https://github.com/BerriAI/litellm/pull/28683)
    - Keep the `team_alias` cache in sync on `_cache_team_object` writes - [PR #28737](https://github.com/BerriAI/litellm/pull/28737)
    - Fix spend-logs v2 route permissions - [PR #28705](https://github.com/BerriAI/litellm/pull/28705)
    - Normalize the Bearer prefix in the safe-hash helper - [PR #29343](https://github.com/BerriAI/litellm/pull/29343)
- **UI**
    - Allow clearing custom pricing on wildcard models - [PR #28719](https://github.com/BerriAI/litellm/pull/28719)
    - Stop `vertex_ai-anthropic_models` from leaking into the Anthropic dropdown - [PR #28723](https://github.com/BerriAI/litellm/pull/28723)
    - Route API Reference back to the query-param page - [PR #28726](https://github.com/BerriAI/litellm/pull/28726)
    - Show 2-decimal precision for `max_budget` on the key overview - [PR #28809](https://github.com/BerriAI/litellm/pull/28809)
    - Break the logout redirect loop across dev and proxy origins - [PR #29360](https://github.com/BerriAI/litellm/pull/29360)
    - Internal refactors: extract auth state into `AuthContext`, remove dead App Router scaffolding - [PR #28910](https://github.com/BerriAI/litellm/pull/28910), [PR #28891](https://github.com/BerriAI/litellm/pull/28891)

## AI Integrations

### Logging

- **[DataDog](https://docs.litellm.ai/docs/proxy/logging#datadog)**
    - Drain the cost-management queue and add an opt-in FinOps tag allowlist - [PR #28487](https://github.com/BerriAI/litellm/pull/28487)
- **Galileo**
    - Support the hosted v2 spans API and string output extraction - [PR #28771](https://github.com/BerriAI/litellm/pull/28771)
- **[OpenTelemetry](https://docs.litellm.ai/docs/proxy/logging#opentelemetry)**
    - Typed, semconv-aligned instrumentation - [PR #28909](https://github.com/BerriAI/litellm/pull/28909)
    - Add `team_metadata`, `http.route`, and model names to inference spans - [PR #29319](https://github.com/BerriAI/litellm/pull/29319)
    - Export the SERVER span on management-endpoint success without an `http_request` - [PR #28794](https://github.com/BerriAI/litellm/pull/28794)
    - Link pass-through success spans to the SERVER root span - [PR #29315](https://github.com/BerriAI/litellm/pull/29315)
- **General**
    - Exclude `proxy_server_request` from its own body snapshot - [PR #28618](https://github.com/BerriAI/litellm/pull/28618)
    - Fix duplicate Claude Code traces - [PR #29311](https://github.com/BerriAI/litellm/pull/29311)

### Guardrails

- **General**
    - Return HTTP 400 for LiteLLM content-filter blocks - [PR #28418](https://github.com/BerriAI/litellm/pull/28418)
    - Wire `apply_guardrail` into proxy logging callbacks - [PR #28970](https://github.com/BerriAI/litellm/pull/28970)
    - Persist `disable_global_guardrails` on keys - [PR #29233](https://github.com/BerriAI/litellm/pull/29233)

## Spend Tracking, Budgets and Rate Limiting

- **Cost Tracking** — [OpenAI](https://docs.litellm.ai/docs/providers/openai) regional-processing cost uplift for EU/US data residency - [PR #28626](https://github.com/BerriAI/litellm/pull/28626)
- **Rate Limiting** — Cap the no-`max_tokens` TPM floor at the smallest configured limit (v3 limiter) - [PR #28805](https://github.com/BerriAI/litellm/pull/28805)
- **Budgets** — Enforce tag budgets for key-level tags - [PR #29108](https://github.com/BerriAI/litellm/pull/29108)
- **Budgets** — Enforce deployment budgets for dynamically added models - [PR #29273](https://github.com/BerriAI/litellm/pull/29273)
- **Budgets** — `reset_budget` writes only `{spend, budget_reset_at}` and stops pre-zeroing the counter - [PR #29358](https://github.com/BerriAI/litellm/pull/29358)

## MCP Gateway

- **Session Routing** — Stateless and stateful clients via session-id routing - [PR #26857](https://github.com/BerriAI/litellm/pull/26857)
- **Access Groups** — Additive key access-group grants with opt-in member assignment - [PR #29313](https://github.com/BerriAI/litellm/pull/29313)
- **Access Groups** — Resolve team `access_group_ids` to MCP servers - [PR #28997](https://github.com/BerriAI/litellm/pull/28997)
- **Access Groups** — Resolve key `access_group_ids` to MCP servers (ungated) - [PR #29195](https://github.com/BerriAI/litellm/pull/29195)
- **Access Groups** — Extend the key access-group union to MCP servers - [PR #28890](https://github.com/BerriAI/litellm/pull/28890)
- **Discovery** — Allow `llm_api_routes` virtual keys to list MCP servers - [PR #28442](https://github.com/BerriAI/litellm/pull/28442)
- **Server CRUD** — Preserve `source_url` on `GET /v1/mcp/server` list responses - [PR #29249](https://github.com/BerriAI/litellm/pull/29249)
- **Server CRUD** — Preserve omitted fields on `PUT /v1/mcp/server` partial updates - [PR #29253](https://github.com/BerriAI/litellm/pull/29253)
- **Virtual Keys** — Ignore stale ids on key save - [PR #29128](https://github.com/BerriAI/litellm/pull/29128)

## Performance / Loadbalancing / Reliability improvements

- **Streaming hot path** — ~30% lower per-chunk overhead on the Anthropic and Bedrock streaming path - [PR #28720](https://github.com/BerriAI/litellm/pull/28720)
- **Docker** — Use system Node in the componentized builders and retry `apk add` - [PR #28888](https://github.com/BerriAI/litellm/pull/28888)
- **Dependencies** — Routine dependency bumps, including a Starlette bad-host fix - [PR #29208](https://github.com/BerriAI/litellm/pull/29208), [PR #29373](https://github.com/BerriAI/litellm/pull/29373)

## Documentation Updates

- Hand-written `CLAUDE.md`; remove `AGENTS.md` and point `GEMINI.md` at it - [PR #29252](https://github.com/BerriAI/litellm/pull/29252)
- Agent guidance: require consent before writing new third-party names - [PR #28908](https://github.com/BerriAI/litellm/pull/28908)
- Cookbook: bump the Go directive to 1.26.3 in the gollem example - [PR #29234](https://github.com/BerriAI/litellm/pull/29234)

## General Proxy Improvements

Testing, CI & build hardening:

- UI e2e coverage across roles and flows — Team-BYOK add-model, Router fallback, MCP add-server, AI Hub make-public, Team Admin, Internal User / Viewer, logout and navbar identity - [PR #29068](https://github.com/BerriAI/litellm/pull/29068), [PR #29069](https://github.com/BerriAI/litellm/pull/29069), [PR #29070](https://github.com/BerriAI/litellm/pull/29070), [PR #29071](https://github.com/BerriAI/litellm/pull/29071), [PR #29072](https://github.com/BerriAI/litellm/pull/29072), [PR #29074](https://github.com/BerriAI/litellm/pull/29074), [PR #29075](https://github.com/BerriAI/litellm/pull/29075), [PR #29076](https://github.com/BerriAI/litellm/pull/29076), [PR #29077](https://github.com/BerriAI/litellm/pull/29077), [PR #29080](https://github.com/BerriAI/litellm/pull/29080), [PR #29083](https://github.com/BerriAI/litellm/pull/29083), [PR #28652](https://github.com/BerriAI/litellm/pull/28652)
- Pass-through `SERVER_ROOT_PATH` login-redirect trailing-slash e2e - [PR #29369](https://github.com/BerriAI/litellm/pull/29369)
- Behavior-pinning harnesses for `proxy_server.py` - [PR #28827](https://github.com/BerriAI/litellm/pull/28827), [PR #29309](https://github.com/BerriAI/litellm/pull/29309)
- Deterministic Redis cassette replay and live Google OAuth token minting for VCR - [PR #28826](https://github.com/BerriAI/litellm/pull/28826), [PR #29229](https://github.com/BerriAI/litellm/pull/29229)
- Reasoning-effort grid test covering Claude Opus 4.8 across provider routes - [PR #29327](https://github.com/BerriAI/litellm/pull/29327)
- Bedrock CI account moves and restore - [PR #28728](https://github.com/BerriAI/litellm/pull/28728), [PR #29326](https://github.com/BerriAI/litellm/pull/29326), [PR #29245](https://github.com/BerriAI/litellm/pull/29245)
- Keep `litellm_internal_staging` green - [PR #29344](https://github.com/BerriAI/litellm/pull/29344)
- Regenerate the admin-ui static export with `trailingSlash: true` - [PR #28112](https://github.com/BerriAI/litellm/pull/28112)

### PR roll-up by ownership area

PRs by ownership area (total: 97)
  - Other (CI / tests / build hardening): 23
  - UI / Auth & Management: 18
  - LLM API Endpoints: 15
  - MCP: 9
  - Models & Providers: 9
  - Logging: 8
  - Spend / Budgets / Rate Limits: 5
  - Performance: 4
  - Documentation: 3
  - Guardrails: 3

## Release candidate changelog (rc.1 → rc.2 → rc.3)

Almost everything above shipped in **rc.1**. The later candidates are small, targeted patches cut by cherry-pick.

**rc.2** added six fixes:

- Resolve managed video model ids for auth - [PR #29545](https://github.com/BerriAI/litellm/pull/29545)
- Allow team members to create keys on org-scoped teams - [PR #29310](https://github.com/BerriAI/litellm/pull/29310)
- Strip `output_config.effort` for Vertex Claude Haiku 4.5 - [PR #29585](https://github.com/BerriAI/litellm/pull/29585)
- De-duplicate pass-through endpoint logs - [PR #29598](https://github.com/BerriAI/litellm/pull/29598)
- Exempt UI/CLI session tokens from team-key budget ceilings - [PR #29612](https://github.com/BerriAI/litellm/pull/29612)
- Harden that exemption against custom `default_key_generate_params` - [PR #29639](https://github.com/BerriAI/litellm/pull/29639)

**rc.3** added one fix:

- Match pass-through registry routes bare-to-bare when `SERVER_ROOT_PATH` is set, fixing pass-through 404s - [PR #29658](https://github.com/BerriAI/litellm/pull/29658)

## New Contributors

No new contributors this release; all 11 authors are returning contributors.

**Full Changelog**: https://github.com/BerriAI/litellm/compare/v1.87.0...v1.88.0

---

## 06/04/2026 (`v1.88.0`)

* New Models / Updated Models: 9
* LLM API Endpoints: 15
* Management Endpoints / UI: 18
* AI Integrations (Logging / Guardrails): 11
* Spend Tracking, Budgets and Rate Limiting: 5
* MCP Gateway: 9
* Performance / Loadbalancing / Reliability improvements: 4
* General Proxy Improvements (testing / CI / build): 23
* Documentation Updates: 3

Total: 97 PRs
