---
title: "v1.89.0 - Claude Fable 5, A2A Agent Providers & MCP Per-Server Controls"
slug: "v1-89-0"
date: 2026-06-10T11:04:00
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
docker.litellm.ai/berriai/litellm:v1.89.0
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.89.0
```

</TabItem>
</Tabs>

## Key Highlights

`v1.89.0` builds on [`v1.88.0`](/release_notes/v1.88.0/v1-88-0).

- **Claude Fable 5** is supported across Anthropic, Bedrock, Azure AI, and Vertex at 1M-token context with adaptive thinking and computer use.
- **Agent-to-agent (A2A)** gains two new agent providers - watsonx Orchestrate and LangFlow (with A2A session bridging) - plus OAuth M2M for Databricks Apps agents.
- **MCP gateway** adds per-server environment variables with global and per-user scopes, per-server RPM rate limiting for keys and teams, OAuth passthrough with issuer-scoped JWT auth, and `oauth2_flow` persistence on server registration.
- **Observability** lands OpenInference rendering parity for Arize/Phoenix (tool calls, cost, passthrough I/O, sessions, multimodal, cache tokens), MCP semantic conventions on the typed OTel v2 spans, and a Galileo logger that uses the ingest-traces API.
- **New search and transcription providers** - APISerpent, You.com, and Soniox - join the gateway, alongside the dashboard's migration to fully typed, OpenAPI-generated API clients.

---

### MCP Credential Store

<Image img={require('../../img/release_notes/mcp_credential_store.png')} style={{ width: '800px', height: 'auto' }} />

<br/>

This release lets you securely store per-server credentials for MCP servers directly on the gateway. Define variables once on a server, scoped either as **Instance** (shared across all users) or **Per-user** (each user supplies their own value), and reference them in static headers or authentication using `${VAR_NAME}` syntax (for example, `${DB_PROTOCOL}://${CORP_USERNAME}:${CORP_PASSWORD}@${DB_HOSTNAME}`), letting each user connect their own identity.

## New Providers and Endpoints

### New Providers (3 new providers)

| Provider                  | Supported LiteLLM Endpoints | Description                           |
| ------------------------- | --------------------------- | ------------------------------------- |
| APISerpent (`apiserpent`) | Search                      | Web search and deep-search API        |
| You.com (`you_com`)       | Search                      | You.com web search API                |
| Soniox (`soniox`)         | Audio Transcription         | Async speech-to-text (`stt-async-v4`) |

## New Models / Updated Models

#### New Model Support (selected)

| Provider       | Model                                                           | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features                                                                  |
| -------------- | --------------------------------------------------------------- | -------------- | ------------------- | -------------------- | ------------------------------------------------------------------------- |
| Anthropic      | `claude-fable-5`                                                | 1,000,000      | $10.00              | $50.00               | Adaptive thinking, computer use, function calling, prompt caching, vision |
| Vertex AI      | `vertex_ai/claude-fable-5`                                      | 1,000,000      | $10.00              | $50.00               | Same as Anthropic direct                                                  |
| Azure AI       | `azure_ai/claude-fable-5`                                       | 1,000,000      | $10.00              | $50.00               | Same as Anthropic direct                                                  |
| Bedrock        | `anthropic.claude-fable-5` (+ `global.` / `us.` / `eu.` routes) | 1,000,000      | $10.00              | $50.00               | Same as Anthropic direct                                                  |
| Bedrock Mantle | `bedrock_mantle/openai.gpt-5.5`                                 | 272,000        | $5.50               | $33.00               | Responses API, reasoning, function calling, prompt caching                |
| Bedrock Mantle | `bedrock_mantle/openai.gpt-5.4`                                 | 272,000        | $2.75               | $16.50               | Responses API, reasoning, function calling, prompt caching                |
| Azure AI       | `azure_ai/kimi-k2.6`                                            | 262,144        | $0.95               | $4.00                | Reasoning, vision, function calling, tool choice                          |
| MiniMax        | `minimax/MiniMax-M3`                                            | 512,000        | $0.60               | $2.40                | Reasoning, prompt caching, function calling                               |
| Inception      | `inception/mercury-2` (+ `mercury-edit-2`)                      | 128,000        | $0.25               | $0.75                | Function calling, prompt caching, response schema                         |

Additional model-map additions: fal.ai Nano Banana and Gemini 2.5 Flash Image generation - [PR #29798](https://github.com/BerriAI/litellm/pull/29798); `mistral/ministral-8b-latest` - [PR #29453](https://github.com/BerriAI/litellm/pull/29453); a batch of new Snowflake Cortex model entries (Claude, GPT, Llama, embeddings); `vertex_ai/google/gemma-4-26b-a4b-it-maas`; APISerpent, You.com, and Soniox catalog entries; and a `jp.` regional route for Claude Opus 4.7.

#### Features

- **[Anthropic](../../docs/providers/anthropic)**
  - Route future Claude models to the Anthropic provider via pattern matching - [PR #29239](https://github.com/BerriAI/litellm/pull/29239)
  - Route Claude Opus 4.8 through adaptive thinking - [PR #29702](https://github.com/BerriAI/litellm/pull/29702)
  - Emit a thinking block for `reasoning_content`-only streaming chunks in the Anthropic adapter - [PR #29600](https://github.com/BerriAI/litellm/pull/29600)
  - Inline legacy `$ref` defs in tool schemas (Anthropic and Fireworks) - [PR #28646](https://github.com/BerriAI/litellm/pull/28646)
- **[Gemini](../../docs/providers/gemini)**
  - Support `googleSearch` with server-side tools and `googleMaps` JSON schema - [PR #29582](https://github.com/BerriAI/litellm/pull/29582)
  - Use GA event names for Pipecat 1.3.x compatibility on Gemini realtime - [PR #29662](https://github.com/BerriAI/litellm/pull/29662)
- **[Vertex AI](../../docs/providers/vertex)**
  - Use a user-supplied `api_base` as-is for the Model Garden OpenAI-compatible path - [PR #29530](https://github.com/BerriAI/litellm/pull/29530)
  - Handle namespace tools and strip `client_metadata` for Codex compatibility on Vertex/Anthropic - [PR #29489](https://github.com/BerriAI/litellm/pull/29489)
- **[Azure AI](../../docs/providers/azure_ai)**
  - Strip tool-level extra fields on a 400 and retry - [PR #29479](https://github.com/BerriAI/litellm/pull/29479)

#### Bug Fixes

- **General**
  - Return a 400 (not 500) on Anthropic context overflow, and seed identity on failed auth - [PR #29848](https://github.com/BerriAI/litellm/pull/29848)
  - Omit the OpenAI `[DONE]` sentinel on google-genai `streamGenerateContent` - [PR #29426](https://github.com/BerriAI/litellm/pull/29426)

## LLM API Endpoints

#### Features

- **[Batches](../../docs/batches)**
  - Skip unnecessary batch input-file reads - [PR #29114](https://github.com/BerriAI/litellm/pull/29114)
  - Resolve credentials correctly when cancelling a managed batch - [PR #29734](https://github.com/BerriAI/litellm/pull/29734)
- **Vector Stores**
  - Resolve vector-store file-list credentials from team deployments - [PR #29739](https://github.com/BerriAI/litellm/pull/29739)
  - Support an engines URL for Vertex AI Search - [PR #27885](https://github.com/BerriAI/litellm/pull/27885)
  - Forward per-request params to Vertex AI Search - [PR #29459](https://github.com/BerriAI/litellm/pull/29459)
- **Realtime**
  - Track realtime audio token cost - [PR #29722](https://github.com/BerriAI/litellm/pull/29722)
  - Allow null transcripts in stream logging payloads - [PR #29625](https://github.com/BerriAI/litellm/pull/29625)
  - WebSocket connection improvements - [PR #29563](https://github.com/BerriAI/litellm/pull/29563)

#### Agents (A2A)

- watsonx Orchestrate agent provider - [PR #29410](https://github.com/BerriAI/litellm/pull/29410)
- LangFlow agent provider with A2A session bridging - [PR #28963](https://github.com/BerriAI/litellm/pull/28963)
- OAuth M2M for Databricks Apps A2A agents - [PR #29586](https://github.com/BerriAI/litellm/pull/29586)
- A2A bug fixes - [PR #29566](https://github.com/BerriAI/litellm/pull/29566)

## Management Endpoints / UI

#### Features

- **Virtual Keys & Auth**
  - JWT-to-virtual-key mapping - [PR #28510](https://github.com/BerriAI/litellm/pull/28510)
  - Let internal users view search tools - [PR #29542](https://github.com/BerriAI/litellm/pull/29542)
  - Expand the all-team-models sentinel in `can_key_call_model` for batch validation - [PR #29746](https://github.com/BerriAI/litellm/pull/29746)
- **Dashboard**
  - Generate dashboard API types from the proxy OpenAPI spec - [PR #29816](https://github.com/BerriAI/litellm/pull/29816)
  - Centralize proxy base-URL resolution into a tested resolver - [PR #29793](https://github.com/BerriAI/litellm/pull/29793)
  - Route networking calls through a shared, location-pinned `apiClient` - [PR #29723](https://github.com/BerriAI/litellm/pull/29723), [PR #29806](https://github.com/BerriAI/litellm/pull/29806), [PR #29815](https://github.com/BerriAI/litellm/pull/29815)
  - Migrate ESLint to flat config and bump `eslint-config-next` to 16 - [PR #29626](https://github.com/BerriAI/litellm/pull/29626)

#### Bug Fixes

- Use the resolved DB `user_id` for spend on legacy email match (JWT) - [PR #29217](https://github.com/BerriAI/litellm/pull/29217)
- Preserve the 401 status for expired JWTs in OTel traces - [PR #29510](https://github.com/BerriAI/litellm/pull/29510)
- Stop team BYOK model-name corruption on model edit - [PR #29731](https://github.com/BerriAI/litellm/pull/29731)
- Drop a deleted team BYOK model name from `team.models` - [PR #29820](https://github.com/BerriAI/litellm/pull/29820)
- Add `default=None` to `LiteLLM_TeamMembership.litellm_budget_table` - [PR #29684](https://github.com/BerriAI/litellm/pull/29684)
- Require a new expiration when regenerating an expired key - [PR #29838](https://github.com/BerriAI/litellm/pull/29838)
- Render caller-supplied filter options in caller order (LIT-3151) - [PR #29462](https://github.com/BerriAI/litellm/pull/29462)
- Make A2A skill tags enterable and validated - [PR #29512](https://github.com/BerriAI/litellm/pull/29512)
- Persist the Tools-tab MCP OAuth token to the DB - [PR #29809](https://github.com/BerriAI/litellm/pull/29809)
- Route MCP playground auth by OAuth2 mode instead of `token_url` - [PR #29714](https://github.com/BerriAI/litellm/pull/29714)
- Stop MCP playground tool calls from sending twice - [PR #29821](https://github.com/BerriAI/litellm/pull/29821)

## AI Integrations

### Logging

- **[Arize / Phoenix](../../docs/proxy/logging)**
  - OpenInference rendering parity: tool calls, cost, passthrough I/O, session/user, multimodal, and cache tokens - [PR #28800](https://github.com/BerriAI/litellm/pull/28800)
- **[Datadog](../../docs/proxy/logging#datadog)**
  - Split oversized batches on a 413 instead of re-queueing forever - [PR #29444](https://github.com/BerriAI/litellm/pull/29444)
- **Galileo**
  - Use the ingest-traces API and the standard logging payload - [PR #29651](https://github.com/BerriAI/litellm/pull/29651)
- **OpenTelemetry**
  - Allowlist `team_metadata` sub-keys promoted to baggage - [PR #29442](https://github.com/BerriAI/litellm/pull/29442)
  - Add MCP semantic conventions to OTel v2 - [PR #29468](https://github.com/BerriAI/litellm/pull/29468)
  - Capture 401 error details in management-endpoint spans - [PR #29535](https://github.com/BerriAI/litellm/pull/29535)
  - Emit the missing MCP span attributes - [PR #29554](https://github.com/BerriAI/litellm/pull/29554)
  - Emit a guardrail span on passthrough, including when a guardrail blocks - [PR #29552](https://github.com/BerriAI/litellm/pull/29552), [PR #29470](https://github.com/BerriAI/litellm/pull/29470)

### Guardrails

- **[Sensitive Data Routing](../../docs/proxy/guardrails/quick_start)**
  - Route sensitive data to on-premise models - [PR #29531](https://github.com/BerriAI/litellm/pull/29531)

## Spend Tracking, Budgets and Rate Limiting

- Strip NUL bytes from spend-log payloads to prevent PostgreSQL `22P05` errors - [PR #29515](https://github.com/BerriAI/litellm/pull/29515)
- Scope the session-token team-key budget exemption to a caller-supplied `team_id` - [PR #29641](https://github.com/BerriAI/litellm/pull/29641)

## MCP Gateway

- Per-server environment variables with global and per-user scopes - [PR #28917](https://github.com/BerriAI/litellm/pull/28917)
- Per-MCP-server RPM rate limiting for keys and teams - [PR #29482](https://github.com/BerriAI/litellm/pull/29482)
- Support MCP OAuth passthrough and issuer-scoped JWT auth - [PR #28356](https://github.com/BerriAI/litellm/pull/28356)
- Persist `oauth2_flow` on MCP server registration - [PR #29690](https://github.com/BerriAI/litellm/pull/29690)
- Clear `allowed_tools` and tool overrides on MCP server edit - [PR #29411](https://github.com/BerriAI/litellm/pull/29411)
- Gate `/public/mcp_hub` strictly on `litellm.public_mcp_servers` - [PR #27764](https://github.com/BerriAI/litellm/pull/27764)

## Performance / Loadbalancing / Reliability improvements

- Native `/health/drain` preStop hook for graceful shutdown - [PR #29439](https://github.com/BerriAI/litellm/pull/29439)
- Disable proxy buffering on streaming SSE responses - [PR #29557](https://github.com/BerriAI/litellm/pull/29557)
- Populate `llm_provider` on internal rate-limit errors - [PR #27707](https://github.com/BerriAI/litellm/pull/27707)
- Hot-reload `.env` in dev when running with `--reload` - [PR #29783](https://github.com/BerriAI/litellm/pull/29783)
- Enable the Helm backend deployment to mount the gateway `config.yaml` - [PR #29605](https://github.com/BerriAI/litellm/pull/29605)
- Convert the AWS and GCP Terraform stacks into reusable modules - [PR #28103](https://github.com/BerriAI/litellm/pull/28103)
- Terraform GCP: abandon the SQL user on destroy - [PR #29855](https://github.com/BerriAI/litellm/pull/29855); prompt for `image_registry` in the DeployStack one-click - [PR #29852](https://github.com/BerriAI/litellm/pull/29852)
- Dependency bumps - [PR #29860](https://github.com/BerriAI/litellm/pull/29860)

## Documentation Updates

- Clarify when to create new test files - [PR #29472](https://github.com/BerriAI/litellm/pull/29472)
- Remove fixed dimensions from the README hero image - [PR #29496](https://github.com/BerriAI/litellm/pull/29496)
- CLAUDE.md nits - [PR #29504](https://github.com/BerriAI/litellm/pull/29504), [PR #29749](https://github.com/BerriAI/litellm/pull/29749)

### PR roll-up by ownership area

```
PRs by ownership area (visible, non-vehicle set; total: 101)
  - UI / Dashboard: 22
  - General Proxy (testing / CI / build): 22
  - Models & Providers: 13
  - Performance / Reliability: 10
  - Logging: 9
  - LLM API Endpoints: 8
  - MCP: 6
  - Auth & Management: 5
  - Agents (A2A): 4
  - Docs: 4
  - Spend / Budgets / Rate Limits: 2
  - Models & Providers (new providers): 3
  - Guardrails: 1
```

## New Contributors

- @someswar177 made their first contribution in https://github.com/BerriAI/litellm/pull/26585
- @trexinc made their first contribution in https://github.com/BerriAI/litellm/pull/26597
- @navnitshukla made their first contribution in https://github.com/BerriAI/litellm/pull/26609
- @tanmay958 made their first contribution in https://github.com/BerriAI/litellm/pull/27580
- @samagana made their first contribution in https://github.com/BerriAI/litellm/pull/27810
- @DrishnaTrivedi made their first contribution in https://github.com/BerriAI/litellm/pull/28330
- @brainsparker made their first contribution in https://github.com/BerriAI/litellm/pull/28370
- @icep87 made their first contribution in https://github.com/BerriAI/litellm/pull/28846
- @adriangomez24 made their first contribution in https://github.com/BerriAI/litellm/pull/29097
- @zzw-math made their first contribution in https://github.com/BerriAI/litellm/pull/29325
- @BeginnerRudy made their first contribution in https://github.com/BerriAI/litellm/pull/29392
- @danisalvaa made their first contribution in https://github.com/BerriAI/litellm/pull/29394
- @kapelame made their first contribution in https://github.com/BerriAI/litellm/pull/29412
- @Zhao73 made their first contribution in https://github.com/BerriAI/litellm/pull/29419
- @suleimanelkhoury made their first contribution in https://github.com/BerriAI/litellm/pull/29420
- @aneeshsangvikar made their first contribution in https://github.com/BerriAI/litellm/pull/29427
- @Ar-maan05 made their first contribution in https://github.com/BerriAI/litellm/pull/29483
- @kingdoooo made their first contribution in https://github.com/BerriAI/litellm/pull/29490
- @dan2k3k4 made their first contribution in https://github.com/BerriAI/litellm/pull/29508
- @yanismiraoui made their first contribution in https://github.com/BerriAI/litellm/pull/29522
- @josx made their first contribution in https://github.com/BerriAI/litellm/pull/29532
- @1qh made their first contribution in https://github.com/BerriAI/litellm/pull/29561
- @tin-berri made their first contribution in https://github.com/BerriAI/litellm/pull/29605
- @mak2508 made their first contribution in https://github.com/BerriAI/litellm/pull/29606
- @VANDRANKI made their first contribution in https://github.com/BerriAI/litellm/pull/29620
- @andrey-dubnik made their first contribution in https://github.com/BerriAI/litellm/pull/29621
- @ErRickow made their first contribution in https://github.com/BerriAI/litellm/pull/29646
- @saswatds made their first contribution in https://github.com/BerriAI/litellm/pull/29650
- @Dinesh-Girbide made their first contribution in https://github.com/BerriAI/litellm/pull/29655
- @BWAAEEEK made their first contribution in https://github.com/BerriAI/litellm/pull/29660
- @hectorc98 made their first contribution in https://github.com/BerriAI/litellm/pull/29672
- @abhay23-AI made their first contribution in https://github.com/BerriAI/litellm/pull/29779

## Full Changelog

https://github.com/BerriAI/litellm/compare/v1.88.0...v1.89.0

---

## 06/10/2026

- New Models / Updated Models: 16
- LLM API Endpoints: 12
- Management Endpoints / UI: 22
- AI Integrations (Logging / Guardrails): 10
- Spend Tracking, Budgets and Rate Limiting: 2
- MCP Gateway: 6
- Performance / Loadbalancing / Reliability improvements: 10
- General Proxy Improvements (testing / CI / build): 22
- Documentation Updates: 4
