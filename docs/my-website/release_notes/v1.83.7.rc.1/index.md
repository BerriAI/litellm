---
title: "[Preview] v1.83.7.rc.1 - Per-User MCP OAuth, Team Spend Logs RBAC"
slug: "v1-83-7-rc-1"
date: 2026-04-12T00:00:00
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
    image_url: https://media.licdn.com/dms/image/v2/D5603AQHt1t9Z4BJ6Gw/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1724453682340?e=1772064000&v=beta&t=VXdmr13rsNB05wyA2F1TENOB5UuDHUZ0FCHTolNyR5M
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
docker.litellm.ai/berriai/litellm:main-v1.83.7.rc.1
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.83.7rc1
```

</TabItem>
</Tabs>

:::warning

**Breaking change — Prometheus latency histogram buckets reduced.** The default `LATENCY_BUCKETS` set has been reduced from 35 to 18 boundaries to lower Prometheus cardinality. Dashboards and PromQL queries that reference specific `le=` bucket values may stop matching. Review your alerts/dashboards before upgrading and use `LATENCY_BUCKETS` env override to restore the previous boundaries if needed — [PR #25527](https://github.com/BerriAI/litellm/pull/25527).

:::

## Key Highlights

- **Per-User MCP OAuth Tokens** — [Each end-user can now hold their own OAuth tokens for interactive MCP server flows, isolating credentials across users](../../docs/mcp)
- **Team Spend Logs RBAC** — Teams with the `/spend/logs` permission can view team-wide spend logs from the UI and API
- **Bulk Team Permissions API** — New `POST /team/permissions_bulk_update` endpoint for updating member permissions across many teams in one call
- **Azure Container Routing** — Container routing, managed container IDs, and delete-response parsing for Azure Responses API containers
- **UI E2E Test Suite** — Playwright-based end-to-end tests for proxy admin, team, and key management flows now run in CI

---

## New Models / Updated Models

#### New Model Support (14 new models)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| AWS Bedrock (GovCloud) | `bedrock/us-gov-east-1/anthropic.claude-sonnet-4-5-20250929-v1:0` | 200K | $3.30 | $16.50 | Chat, vision, tool use, prompt caching, reasoning |
| AWS Bedrock (GovCloud) | `bedrock/us-gov-west-1/anthropic.claude-sonnet-4-5-20250929-v1:0` | 200K | $3.30 | $16.50 | Chat, vision, tool use, prompt caching, reasoning |
| AWS Bedrock (GovCloud) | `us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0` | 200K | $3.30 | $16.50 | Bedrock Converse, with above-200K tier pricing |
| Baseten | `baseten/MiniMaxAI/MiniMax-M2.5` | - | $0.30 | $1.20 | Chat |
| Baseten | `baseten/nvidia/Nemotron-120B-A12B` | - | $0.30 | $0.75 | Chat |
| Baseten | `baseten/zai-org/GLM-5` | - | $0.95 | $3.15 | Chat |
| Baseten | `baseten/zai-org/GLM-4.7` | - | $0.60 | $2.20 | Chat |
| Baseten | `baseten/zai-org/GLM-4.6` | - | $0.60 | $2.20 | Chat |
| Baseten | `baseten/moonshotai/Kimi-K2.5` | - | $0.60 | $3.00 | Chat |
| Baseten | `baseten/moonshotai/Kimi-K2-Thinking` | - | $0.60 | $2.50 | Chat |
| Baseten | `baseten/moonshotai/Kimi-K2-Instruct-0905` | - | $0.60 | $2.50 | Chat |
| Baseten | `baseten/openai/gpt-oss-120b` | - | $0.10 | $0.50 | Chat |
| Baseten | `baseten/deepseek-ai/DeepSeek-V3.1` | - | $0.50 | $1.50 | Chat |
| Baseten | `baseten/deepseek-ai/DeepSeek-V3-0324` | - | $0.77 | $0.77 | Chat |

#### Features

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Update GovCloud Claude Sonnet 4.5 pricing, raise `max_tokens` to 8192, and add prompt-caching costs
    - Skip dummy `user` continue message when assistant prefix prefill is set - [PR #25419](https://github.com/BerriAI/litellm/pull/25419)
    - Avoid double-counting cache tokens in Anthropic Messages streaming usage - [PR #25517](https://github.com/BerriAI/litellm/pull/25517)
- **[Anthropic](../../docs/providers/anthropic)**
    - Support `advisor_20260301` tool type - [PR #25525](https://github.com/BerriAI/litellm/pull/25525)
- **[Google Gemini / Vertex AI](../../docs/providers/gemini)**
    - Mark applicable Gemini 2.5/3 models with `supports_service_tier`

### Bug Fixes

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Pass-through fix for Bedrock JSON body and multipart uploads - [PR #25464](https://github.com/BerriAI/litellm/pull/25464)
- **[OpenAI](../../docs/providers/openai)**
    - Mock headers in `test_completion_fine_tuned_model` to stabilize tests - [PR #25444](https://github.com/BerriAI/litellm/pull/25444)

## LLM API Endpoints

#### Features

- **[Responses API](../../docs/response_api)**
    - Containers: Azure routing, managed container IDs, and delete-response parsing - [PR #25287](https://github.com/BerriAI/litellm/pull/25287)
    - WebSocket: append `?model=` to backend WebSocket URL so model selection routes correctly - [PR #25437](https://github.com/BerriAI/litellm/pull/25437)
- **[OpenAI / Files API](../../docs/providers/openai)**
    - Add file content streaming support for OpenAI and related utilities - [PR #25450](https://github.com/BerriAI/litellm/pull/25450)
- **[A2A](../../docs/mcp)**
    - Default 60-second timeout when creating an A2A client - [PR #25514](https://github.com/BerriAI/litellm/pull/25514)

#### Bugs

- **[Responses API](../../docs/response_api)**
    - Map refusal `stop_reason` to `incomplete` status in streaming - [PR #25498](https://github.com/BerriAI/litellm/pull/25498)
    - Fix duplicate keyword argument error in Responses WebSocket path - [PR #25513](https://github.com/BerriAI/litellm/pull/25513)
- **General**
    - Ensure spend/cost logging runs when `stream=True` for web-search interception - [PR #25424](https://github.com/BerriAI/litellm/pull/25424)

## Management Endpoints / UI

#### Features

- **Teams + Organizations**
    - New `POST /team/permissions_bulk_update` endpoint for bulk permission updates across teams - [PR #25239](https://github.com/BerriAI/litellm/pull/25239)
    - Team member permission `/spend/logs` to view team-wide spend logs (UI + RBAC) - [PR #25458](https://github.com/BerriAI/litellm/pull/25458)
    - Align org and team endpoint permission checks - [PR #25554](https://github.com/BerriAI/litellm/pull/25554)
- **Virtual Keys**
    - Align `/v2/key/info` response handling with v1 - [PR #25313](https://github.com/BerriAI/litellm/pull/25313)
- **Authentication / Routing**
    - Consolidate route auth for UI and API tokens - [PR #25473](https://github.com/BerriAI/litellm/pull/25473)
    - Use parameterized query for `combined_view` token lookup - [PR #25467](https://github.com/BerriAI/litellm/pull/25467)
- **Provider Credentials**
    - Per-team / per-project credential overrides via `model_config` metadata - [PR #24438](https://github.com/BerriAI/litellm/pull/24438)
- **UI**
    - Improve browser storage handling and Dockerfile consistency - [PR #25384](https://github.com/BerriAI/litellm/pull/25384)
    - Align v1 guardrail and agent list responses with v2 field handling - [PR #25478](https://github.com/BerriAI/litellm/pull/25478)
    - Flush Tremor Tooltip timers in `user_edit_view` tests - [PR #25480](https://github.com/BerriAI/litellm/pull/25480)

#### Bugs

- Improve input validation on management endpoints - [PR #25445](https://github.com/BerriAI/litellm/pull/25445)
- Harden file path resolution in skill archive extraction - [PR #25475](https://github.com/BerriAI/litellm/pull/25475)

## AI Integrations

### Logging

- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Preserve proxy key-auth metadata on `/v1/messages` Langfuse traces - [PR #25448](https://github.com/BerriAI/litellm/pull/25448)
- **[Prometheus](../../docs/proxy/logging#prometheus)**
    - Reduce default `LATENCY_BUCKETS` from 35 → 18 boundaries (see breaking-change note above) - [PR #25527](https://github.com/BerriAI/litellm/pull/25527)
- **General**
    - S3 logging: retry with exponential backoff for transient 503/500 errors - [PR #25530](https://github.com/BerriAI/litellm/pull/25530)

### Guardrails

- Optional skip system message in unified guardrail inputs - [PR #25481](https://github.com/BerriAI/litellm/pull/25481)
- Inline IAM: apply guardrail support - [PR #25241](https://github.com/BerriAI/litellm/pull/25241)
- Preserve `dict` `HTTPException.detail` and Bedrock context in guardrail errors - [PR #25558](https://github.com/BerriAI/litellm/pull/25558)

## Spend Tracking, Budgets and Rate Limiting

- Session-TZ-independent date filtering for spend / error log queries - [PR #25542](https://github.com/BerriAI/litellm/pull/25542)

## MCP Gateway

- **Per-user OAuth token storage for interactive MCP flows** - [PR #25441](https://github.com/BerriAI/litellm/pull/25441)
- Block arbitrary command execution via MCP `stdio` transport - [PR #25343](https://github.com/BerriAI/litellm/pull/25343)
- Document missing MCP per-user token environment variables in `config_settings` - [PR #25471](https://github.com/BerriAI/litellm/pull/25471)

## Performance / Loadbalancing / Reliability improvements

- Reduce Prometheus latency histogram cardinality (default buckets 35 → 18) - [PR #25527](https://github.com/BerriAI/litellm/pull/25527)
- S3 retry with exponential backoff for transient errors - [PR #25530](https://github.com/BerriAI/litellm/pull/25530)

## Documentation Updates

- Add Docker Image Security Guide covering cosign verification and deployment best practices - [PR #25439](https://github.com/BerriAI/litellm/pull/25439)
- Document April townhall announcements - [PR #25537](https://github.com/BerriAI/litellm/pull/25537)
- Document missing MCP per-user token env vars - [PR #25471](https://github.com/BerriAI/litellm/pull/25471)
- Add "Screenshots / Proof of Fix" section to PR template - [PR #25564](https://github.com/BerriAI/litellm/pull/25564)

## Infrastructure / Security Notes

- Pin cosign.pub verification to initial commit hash - [PR #25273](https://github.com/BerriAI/litellm/pull/25273)
- Fix node-gyp symlink path after npm upgrade in Dockerfile - [PR #25048](https://github.com/BerriAI/litellm/pull/25048)
- `Dockerfile.non_root`: handle missing `.npmrc` gracefully - [PR #25307](https://github.com/BerriAI/litellm/pull/25307)
- Add Playwright E2E tests with local PostgreSQL - [PR #25126](https://github.com/BerriAI/litellm/pull/25126)
- UI E2E tests for proxy admin team and key management - [PR #25365](https://github.com/BerriAI/litellm/pull/25365)
- Migrate Redis caching tests from GHA to CircleCI - [PR #25354](https://github.com/BerriAI/litellm/pull/25354)
- Update `check_responses_cost` tests for `_expire_stale_rows` - [PR #25299](https://github.com/BerriAI/litellm/pull/25299)
- Raise global vitest timeout and remove per-test overrides - [PR #25468](https://github.com/BerriAI/litellm/pull/25468)
- Version bumps and UI rebuilds: [PR #25316](https://github.com/BerriAI/litellm/pull/25316), [PR #25528](https://github.com/BerriAI/litellm/pull/25528), [PR #25578](https://github.com/BerriAI/litellm/pull/25578), [PR #25571](https://github.com/BerriAI/litellm/pull/25571), [PR #25573](https://github.com/BerriAI/litellm/pull/25573), [PR #25577](https://github.com/BerriAI/litellm/pull/25577)

## New Contributors

* @csoni-cweave made their first contribution in https://github.com/BerriAI/litellm/pull/25441
* @jimmychen-p72 made their first contribution in https://github.com/BerriAI/litellm/pull/25530

**Full Changelog**: https://github.com/BerriAI/litellm/compare/v1.83.3.rc.1...v1.83.7.rc.1
