---
title: "[Preview] v1.83.7.rc.1 - Anthropic Advisor Tool, MCP Per-User OAuth, AWS GovCloud Bedrock"
slug: "v1-83-7-rc-1"
date: 2026-04-11T00:00:00
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
  - name: Harish Mohan
    title: Engineer, LiteLLM
    url: https://github.com/harish876
    image_url: https://github.com/harish876.png
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

## Key Highlights

- **Anthropic Advisor Tool** — [Support for the new `advisor_20260301` tool type, enabling Anthropic's structured advisor capabilities](../../docs/completion/anthropic_advisor_tool)
- **MCP Per-User OAuth Token Storage** — [Store and reuse per-user OAuth tokens for interactive MCP flows — no repeated auth prompts](../../docs/mcp)
- **AWS GovCloud Bedrock Claude Sonnet 4.5** — Support for `claude-sonnet-4-5` in `us-gov-east-1` and `us-gov-west-1` regions with correct pricing and prompt caching
- **11 New Baseten Models** — MiniMax, Nvidia Nemotron, GLM, Kimi, DeepSeek and GPT-OSS models added to the Baseten catalog
- **MCP stdio RCE Fix** — [Blocked arbitrary command execution via stdio transport](../../docs/mcp)
- **Team Spend Logs** — Team members can now view team-wide spend logs directly from the UI with proper RBAC enforcement
- **File Content Streaming** — OpenAI file content endpoint now supports streaming responses

---

## New Models / Updated Models

#### New Model Support (14 new models)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| AWS Bedrock (GovCloud) | `bedrock/us-gov-east-1/anthropic.claude-sonnet-4-5-20250929-v1:0` | 200K | $3.30 | $16.50 | Prompt caching, vision, function calling, computer use, reasoning, native structured output |
| AWS Bedrock (GovCloud) | `bedrock/us-gov-west-1/anthropic.claude-sonnet-4-5-20250929-v1:0` | 200K | $3.30 | $16.50 | Prompt caching, vision, function calling, computer use, reasoning, native structured output |
| AWS Bedrock (GovCloud) | `us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0` | 200K | $3.30 | $16.50 | Prompt caching, vision, function calling, computer use, reasoning, native structured output |
| Baseten | `baseten/MiniMaxAI/MiniMax-M2.5` | - | $0.30 | $1.20 | Chat completions |
| Baseten | `baseten/nvidia/Nemotron-120B-A12B` | - | $0.30 | $0.75 | Chat completions |
| Baseten | `baseten/zai-org/GLM-5` | - | $0.95 | $3.15 | Chat completions |
| Baseten | `baseten/zai-org/GLM-4.7` | - | $0.60 | $2.20 | Chat completions |
| Baseten | `baseten/zai-org/GLM-4.6` | - | $0.60 | $2.20 | Chat completions |
| Baseten | `baseten/moonshotai/Kimi-K2.5` | - | $0.60 | $3.00 | Chat completions |
| Baseten | `baseten/moonshotai/Kimi-K2-Thinking` | - | $0.60 | $2.50 | Chat completions |
| Baseten | `baseten/moonshotai/Kimi-K2-Instruct-0905` | - | $0.60 | $2.50 | Chat completions |
| Baseten | `baseten/openai/gpt-oss-120b` | - | $0.10 | $0.50 | Chat completions |
| Baseten | `baseten/deepseek-ai/DeepSeek-V3.1` | - | $0.50 | $1.50 | Chat completions |
| Baseten | `baseten/deepseek-ai/DeepSeek-V3-0324` | - | $0.77 | $0.77 | Chat completions |

#### Features

- **[Anthropic](../../docs/providers/anthropic)**
    - Support `advisor_20260301` tool type for structured advisor responses - [PR #25525](https://github.com/BerriAI/litellm/pull/25525), [PR #25545](https://github.com/BerriAI/litellm/pull/25545)

- **[Google Gemini](../../docs/providers/gemini)**
    - Add `supports_service_tier` flag to `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-pro`, `gemini-3-flash`, `gemini-2.5-flash-lite`, and related image preview models - [PR #25562](https://github.com/BerriAI/litellm/pull/25562)

### Bug Fixes

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Fix double-counting of cache tokens in Anthropic Messages streaming usage - [PR #25517](https://github.com/BerriAI/litellm/pull/25517)
    - Update `us-gov` Claude Sonnet 4.5 pricing: $3.30/$16.50 per 1M tokens, 8K max output, add native structured output - [PR #25562](https://github.com/BerriAI/litellm/pull/25562)

## LLM API Endpoints

#### Features

- **[Responses API (/responses)](../../docs/response_api)**
    - Append `?model=` query param to backend WebSocket URL for correct model routing - [PR #25437](https://github.com/BerriAI/litellm/pull/25437)
    - Fix duplicate keyword argument error in Responses WebSocket handler - [PR #25513](https://github.com/BerriAI/litellm/pull/25513)

- **[Files API (/files)](../../docs/proxy/pass_through)**
    - Add file content streaming support for OpenAI-compatible file endpoints - [PR #25450](https://github.com/BerriAI/litellm/pull/25450), [PR #25569](https://github.com/BerriAI/litellm/pull/25569)

#### Bugs

- **General**
    - Fix pass-through of multipart uploads and Bedrock JSON body in proxy requests - [PR #25464](https://github.com/BerriAI/litellm/pull/25464)
    - Align v1 guardrail and agent list response field handling with v2 - [PR #25478](https://github.com/BerriAI/litellm/pull/25478)

## Management Endpoints / UI

#### Features

- **Teams + Organizations**
    - Add `POST /team/permissions_bulk_update` endpoint for bulk team permission management - [PR #25239](https://github.com/BerriAI/litellm/pull/25239)
    - Align org and team endpoint permission checks for consistent RBAC enforcement - [PR #25554](https://github.com/BerriAI/litellm/pull/25554)
    - Consolidate route auth for UI and API tokens - [PR #25473](https://github.com/BerriAI/litellm/pull/25473)

- **Spend Logs**
    - Team members can view team-wide spend logs from the UI with RBAC enforcement - [PR #25458](https://github.com/BerriAI/litellm/pull/25458)

- **Virtual Keys**
    - Align `/v2/key/info` response field handling with `/v1/key/info` - [PR #25313](https://github.com/BerriAI/litellm/pull/25313)

#### Bugs

- Improve input validation on management endpoints to reject malformed requests - [PR #25445](https://github.com/BerriAI/litellm/pull/25445)
- Use parameterized query for combined-view token lookup to prevent SQL injection - [PR #25467](https://github.com/BerriAI/litellm/pull/25467)
- Fix session-timezone-independent date filtering for spend/error log queries - [PR #25542](https://github.com/BerriAI/litellm/pull/25542)
- Improve UI storage handling and Dockerfile consistency - [PR #25384](https://github.com/BerriAI/litellm/pull/25384)

## AI Integrations

### Logging

- **General**
    - Ensure spend/cost logging runs correctly when `stream=True` in websearch interception - [PR #25424](https://github.com/BerriAI/litellm/pull/25424)

### Guardrails

- Add `applyGuardrail` support for inline IAM-based guardrails - [PR #25241](https://github.com/BerriAI/litellm/pull/25241)
- Add UI option to skip system messages in guardrail processing - [PR #25562](https://github.com/BerriAI/litellm/pull/25562)

## Spend Tracking, Budgets and Rate Limiting

- Team members can now access team-wide spend logs with proper RBAC enforcement - [PR #25458](https://github.com/BerriAI/litellm/pull/25458)
- Fix timezone-independent date filtering in spend/error log queries - [PR #25542](https://github.com/BerriAI/litellm/pull/25542)

## MCP Gateway

- **Security**: Block arbitrary command execution via stdio transport - [PR #25343](https://github.com/BerriAI/litellm/pull/25343)
- Add per-user OAuth token storage for interactive MCP flows - [PR #25441](https://github.com/BerriAI/litellm/pull/25441)
- Set default 60-second timeout for A2A client creation - [PR #25514](https://github.com/BerriAI/litellm/pull/25514)

## Performance / Loadbalancing / Reliability improvements

- Fix node-gyp symlink path after npm upgrade in Dockerfile - [PR #25048](https://github.com/BerriAI/litellm/pull/25048)
- Handle missing `.npmrc` gracefully in `Dockerfile.non_root` - [PR #25307](https://github.com/BerriAI/litellm/pull/25307)
- Harden file path resolution in skill archive extraction - [PR #25475](https://github.com/BerriAI/litellm/pull/25475)

## Documentation Updates

- Add Anthropic Advisor Tool guide - [PR #25545](https://github.com/BerriAI/litellm/pull/25545)
- Add missing MCP per-user token env vars to `config_settings` reference - [PR #25471](https://github.com/BerriAI/litellm/pull/25471)
- Update guardrails quick-start and proxy config settings docs - [PR #25562](https://github.com/BerriAI/litellm/pull/25562)

## New Contributors

* @csoni-cweave made their first contribution in https://github.com/BerriAI/litellm/pull/25441
* @jaydns made their first contribution in https://github.com/BerriAI/litellm/pull/25445

**Full Changelog**: https://github.com/BerriAI/litellm/compare/v1.83.3.rc.1...v1.83.7.rc.1
