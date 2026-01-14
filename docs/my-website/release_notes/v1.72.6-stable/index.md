---
title: "v1.72.6-stable - MCP Gateway Permission Management"
slug: "v1-72-6-stable"
date: 2025-06-14T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaffer
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
docker run
-e STORE_MODEL_IN_DB=True
-p 4000:4000
docker.litellm.ai/berriai/litellm:main-v1.72.6-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.72.6.post2
```

</TabItem>
</Tabs>


## TLDR


* **Why Upgrade**
    - Codex-mini on Claude Code: You can now use `codex-mini` (OpenAI’s code assistant model) via Claude Code.
    - MCP Permissions Management: Manage permissions for MCP Servers by Keys, Teams, Organizations (entities) on LiteLLM.
    - UI: Turn on/off auto refresh on logs view. 
    - Rate Limiting: Support for output token-only rate limiting.  
* **Who Should Read**
    - Teams using `/v1/messages` API (Claude Code)
    - Teams using **MCP**
    - Teams giving access to self-hosted models and setting rate limits
* **Risk of Upgrade**
    - **Low**
        - No major changes to existing functionality or package updates.


---

## Key Highlights


### MCP Permissions Management

<Image img={require('../../img/release_notes/mcp_permissions.png')}/>

This release brings support for managing permissions for MCP Servers by Keys, Teams, Organizations (entities) on LiteLLM. When a MCP client attempts to list tools, LiteLLM will only return the tools the entity has permissions to access.

This is great for use cases that require access to restricted data (e.g Jira MCP) that you don't want everyone to use.

For Proxy Admins, this enables centralized management of all MCP Servers with access control. For developers, this means you'll only see the MCP tools assigned to you.




### Codex-mini on Claude Code

<Image img={require('../../img/release_notes/codex_on_claude_code.jpg')} />

This release brings support for calling `codex-mini` (OpenAI’s code assistant model) via Claude Code.

This is done by LiteLLM enabling any Responses API model (including `o3-pro`) to be called via `/chat/completions` and `/v1/messages` endpoints. This includes:

- Streaming calls
- Non-streaming calls
- Cost Tracking on success + failure for Responses API models

Here's how to use it [today](../../docs/tutorials/claude_responses_api)




---


## New / Updated Models

### Pricing / Context Window Updates

| Provider    | Model                                  | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Type |
| ----------- | -------------------------------------- | -------------- | ------------------- | -------------------- | -------------------- |
| VertexAI   | `vertex_ai/claude-opus-4`               | 200K           | $15.00              | $75.00               | New |
| OpenAI   | `gpt-4o-audio-preview-2025-06-03`             | 128k           | $2.5 (text), $40 (audio)              | $10 (text), $80 (audio)               | New |
| OpenAI | `o3-pro` | 200k | 20 | 80 | New |
| OpenAI | `o3-pro-2025-06-10` | 200k | 20 | 80 | New |
| OpenAI | `o3` | 200k | 2 | 8 | Updated |
| OpenAI | `o3-2025-04-16` | 200k | 2 | 8 | Updated |
| Azure | `azure/gpt-4o-mini-transcribe` | 16k | 1.25 (text), 3 (audio) | 5 (text) | New |
| Mistral | `mistral/magistral-medium-latest` | 40k | 2 | 5 | New |
| Mistral | `mistral/magistral-small-latest` | 40k | 0.5 | 1.5 | New |

- Deepgram: `nova-3` cost per second pricing is [now supported](https://github.com/BerriAI/litellm/pull/11634).

### Updated Models
#### Bugs
- **[Watsonx](../../docs/providers/watsonx)**
    - Ignore space id on Watsonx deployments (throws json errors) - [PR](https://github.com/BerriAI/litellm/pull/11527)
- **[Ollama](../../docs/providers/ollama)**
    - Set tool call id for streaming calls - [PR](https://github.com/BerriAI/litellm/pull/11528)
- **Gemini ([VertexAI](../../docs/providers/vertex) + [Google AI Studio](../../docs/providers/gemini))**
    - Fix tool call indexes - [PR](https://github.com/BerriAI/litellm/pull/11558)
    - Handle empty string for arguments in function calls - [PR](https://github.com/BerriAI/litellm/pull/11601)
    - Add audio/ogg mime type support when inferring from file url’s - [PR](https://github.com/BerriAI/litellm/pull/11635)
- **[Custom LLM](../../docs/providers/custom_llm_server)**
    - Fix passing api_base, api_key, litellm_params_dict to custom_llm embedding methods - [PR](https://github.com/BerriAI/litellm/pull/11450) s/o [ElefHead](https://github.com/ElefHead)
- **[Huggingface](../../docs/providers/huggingface)**
    - Add /chat/completions to endpoint url when missing - [PR](https://github.com/BerriAI/litellm/pull/11630)
- **[Deepgram](../../docs/providers/deepgram)**
    - Support async httpx calls - [PR](https://github.com/BerriAI/litellm/pull/11641)
- **[Anthropic](../../docs/providers/anthropic)**
    - Append prefix (if set) to assistant content start - [PR](https://github.com/BerriAI/litellm/pull/11719)

#### Features
- **[VertexAI](../../docs/providers/vertex)**
    - Support vertex credentials set via env var on passthrough - [PR](https://github.com/BerriAI/litellm/pull/11527)
    - Support for choosing ‘global’ region when model is only available there - [PR](https://github.com/BerriAI/litellm/pull/11566)
    - Anthropic passthrough cost calculation + token tracking - [PR](https://github.com/BerriAI/litellm/pull/11611)
    - Support ‘global’ vertex region on passthrough - [PR](https://github.com/BerriAI/litellm/pull/11661)
- **[Anthropic](../../docs/providers/anthropic)**
    - ‘none’ tool choice param support - [PR](https://github.com/BerriAI/litellm/pull/11695), [Get Started](../../docs/providers/anthropic#disable-tool-calling)
- **[Perplexity](../../docs/providers/perplexity)**
    - Add ‘reasoning_effort’ support - [PR](https://github.com/BerriAI/litellm/pull/11562), [Get Started](../../docs/providers/perplexity#reasoning-effort)
- **[Mistral](../../docs/providers/mistral)**
    - Add mistral reasoning support - [PR](https://github.com/BerriAI/litellm/pull/11642), [Get Started](../../docs/providers/mistral#reasoning)
- **[SGLang](../../docs/providers/openai_compatible)**
    - Map context window exceeded error for proper handling - [PR](https://github.com/BerriAI/litellm/pull/11575/)
- **[Deepgram](../../docs/providers/deepgram)**
    - Provider specific params support - [PR](https://github.com/BerriAI/litellm/pull/11638)
- **[Azure](../../docs/providers/azure)**
    - Return content safety filter results - [PR](https://github.com/BerriAI/litellm/pull/11655)
---

## LLM API Endpoints

#### Bugs
- **[Chat Completion](../../docs/completion/input)**
    - Streaming - Ensure consistent ‘created’ across chunks - [PR](https://github.com/BerriAI/litellm/pull/11528)
#### Features
- **MCP**
    - Add controls for MCP Permission Management - [PR](https://github.com/BerriAI/litellm/pull/11598), [Docs](../../docs/mcp#-mcp-permission-management)
    - Add permission management for MCP List + Call Tool operations - [PR](https://github.com/BerriAI/litellm/pull/11682), [Docs](../../docs/mcp#-mcp-permission-management)
    - Streamable HTTP server support - [PR](https://github.com/BerriAI/litellm/pull/11628), [PR](https://github.com/BerriAI/litellm/pull/11645), [Docs](../../docs/mcp#using-your-mcp)
    - Use Experimental dedicated Rest endpoints for list, calling MCP tools - [PR](https://github.com/BerriAI/litellm/pull/11684)
- **[Responses API](../../docs/response_api)**
    - NEW API Endpoint - List input items - [PR](https://github.com/BerriAI/litellm/pull/11602) 
    - Background mode for OpenAI + Azure OpenAI - [PR](https://github.com/BerriAI/litellm/pull/11640)
    - Langfuse/other Logging support on responses api requests - [PR](https://github.com/BerriAI/litellm/pull/11685)
- **[Chat Completions](../../docs/completion/input)**
    - Bridge for Responses API - allows calling codex-mini via `/chat/completions` and `/v1/messages` - [PR](https://github.com/BerriAI/litellm/pull/11632), [PR](https://github.com/BerriAI/litellm/pull/11685)


---

## Spend Tracking

#### Bugs
- **[End Users](../../docs/proxy/customers)**
    - Update enduser spend and budget reset date based on budget duration - [PR](https://github.com/BerriAI/litellm/pull/8460) (s/o [laurien16](https://github.com/laurien16))
- **[Custom Pricing](../../docs/proxy/custom_pricing)**
    - Convert scientific notation str to int - [PR](https://github.com/BerriAI/litellm/pull/11655)

---

## Management Endpoints / UI

#### Bugs
- **[Users](../../docs/proxy/users)**
    - `/user/info` - fix passing user with `+` in user id
    - Add admin-initiated password reset flow - [PR](https://github.com/BerriAI/litellm/pull/11618)
    - Fixes default user settings UI rendering error - [PR](https://github.com/BerriAI/litellm/pull/11674)
- **[Budgets](../../docs/proxy/users)**
    - Correct success message when new user budget is created - [PR](https://github.com/BerriAI/litellm/pull/11608)

#### Features
- **Leftnav**
    - Show remaining Enterprise users on UI
- **MCP**
    - New server add form - [PR](https://github.com/BerriAI/litellm/pull/11604)
    - Allow editing mcp servers - [PR](https://github.com/BerriAI/litellm/pull/11693)
- **Models**
    - Add deepgram models on UI
    - Model Access Group support on UI - [PR](https://github.com/BerriAI/litellm/pull/11719)
- **Keys**
    - Trim long user id’s - [PR](https://github.com/BerriAI/litellm/pull/11488)
- **Logs**
    - Add live tail feature to logs view, allows user to disable auto refresh in high traffic - [PR](https://github.com/BerriAI/litellm/pull/11712)
    - Audit Logs - preview screenshot - [PR](https://github.com/BerriAI/litellm/pull/11715)

---

## Logging / Guardrails Integrations

#### Bugs
- **[Arize](../../docs/observability/arize_integration)**
    - Change space_key header to space_id - [PR](https://github.com/BerriAI/litellm/pull/11595) (s/o [vanities](https://github.com/vanities))
- **[Prometheus](../../docs/proxy/prometheus)**
    - Fix total requests increment - [PR](https://github.com/BerriAI/litellm/pull/11718)

#### Features
- **[Lasso Guardrails](../../docs/proxy/guardrails/lasso_security)**
    - [NEW] Lasso Guardrails support - [PR](https://github.com/BerriAI/litellm/pull/11565)
- **[Users](../../docs/proxy/users)**
    - New `organizations` param on `/user/new` - allows adding users to orgs on creation - [PR](https://github.com/BerriAI/litellm/pull/11572/files)
- **Prevent double logging when using bridge logic** - [PR](https://github.com/BerriAI/litellm/pull/11687)

---

## Performance / Reliability Improvements

#### Bugs
- **[Tag based routing](../../docs/proxy/tag_routing)**
    - Do not consider ‘default’ models when request specifies a tag - [PR](https://github.com/BerriAI/litellm/pull/11454) (s/o [thiagosalvatore](https://github.com/thiagosalvatore))

#### Features
- **[Caching](../../docs/caching/all_caches)**
    - New optional ‘litellm[caching]’ pip install for adding disk cache dependencies - [PR](https://github.com/BerriAI/litellm/pull/11600)

---

## General Proxy Improvements

#### Bugs
- **aiohttp**
    - fixes for transfer encoding error on aiohttp transport - [PR](https://github.com/BerriAI/litellm/pull/11561)

#### Features
- **aiohttp**
    - Enable System Proxy Support for aiohttp transport - [PR](https://github.com/BerriAI/litellm/pull/11616) (s/o [idootop](https://github.com/idootop))
- **CLI**
    - Make all commands show server URL - [PR](https://github.com/BerriAI/litellm/pull/10801)
- **Unicorn**
    - Allow setting keep alive timeout - [PR](https://github.com/BerriAI/litellm/pull/11594)
- **Experimental Rate Limiting v2** (enable via `EXPERIMENTAL_MULTI_INSTANCE_RATE_LIMITING="True"`)
    - Support specifying rate limit by output_tokens only - [PR](https://github.com/BerriAI/litellm/pull/11646)
    - Decrement parallel requests on call failure - [PR](https://github.com/BerriAI/litellm/pull/11646)
    - In-memory only rate limiting support - [PR](https://github.com/BerriAI/litellm/pull/11646)
    - Return remaining rate limits by key/user/team - [PR](https://github.com/BerriAI/litellm/pull/11646)
- **Helm**
    - support extraContainers in migrations-job.yaml - [PR](https://github.com/BerriAI/litellm/pull/11649)




---

## New Contributors
* @laurien16 made their first contribution in https://github.com/BerriAI/litellm/pull/8460
* @fengbohello made their first contribution in https://github.com/BerriAI/litellm/pull/11547
* @lapinek made their first contribution in https://github.com/BerriAI/litellm/pull/11570
* @yanwork made their first contribution in https://github.com/BerriAI/litellm/pull/11586
* @dhs-shine made their first contribution in https://github.com/BerriAI/litellm/pull/11575
* @ElefHead made their first contribution in https://github.com/BerriAI/litellm/pull/11450
* @idootop made their first contribution in https://github.com/BerriAI/litellm/pull/11616
* @stevenaldinger made their first contribution in https://github.com/BerriAI/litellm/pull/11649
* @thiagosalvatore made their first contribution in https://github.com/BerriAI/litellm/pull/11454
* @vanities made their first contribution in https://github.com/BerriAI/litellm/pull/11595
* @alvarosevilla95 made their first contribution in https://github.com/BerriAI/litellm/pull/11661

---

## Demo Instance

Here's a Demo Instance to test changes:

- Instance: https://demo.litellm.ai/
- Login Credentials:
    - Username: admin
    - Password: sk-1234

## [Git Diff](https://github.com/BerriAI/litellm/compare/v1.72.2-stable...1.72.6.rc)
