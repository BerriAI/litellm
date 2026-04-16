---
title: "v1.83.3-stable - MCP Toolsets & Skills Marketplace"
slug: "v1-83-3-stable"
date: 2026-04-04T00:00:00
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
docker.litellm.ai/berriai/litellm:main-v1.83.3-stable
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.83.3
```

</TabItem>
</Tabs>

## Key Highlights

- **MCP Toolsets** — [Create curated tool subsets from one or more MCP servers with scoped permissions, and manage them from the UI or API](../../docs/mcp)
- **Skills Marketplace** — [Browse, install, and publish Claude Code skills from a self-hosted marketplace — works across Anthropic, Vertex AI, Azure, and Bedrock](../../docs/proxy/skills)
- **Guardrail Fallbacks** — [Configure `on_error` behavior so guardrail failures degrade gracefully instead of blocking the request](../../docs/proxy/guardrails)
- **Team Bring Your Own Guardrails** — [Teams can now attach and manage their own guardrails directly from team settings in the UI](../../docs/proxy/guardrails)

---


### Skills Marketplace

The Skills Marketplace gives teams a self-hosted catalog for discovering, installing, and publishing Claude Code skills. Skills are portable across Anthropic, Vertex AI, Azure, and Bedrock — so a skill published once works everywhere your gateway routes to.

![Skills Marketplace](../../img/release_notes/skills_marketplace.png)

[Get Started](../../docs/proxy/skills)

### Guardrail Fallbacks

![Guardrail Fallbacks](../../img/release_notes/guardrail_fallbacks.png)

Guardrail pipelines now support an optional `on_error` behavior. When a guardrail check fails or errors out, you can configure the pipeline to fall back gracefully — logging the failure and continuing the request — instead of returning a hard 500 to the caller. This is especially useful for non-critical guardrails where availability matters more than enforcement.

[Get Started](../../docs/proxy/guardrails/policy_flow_builder)

### Team Bring Your Own Guardrails

Teams can now attach guardrails directly from the team management UI. Admins configure available guardrails at the project or proxy level, and individual teams select which ones apply to their traffic — no config file changes or proxy restarts needed. This also ships with project-level guardrail support in the project create/edit flows.

### MCP Toolsets

MCP Toolsets let AI platform admins create curated subsets of tools from one or more MCP servers and assign them to teams and keys with scoped permissions. Instead of granting access to an entire MCP server, you can now bundle specific tools into a named toolset — controlling exactly which tools each team or API key can invoke. Toolsets are fully managed through the UI (new Toolsets tab) and API, and work seamlessly with the Responses API and Playground.

![MCP Toolsets](../../img/release_notes/mcp_toolsets.jpeg)

[Get Started](../../docs/mcp)

---

## New Models / Updated Models

#### New Model Support (60 new models)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| OpenAI | `gpt-5.4-mini` | 272K | $0.75 | $4.50 | Chat, cache read, flex/batch/priority tiers |
| OpenAI | `gpt-5.4-nano` | 272K | $0.20 | - | Chat, flex/batch tiers |
| OpenAI | `gpt-4-0314` | 8K | $30.00 | $60.00 | Re-added legacy entry (deprecation 2026-03-26) |
| Azure OpenAI | `azure/gpt-5.4-mini` | 1.05M | $0.75 | $4.50 | Chat completions, cache read |
| Azure OpenAI | `azure/gpt-5.4-nano` | - | - | - | Chat completions |
| AWS Bedrock | `us.amazon.nova-canvas-v1:0` | 2.6K | - | $0.06 / image | Nova Canvas image edit support |
| AWS Bedrock | `nvidia.nemotron-super-3-120b` | 256K | $0.15 | $0.65 | Function calling, reasoning, system messages |
| AWS Bedrock | `minimax.minimax-m2.5` (12 regions) | 1M | $0.30 | $1.20 | Function calling, reasoning, system messages |
| AWS Bedrock | `zai.glm-5` | 200K | $1.00 | $3.20 | Function calling, reasoning |
| AWS Bedrock | `bedrock/us-gov-{east,west}-1/anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | $1.20 | $6.00 | GovCloud Claude Haiku 4.5 |
| Vertex AI | `vertex_ai/claude-haiku-4-5` | 200K | $1.00 | $5.00 | Chat, cache creation/read |
| Gemini | `gemini-3.1-flash-live-preview` / `gemini/gemini-3.1-flash-live-preview` | 131K | $0.75 | - | Live audio/video/image/text |
| Gemini | `gemini/lyria-3-pro-preview`, `gemini/lyria-3-clip-preview` | 131K | - | - | Music generation preview |
| xAI | `xai/grok-4.20-beta-0309-reasoning` | 2M | $2.00 | $6.00 | Function calling, reasoning |
| xAI | `xai/grok-4.20-beta-0309-non-reasoning` | 2M | - | - | Function calling |
| xAI | `xai/grok-4.20-multi-agent-beta-0309` | 2M | - | - | Multi-agent preview |
| OCI GenAI | `oci/cohere.command-a-reasoning-08-2025`, `oci/cohere.command-a-vision-07-2025`, `oci/cohere.command-a-translate-08-2025`, `oci/cohere.command-r-08-2024`, `oci/cohere.command-r-plus-08-2024` | 256K | $1.56 | $1.56 | Cohere chat family on OCI |
| OCI GenAI | `oci/meta.llama-3.1-70b-instruct`, `oci/meta.llama-3.2-11b-vision-instruct`, `oci/meta.llama-3.3-70b-instruct-fp8-dynamic` | Varies | Varies | Varies | Llama chat family on OCI |
| OCI GenAI | `oci/xai.grok-4-fast`, `oci/xai.grok-4.1-fast`, `oci/xai.grok-4.20`, `oci/xai.grok-4.20-multi-agent`, `oci/xai.grok-code-fast-1` | 131K | $3.00 | $15.00 | Grok family on OCI |
| OCI GenAI | `oci/google.gemini-2.5-pro`, `oci/google.gemini-2.5-flash`, `oci/google.gemini-2.5-flash-lite` | 1M+ | $1.25 | $10.00 | Gemini family on OCI |
| OCI GenAI | `oci/cohere.embed-english-v3.0`, `oci/cohere.embed-english-light-v3.0`, `oci/cohere.embed-multilingual-v3.0`, `oci/cohere.embed-multilingual-light-v3.0`, `oci/cohere.embed-english-image-v3.0`, `oci/cohere.embed-english-light-image-v3.0`, `oci/cohere.embed-multilingual-light-image-v3.0`, `oci/cohere.embed-v4.0` | Varies | Varies | - | Embeddings on OCI |
| Volcengine | `volcengine/doubao-seed-2-0-pro-260215`, `doubao-seed-2-0-lite-260215`, `doubao-seed-2-0-mini-260215`, `doubao-seed-2-0-code-preview-260215` | 256K | - | - | Doubao Seed 2.0 family |

#### Features

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Add Nova Canvas image edit support - [PR #24869](https://github.com/BerriAI/litellm/pull/24869), [PR #25110](https://github.com/BerriAI/litellm/pull/25110)
    - Add `nvidia.nemotron-super-3-120b` entries and Bedrock model catalog updates - [PR #24588](https://github.com/BerriAI/litellm/pull/24588), [PR #24645](https://github.com/BerriAI/litellm/pull/24645)
    - Add MiniMax M2.5 cross-region entries - cost map additions
    - Add `zai.glm-5` pricing entry
    - Improve cache usage exposure for Claude-compatible streaming paths - [PR #24850](https://github.com/BerriAI/litellm/pull/24850)
    - Structured output cost tracking fix for Bedrock JSON mode - [PR #23794](https://github.com/BerriAI/litellm/pull/23794)
    - Preserve JSON-RPC envelope for AgentCore A2A-native agents - [PR #25092](https://github.com/BerriAI/litellm/pull/25092)
    - Fix Bedrock Anthropic file/document handling - [PR #25047](https://github.com/BerriAI/litellm/pull/25047), [PR #25050](https://github.com/BerriAI/litellm/pull/25050)
    - Fix Bedrock count-tokens with custom endpoint - [PR #24199](https://github.com/BerriAI/litellm/pull/24199)

- **[Fireworks AI](../../docs/providers/fireworks_ai)**
    - Skip `#transform=inline` for base64 data URLs - [PR #23818](https://github.com/BerriAI/litellm/pull/23818)

- **[DeepInfra](../../docs/providers/deepinfra)**
    - Mock DeepInfra completion tests to avoid real API calls - [PR #24805](https://github.com/BerriAI/litellm/pull/24805)

- **[WatsonX](../../docs/providers/watsonx)**
    - Fix WatsonX tests failing in CI due to missing env vars - [PR #24814](https://github.com/BerriAI/litellm/pull/24814)

- **[Snowflake Cortex](../../docs/providers/snowflake)**
    - Move Snowflake mocked tests to unit test directory - [PR #24822](https://github.com/BerriAI/litellm/pull/24822)

- **[Anthropic](../../docs/providers/anthropic)**
    - Surface Anthropic tool results in Responses API - [PR #23784](https://github.com/BerriAI/litellm/pull/23784)
    - Auth token and custom `api_base` support - [PR #24140](https://github.com/BerriAI/litellm/pull/24140)
    - Preserve beta header order - [PR #23715](https://github.com/BerriAI/litellm/pull/23715)
    - Cache-control support for Anthropic document/file message blocks - [PR #23906](https://github.com/BerriAI/litellm/pull/23906), [PR #23911](https://github.com/BerriAI/litellm/pull/23911)
    - Map Anthropic refusal finish_reason - [PR #23899](https://github.com/BerriAI/litellm/pull/23899)
    - Cache-control on tool config - [PR #24076](https://github.com/BerriAI/litellm/pull/24076)
    - Remove 200K pricing entries for Opus/Sonnet 4.6 - [PR #24689](https://github.com/BerriAI/litellm/pull/24689)

- **[OpenAI](../../docs/providers/openai)**
    - Add `gpt-5.4-mini` / `gpt-5.4-nano` with flex/batch/priority tiers - [PR #23958](https://github.com/BerriAI/litellm/pull/23958)
    - Restore `gpt-4-0314` cost entry with deprecation metadata - [PR #23753](https://github.com/BerriAI/litellm/pull/23753)
    - OpenAI reasoning items in chat completions - [PR #24690](https://github.com/BerriAI/litellm/pull/24690)

- **[Google Vertex AI](../../docs/providers/vertex)**
    - Add `vertex_ai/claude-haiku-4-5` pricing entry - [PR #25151](https://github.com/BerriAI/litellm/pull/25151)
    - Vertex `count_tokens` location override - [PR #23907](https://github.com/BerriAI/litellm/pull/23907)
    - Vertex cancel batch endpoint - [PR #23957](https://github.com/BerriAI/litellm/pull/23957)
    - Vertex PAYGO tutorial - [PR #24009](https://github.com/BerriAI/litellm/pull/24009)
    - Fix Vertex AI batch - [PR #23718](https://github.com/BerriAI/litellm/pull/23718)
    - DeepSeek v3.2 Vertex region mapping - [PR #23864](https://github.com/BerriAI/litellm/pull/23864)

- **[Google Gemini](../../docs/providers/gemini)**
    - Add `gemini-3.1-flash-live-preview` model - [PR #24665](https://github.com/BerriAI/litellm/pull/24665)
    - Add Lyria 3 Pro / Clip preview entries + docs - [PR #24610](https://github.com/BerriAI/litellm/pull/24610)
    - Normalize Gemini retrieve-file URL - [PR #24662](https://github.com/BerriAI/litellm/pull/24662)
    - Gemini context caching with custom `api_base` - [PR #23928](https://github.com/BerriAI/litellm/pull/23928)
    - Strict `additional_properties` cleanup - [PR #24072](https://github.com/BerriAI/litellm/pull/24072)
    - Gemini context circulation - [PR #24073](https://github.com/BerriAI/litellm/pull/24073)

- **[Azure OpenAI](../../docs/providers/azure)**
    - Add `azure/gpt-5.4-mini` / `azure/gpt-5.4-nano` pricing - model catalog
    - Bump proxy Azure API version - [PR #24120](https://github.com/BerriAI/litellm/pull/24120)
    - Azure fine-tuning fixes - [PR #24687](https://github.com/BerriAI/litellm/pull/24687)
    - Azure gpt-5.4 Responses API routing fix - [PR #23926](https://github.com/BerriAI/litellm/pull/23926)
    - Azure AI annotations - [PR #23939](https://github.com/BerriAI/litellm/pull/23939)

- **[xAI](../../docs/providers/xai)**
    - Add Grok 4.20 reasoning / non-reasoning / multi-agent preview entries - cost map

- **[OCI GenAI](../../docs/providers/oci)**
    - Native embeddings support and expanded chat + embedding model catalog - [PR #24887](https://github.com/BerriAI/litellm/pull/24887), [PR #25151](https://github.com/BerriAI/litellm/pull/25151)

- **[Volcengine](../../docs/providers/volcengine)**
    - Add Doubao Seed 2.0 pro/lite/mini/code-preview entries - cost map

- **[Mistral](../../docs/providers/mistral)**
    - Fix Mistral diarize segments response - [PR #23925](https://github.com/BerriAI/litellm/pull/23925)

- **[OpenRouter](../../docs/providers/openrouter)**
    - Strip prefix on OpenRouter wildcard routing - [PR #24603](https://github.com/BerriAI/litellm/pull/24603)

- **[Deepgram](../../docs/providers/deepgram)**
    - Revert problematic cost-per-second change - [PR #24297](https://github.com/BerriAI/litellm/pull/24297)

- **[GitHub Copilot](../../docs/providers/github_copilot)**
    - Short-circuit web search when not supported by Copilot model - [PR #24143](https://github.com/BerriAI/litellm/pull/24143)

- **[Snowflake Cortex](../../docs/providers/snowflake)**
    - Test conflict resolution and reliability fixes - merges across release window

- **[Quora / Poe](../../docs/providers/poe)**
    - Fix missing content-part added event - [PR #24445](https://github.com/BerriAI/litellm/pull/24445)

### Bug Fixes

- **General**
    - Fix `gpt-5.4` pricing metadata - [PR #24748](https://github.com/BerriAI/litellm/pull/24748)
    - Fix gov pricing tests and Bedrock model test follow-ups - [PR #24931](https://github.com/BerriAI/litellm/pull/24931), [PR #24947](https://github.com/BerriAI/litellm/pull/24947), [PR #25022](https://github.com/BerriAI/litellm/pull/25022)
    - Fix thinking blocks null handling - [PR #24070](https://github.com/BerriAI/litellm/pull/24070)
    - Streaming tool-call finish reason with empty content - [PR #23895](https://github.com/BerriAI/litellm/pull/23895)
    - Ensure alternating roles in conversion paths - [PR #24015](https://github.com/BerriAI/litellm/pull/24015)
    - File → input_file mapping fix - [PR #23618](https://github.com/BerriAI/litellm/pull/23618)
    - File-search emulated alignment - [PR #23969](https://github.com/BerriAI/litellm/pull/23969)
    - Preserve final streaming attributes - [PR #23530](https://github.com/BerriAI/litellm/pull/23530)
    - Streaming metadata hidden params - [PR #24220](https://github.com/BerriAI/litellm/pull/24220)
    - Improve LLM repeated message detection performance - [PR #18120](https://github.com/BerriAI/litellm/pull/18120)

## LLM API Endpoints

#### Features

- **[Responses API](../../docs/response_api)**
    - File Search support — Phase 1 native passthrough and Phase 2 emulated fallback for non-OpenAI models - [PR #23969](https://github.com/BerriAI/litellm/pull/23969)
    - Prompt management support for Responses API - [PR #23999](https://github.com/BerriAI/litellm/pull/23999)
    - Encrypted-content affinity across model versions - [PR #23854](https://github.com/BerriAI/litellm/pull/23854), [PR #24110](https://github.com/BerriAI/litellm/pull/24110)
    - Round-trip Responses API `reasoning_items` in chat completions - [PR #24690](https://github.com/BerriAI/litellm/pull/24690)
    - Emit `content_part.added` streaming event for non-OpenAI models - [PR #24445](https://github.com/BerriAI/litellm/pull/24445)
    - Surface Anthropic code execution results as `code_interpreter_call` - [PR #23784](https://github.com/BerriAI/litellm/pull/23784)
    - Preserve Anthropic `thinking.summary` when routing to OpenAI Responses API - [PR #21441](https://github.com/BerriAI/litellm/pull/21441)
    - Auto-route Azure `gpt-5.4+` tools + reasoning to Responses API - [PR #23926](https://github.com/BerriAI/litellm/pull/23926)
    - Preserve annotations in Azure AI Foundry Agents responses - [PR #23939](https://github.com/BerriAI/litellm/pull/23939)
    - API reference path routing updates - [PR #24155](https://github.com/BerriAI/litellm/pull/24155)
    - Map Chat Completion `file` type to Responses API `input_file` - [PR #23618](https://github.com/BerriAI/litellm/pull/23618)
    - Map `file_url` → `file_id` in Responses→Completions translation - [PR #24874](https://github.com/BerriAI/litellm/pull/24874)

- **[Batch API](../../docs/batches)**
    - Vertex AI batch cancel support - [PR #23957](https://github.com/BerriAI/litellm/pull/23957)

- **Token Counting**
    - Bedrock: respect `api_base` and `aws_bedrock_runtime_endpoint` - [PR #24199](https://github.com/BerriAI/litellm/pull/24199)
    - Vertex: respect `vertex_count_tokens_location` for Claude - [PR #23907](https://github.com/BerriAI/litellm/pull/23907)

- **[Audio / Transcription API](../../docs/audio_transcription)**
    - Mistral: preserve diarization segments in transcription response - [PR #23925](https://github.com/BerriAI/litellm/pull/23925)

- **[Embeddings API](../../docs/embedding/supported_embedding)**
    - Gemini: convert `task_type` to camelCase `taskType` for Gemini API - [PR #24191](https://github.com/BerriAI/litellm/pull/24191)

- **[Video Generation](../../docs/video_generation)**
    - New reusable video character endpoints (create / edit / extension / get) with router-first routing - [PR #23737](https://github.com/BerriAI/litellm/pull/23737)

- **[Search API](../../docs/search)**
    - Support self-hosted Firecrawl response format - [PR #24866](https://github.com/BerriAI/litellm/pull/24866)

- **[A2A / MCP Gateway API](../../docs/mcp)**
    - Preserve JSON-RPC envelope for AgentCore A2A-native agents - [PR #25092](https://github.com/BerriAI/litellm/pull/25092)

- **[Pass-Through Endpoints](../../docs/pass_through/intro)**
    - Support `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_BASE_URL` env vars and custom `api_base` in experimental passthrough - [PR #24140](https://github.com/BerriAI/litellm/pull/24140)

#### Bugs

- **[Responses API](../../docs/response_api)**
    - Use real `request_data` in Responses API streaming fallback path - [PR #23910](https://github.com/BerriAI/litellm/pull/23910)
    - Fix Responses API cost calculation - [PR #24080](https://github.com/BerriAI/litellm/pull/24080)

- **[Pass-Through Endpoints](../../docs/pass_through/intro)**
    - Allow non-admin users to access pass-through subpath routes with auth - [PR #24079](https://github.com/BerriAI/litellm/pull/24079)
    - Prevent duplicate callback logs for pass-through endpoint failures - [PR #23509](https://github.com/BerriAI/litellm/pull/23509)

- **General**
    - Proxy-only failure call-type handling - [PR #24050](https://github.com/BerriAI/litellm/pull/24050)
    - Generic API model-group logging fix - [PR #24044](https://github.com/BerriAI/litellm/pull/24044)

## Management Endpoints / UI

#### Features

- **Virtual Keys**
    - Substring search for `user_id` and `key_alias` on `/key/list` - [PR #24746](https://github.com/BerriAI/litellm/pull/24746), [PR #24751](https://github.com/BerriAI/litellm/pull/24751)
    - Wire `team_id` filter to key alias dropdown - [PR #25114](https://github.com/BerriAI/litellm/pull/25114), [PR #25119](https://github.com/BerriAI/litellm/pull/25119)
    - Allow hashed `token_id` in `/key/update` - [PR #24969](https://github.com/BerriAI/litellm/pull/24969)
    - Enforce upper-bound key params on `/key/update` and bulk update hook paths - [PR #25103](https://github.com/BerriAI/litellm/pull/25103), [PR #25110](https://github.com/BerriAI/litellm/pull/25110)
    - Fix create-key tags dropdown - [PR #24273](https://github.com/BerriAI/litellm/pull/24273)
    - Fix key-update 404 - [PR #24063](https://github.com/BerriAI/litellm/pull/24063)
    - Fix key admin privilege escalation - [PR #23781](https://github.com/BerriAI/litellm/pull/23781)
    - Key-endpoint authentication hardening - [PR #23977](https://github.com/BerriAI/litellm/pull/23977)
    - Disable custom API keys flag - [PR #23812](https://github.com/BerriAI/litellm/pull/23812)
    - Skip alias revalidation on key update - [PR #23798](https://github.com/BerriAI/litellm/pull/23798)
    - Fix invalid keys for internal users - [PR #23795](https://github.com/BerriAI/litellm/pull/23795)
    - Distributed lock for scheduled key rotation job execution - [PR #23364](https://github.com/BerriAI/litellm/pull/23364), [PR #23834](https://github.com/BerriAI/litellm/pull/23834), [PR #25150](https://github.com/BerriAI/litellm/pull/25150)

- **Teams + Organizations**
    - Resolve access-group models / MCP servers / agents in team endpoints and UI - [PR #25027](https://github.com/BerriAI/litellm/pull/25027), [PR #25119](https://github.com/BerriAI/litellm/pull/25119)
    - Allow changing team organization from team settings - [PR #25095](https://github.com/BerriAI/litellm/pull/25095)
    - Per-model rate limits in team edit/info views - [PR #25144](https://github.com/BerriAI/litellm/pull/25144), [PR #25156](https://github.com/BerriAI/litellm/pull/25156)
    - Fix team model update 500 due to unsupported Prisma JSON path filter - [PR #25152](https://github.com/BerriAI/litellm/pull/25152)
    - Team model-group name routing fix - [PR #24688](https://github.com/BerriAI/litellm/pull/24688)
    - Modernize teams table - [PR #24189](https://github.com/BerriAI/litellm/pull/24189)
    - Team-member budget duration on create - [PR #23484](https://github.com/BerriAI/litellm/pull/23484)
    - Add missing `team_member_budget_duration` param to `new_team` docstring - [PR #24243](https://github.com/BerriAI/litellm/pull/24243)
    - Fix teams table refresh, infinite dropdown, and leftnav migration - [PR #24342](https://github.com/BerriAI/litellm/pull/24342)

- **Usage + Analytics**
    - Paginated team search on usage page filters - [PR #25107](https://github.com/BerriAI/litellm/pull/25107)
    - Use entity key for usage export display correctness - [PR #25153](https://github.com/BerriAI/litellm/pull/25153)
    - Aggregated activity entity breakdown - [PR #23471](https://github.com/BerriAI/litellm/pull/23471)
    - CSV export fixes - [PR #23819](https://github.com/BerriAI/litellm/pull/23819)
    - Audit log S3 export - [PR #23167](https://github.com/BerriAI/litellm/pull/23167)
    - Audit log export UI - [PR #24486](https://github.com/BerriAI/litellm/pull/24486)

- **Models + Providers**
    - Include access-group models in UI model listing - [PR #24743](https://github.com/BerriAI/litellm/pull/24743)
    - Expose Azure Entra ID credential fields in provider forms - [PR #25137](https://github.com/BerriAI/litellm/pull/25137)
    - Do not inject `vector_store_ids: []` when editing a model - [PR #25133](https://github.com/BerriAI/litellm/pull/25133)

- **Guardrails UI**
    - Project-level guardrails in project create/edit flows - [PR #25100](https://github.com/BerriAI/litellm/pull/25100)
    - Project-level guardrails support in the proxy - [PR #25087](https://github.com/BerriAI/litellm/pull/25087)
    - Allow adding team guardrails from the UI - [PR #25038](https://github.com/BerriAI/litellm/pull/25038)

- **MCP Toolsets UI**
    - New Toolsets tab for curated MCP tool subsets with scoped permissions - [PR #25155](https://github.com/BerriAI/litellm/pull/25155)

- **Auth / SSO**
    - Fix SSO return-to validation - [PR #24475](https://github.com/BerriAI/litellm/pull/24475)
    - Fix JWT role mappings - [PR #24701](https://github.com/BerriAI/litellm/pull/24701)
    - JWT `none` guard hardening - [PR #24706](https://github.com/BerriAI/litellm/pull/24706)
    - JWT to Virtual Key mapping docs - [PR #24882](https://github.com/BerriAI/litellm/pull/24882)
    - Remove login asterisks display - [PR #24318](https://github.com/BerriAI/litellm/pull/24318)
    - Copy `user_id` on click - [PR #24315](https://github.com/BerriAI/litellm/pull/24315)
    - Fix default user perms not synced with UI - [PR #23666](https://github.com/BerriAI/litellm/pull/23666)

- **UI Cleanup / Migration**
    - Migrate Tremor Text/Badge to antd Tag and native spans - [PR #24750](https://github.com/BerriAI/litellm/pull/24750)
    - Migrate default user settings to antd - [PR #23787](https://github.com/BerriAI/litellm/pull/23787)
    - Migrate route preview Tremor → antd - [PR #24485](https://github.com/BerriAI/litellm/pull/24485)
    - Migrate antd message to context API - [PR #24192](https://github.com/BerriAI/litellm/pull/24192)
    - Extract `useChatHistory` hook - [PR #24172](https://github.com/BerriAI/litellm/pull/24172)
    - Left-nav external icon - [PR #24069](https://github.com/BerriAI/litellm/pull/24069)
    - Vitest coverage for UI - [PR #24144](https://github.com/BerriAI/litellm/pull/24144)

#### Bugs

- Fix logs page showing unfiltered results when backend filter returns zero rows - [PR #24745](https://github.com/BerriAI/litellm/pull/24745)
- Fix UI logs filter - [PR #23792](https://github.com/BerriAI/litellm/pull/23792)
- Fix edit budget flow - [PR #24711](https://github.com/BerriAI/litellm/pull/24711)
- Fix bulk update - [PR #24708](https://github.com/BerriAI/litellm/pull/24708)
- Fix user cache invalidation - [PR #24717](https://github.com/BerriAI/litellm/pull/24717)
- Fix guardrail mode type crash - [PR #24035](https://github.com/BerriAI/litellm/pull/24035)
- Sanitize proxy inputs - [PR #24624](https://github.com/BerriAI/litellm/pull/24624)

## AI Integrations

### Logging

- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Fix Langfuse usage metadata - [PR #24043](https://github.com/BerriAI/litellm/pull/24043)
    - Fix Langfuse OTEL traceparent propagation - [PR #24048](https://github.com/BerriAI/litellm/pull/24048)
    - Re-apply Langfuse key-leakage fix - [PR #22188](https://github.com/BerriAI/litellm/pull/22188), revert [PR #23868](https://github.com/BerriAI/litellm/pull/23868)

- **[Prometheus](../../docs/proxy/logging#prometheus)**
    - Organization budget metrics - [PR #24449](https://github.com/BerriAI/litellm/pull/24449)
    - Prometheus spend metadata - [PR #24434](https://github.com/BerriAI/litellm/pull/24434)

- **General**
    - Centralize logging kwarg updates via a single update function - [PR #23659](https://github.com/BerriAI/litellm/pull/23659)
    - Fix failure callbacks silently skipped when customLogger is not initialized - [PR #24826](https://github.com/BerriAI/litellm/pull/24826)
    - Eliminate race condition in streaming `guardrail_information` logging - [PR #24592](https://github.com/BerriAI/litellm/pull/24592)
    - Use actual `start_time` in failed request spend logs - [PR #24906](https://github.com/BerriAI/litellm/pull/24906)
    - Harden credential redaction and stop logging raw sensitive auth values - [PR #25151](https://github.com/BerriAI/litellm/pull/25151), [PR #24305](https://github.com/BerriAI/litellm/pull/24305)
    - Filter metadata by `user_id` - [PR #24661](https://github.com/BerriAI/litellm/pull/24661)
    - Batch metrics improvements - [PR #24691](https://github.com/BerriAI/litellm/pull/24691)
    - Filter metadata hidden params in streaming - [PR #24220](https://github.com/BerriAI/litellm/pull/24220)
    - Shared aiohttp session auto-recovery - [PR #23808](https://github.com/BerriAI/litellm/pull/23808)
    - Deferred guardrail logging v2 - [PR #24135](https://github.com/BerriAI/litellm/pull/24135)

### Guardrails

- Register DynamoAI guardrail initializer and enum entry - [PR #23752](https://github.com/BerriAI/litellm/pull/23752)
- Extract helper methods in guardrail handlers to fix PLR0915 - [PR #24802](https://github.com/BerriAI/litellm/pull/24802)
- Add optional `on_error` fallback for guardrail pipeline failures - [PR #24831](https://github.com/BerriAI/litellm/pull/24831), [PR #25150](https://github.com/BerriAI/litellm/pull/25150)
- Allow teams to attach/manage their own guardrails from team settings - [PR #25038](https://github.com/BerriAI/litellm/pull/25038)
- Project-level guardrail config in create/edit flows - [PR #25100](https://github.com/BerriAI/litellm/pull/25100)
- Return HTTP 400 (vs 500) for Model Armor streaming blocks - [PR #24693](https://github.com/BerriAI/litellm/pull/24693)
- Deferred guardrail logging v2 - [PR #24135](https://github.com/BerriAI/litellm/pull/24135)
- Eliminate race condition in streaming `guardrail_information` logging - [PR #24592](https://github.com/BerriAI/litellm/pull/24592)
- Model-level guardrails on non-streaming post-call - [PR #23774](https://github.com/BerriAI/litellm/pull/23774)
- Guardrail post-call logging fix - [PR #23910](https://github.com/BerriAI/litellm/pull/23910)
- Missing guardrails docs - [PR #24083](https://github.com/BerriAI/litellm/pull/24083)

### Prompt Management

- Environment + user tracking for prompts (`development/staging/production`) in CRUD + UI flows - [PR #24855](https://github.com/BerriAI/litellm/pull/24855), [PR #25110](https://github.com/BerriAI/litellm/pull/25110)
- Prompt-to-responses integration - [PR #23999](https://github.com/BerriAI/litellm/pull/23999)

### Secret Managers

- No new secret manager provider additions in this release.

## Spend Tracking, Budgets and Rate Limiting

- Enforce budget for models not directly present in the cost map - [PR #24949](https://github.com/BerriAI/litellm/pull/24949)
- Per-model rate limits in team settings/info UI - [PR #25144](https://github.com/BerriAI/litellm/pull/25144), [PR #25156](https://github.com/BerriAI/litellm/pull/25156)
- Prometheus organization budget metrics - [PR #24449](https://github.com/BerriAI/litellm/pull/24449)
- Prometheus spend metadata - [PR #24434](https://github.com/BerriAI/litellm/pull/24434)
- Fix unversioned Vertex Claude Haiku pricing entry to avoid `$0.00` accounting - [PR #25151](https://github.com/BerriAI/litellm/pull/25151)
- Fix budget/spend counters - [PR #24682](https://github.com/BerriAI/litellm/pull/24682)
- Project ID tracking in spend logs - [PR #24432](https://github.com/BerriAI/litellm/pull/24432)
- Dynamic rate-limit pre-ratelimit background refresh - [PR #24106](https://github.com/BerriAI/litellm/pull/24106)
- Point72 limits changes - [PR #24088](https://github.com/BerriAI/litellm/pull/24088)
- Model-level affinity in router - [PR #24110](https://github.com/BerriAI/litellm/pull/24110)

## MCP Gateway

- Introduce **MCP Toolsets** with DB types, CRUD APIs, scoped permissions, and UI management tab - [PR #25155](https://github.com/BerriAI/litellm/pull/25155)
- Resolve toolset names and enforce toolset access correctly in Responses API and streamable MCP paths - [PR #25155](https://github.com/BerriAI/litellm/pull/25155)
- Switch toolset permission caching to shared cache path and improve cache invalidation behavior - [PR #25155](https://github.com/BerriAI/litellm/pull/25155)
- Allow JWT auth for `/v1/mcp/server/*` sub-paths - [PR #24698](https://github.com/BerriAI/litellm/pull/24698), [PR #25113](https://github.com/BerriAI/litellm/pull/25113)
- Add STS AssumeRole support for MCP SigV4 auth - [PR #25151](https://github.com/BerriAI/litellm/pull/25151)
- Tag query fix + MCP metadata support cherry-pick - [PR #25145](https://github.com/BerriAI/litellm/pull/25145)
- MCP REST M2M OAuth2 flow - [PR #23468](https://github.com/BerriAI/litellm/pull/23468)
- Upgrade MCP SDK to 1.26.0 - [PR #24179](https://github.com/BerriAI/litellm/pull/24179)
- Restore MCP server fields dropped by schema sync migration - [PR #24078](https://github.com/BerriAI/litellm/pull/24078)

## Performance / Loadbalancing / Reliability improvements

- Add control plane for multi-proxy worker management - [PR #24217](https://github.com/BerriAI/litellm/pull/24217)
- Make DB migration failure exit opt-in via `--enforce_prisma_migration_check` - [PR #23675](https://github.com/BerriAI/litellm/pull/23675)
- Return the picked model (not a comma-separated list) when batch completions is used - [PR #24753](https://github.com/BerriAI/litellm/pull/24753)
- Fix mypy type errors in Responses transformation, spend tracking, and PagerDuty - [PR #24803](https://github.com/BerriAI/litellm/pull/24803)
- Fix router code coverage CI failure for health check filter tests - [PR #24812](https://github.com/BerriAI/litellm/pull/24812)
- Integrate router health-check failures with cooldown behavior and transient 429/408 handling - [PR #24988](https://github.com/BerriAI/litellm/pull/24988), [PR #25150](https://github.com/BerriAI/litellm/pull/25150)
- Add distributed lock for key rotation job execution - [PR #23364](https://github.com/BerriAI/litellm/pull/23364), [PR #23834](https://github.com/BerriAI/litellm/pull/23834), [PR #25150](https://github.com/BerriAI/litellm/pull/25150)
- Improve team routing reliability with deterministic grouping, isolation fixes, stale alias controls, and order-based fallback - [PR #25148](https://github.com/BerriAI/litellm/pull/25148), [PR #25154](https://github.com/BerriAI/litellm/pull/25154)
- Regenerate GCP IAM token per async Redis cluster connection (fix token TTL failures) - [PR #24426](https://github.com/BerriAI/litellm/pull/24426), [PR #25155](https://github.com/BerriAI/litellm/pull/25155)
- Proxy server reliability hardening with bounded queue usage - [PR #25155](https://github.com/BerriAI/litellm/pull/25155)
- Auto schema sync on startup - [PR #24705](https://github.com/BerriAI/litellm/pull/24705)
- Kill orphaned Prisma engine on reconnect - [PR #24149](https://github.com/BerriAI/litellm/pull/24149)
- Use dynamic DB URL - [PR #24827](https://github.com/BerriAI/litellm/pull/24827)
- Migration corrections - [PR #24105](https://github.com/BerriAI/litellm/pull/24105)

## Documentation Updates

- MCP zero trust auth guide - [PR #23918](https://github.com/BerriAI/litellm/pull/23918)
- Week 1 onboarding checklist - [PR #25083](https://github.com/BerriAI/litellm/pull/25083)
- Remove `NLP_CLOUD_API_KEY` requirement from `test_exceptions` - [PR #24756](https://github.com/BerriAI/litellm/pull/24756)
- Update `gemini-2.0-flash` to `gemini-2.5-flash` in `test_gemini` - [PR #24817](https://github.com/BerriAI/litellm/pull/24817)
- HA control-plane diagram clarity + mobile rendering updates - [PR #24747](https://github.com/BerriAI/litellm/pull/24747)
- Document `default_team_params` in config reference and examples - [PR #25032](https://github.com/BerriAI/litellm/pull/25032)
- JWT to Virtual Key mapping guide - [PR #24882](https://github.com/BerriAI/litellm/pull/24882)
- MCP Toolsets docs and sidebar updates - [PR #25155](https://github.com/BerriAI/litellm/pull/25155)
- Security docs updates and April hardening blog - [PR #24867](https://github.com/BerriAI/litellm/pull/24867), [PR #24868](https://github.com/BerriAI/litellm/pull/24868), [PR #24871](https://github.com/BerriAI/litellm/pull/24871), [PR #25102](https://github.com/BerriAI/litellm/pull/25102)
- Security incident blog - [PR #24537](https://github.com/BerriAI/litellm/pull/24537)
- Security townhall blog - [PR #24692](https://github.com/BerriAI/litellm/pull/24692)
- WebRTC blog - [PR #23547](https://github.com/BerriAI/litellm/pull/23547)
- Vanta announcement - [PR #24800](https://github.com/BerriAI/litellm/pull/24800)
- Prompt caching Gemini support docs - [PR #24222](https://github.com/BerriAI/litellm/pull/24222)
- OpenCode / reasoningSummary docs - [PR #24468](https://github.com/BerriAI/litellm/pull/24468)
- Thinking summary docs - [PR #22823](https://github.com/BerriAI/litellm/pull/22823)
- v0 docs contributions - [PR #24023](https://github.com/BerriAI/litellm/pull/24023)
- Blog posts RSS update - [PR #23791](https://github.com/BerriAI/litellm/pull/23791)
- General docs cleanup + townhall announcements - [PR #24839](https://github.com/BerriAI/litellm/pull/24839), [PR #25021](https://github.com/BerriAI/litellm/pull/25021), [PR #25026](https://github.com/BerriAI/litellm/pull/25026)

## Infrastructure / Security Notes

- Optimize CI pipeline - [PR #23721](https://github.com/BerriAI/litellm/pull/23721)
- Add zizmor to CI/CD - [PR #24663](https://github.com/BerriAI/litellm/pull/24663)
- Remove `.claude/settings.json` and block re-adding via semgrep - [PR #24584](https://github.com/BerriAI/litellm/pull/24584)
- Harden npm and Docker supply chain workflows and release pipeline checks - [PR #24838](https://github.com/BerriAI/litellm/pull/24838), [PR #24877](https://github.com/BerriAI/litellm/pull/24877), [PR #24881](https://github.com/BerriAI/litellm/pull/24881), [PR #24905](https://github.com/BerriAI/litellm/pull/24905), [PR #24951](https://github.com/BerriAI/litellm/pull/24951), [PR #25023](https://github.com/BerriAI/litellm/pull/25023), [PR #25034](https://github.com/BerriAI/litellm/pull/25034), [PR #25036](https://github.com/BerriAI/litellm/pull/25036), [PR #25037](https://github.com/BerriAI/litellm/pull/25037), [PR #25136](https://github.com/BerriAI/litellm/pull/25136), [PR #25158](https://github.com/BerriAI/litellm/pull/25158)
- Resolve CodeQL/security workflow issues and fix broken action SHA references - [PR #24815](https://github.com/BerriAI/litellm/pull/24815), [PR #24880](https://github.com/BerriAI/litellm/pull/24880), [PR #24697](https://github.com/BerriAI/litellm/pull/24697)
- Pin axios and tool versions - [PR #24829](https://github.com/BerriAI/litellm/pull/24829), [PR #24594](https://github.com/BerriAI/litellm/pull/24594), [PR #24607](https://github.com/BerriAI/litellm/pull/24607), [PR #24525](https://github.com/BerriAI/litellm/pull/24525), [PR #24696](https://github.com/BerriAI/litellm/pull/24696)
- Re-add Codecov reporting in GHA matrix workflows - [PR #24804](https://github.com/BerriAI/litellm/pull/24804), [PR #24815](https://github.com/BerriAI/litellm/pull/24815)
- Fix(docker): load enterprise hooks in non-root runtime image - [PR #24917](https://github.com/BerriAI/litellm/pull/24917), [PR #25037](https://github.com/BerriAI/litellm/pull/25037)
- OSSF scorecard workflow - [PR #24792](https://github.com/BerriAI/litellm/pull/24792)
- Skip scheduled workflows on forks - [PR #24460](https://github.com/BerriAI/litellm/pull/24460)
- CI/CD improvements - [PR #24839](https://github.com/BerriAI/litellm/pull/24839), [PR #24837](https://github.com/BerriAI/litellm/pull/24837), [PR #24740](https://github.com/BerriAI/litellm/pull/24740), [PR #24741](https://github.com/BerriAI/litellm/pull/24741), [PR #24742](https://github.com/BerriAI/litellm/pull/24742), [PR #24754](https://github.com/BerriAI/litellm/pull/24754)
- Remove neon CLI dependency - [PR #24951](https://github.com/BerriAI/litellm/pull/24951)
- Workflow deletions - [PR #24541](https://github.com/BerriAI/litellm/pull/24541)
- Publish to PyPI migration - [PR #24654](https://github.com/BerriAI/litellm/pull/24654)
- Poetry lock / content-hash checks - [PR #24082](https://github.com/BerriAI/litellm/pull/24082), [PR #24159](https://github.com/BerriAI/litellm/pull/24159)
- Apply Black formatting to 14 files - [PR #24532](https://github.com/BerriAI/litellm/pull/24532), [PR #24092](https://github.com/BerriAI/litellm/pull/24092), [PR #24153](https://github.com/BerriAI/litellm/pull/24153), [PR #24167](https://github.com/BerriAI/litellm/pull/24167), [PR #24173](https://github.com/BerriAI/litellm/pull/24173), [PR #24187](https://github.com/BerriAI/litellm/pull/24187)
- Fix lint issues - [PR #24932](https://github.com/BerriAI/litellm/pull/24932)
- Version bump to 1.83.0 - [PR #24840](https://github.com/BerriAI/litellm/pull/24840)
- Test cleanup and reliability fixes - [PR #24755](https://github.com/BerriAI/litellm/pull/24755), [PR #24820](https://github.com/BerriAI/litellm/pull/24820), [PR #24824](https://github.com/BerriAI/litellm/pull/24824), [PR #24258](https://github.com/BerriAI/litellm/pull/24258)
- License key environment handling - [PR #24168](https://github.com/BerriAI/litellm/pull/24168)
- Remove phone numbers from repo - [PR #24587](https://github.com/BerriAI/litellm/pull/24587)

## New Contributors

* @voidborne-d made their first contribution in https://github.com/BerriAI/litellm/pull/23808
* @vanhtuan0409 made their first contribution in https://github.com/BerriAI/litellm/pull/24078
* @devin-petersohn made their first contribution in https://github.com/BerriAI/litellm/pull/24140
* @benlangfeld made their first contribution in https://github.com/BerriAI/litellm/pull/24413
* @J-Byron made their first contribution in https://github.com/BerriAI/litellm/pull/24449
* @jaydns made their first contribution in https://github.com/BerriAI/litellm/pull/24823
* @stuxf made their first contribution in https://github.com/BerriAI/litellm/pull/24838
* @clfhhc made their first contribution in https://github.com/BerriAI/litellm/pull/24932

**Full Changelog**: https://github.com/BerriAI/litellm/compare/v1.82.3-stable...v1.83.3-stable

---

## 04/04/2026

* New Models / Updated Models: 59
* LLM API Endpoints: 28
* Management Endpoints / UI: 61
* Logging / Guardrail / Prompt Management Integrations: 30
* Spend Tracking, Budgets and Rate Limiting: 11
* MCP Gateway: 8
* Performance / Loadbalancing / Reliability improvements: 17
* Documentation Updates: 24
* Infrastructure / Security: 50
