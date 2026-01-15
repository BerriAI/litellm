---
title: "v1.74.9-stable - Auto-Router"
slug: "v1-74-9"
date: 2025-07-27T10:00:00
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
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
docker.litellm.ai/berriai/litellm:v1.74.9-stable.patch.1
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.74.9.post2
```

</TabItem>
</Tabs>

---

## Key Highlights

- **Auto-Router** - Automatically route requests to specific models based on request content.
- **Model-level Guardrails** - Only run guardrails when specific models are used.
- **MCP Header Propagation** - Propagate headers from client to backend MCP.
- **New LLM Providers** - Added Bedrock inpainting support and Recraft API image generation  / image edits support.

---

## Auto-Router

<Image img={require('../../img/release_notes/auto_router.png')} />

<br/>

This release introduces auto-routing to models based on request content. This means **Proxy Admins** can define a set of keywords that always routes to specific models when **users** opt in to using the auto-router.

This is great for internal use cases where you don't want **users** to think about which model to use - for example, use Claude models for coding vs GPT models for generating ad copy.


[Read More](../../docs/proxy/auto_routing)

---

## Model-level Guardrails

<Image img={require('../../img/release_notes/model_level_guardrails.jpg')} />

<br/>

This release brings model-level guardrails support to your config.yaml + UI. This is great for cases when you have an on-prem and hosted model, and just want to run prevent sending PII to the hosted model.

```yaml
model_list:
  - model_name: claude-sonnet-4
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
      api_base: https://api.anthropic.com/v1
      guardrails: ["azure-text-moderation"] # 👈 KEY CHANGE

guardrails:
  - guardrail_name: azure-text-moderation
    litellm_params:
      guardrail: azure/text_moderations
      mode: "post_call" 
      api_key: os.environ/AZURE_GUARDRAIL_API_KEY
      api_base: os.environ/AZURE_GUARDRAIL_API_BASE 
```


[Read More](../../docs/proxy/guardrails/quick_start#model-level-guardrails)

---
## MCP Header Propagation

<Image img={require('../../img/release_notes/mcp_header_propogation.png')} />

<br/>

v1.74.9-stable allows you to propagate MCP server specific authentication headers via LiteLLM

- Allowing users to specify which `header_name` is to be propagated to which `mcp_server` via headers
- Allows adding of different deployments of same MCP server type to use different authentication headers


[Read More](https://docs.litellm.ai/docs/mcp#new-server-specific-auth-headers-recommended)

---
## New Models / Updated Models

#### Pricing / Context Window Updates

| Provider    | Model                                  | Context Window | Input ($/1M tokens) | Output ($/1M tokens) |
| ----------- | -------------------------------------- | -------------- | ------------------- | -------------------- |
| Fireworks AI | `fireworks/models/kimi-k2-instruct` | 131k | $0.6 | $2.5 | 
| OpenRouter | `openrouter/qwen/qwen-vl-plus` | 8192 | $0.21 | $0.63 | 
| OpenRouter | `openrouter/qwen/qwen3-coder` | 8192 | $1 | $5 | 
| OpenRouter | `openrouter/bytedance/ui-tars-1.5-7b` | 128k | $0.10 | $0.20 | 
| Groq | `groq/qwen/qwen3-32b` | 131k | $0.29 | $0.59 | 
| VertexAI | `vertex_ai/meta/llama-3.1-8b-instruct-maas` | 128k | $0.00 | $0.00 | 
| VertexAI | `vertex_ai/meta/llama-3.1-405b-instruct-maas` | 128k | $5 | $16 | 
| VertexAI | `vertex_ai/meta/llama-3.2-90b-vision-instruct-maas` | 128k | $0.00 | $0.00 | 
| Google AI Studio | `gemini/gemini-2.0-flash-live-001` | 1,048,576 | $0.35 | $1.5 | 
| Google AI Studio | `gemini/gemini-2.5-flash-lite` | 1,048,576 | $0.1 | $0.4 | 
| VertexAI | `vertex_ai/gemini-2.0-flash-lite-001` | 1,048,576 | $0.35 | $1.5 | 
| OpenAI | `gpt-4o-realtime-preview-2025-06-03` | 128k | $5 | $20 |

#### Features

- **[Lambda AI](../../docs/providers/lambda_ai)**
    - New LLM API provider - [PR #12817](https://github.com/BerriAI/litellm/pull/12817)
- **[Github Copilot](../../docs/providers/github_copilot)**
    - Dynamic endpoint support - [PR #12827](https://github.com/BerriAI/litellm/pull/12827)
- **[Morph](../../docs/providers/morph)**
    - New LLM API provider - [PR #12821](https://github.com/BerriAI/litellm/pull/12821)
- **[Groq](../../docs/providers/groq)**
    - Remove deprecated groq/qwen-qwq-32b - [PR #12832](https://github.com/BerriAI/litellm/pull/12831)
- **[Recraft](../../docs/providers/recraft)**
    - New image generation API - [PR #12832](https://github.com/BerriAI/litellm/pull/12832)
    - New image edits api - [PR #12874](https://github.com/BerriAI/litellm/pull/12874)
- **[Azure OpenAI](../../docs/providers/azure/azure)**
    - Support DefaultAzureCredential without hard-coded environment variables - [PR #12841](https://github.com/BerriAI/litellm/pull/12841)
- **[Hyperbolic](../../docs/providers/hyperbolic)**
    - New LLM API provider - [PR #12826](https://github.com/BerriAI/litellm/pull/12826)
- **[OpenAI](../../docs/providers/openai)**
    - `/realtime` API - pass through intent query param - [PR #12838](https://github.com/BerriAI/litellm/pull/12838)
- **[Bedrock](../../docs/providers/bedrock)**
    - Add inpainting support for Amazon Nova Canvas - [PR #12949](https://github.com/BerriAI/litellm/pull/12949) s/o @[SantoshDhaladhuli](https://github.com/SantoshDhaladhuli)

#### Bugs
- **Gemini ([Google AI Studio](../../docs/providers/gemini) + [VertexAI](../../docs/providers/vertex))**
    - Fix leaking file descriptor error on sync calls - [PR #12824](https://github.com/BerriAI/litellm/pull/12824)
- **IBM Watsonx**
    - use correct parameter name for tool choice - [PR #9980](https://github.com/BerriAI/litellm/pull/9980)
- **[Anthropic](../../docs/providers/anthropic)**
    - Only show ‘reasoning_effort’ for supported models - [PR #12847](https://github.com/BerriAI/litellm/pull/12847)
    - Handle $id and $schema in tool call requests (Anthropic API stopped accepting them) - [PR #12959](https://github.com/BerriAI/litellm/pull/12959)
- **[Openrouter](../../docs/providers/openrouter)**
    - filter out cache_control flag for non-anthropic models (allows usage with claude code) https://github.com/BerriAI/litellm/pull/12850
- **[Gemini](../../docs/providers/gemini)**
    - Shorten Gemini tool_call_id for Open AI compatibility - [PR #12941](https://github.com/BerriAI/litellm/pull/12941) s/o @[tonga54](https://github.com/tonga54)

---

## LLM API Endpoints

#### Features

- **[Passthrough endpoints](../../docs/pass_through/)**
    - Make key/user/team cost tracking OSS - [PR #12847](https://github.com/BerriAI/litellm/pull/12847)
- **[/v1/models](../../docs/providers/passthrough)**
    - Return fallback models as part of api response - [PR #12811](https://github.com/BerriAI/litellm/pull/12811) s/o @[murad-khafizov](https://github.com/murad-khafizov)
- **[/vector_stores](../../docs/providers/passthrough)**
    - Make permission management OSS - [PR #12990](https://github.com/BerriAI/litellm/pull/12990)

#### Bugs
1. `/batches`
    1. Skip invalid batch during cost tracking check (prev. Would stop all checks) - [PR #12782](https://github.com/BerriAI/litellm/pull/12782)
2. `/chat/completions`
    1. Fix async retryer on .acompletion() - [PR #12886](https://github.com/BerriAI/litellm/pull/12886)

---

## [MCP Gateway](../../docs/mcp)

#### Features
- **[Permission Management](../../docs/mcp#grouping-mcps-access-groups)**
    - Make permission management by key/team OSS - [PR #12988](https://github.com/BerriAI/litellm/pull/12988)
- **[MCP Alias](../../docs/mcp#mcp-aliases)**
    - Support mcp server aliases (useful for calling long mcp server names on Cursor) - [PR #12994](https://github.com/BerriAI/litellm/pull/12994)
- **Header Propagation**
    - Support propagating headers from client to backend MCP (useful for sending personal access tokens to backend MCP) - [PR #13003](https://github.com/BerriAI/litellm/pull/13003)

---

## Management Endpoints / UI

#### Features
- **Usage**
    - Support viewing usage by model group - [PR #12890](https://github.com/BerriAI/litellm/pull/12890)
- **Virtual Keys**
    - New `key_type` field on `/key/generate` - allows specifying if key can call LLM API vs. Management routes - [PR #12909](https://github.com/BerriAI/litellm/pull/12909)
- **Models**
    - Add ‘auto router’ on UI - [PR #12960](https://github.com/BerriAI/litellm/pull/12960)
    - Show global retry policy on UI - [PR #12969](https://github.com/BerriAI/litellm/pull/12969)
    - Add model-level guardrails on create + update - [PR #13006](https://github.com/BerriAI/litellm/pull/13006)

#### Bugs
- **SSO**
    - Fix logout when SSO is enabled - [PR #12703](https://github.com/BerriAI/litellm/pull/12703)
    - Fix reset SSO when ui_access_mode is updated - [PR #13011](https://github.com/BerriAI/litellm/pull/13011)
- **Guardrails**
    - Show correct guardrails when editing a team - [PR #12823](https://github.com/BerriAI/litellm/pull/12823)
- **Virtual Keys**
    - Get updated token on regenerate key - [PR #12788](https://github.com/BerriAI/litellm/pull/12788)
    - Fix CVE with key injection - [PR #12840](https://github.com/BerriAI/litellm/pull/12840)
---

## Logging / Guardrail Integrations

#### Features
- **[Google Cloud Model Armor](../../docs/proxy/guardrails/model_armor)**
    - Document new guardrail - [PR #12492](https://github.com/BerriAI/litellm/pull/12492)
- **[Pillar Security](../../docs/proxy/guardrails/pillar_security)**
    - New LLM Guardrail - [PR #12791](https://github.com/BerriAI/litellm/pull/12791)
- **CloudZero**
    - Allow exporting spend to cloudzero - [PR #12908](https://github.com/BerriAI/litellm/pull/12908)
- **Model-level Guardrails**
    - Support model-level guardrails - [PR #12968](https://github.com/BerriAI/litellm/pull/12968)

#### Bugs
- **[Prometheus](../../docs/proxy/prometheus)**
    - Fix `[tag]=false` when tag is set for tag-based metrics - [PR #12916](https://github.com/BerriAI/litellm/pull/12916)
- **[Guardrails AI](../../docs/proxy/guardrails/guardrails_ai)**
    - Use ‘validatedOutput’ to allow usage of “fix” guards - [PR #12891](https://github.com/BerriAI/litellm/pull/12891) s/o @[DmitriyAlergant](https://github.com/DmitriyAlergant)

---

## Performance / Loadbalancing / Reliability improvements

#### Features
- **[Auto-Router](../../docs/proxy/auto_routing)**
    - New auto-router powered by `semantic-router` - [PR #12955](https://github.com/BerriAI/litellm/pull/12955)

#### Bugs
- **forward_clientside_headers**
    - Filter out `content-length` from headers (caused backend requests to hang) - [PR #12886](https://github.com/BerriAI/litellm/pull/12886/files)
- **Message Redaction**
    - Fix cannot pickle coroutine object error - [PR #13005](https://github.com/BerriAI/litellm/pull/13005)
---

## General Proxy Improvements

#### Features
- **Benchmarks**
    - Updated litellm proxy benchmarks (p50, p90, p99 overhead) - [PR #12842](https://github.com/BerriAI/litellm/pull/12842)
- **Request Headers**
    - Added new `x-litellm-num-retries` request header 
- **Swagger**
    - Support local swagger on custom root paths - [PR #12911](https://github.com/BerriAI/litellm/pull/12911)
- **Health**
    - Track cost + add tags for health checks done by LiteLLM Proxy - [PR #12880](https://github.com/BerriAI/litellm/pull/12880)
#### Bugs

- **Proxy Startup**
    - Fixes issue on startup where team member budget is None would block startup - [PR #12843](https://github.com/BerriAI/litellm/pull/12843)
- **Docker**
    - Move non-root docker to chain guard image (fewer vulnerabilities) - [PR #12707](https://github.com/BerriAI/litellm/pull/12707)
    - add azure-keyvault==4.2.0 to Docker img - [PR #12873](https://github.com/BerriAI/litellm/pull/12873)
- **Separate Health App**
    - Pass through cmd args via supervisord (enables user config to still work via docker) - [PR #12871](https://github.com/BerriAI/litellm/pull/12871)
- **Swagger**
    - Bump DOMPurify version (fixes vulnerability) - [PR #12911](https://github.com/BerriAI/litellm/pull/12911)
    - Add back local swagger bundle (enables swagger to work in air gapped env.) - [PR #12911](https://github.com/BerriAI/litellm/pull/12911)
- **Request Headers**
    - Make ‘user_header_name’ field check case insensitive (fixes customer budget enforcement for OpenWebUi) - [PR #12950](https://github.com/BerriAI/litellm/pull/12950)
- **SpendLogs**
    - Fix issues writing to DB when custom_llm_provider is None - [PR #13001](https://github.com/BerriAI/litellm/pull/13001)

---

## New Contributors
* @magicalne made their first contribution in https://github.com/BerriAI/litellm/pull/12804
* @pavangudiwada made their first contribution in https://github.com/BerriAI/litellm/pull/12798
* @mdiloreto made their first contribution in https://github.com/BerriAI/litellm/pull/12707
* @murad-khafizov made their first contribution in https://github.com/BerriAI/litellm/pull/12811
* @eagle-p made their first contribution in https://github.com/BerriAI/litellm/pull/12791
* @apoorv-sharma made their first contribution in https://github.com/BerriAI/litellm/pull/12920
* @SantoshDhaladhuli made their first contribution in https://github.com/BerriAI/litellm/pull/12949
* @tonga54 made their first contribution in https://github.com/BerriAI/litellm/pull/12941
* @sings-to-bees-on-wednesdays made their first contribution in https://github.com/BerriAI/litellm/pull/12950

## **[Full Changelog](https://github.com/BerriAI/litellm/compare/v1.74.7-stable...v1.74.9.rc-draft)**
