---
title: "v1.82.3 - Nebius AI, gpt-5.4, Gemini 3.x, FLUX Kontext, and 116 New Models"
slug: "v1-82-3"
date: 2026-03-16T00:00:00
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

## Deploy this version

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:main-1.82.3-stable
```

</TabItem>
<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.82.3
```

</TabItem>
</Tabs>

## Key Highlights

- **Nebius AI — new provider** — [30 models across DeepSeek, Qwen, Llama, Mistral, NVIDIA, and BAAI available via Nebius AI cloud](../../docs/providers/nebius) - [PR #22614](https://github.com/BerriAI/litellm/pull/22614)
- **OpenAI gpt-5.4 / gpt-5.4-pro — day 0** — Full pricing and routing support for `gpt-5.4` (1M context, $2.50/$15.00) and `gpt-5.4-pro` ($30.00/$180.00) on OpenAI and Azure
- **Gemini 3.x models** — `gemini-3-flash-preview`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-image-preview`, and `gemini-embedding-2-preview` added to cost map for Google AI and Vertex AI
- **FLUX Kontext image editing** — `flux-kontext-pro` and `flux-kontext-max` added to Black Forest Labs, alongside `flux-pro-1.0-fill` and `flux-pro-1.0-expand` for inpainting and outpainting
- **116 new models, 132 deprecated models cleaned up** — Major model map refresh including Mistral Magistral, Dashscope Qwen3 VL, xAI Grok via Azure AI, ZAI GLM-5, Serper Search; removal of OpenAI GPT-3.5/GPT-4 legacy variants, Gemini 1.5, and Vertex AI PaLM2
- **SageMaker Nova provider** — [New `sagemaker_nova` provider for Amazon Nova models on SageMaker](../../docs/providers/aws_sagemaker) - [PR #21542](https://github.com/BerriAI/litellm/pull/21542)
- **Secret redaction in logs** — API keys, tokens, and credentials automatically scrubbed from all proxy log output. Enabled by default; opt out with `LITELLM_DISABLE_REDACT_SECRETS=true` - [PR #23668](https://github.com/BerriAI/litellm/pull/23668)
- **Streaming stability fix** — Critical fix for `RuntimeError: Cannot send a request, as the client has been closed.` crashes after ~1 hour in production - [PR #22926](https://github.com/BerriAI/litellm/pull/22926)

---

## New Providers and Endpoints

### New Providers (5 new providers)

| Provider | Supported LiteLLM Endpoints | Description |
| -------- | --------------------------- | ----------- |
| [Nebius AI](../../docs/providers/nebius) (`nebius/`) | `/chat/completions`, `/embeddings` | EU-based AI cloud with 30+ open models — DeepSeek, Qwen3, Llama 3.1/3.3, NVIDIA Nemotron, BAAI embeddings |
| [ZAI](../../docs/providers/zai) (`zai/`) | `/chat/completions` | ZhipuAI GLM-5 models via ZAI cloud |
| [Black Forest Labs](../../docs/providers/black_forest_labs) (`black_forest_labs/`) | `/images/generations`, `/images/edits` | FLUX image generation and editing — Kontext Pro/Max, Pro 1.0 Fill/Expand |
| [Serper](../../docs/providers/serper) (`serper/`) | `/search` | Web search via Serper API |
| [SageMaker Nova](../../docs/providers/aws_sagemaker) (`sagemaker_nova/`) | `/chat/completions` | Amazon Nova models via SageMaker endpoint |

---

## New Models / Updated Models

#### New Model Support (116 new models)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| OpenAI | `gpt-5.4` | 1.05M | $2.50 | $15.00 | chat, vision, tools, reasoning |
| OpenAI | `gpt-5.4-pro` | 1.05M | $30.00 | $180.00 | responses, vision, tools, reasoning |
| OpenAI | `gpt-5.3-chat-latest` | 128K | $1.75 | $14.00 | chat, vision, tools, reasoning |
| Azure OpenAI | `azure/gpt-5.4` | 1.05M | $2.50 | $15.00 | chat, vision, tools, reasoning |
| Azure OpenAI | `azure/gpt-5.4-pro` | 1.05M | $30.00 | $180.00 | responses, vision, tools, reasoning |
| Azure OpenAI | `azure/gpt-5.3-chat` | 128K | $1.75 | $14.00 | chat, vision, tools, reasoning |
| Google Gemini | `gemini/gemini-3-flash-preview` | 1M | $0.50 | $3.00 | chat, vision, tools, reasoning |
| Google Gemini | `gemini/gemini-3.1-pro-preview` | 1M | $2.00 | $12.00 | chat, vision, tools, reasoning |
| Google Gemini | `gemini/gemini-3.1-flash-image-preview` | 65K | $0.25 | $1.50 | image generation, vision |
| Google Gemini | `gemini/gemini-3.1-flash-lite-preview` | - | - | - | chat |
| Google Gemini | `gemini/gemini-3-pro-image-preview` | - | - | - | image generation |
| Google Gemini | `gemini/gemini-embedding-2-preview` | 8K | $0.20 | - | embeddings |
| Google Vertex AI | `vertex_ai/gemini-3-flash-preview` | - | - | - | chat |
| Google Vertex AI | `vertex_ai/gemini-3.1-pro-preview` | - | - | - | chat |
| Google Vertex AI | `vertex_ai/gemini-3.1-flash-lite-preview` | - | - | - | chat |
| Google Vertex AI | `vertex_ai/gemini-embedding-2-preview` | - | $0.20 | - | embeddings |
| Mistral | `mistral/magistral-medium-1-2-2509` | 40K | $2.00 | $5.00 | chat, tools, reasoning |
| Mistral | `mistral/magistral-small-1-2-2509` | 40K | $0.50 | $1.50 | chat, tools, reasoning |
| Mistral | `mistral/mistral-large-2512` | 262K | $0.50 | $1.50 | chat, vision, tools |
| Mistral | `mistral/mistral-medium-3-1-2508` | - | - | - | chat |
| Mistral | `mistral/mistral-small-3-2-2506` | - | - | - | chat |
| Mistral | `mistral/ministral-3-3b-2512` | - | - | - | chat |
| Mistral | `mistral/ministral-3-8b-2512` | - | - | - | chat |
| Mistral | `mistral/ministral-3-14b-2512` | - | - | - | chat |
| Black Forest Labs | `black_forest_labs/flux-kontext-pro` | - | - | - | image editing |
| Black Forest Labs | `black_forest_labs/flux-kontext-max` | - | - | - | image editing |
| Black Forest Labs | `black_forest_labs/flux-pro-1.0-fill` | - | - | - | image editing (inpaint) |
| Black Forest Labs | `black_forest_labs/flux-pro-1.0-expand` | - | - | - | image editing (outpaint) |
| Black Forest Labs | `black_forest_labs/flux-pro-1.1` | - | - | - | image generation |
| Black Forest Labs | `black_forest_labs/flux-pro-1.1-ultra` | - | - | - | image generation |
| Black Forest Labs | `black_forest_labs/flux-dev` | - | - | - | image generation |
| Black Forest Labs | `black_forest_labs/flux-pro` | - | - | - | image generation |
| Azure AI | `azure_ai/grok-4-1-fast-non-reasoning` | 131K | $0.20 | $0.50 | chat, tools |
| Azure AI | `azure_ai/grok-4-1-fast-reasoning` | 131K | $0.20 | $0.50 | chat, tools, reasoning |
| Azure AI | `azure_ai/mistral-document-ai-2512` | - | - | - | OCR |
| Dashscope | `dashscope/qwen3-next-80b-a3b-instruct` | 262K | $0.15 | $1.20 | chat |
| Dashscope | `dashscope/qwen3-next-80b-a3b-thinking` | 262K | $0.15 | $1.20 | chat, reasoning |
| Dashscope | `dashscope/qwen3-vl-235b-a22b-instruct` | 131K | $0.40 | $1.60 | chat, vision |
| Dashscope | `dashscope/qwen3-vl-235b-a22b-thinking` | 131K | $0.40 | $4.00 | chat, vision, reasoning |
| Dashscope | `dashscope/qwen3-vl-32b-instruct` | 131K | $0.16 | $0.64 | chat, vision |
| Dashscope | `dashscope/qwen3-vl-32b-thinking` | 131K | $0.16 | $2.87 | chat, vision, reasoning |
| Dashscope | `dashscope/qwen3-vl-plus` | 260K | - | - | chat, vision |
| Dashscope | `dashscope/qwen3.5-plus` | 992K | - | - | chat |
| Dashscope | `dashscope/qwen3-max-2026-01-23` | 258K | - | - | chat |
| Nebius AI | `nebius/deepseek-ai/DeepSeek-R1` | 128K | $0.80 | $2.40 | chat, reasoning |
| Nebius AI | `nebius/deepseek-ai/DeepSeek-R1-0528` | 164K | $0.80 | $2.40 | chat, reasoning |
| Nebius AI | `nebius/deepseek-ai/DeepSeek-V3` | 128K | $0.50 | $1.50 | chat |
| Nebius AI | `nebius/deepseek-ai/DeepSeek-V3-0324` | 128K | $0.50 | $1.50 | chat |
| Nebius AI | `nebius/deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 128K | $0.25 | $0.75 | chat |
| Nebius AI | `nebius/Qwen/Qwen3-235B-A22B` | 262K | $0.20 | $0.60 | chat |
| Nebius AI | `nebius/Qwen/Qwen3-32B` | 32K | $0.10 | $0.30 | chat |
| Nebius AI | `nebius/Qwen/Qwen3-30B-A3B` | 32K | $0.10 | $0.30 | chat |
| Nebius AI | `nebius/Qwen/Qwen3-14B` | 32K | $0.08 | $0.24 | chat |
| Nebius AI | `nebius/Qwen/Qwen3-4B` | 32K | $0.08 | $0.24 | chat |
| Nebius AI | `nebius/Qwen/QwQ-32B` | 32K | $0.15 | $0.45 | chat |
| Nebius AI | `nebius/Qwen/Qwen2.5-72B-Instruct` | 128K | $0.13 | $0.40 | chat |
| Nebius AI | `nebius/Qwen/Qwen2.5-32B-Instruct` | 128K | $0.06 | $0.20 | chat |
| Nebius AI | `nebius/Qwen/Qwen2.5-VL-72B-Instruct` | 131K | $0.13 | $0.40 | chat, vision |
| Nebius AI | `nebius/Qwen/Qwen2-VL-72B-Instruct` | 131K | $0.13 | $0.40 | chat, vision |
| Nebius AI | `nebius/Qwen/Qwen2-VL-7B-Instruct` | 131K | $0.02 | $0.06 | chat, vision |
| Nebius AI | `nebius/meta-llama/Meta-Llama-3.1-405B-Instruct` | 128K | $1.00 | $3.00 | chat |
| Nebius AI | `nebius/meta-llama/Meta-Llama-3.1-70B-Instruct` | 128K | $0.13 | $0.40 | chat |
| Nebius AI | `nebius/meta-llama/Meta-Llama-3.1-8B-Instruct` | 128K | $0.02 | $0.06 | chat |
| Nebius AI | `nebius/meta-llama/Llama-3.3-70B-Instruct` | 128K | $0.13 | $0.40 | chat |
| Nebius AI | `nebius/meta-llama/Llama-Guard-3-8B` | 128K | $0.02 | $0.06 | chat |
| Nebius AI | `nebius/nvidia/Llama-3.1-Nemotron-Ultra-253B-v1` | 128K | $0.60 | $1.80 | chat |
| Nebius AI | `nebius/nvidia/Llama-3.3-Nemotron-Super-49B-v1` | 131K | $0.10 | $0.40 | chat |
| Nebius AI | `nebius/NousResearch/Hermes-3-Llama-3.1-405B` | 128K | $1.00 | $3.00 | chat |
| Nebius AI | `nebius/google/gemma-3-27b-it` | 128K | $0.06 | $0.20 | chat |
| Nebius AI | `nebius/mistralai/Mistral-Nemo-Instruct-2407` | 128K | $0.04 | $0.12 | chat |
| Nebius AI | `nebius/Qwen/Qwen2.5-Coder-7B` | 32K | $0.01 | $0.03 | chat |
| Nebius AI | `nebius/BAAI/bge-en-icl` | 32K | $0.01 | - | embeddings |
| Nebius AI | `nebius/BAAI/bge-multilingual-gemma2` | 8K | $0.01 | - | embeddings |
| Nebius AI | `nebius/intfloat/e5-mistral-7b-instruct` | 32K | $0.01 | - | embeddings |
| AWS Bedrock | `mistral.devstral-2-123b` | 256K | $0.40 | $2.00 | chat, tools |
| AWS Bedrock | `zai.glm-4.7-flash` | 200K | $0.07 | $0.40 | chat, tools, reasoning |
| ZAI | `zai/glm-5` | 200K | $1.00 | $3.20 | chat, tools, reasoning |
| ZAI | `zai/glm-5-code` | 200K | $1.20 | $5.00 | chat, tools, reasoning |
| OpenRouter | `openrouter/anthropic/claude-sonnet-4.6` | - | - | - | chat |
| OpenRouter | `openrouter/google/gemini-3.1-pro-preview` | - | - | - | chat |
| OpenRouter | `openrouter/openai/gpt-5.1-codex-max` | - | - | - | chat |
| OpenRouter | `openrouter/qwen/qwen3-coder-plus` | - | - | - | chat |
| OpenRouter | `openrouter/qwen/qwen3.5-*` (5 models) | - | - | - | chat |
| OpenRouter | `openrouter/z-ai/glm-5` | - | - | - | chat |
| Together AI | `together_ai/Qwen/Qwen3.5-397B-A17B` | - | - | - | chat |
| Perplexity | `perplexity/pplx-embed-v1-0.6b` | 32K | $0.00 | - | embeddings |
| Perplexity | `perplexity/pplx-embed-v1-4b` | 32K | $0.03 | - | embeddings |
| Serper | `serper/search` | - | - | - | search |

#### Updated Models

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Add `cache_read_input_token_cost` and `cache_creation_input_token_cost` to Bedrock-hosted Anthropic models (`claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`, and APAC/EU variants) — prompt caching is now tracked for cost estimation
    - Rename `apac.anthropic.claude-sonnet-4-6` → `au.anthropic.claude-sonnet-4-6` to reflect correct regional identifier

- **[Azure OpenAI](../../docs/providers/azure)**
    - Add `supports_none_reasoning_effort` to all `gpt-5.1-chat`, `gpt-5.1-codex`, and `gpt-5.4` variants (global, EU, standard deployments) — allows passing `reasoning_effort: null` to disable reasoning

- **[Azure OpenAI](../../docs/providers/azure)** — Removed deprecated models
    - Remove `azure/gpt-35-turbo-0301` (deprecated 2025-02-13)
    - Remove `azure/gpt-35-turbo-0613` (deprecated 2025-02-13)

#### Features

- **[OpenAI](../../docs/providers/openai)**
    - Day 0 support for `gpt-5.4` and `gpt-5.4-pro` on OpenAI and Azure

- **[Google Gemini](../../docs/providers/gemini)**
    - Add Gemini 3.x model cost map entries — `gemini-3-flash-preview`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3-pro-image-preview`, `gemini-embedding-2-preview`
    - Add Gemini 2.0 Flash and Flash Lite to cost map (re-added with updated pricing)

- **[Google Vertex AI](../../docs/providers/vertex)**
    - Add `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-flash-experimental`, and `gemini-embedding-2-preview` to Vertex AI model cost map

- **[Mistral](../../docs/providers/mistral)**
    - Add Magistral reasoning models (`magistral-medium-1-2-2509`, `magistral-small-1-2-2509`)
    - Add `mistral-large-2512`, `mistral-medium-3-1-2508`, `mistral-small-3-2-2506`, `ministral-3-*` variants

- **[Dashscope / Qwen](../../docs/providers/dashscope)**
    - Add Qwen3 VL multimodal models (`qwen3-vl-235b`, `qwen3-vl-32b` — instruct and thinking variants)
    - Add `qwen3-next-80b-a3b` (instruct + thinking), `qwen3.5-plus`, `qwen3-max-2026-01-23`

- **[Black Forest Labs](../../docs/providers/black_forest_labs)**
    - Add FLUX Kontext image editing models (`flux-kontext-pro`, `flux-kontext-max`)
    - Add FLUX Pro 1.0 Fill (inpainting) and Expand (outpainting)
    - Add `flux-pro-1.1`, `flux-pro-1.1-ultra`, `flux-dev`, `flux-pro`

- **[Azure AI](../../docs/providers/azure_ai)**
    - Add xAI Grok models via Azure AI Foundry (`grok-4-1-fast-non-reasoning`, `grok-4-1-fast-reasoning`)
    - Add Mistral Document AI (`mistral-document-ai-2512`) — OCR mode

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Add `mistral.devstral-2-123b` (256K context, tools)
    - Add `zai.glm-4.7-flash` via Bedrock Converse (200K context, tools, reasoning)

- **[SageMaker](../../docs/providers/aws_sagemaker)**
    - Add `sagemaker_nova` provider for Amazon Nova models on SageMaker - [PR #21542](https://github.com/BerriAI/litellm/pull/21542)

#### Deprecated / Removed Models

**OpenAI** — Legacy models removed from cost map:
- `gpt-3.5-turbo-0301`, `gpt-3.5-turbo-0613`, `gpt-3.5-turbo-16k-0613`
- `gpt-4-0314`, `gpt-4-32k`, `gpt-4-32k-0314`, `gpt-4-32k-0613`, `gpt-4-1106-vision-preview`, `gpt-4-vision-preview`
- `gpt-4.5-preview`, `gpt-4.5-preview-2025-02-27`
- `gpt-4o-audio-preview-2024-10-01`, `gpt-4o-realtime-preview-2024-10-01`
- `o1-mini`, `o1-mini-2024-09-12`, `o1-preview`, `o1-preview-2024-09-12`

**Google Gemini** — Gemini 1.5 and legacy 2.0 variants removed:
- All `gemini-1.5-*` variants (flash, flash-8b, pro, and dated versions)
- `gemini-2.0-flash-exp`, `gemini-2.0-pro-exp-02-05`, `gemini-2.5-flash-preview-04-17`, `gemini-2.5-flash-preview-05-20`

**Google Vertex AI** — PaLM 2 / legacy models removed:
- All `chat-bison`, `text-bison`, `codechat-bison`, `code-bison`, `code-gecko` variants
- Gemini 1.0 Pro, 1.5 Flash/Pro, 2.0 Flash experimental, and preview variants

**Perplexity** — Legacy Llama-sonar models removed:
- `llama-3.1-sonar-huge-128k-online`, `llama-3.1-sonar-large/small-128k-chat/online`

---

## LLM API Endpoints

#### Features

- **[Responses API](../../docs/response_api)**
    - Handle `response.failed`, `response.incomplete`, and `response.cancelled` terminal event types in background streaming — previously only `response.completed` was handled - [PR #23492](https://github.com/BerriAI/litellm/pull/23492)

#### Bug Fixes

- **[Anthropic](../../docs/providers/anthropic)**
    - Preserve native tool format (web_search, bash, tool_search, etc.) when guardrails convert tools for the Anthropic Messages API - [PR #23526](https://github.com/BerriAI/litellm/pull/23526)

- **[Moonshot / Kimi](../../docs/providers/openai_compatible)**
    - Auto-fill `reasoning_content` for Moonshot Kimi reasoning models - [PR #23580](https://github.com/BerriAI/litellm/pull/23580)

- **[HuggingFace](../../docs/providers/huggingface)**
    - Forward `extra_headers` to HuggingFace embedding API - [PR #23525](https://github.com/BerriAI/litellm/pull/23525)

- **General**
    - Normalize `content_filtered` finish reason across providers - [PR #23564](https://github.com/BerriAI/litellm/pull/23564)
    - Fix custom cost tracking on deployments for `/v1/messages` and `/v1/responses` - [PR #23647](https://github.com/BerriAI/litellm/pull/23647)
    - Fix per-request custom pricing when `router_model_id` has no pricing data — now falls back to model name

---

## Management Endpoints / UI

#### Features

- **Virtual Keys**
    - Add Organization dropdown to Create/Edit Key form — `organization_id` is now a first-class field in Key Ownership - [PR #23595](https://github.com/BerriAI/litellm/pull/23595)
    - Allow setting `organization_id` on `/key/update` — keys can be assigned or moved to a different organization after creation - [PR #23557](https://github.com/BerriAI/litellm/pull/23557)

- **Internal Users**
    - Add/Remove Team Membership directly from the Internal Users info page — includes searchable dropdown and role selector; no longer requires navigating to each team - [PR #23638](https://github.com/BerriAI/litellm/pull/23638)

- **Default Team Settings**
    - Modernize page to antd (consistent with rest of app) - [PR #23614](https://github.com/BerriAI/litellm/pull/23614)
    - Fix: default team params (budget, duration, tpm, rpm, permissions) now correctly applied on `/team/new` - [PR #23614](https://github.com/BerriAI/litellm/pull/23614)
    - Fix: settings persist across proxy restarts (`default_team_params` added to `LITELLM_SETTINGS_SAFE_DB_OVERRIDES`) - [PR #23614](https://github.com/BerriAI/litellm/pull/23614)
    - Fix: resolved race condition in `_update_litellm_setting` where `get_config()` could overwrite freshly saved values - [PR #23614](https://github.com/BerriAI/litellm/pull/23614)

- **Usage**
    - Auto-paginate daily spend data — all entity views (teams, orgs, customers, tags, agents, users) fetch pages progressively with charts updating after each page - [PR #23622](https://github.com/BerriAI/litellm/pull/23622)

- **Models / Cost**
    - Azure Model Router cost breakdown in UI — show per-sub-model `additional_costs` from `hidden_params` in `CostBreakdownViewer` - [PR #23550](https://github.com/BerriAI/litellm/pull/23550)

- **User Management**
    - New `/user/info/v2` endpoint — scoped, paginated replacement for the existing god endpoint that caused memory and stability issues on large installs - [PR #23437](https://github.com/BerriAI/litellm/pull/23437)

#### Bugs

- Fix Tag list endpoint returning 500 due to invalid Prisma `group_by` kwargs - [PR #23606](https://github.com/BerriAI/litellm/pull/23606)
- Fix Team Admin getting 403 on `/user/filter/ui` when `scope_user_search_to_org` is enabled - [PR #23671](https://github.com/BerriAI/litellm/pull/23671)
- Fix Public Model Hub not showing config-defined models after save - [PR #23501](https://github.com/BerriAI/litellm/pull/23501)
- Fix fallback popup model dropdown z-index issue - [PR #23516](https://github.com/BerriAI/litellm/pull/23516)
- Fix double-counting bug in org/team key limit checks on `/key/update`

---

## AI Integrations

### Logging

- **[Vantage](https://vantage.sh)**
    - Add Vantage integration for FOCUS 1.2 CSV export — export LiteLLM proxy spend data as FinOps Open Cost & Usage Specification reports, with time-windowed filenames to prevent overwrites - [PR #23333](https://github.com/BerriAI/litellm/pull/23333)

- **General**
    - Fix silent metrics race condition causing metric collision across experiments - [PR #23542](https://github.com/BerriAI/litellm/pull/23542)

### Guardrails

No major guardrail changes in this release.

### Prompt Management

No major prompt management changes in this release.

### Secret Managers

No major secret manager changes in this release.

---

## Performance / Loadbalancing / Reliability improvements

- **Fix streaming crashes after ~1 hour** — `LLMClientCache._remove_key()` no longer calls `close()`/`aclose()` on evicted HTTP/SDK clients. In-flight requests were crashing with `RuntimeError: Cannot send a request, as the client has been closed.` after the 1-hour TTL expired. Cleanup now happens only at shutdown via `close_litellm_async_clients()` - [PR #22926](https://github.com/BerriAI/litellm/pull/22926)
- **Fix OOM / Prisma connection loss** on large installs — unbounded managed-object poll was exhausting Prisma connections after ~60–70 minutes on instances with 336K+ queued response rows - [PR #23472](https://github.com/BerriAI/litellm/pull/23472)
- **Centralize logging kwarg updates** — root cause fix migrating all logging updates to a single function, eliminating kwarg inconsistencies across logging paths - [PR #23659](https://github.com/BerriAI/litellm/pull/23659)
- **Fix tiktoken cache for non-root offline containers** — tiktoken cache now works correctly in offline environments running as non-root users - [PR #23498](https://github.com/BerriAI/litellm/pull/23498)
- **Add CodSpeed continuous performance benchmarks** — automated performance regression tracking on CI - [PR #23676](https://github.com/BerriAI/litellm/pull/23676)

---

## Security

- **Secret redaction in proxy logs** — Adds a `SecretRedactionFilter` to all LiteLLM loggers that scrubs API keys, tokens, and credentials from log messages, format args, exception tracebacks, and extra fields. Enabled by default; opt out with `LITELLM_DISABLE_REDACT_SECRETS=true` - [PR #23668](https://github.com/BerriAI/litellm/pull/23668), [PR #23667](https://github.com/BerriAI/litellm/pull/23667)
- **Bump PyJWT to `^2.12.0`** — addresses security vulnerability in `^2.10.1` - [PR #23678](https://github.com/BerriAI/litellm/pull/23678)
- **Bump `tar` to 7.5.11 and `tornado` to 6.5.5** — addresses CVEs in transitive dependencies - [PR #23602](https://github.com/BerriAI/litellm/pull/23602)

---

## Database / Proxy Operations

- **Fix Prisma migrate deploy on pre-existing instances** — resolved multiple bugs in migration recovery logic: missing return in the P3018 idempotent error handler and unhandled exceptions in `_roll_back_migration` that caused silent failures even after successful recovery - [PR #23655](https://github.com/BerriAI/litellm/pull/23655)
- **Make DB migration failure exit opt-in** — proxy no longer exits on `prisma migrate deploy` failure by default; enable with `--enforce_prisma_migration_check` - [PR #23675](https://github.com/BerriAI/litellm/pull/23675)

---

## New Contributors

* @ryanh-ai made their first contribution in [PR #21542](https://github.com/BerriAI/litellm/pull/21542)
* @ryan-crabbe made their first contribution in [PR #23668](https://github.com/BerriAI/litellm/pull/23668)
* @Jah-yee made their first contribution in [PR #23525](https://github.com/BerriAI/litellm/pull/23525)
* @gambletan made their first contribution in [PR #23516](https://github.com/BerriAI/litellm/pull/23516)
* @awais786 made their first contribution in [PR #23183](https://github.com/BerriAI/litellm/pull/23183)
* @pradyyadav made their first contribution in [PR #23580](https://github.com/BerriAI/litellm/pull/23580)
* @xianzongxie-stripe made their first contribution in [PR #23492](https://github.com/BerriAI/litellm/pull/23492)
* @Harshit28j made their first contribution in [PR #23333](https://github.com/BerriAI/litellm/pull/23333)
* @codspeed-hq[bot] made their first contribution in [PR #23676](https://github.com/BerriAI/litellm/pull/23676)

---

## Diff Summary

## 03/16/2026
* New Providers: 5
* New Models / Updated Models: 116 new, 132 removed
* LLM API Endpoints: 5
* Management Endpoints / UI: 11
* AI Integrations: 2
* Performance / Reliability: 5
* Security: 3
* Database / Proxy Operations: 2

---

## Full Changelog
[v1.82.0-stable...v1.82.3-stable](https://github.com/BerriAI/litellm/compare/v1.82.0-stable...v1.82.3-stable)
