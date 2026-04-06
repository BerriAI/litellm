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
ghcr.io/berriai/litellm:v1.82.3-stable
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
- **Hashicorp Vault secret manager** — Config override backend powered by Hashicorp Vault, with full UI for managing vault-sourced credentials - [PR #22939](https://github.com/BerriAI/litellm/pull/22939), [PR #23036](https://github.com/BerriAI/litellm/pull/23036)
- **Responses API WebSocket streaming** — Real-time WebSocket streaming for the Responses API, including support across all providers - [PR #22559](https://github.com/BerriAI/litellm/pull/22559), [PR #22771](https://github.com/BerriAI/litellm/pull/22771)
- **Org Admin RBAC expansion** — Org Admins can now access team management endpoints, view and invite internal users, and manage team membership without requiring a global admin role - [PR #23085](https://github.com/BerriAI/litellm/pull/23085), [PR #23080](https://github.com/BerriAI/litellm/pull/23080)
- **Guardrail mode defaults and tag-based modes** — Set a default guardrail mode list globally, and specify a list of modes in tag-based guardrail configs - [PR #22676](https://github.com/BerriAI/litellm/pull/22676), [PR #23020](https://github.com/BerriAI/litellm/pull/23020)
- **Secret redaction in logs** — API keys, tokens, and credentials automatically scrubbed from all proxy log output. Enabled by default; opt out with `LITELLM_DISABLE_REDACT_SECRETS=true` - [PR #23668](https://github.com/BerriAI/litellm/pull/23668)
- **Streaming stability fix** — Critical fix for `RuntimeError: Cannot send a request, as the client has been closed.` crashes after ~1 hour in production - [PR #22926](https://github.com/BerriAI/litellm/pull/22926)

---

## New Providers and Endpoints

### New Providers (7 new providers)

| Provider | Supported LiteLLM Endpoints | Description |
| -------- | --------------------------- | ----------- |
| [Nebius AI](../../docs/providers/nebius) (`nebius/`) | `/chat/completions`, `/embeddings` | EU-based AI cloud with 30+ open models — DeepSeek, Qwen3, Llama 3.1/3.3, NVIDIA Nemotron, BAAI embeddings |
| [ZAI](../../docs/providers/zai) (`zai/`) | `/chat/completions` | ZhipuAI GLM-5 models via ZAI cloud |
| [Black Forest Labs](../../docs/providers/black_forest_labs) (`black_forest_labs/`) | `/images/generations`, `/images/edits` | FLUX image generation and editing — Kontext Pro/Max, Pro 1.0 Fill/Expand |
| [Serper](../../docs/providers/serper) (`serper/`) | `/search` | Web search via Serper API |
| [SageMaker Nova](../../docs/providers/aws_sagemaker) (`sagemaker_nova/`) | `/chat/completions` | Amazon Nova models via SageMaker endpoint |
| [Google Search API](../../docs/providers/google_search) (`google_search/`) | `/search` | Google Search API integration - [PR #22752](https://github.com/BerriAI/litellm/pull/22752) |
| [Bedrock Mantle](../../docs/providers/bedrock) (`bedrock_mantle/`) | `/chat/completions` | Amazon Bedrock via Mantle — alternative auth and routing path for Bedrock models - [PR #22866](https://github.com/BerriAI/litellm/pull/22866) |

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
    - WebSocket streaming support for Responses API — real-time streaming via WebSocket for all providers - [PR #22559](https://github.com/BerriAI/litellm/pull/22559), [PR #22771](https://github.com/BerriAI/litellm/pull/22771)
    - WebRTC support for real-time audio/video communication - [PR #23446](https://github.com/BerriAI/litellm/pull/23446)
    - Responses API support for OpenAI-compatible JSON providers (`openai_like`) - [PR #21398](https://github.com/BerriAI/litellm/pull/21398)
    - Route `gpt-5.4+` calls using both tools and reasoning to the Responses API automatically - [PR #23577](https://github.com/BerriAI/litellm/pull/23577)

- **[Anthropic Files API](../../docs/providers/anthropic)**
    - Full Anthropic Files API support — upload, retrieve, list, and delete files; use file references in messages - [PR #16594](https://github.com/BerriAI/litellm/pull/16594)

- **[Mistral](../../docs/providers/mistral)**
    - Voxtral audio transcription support — `mistral/voxtral-mini-*` and `mistral/voxtral-*` for audio transcription via Mistral - [PR #22801](https://github.com/BerriAI/litellm/pull/22801)

- **[OpenAI](../../docs/providers/openai)**
    - `litellm.acount_tokens()` public API — async token counting with full OpenAI provider support - [PR #22809](https://github.com/BerriAI/litellm/pull/22809)
    - Normalize `reasoning_effort` dict to string for chat completion API - [PR #22981](https://github.com/BerriAI/litellm/pull/22981)

- **[OpenRouter](../../docs/providers/openrouter)**
    - Image edit support for OpenRouter models - [PR #22403](https://github.com/BerriAI/litellm/pull/22403)

- **[Google Vertex AI](../../docs/providers/vertex)**
    - VIDEO modality token usage tracking in `completion_tokens_details` - [PR #22550](https://github.com/BerriAI/litellm/pull/22550)

- **Images API**
    - `input_fidelity` parameter for image edit API - [PR #23201](https://github.com/BerriAI/litellm/pull/23201)

- **General**
    - Per-request `enable_json_schema_validation` flag for thread-safe JSON schema validation - [PR #21233](https://github.com/BerriAI/litellm/pull/21233)
    - Model cost aliases expansion — define aliases in the cost map that inherit pricing from a parent model - [PR #23314](https://github.com/BerriAI/litellm/pull/23314), [PR #23457](https://github.com/BerriAI/litellm/pull/23457)
    - Wildcards model support for the Files API - [PR #22740](https://github.com/BerriAI/litellm/pull/22740)

#### Bugs

- **[Anthropic](../../docs/providers/anthropic)**
    - Preserve native tool format (web_search, bash, tool_search, etc.) when guardrails convert tools for the Anthropic Messages API - [PR #23526](https://github.com/BerriAI/litellm/pull/23526)
    - Enforce `type: "object"` on tool input schemas in `_map_tool_helper` — fixes tool call failures for strict-schema providers - [PR #23103](https://github.com/BerriAI/litellm/pull/23103)
    - Deduplicate `tool_result` messages by `tool_call_id` — prevents duplicate tool result errors in multi-turn conversations - [PR #23104](https://github.com/BerriAI/litellm/pull/23104)
    - Map `reasoning_effort` to `output_config` for Claude 4.6 models - [PR #22220](https://github.com/BerriAI/litellm/pull/22220)

- **[Google Gemini](../../docs/providers/gemini)**
    - Correct streaming `finish_reason` for tool calls — was incorrectly returning `null` instead of `tool_calls` - [PR #21577](https://github.com/BerriAI/litellm/pull/21577)
    - Preserve `$ref` in JSON Schema for Gemini 2.0+ — schema references were being stripped, breaking structured output - [PR #21597](https://github.com/BerriAI/litellm/pull/21597)
    - Handle `minimal` `reasoning_effort` param for Gemini 3.1 models - [PR #22920](https://github.com/BerriAI/litellm/pull/22920)

- **[Google Vertex AI](../../docs/providers/vertex)**
    - Pass through native Gemini `imageConfig` params for image generation - [PR #21585](https://github.com/BerriAI/litellm/pull/21585)
    - Prevent content truncation when `finish_reason` races ahead of content chunks in streaming - [PR #22692](https://github.com/BerriAI/litellm/pull/22692)
    - Strip LiteLLM-internal keys from `extra_body` before merging to Gemini request body - [PR #23131](https://github.com/BerriAI/litellm/pull/23131)
    - Drop unsupported `output_config` parameter from all Vertex AI requests - [PR #22884](https://github.com/BerriAI/litellm/pull/22884)
    - Skip schema transforms for Gemini 2.0+ tool parameters — avoids breaking native Gemini schema handling - [PR #23265](https://github.com/BerriAI/litellm/pull/23265)

- **[OpenRouter](../../docs/providers/openrouter)**
    - Pattern-based fix for native model double-stripping when provider prefix matches model name - [PR #22320](https://github.com/BerriAI/litellm/pull/22320)
    - Use provider-reported usage in streaming responses when `stream_options` is not set - [PR #21592](https://github.com/BerriAI/litellm/pull/21592)

- **[AWS Bedrock](../../docs/providers/bedrock)**
    - Extract region and model ID from `bedrock/{region}/{model}` path format - [PR #22546](https://github.com/BerriAI/litellm/pull/22546)
    - Strip `scope` from `cache_control` for Anthropic messages on Bedrock and Azure AI - [PR #22867](https://github.com/BerriAI/litellm/pull/22867)
    - Populate `completion_tokens_details` in Responses API responses - [PR #23243](https://github.com/BerriAI/litellm/pull/23243)

- **[Azure AI](../../docs/providers/azure_ai)**
    - Resolve `api_base` from environment variable in Document Intelligence OCR - [PR #21581](https://github.com/BerriAI/litellm/pull/21581)

- **[Moonshot / Kimi](../../docs/providers/openai_compatible)**
    - Auto-fill `reasoning_content` for Moonshot Kimi reasoning models - [PR #23580](https://github.com/BerriAI/litellm/pull/23580)
    - Preserve `image_url` blocks in multimodal messages for Moonshot - [PR #21595](https://github.com/BerriAI/litellm/pull/21595)

- **[HuggingFace](../../docs/providers/huggingface)**
    - Forward `extra_headers` to HuggingFace embedding API - [PR #23525](https://github.com/BerriAI/litellm/pull/23525)

- **Token Counting / Cost**
    - Fix `count_tokens` to include system prompts and tools in token counting API requests - [PR #22301](https://github.com/BerriAI/litellm/pull/22301)
    - Pass all custom pricing fields to `register_model` in `completion()` and `embedding()` - [PR #22552](https://github.com/BerriAI/litellm/pull/22552)

- **Tools / Function Calling**
    - Gracefully repair truncated JSON in tool call arguments — prevents crashes on malformed tool responses - [PR #22503](https://github.com/BerriAI/litellm/pull/22503)
    - Fix `output_item.done` for function calls not emitting `finish_reason` in streaming - [PR #22553](https://github.com/BerriAI/litellm/pull/22553)
    - Preserve thinking block order with multiple web searches - [PR #23093](https://github.com/BerriAI/litellm/pull/23093)

- **General**
    - Normalize `content_filtered` finish reason across providers - [PR #23564](https://github.com/BerriAI/litellm/pull/23564)
    - Unify `finish_reason` mapping to OpenAI-compatible values across all providers - [PR #22138](https://github.com/BerriAI/litellm/pull/22138)
    - Fix custom cost tracking on deployments for `/v1/messages` and `/v1/responses` - [PR #23647](https://github.com/BerriAI/litellm/pull/23647)
    - Fix per-request custom pricing when `router_model_id` has no pricing data — now falls back to model name
    - Fix batch list showing stale `validating` status after completion - [PR #22982](https://github.com/BerriAI/litellm/pull/22982)
    - Fix batch retrieve returning raw `output_file_id` when `model_id` is missing - [PR #23194](https://github.com/BerriAI/litellm/pull/23194)
    - Encode batch IDs when `x-litellm-model` header is used - [PR #22653](https://github.com/BerriAI/litellm/pull/22653)
    - Map `reasoning` to `reasoning_content` in streaming Delta for gpt-oss providers - [PR #22803](https://github.com/BerriAI/litellm/pull/22803)

---

## Management Endpoints / UI

#### Features

- **Virtual Keys**
    - Add Organization dropdown to Create/Edit Key form — `organization_id` is now a first-class field in Key Ownership - [PR #23595](https://github.com/BerriAI/litellm/pull/23595)
    - Allow setting `organization_id` on `/key/update` — keys can be assigned or moved to a different organization after creation - [PR #23557](https://github.com/BerriAI/litellm/pull/23557)
    - Manual Spend Reset for virtual keys from the UI — admins can reset key spend to zero on demand - [PR #22715](https://github.com/BerriAI/litellm/pull/22715)
    - BYOK (Bring Your Own Key) — client-side provider API key takes precedence over proxy key for Anthropic `/v1/messages` - [PR #22964](https://github.com/BerriAI/litellm/pull/22964)
    - UI login session duration configurable via `LITELLM_UI_SESSION_DURATION` environment variable - [PR #22182](https://github.com/BerriAI/litellm/pull/22182)
    - Auto-redirect UI login to SSO via `auto_redirect_ui_login_to_sso: true` in config.yaml - [PR #23367](https://github.com/BerriAI/litellm/pull/23367)

- **Access Control (RBAC)**
    - Org Admins can now access team management endpoints — `/team/new`, `/team/update`, `/team/delete`, `/team/member_add`, `/team/member_delete` - [PR #23085](https://github.com/BerriAI/litellm/pull/23085), [PR #23095](https://github.com/BerriAI/litellm/pull/23095)
    - Org Admins can view and invite internal users — full user management without requiring global admin role - [PR #23080](https://github.com/BerriAI/litellm/pull/23080)
    - Allow Admin Viewers to access Audit Logs — view-only admin role now includes audit log access - [PR #23419](https://github.com/BerriAI/litellm/pull/23419)
    - RBAC for Vector Stores and Agents — key/team-level access control for vector store and agent resources - [PR #22858](https://github.com/BerriAI/litellm/pull/22858)
    - User filter scope (`scope_user_search_to_org`) is now opt-in — previously default-on, causing unintended restriction - [PR #23057](https://github.com/BerriAI/litellm/pull/23057)

- **Vector Stores**
    - Vector Store management endpoints — retrieve, list, update, and delete vector stores via `/v1/vector_stores/*` - [PR #23435](https://github.com/BerriAI/litellm/pull/23435)

- **Teams**
    - Batch expiry setting for teams — configure a default expiry duration for all team keys - [PR #22705](https://github.com/BerriAI/litellm/pull/22705)
    - Team Admin can reset key spend - [PR #22725](https://github.com/BerriAI/litellm/pull/22725)

- **Internal Users**
    - Add/Remove Team Membership directly from the Internal Users info page — includes searchable dropdown and role selector; no longer requires navigating to each team - [PR #23638](https://github.com/BerriAI/litellm/pull/23638)

- **Models**
    - Attach knowledge base to model via UI - [PR #22656](https://github.com/BerriAI/litellm/pull/22656)

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
- Fix invite link allowing multiple password resets for the same link - [PR #22462](https://github.com/BerriAI/litellm/pull/22462)
- Fix key expiry default duration not being applied when `duration` is not set - [PR #22956](https://github.com/BerriAI/litellm/pull/22956)
- Fix all proxy models not including model access groups in key creation - [PR #23236](https://github.com/BerriAI/litellm/pull/23236)
- Fix admin viewers unable to see all organizations - [PR #22940](https://github.com/BerriAI/litellm/pull/22940)
- Fix Audit Logs UI: added server-side pagination, filtering, and drawer view - [PR #22476](https://github.com/BerriAI/litellm/pull/22476)
- Fix virtual keys in teams view not applying the team filter correctly - [PR #23065](https://github.com/BerriAI/litellm/pull/23065)
- Fix team expiry enforcement validation - [PR #22728](https://github.com/BerriAI/litellm/pull/22728)

---

## AI Integrations

### Logging

- **[Helicone](../../docs/observability/helicone_integration)**
    - Add Gemini and Vertex AI support to HeliconeLogger — routes Gemini and Vertex AI requests through the correct Helicone provider URL - [PR #19288](https://github.com/BerriAI/litellm/pull/19288)
    - Fix correct provider URL for Vertex AI Gemini models - [PR #22603](https://github.com/BerriAI/litellm/pull/22603)

- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Fix failure path kwargs inconsistency causing dropped traces on failed requests - [PR #22390](https://github.com/BerriAI/litellm/pull/22390)

- **[Vantage](https://vantage.sh)**
    - Add Vantage integration for FOCUS 1.2 CSV export — export LiteLLM proxy spend data as FinOps Open Cost & Usage Specification reports, with time-windowed filenames to prevent overwrites - [PR #23333](https://github.com/BerriAI/litellm/pull/23333)

- **General**
    - Fix silent metrics race condition causing metric collision across experiments - [PR #23542](https://github.com/BerriAI/litellm/pull/23542)

### Guardrails

- **Guardrail mode default list** — Configure a default list of guardrail modes applied globally when no per-request mode is specified - [PR #22676](https://github.com/BerriAI/litellm/pull/22676)
- **Tag-based guardrail mode lists** — Specify a list of modes in tag-based guardrail configs instead of a single mode - [PR #23020](https://github.com/BerriAI/litellm/pull/23020)
- **Fix presidio PII token leak** — Edge case where Anthropic handle in Presidio caused PII data exposure in token response - [PR #22627](https://github.com/BerriAI/litellm/pull/22627)
- **Fix OTEL orphaned guardrail traces** — Span redundancy and missing response IDs in OpenTelemetry guardrail traces - [PR #23001](https://github.com/BerriAI/litellm/pull/23001)

### Prompt Management

No major prompt management changes in this release.

### Secret Managers

- **[Hashicorp Vault](../../docs/secret_managers)** — Full Hashicorp Vault integration as a config override backend — secrets defined in Vault are fetched at startup and override `config.yaml` values. UI support for managing vault-sourced credentials included - [PR #22939](https://github.com/BerriAI/litellm/pull/22939), [PR #23036](https://github.com/BerriAI/litellm/pull/23036)

---

## MCP Gateway

#### Features

- **Token authentication for MCP servers** — configure `auth_type: "bearer"` per MCP server to require token-based auth on tool calls - [PR #23260](https://github.com/BerriAI/litellm/pull/23260)
- **Team-scoped MCP server filtering** — keys created under a team only see MCP servers available to that team - [PR #23323](https://github.com/BerriAI/litellm/pull/23323)
- **Per-server health recheck in UI** — trigger a health check for individual MCP servers without reloading all servers - [PR #23328](https://github.com/BerriAI/litellm/pull/23328)

#### Bugs

- Fix MCP server URL and tools management issues causing tool discovery to fail - [PR #22751](https://github.com/BerriAI/litellm/pull/22751)
- Fix MCP server health checks triggering on server deletion - [PR #23063](https://github.com/BerriAI/litellm/pull/23063)

---

## Spend Tracking, Budgets and Rate Limiting

- **Fix budget-linked keys never having spend reset** — Keys linked to budget objects were not having their spend reset on the configured reset interval - [PR #20688](https://github.com/BerriAI/litellm/pull/20688)
- **Flex pricing support** — Add `flex_pricing` field to cost map for providers that offer dynamic pricing tiers - [PR #22992](https://github.com/BerriAI/litellm/pull/22992)
- **Fix spend log cleanup** — Resolved lock tracking, integer retention, and skip-log-level issues in spend log cleanup job - [PR #22687](https://github.com/BerriAI/litellm/pull/22687)
- **Fix WebSearch spend log deduplication** — WebSearch interception was failing with thinking enabled; fixed along with spend log dedup - [PR #22679](https://github.com/BerriAI/litellm/pull/22679)
- **Fix TypeError when request has no API key** — Spend tracking was throwing unhandled exception when API key was absent from request - [PR #23363](https://github.com/BerriAI/litellm/pull/23363)

---

## Performance / Loadbalancing / Reliability improvements

- **Fix streaming crashes after ~1 hour** — `LLMClientCache._remove_key()` no longer calls `close()`/`aclose()` on evicted HTTP/SDK clients. In-flight requests were crashing with `RuntimeError: Cannot send a request, as the client has been closed.` after the 1-hour TTL expired. Cleanup now happens only at shutdown via `close_litellm_async_clients()` - [PR #22926](https://github.com/BerriAI/litellm/pull/22926)
- **Fix OOM / Prisma connection loss** on large installs — unbounded managed-object poll was exhausting Prisma connections after ~60–70 minutes on instances with 336K+ queued response rows - [PR #23472](https://github.com/BerriAI/litellm/pull/23472)
- **Centralize logging kwarg updates** — root cause fix migrating all logging updates to a single function, eliminating kwarg inconsistencies across logging paths - [PR #23659](https://github.com/BerriAI/litellm/pull/23659)
- **Fix tiktoken cache for non-root offline containers** — tiktoken cache now works correctly in offline environments running as non-root users - [PR #23498](https://github.com/BerriAI/litellm/pull/23498)
- **Block proxy startup when Redis transaction buffer has no Redis** — prevents silent data loss when `use_redis_transaction_buffer: true` is set without a Redis connection - [PR #23019](https://github.com/BerriAI/litellm/pull/23019)
- **Fix `InFlightRequestsMiddleware` crash** — undefined kwargs in middleware were causing request failures - [PR #22523](https://github.com/BerriAI/litellm/pull/22523)
- **Fix `BaseModelResponseIterator` crash on non-string stream chunks** — streaming was crashing when providers returned non-string chunk data - [PR #23497](https://github.com/BerriAI/litellm/pull/23497)
- **Fix `SERVER_ROOT_PATH` prefix handling** — strip prefix before checking mapped pass-through routes to prevent double-prefix issues - [PR #23414](https://github.com/BerriAI/litellm/pull/23414)
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

## Documentation Updates

- Add Anthropic `/v1/messages` → `/responses` parameter mapping reference - [PR #22893](https://github.com/BerriAI/litellm/pull/22893)
- Update Okta SSO docs and custom SSO handler example - [PR #22786](https://github.com/BerriAI/litellm/pull/22786)
- Add `LITELLM_MAX_BUDGET_PER_SESSION_TTL` to environment variables reference - [PR #23186](https://github.com/BerriAI/litellm/pull/23186)
- Add DB query performance guidelines to `CLAUDE.md` - [PR #23196](https://github.com/BerriAI/litellm/pull/23196)
- Add Gemini Vertex AI PayGo/priority cost tracking docs - [PR #22948](https://github.com/BerriAI/litellm/pull/22948)

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
* New Providers: 7
* New Models / Updated Models: 116 new, 132 removed
* LLM API Endpoints: 37
* Management Endpoints / UI: 31
* AI Integrations: 8
* MCP Gateway: 5
* Spend Tracking, Budgets and Rate Limiting: 5
* Performance / Loadbalancing / Reliability improvements: 9
* Security: 3
* Database / Proxy Operations: 2
* Documentation Updates: 5

---

## Full Changelog
[v1.82.0-stable...v1.82.3-stable](https://github.com/BerriAI/litellm/compare/v1.82.0-stable...v1.82.3-stable)
