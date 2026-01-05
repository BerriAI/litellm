---
title: "v1.78.5-stable - Native OCR Support"
slug: "v1-78-5"
date: 2025-10-18T10:00:00
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
docker.litellm.ai/berriai/litellm:v1.78.5-stable
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.78.5
```

</TabItem>
</Tabs>

---

## Key Highlights

- **Native OCR Endpoints** - Native `/v1/ocr` endpoint support with cost tracking for Mistral OCR and Azure AI OCR
- **Global Vendor Discounts** - Specify global vendor discount percentages for accurate cost tracking and reporting
- **Team Spending Reports** - Team admins can now export detailed spending reports for their teams
- **Claude Haiku 4.5** - Day 0 support for Claude Haiku 4.5 across Bedrock, Vertex AI, and OpenRouter with 200K context window
- **GPT-5-Codex** - Support for GPT-5-Codex via Responses API on OpenAI and Azure
- **Performance Improvements** - Major router optimizations: O(1) model lookups, 10-100x faster shallow copy, 30-40% faster timing calls, and O(n) to O(1) hash generation

---

## New Models / Updated Models

#### New Model Support

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| Anthropic | `claude-haiku-4-5` | 200K | $1.00 | $5.00 | Chat, reasoning, vision, function calling, prompt caching, computer use |
| Anthropic | `claude-haiku-4-5-20251001` | 200K | $1.00 | $5.00 | Chat, reasoning, vision, function calling, prompt caching, computer use |
| Bedrock | `anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | $1.00 | $5.00 | Chat, reasoning, vision, function calling, prompt caching |
| Bedrock | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | $1.00 | $5.00 | Chat, reasoning, vision, function calling, prompt caching |
| Bedrock | `jp.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | $1.10 | $5.50 | Chat, reasoning, vision, function calling, prompt caching (JP Cross-Region) |
| Bedrock | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | $1.10 | $5.50 | Chat, reasoning, vision, function calling, prompt caching (US region) |
| Bedrock | `eu.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | $1.10 | $5.50 | Chat, reasoning, vision, function calling, prompt caching (EU region) |
| Bedrock | `apac.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | $1.10 | $5.50 | Chat, reasoning, vision, function calling, prompt caching (APAC region) |
| Bedrock | `au.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | $1.10 | $5.50 | Chat, reasoning, vision, function calling, prompt caching (AU region) |
| Vertex AI | `vertex_ai/claude-haiku-4-5@20251001` | 200K | $1.00 | $5.00 | Chat, reasoning, vision, function calling, prompt caching |
| OpenAI | `gpt-5` | 272K | $1.25 | $10.00 | Chat, responses API, reasoning, vision, function calling, prompt caching |
| OpenAI | `gpt-5-codex` | 272K | $1.25 | $10.00 | Responses API mode |
| Azure | `azure/gpt-5-codex` | 272K | $1.25 | $10.00 | Responses API mode |
| Gemini | `gemini-2.5-flash-image` | 32K | $0.30 | $2.50 | Image generation (GA - Nano Banana) - $0.039/image |
| ZhipuAI | `glm-4.6` | - | - | - | Chat completions |

#### Features

- **[OpenAI](../../docs/providers/openai)**
    - GPT-5 return reasoning content via /chat/completions + GPT-5-Codex working on Claude Code - [PR #15441](https://github.com/BerriAI/litellm/pull/15441)

- **[Anthropic](../../docs/providers/anthropic)**
    - Reduce claude-4-sonnet max_output_tokens to 64k - [PR #15409](https://github.com/BerriAI/litellm/pull/15409)
    - Added claude-haiku-4.5 - [PR #15579](https://github.com/BerriAI/litellm/pull/15579)
    - Add support for thinking blocks and redacted thinking blocks in Anthropic v1/messages API - [PR #15501](https://github.com/BerriAI/litellm/pull/15501)

- **[Bedrock](../../docs/providers/bedrock)**
    - Add anthropic.claude-haiku-4-5-20251001-v1:0 on Bedrock, VertexAI - [PR #15581](https://github.com/BerriAI/litellm/pull/15581)
    - Add Claude Haiku 4.5 support for Bedrock global and US regions - [PR #15650](https://github.com/BerriAI/litellm/pull/15650)
    - Add Claude Haiku 4.5 support for Bedrock Other regions - [PR #15653](https://github.com/BerriAI/litellm/pull/15653)
    - Add JP Cross-Region Inference jp.anthropic.claude-haiku-4-5-20251001 - [PR #15598](https://github.com/BerriAI/litellm/pull/15598)
    - Fix: bedrock-pricing-geo-inregion-cross-region / add Global Cross-Region Inference - [PR #15685](https://github.com/BerriAI/litellm/pull/15685)
    - Fix: Support us-gov prefix for AWS GovCloud Bedrock models - [PR #15626](https://github.com/BerriAI/litellm/pull/15626)
    - Fix GPT-OSS in Bedrock now supports streaming. Revert fake streaming - [PR #15668](https://github.com/BerriAI/litellm/pull/15668)

- **[Gemini](../../docs/providers/gemini)**
    - Feat(pricing): Add Gemini 2.5 Flash Image (Nano Banana) in GA - [PR #15557](https://github.com/BerriAI/litellm/pull/15557)
    - Fix: Gemini 2.5 Flash Image should not have supports_web_search=true - [PR #15642](https://github.com/BerriAI/litellm/pull/15642)
    - Remove penalty params as supported params for gemini preview model - [PR #15503](https://github.com/BerriAI/litellm/pull/15503)

- **[Ollama](../../docs/providers/ollama)**
    - Fix(ollama/chat): correctly map reasoning_effort to think in requests - [PR #15465](https://github.com/BerriAI/litellm/pull/15465)

- **[OpenRouter](../../docs/providers/openrouter)**
    - Add anthropic/claude-sonnet-4.5 to OpenRouter cost map - [PR #15472](https://github.com/BerriAI/litellm/pull/15472)
    - Prompt caching for anthropic models with OpenRouter - [PR #15535](https://github.com/BerriAI/litellm/pull/15535)
    - Get completion cost directly from OpenRouter - [PR #15448](https://github.com/BerriAI/litellm/pull/15448)
    - Fix OpenRouter Claude Opus 4 model naming - [PR #15495](https://github.com/BerriAI/litellm/pull/15495)

- **[CometAPI](../../docs/providers/comet)**
    - Fix(cometapi): improve CometAPI provider support (embeddings, image generation, docs) - [PR #15591](https://github.com/BerriAI/litellm/pull/15591)

- **[Lemonade](../../docs/providers/lemonade)**
    - Adding new models to the lemonade provider - [PR #15554](https://github.com/BerriAI/litellm/pull/15554)

- **[Watson X](../../docs/providers/watsonx)**
    - Fix (pricing): Fix pricing for watsonx model family for various models - [PR #15670](https://github.com/BerriAI/litellm/pull/15670)

- **[Vercel AI Gateway](../../docs/providers/vercel_ai_gateway)**
    - Add glm-4.6 model to pricing configuration - [PR #15679](https://github.com/BerriAI/litellm/pull/15679)

- **[Vertex AI](../../docs/providers/vertex)**
    - Add Vertex AI Discovery Engine Rerank Support - [PR #15532](https://github.com/BerriAI/litellm/pull/15532)

### Bug Fixes

- **[Anthropic](../../docs/providers/anthropic)**
    - Fix: Pricing for Claude Sonnet 4.5 in US regions is 10x too high - [PR #15374](https://github.com/BerriAI/litellm/pull/15374)

- **[OpenRouter](../../docs/providers/openrouter)**
    - Change gpt-5-codex support in model_price json - [PR #15540](https://github.com/BerriAI/litellm/pull/15540)

- **[Bedrock](../../docs/providers/bedrock)**
    - Fix filtering headers for signature calcs - [PR #15590](https://github.com/BerriAI/litellm/pull/15590)

- **General**
    - Add native reasoning and streaming support flag for gpt-5-codex - [PR #15569](https://github.com/BerriAI/litellm/pull/15569)

---

## LLM API Endpoints

#### Features

- **[Responses API](../../docs/response_api)**
    - Responses API - enable calling anthropic/gemini models in Responses API streaming in openai ruby sdk + DB - sanity check pending migrations before startup - [PR #15432](https://github.com/BerriAI/litellm/pull/15432)
    - Add support for responses mode in health check - [PR #15658](https://github.com/BerriAI/litellm/pull/15658)

- **[OCR API](../../docs/ocr)**
    - Feat: Add native litellm.ocr() functions - [PR #15567](https://github.com/BerriAI/litellm/pull/15567)
    - Feat: Add /ocr route on LiteLLM AI Gateway - Adds support for native Mistral OCR calling - [PR #15571](https://github.com/BerriAI/litellm/pull/15571)
    - Feat: Add Azure AI Mistral OCR Integration - [PR #15572](https://github.com/BerriAI/litellm/pull/15572)
    - Feat: Native /ocr endpoint support - [PR #15573](https://github.com/BerriAI/litellm/pull/15573)
    - Feat: Add Cost Tracking for /ocr endpoints - [PR #15678](https://github.com/BerriAI/litellm/pull/15678)

- **[/generateContent](../../docs/providers/gemini)**
    - Fix: GEMINI - CLI - add google_routes to llm_api_routes - [PR #15500](https://github.com/BerriAI/litellm/pull/15500)
    - Fix Pydantic validation error for citationMetadata.citationSources in Google GenAI responses - [PR #15592](https://github.com/BerriAI/litellm/pull/15592)

- **[Images API](../../docs/image_generation)**
    - Fix: Dall-e-2 for Image Edits API - [PR #15604](https://github.com/BerriAI/litellm/pull/15604)

- **[Bedrock Passthrough](../../docs/pass_through/bedrock)**
    - Feat: Allow calling /invoke, /converse routes through AI Gateway + models on config.yaml - [PR #15618](https://github.com/BerriAI/litellm/pull/15618)

#### Bugs

- **General**
    - Fix: Convert object to a correct type - [PR #15634](https://github.com/BerriAI/litellm/pull/15634)
    - Bug Fix: Tags as metadata dicts were raising exceptions - [PR #15625](https://github.com/BerriAI/litellm/pull/15625)
    - Add type hint to function_to_dict and fix typo - [PR #15580](https://github.com/BerriAI/litellm/pull/15580)

---

## Management Endpoints / UI

#### Features

- **Virtual Keys**
    - Docs: Key Rotations - [PR #15455](https://github.com/BerriAI/litellm/pull/15455)
    - Fix: UI - Key Max Budget Removal Error Fix - [PR #15672](https://github.com/BerriAI/litellm/pull/15672)
    - litellm_Key Settings Max Budget Removal Error Fix - [PR #15669](https://github.com/BerriAI/litellm/pull/15669)

- **Teams**
    - Feat: Allow Team Admins to export a report of the team spending - [PR #15542](https://github.com/BerriAI/litellm/pull/15542)

- **Passthrough**
    - Feat: Passthrough - allow admin to give access to specific passthrough endpoints - [PR #15401](https://github.com/BerriAI/litellm/pull/15401)

- **SCIM v2**
    - Feat(scim_v2.py): if group.id doesn't exist, use external id + Passthrough - ensure updates and deletions persist across instances - [PR #15276](https://github.com/BerriAI/litellm/pull/15276)

- **SSO**
    - Feat: UI SSO - Add PKCE for OKTA SSO - [PR #15608](https://github.com/BerriAI/litellm/pull/15608)
    - Fix: Separate OAuth M2M authentication from UI SSO + Handle Introspection endpoint for Oauth2 - [PR #15667](https://github.com/BerriAI/litellm/pull/15667)
    - Fix/entraid app roles jwt claim clean - [PR #15583](https://github.com/BerriAI/litellm/pull/15583)

---

## Logging / Guardrail / Prompt Management Integrations

#### Guardrails

- **General**
    - Fix apply_guardrail endpoint returning raw string instead of ApplyGuardrailResponse - [PR #15436](https://github.com/BerriAI/litellm/pull/15436)
    - Fix: Ensure guardrail memory sync after database updates - [PR #15633](https://github.com/BerriAI/litellm/pull/15633)
    - Feat: add guardrail for image generation - [PR #15619](https://github.com/BerriAI/litellm/pull/15619)
    - Feat: Add Guardrails for /v1/messages and /v1/responses API - [PR #15686](https://github.com/BerriAI/litellm/pull/15686)

- **[Pillar Security](../../docs/proxy/guardrails)**
    - Feature: update pillar security integration to support no persistence mode in litellm proxy - [PR #15599](https://github.com/BerriAI/litellm/pull/15599)

#### Prompt Management

- **General**
    - Small fix code snippet custom_prompt_management.md - [PR #15544](https://github.com/BerriAI/litellm/pull/15544)

---

## Spend Tracking, Budgets and Rate Limiting

- **Cost Tracking**
    - Feat: Cost Tracking - specify a global vendor discount for costs - [PR #15546](https://github.com/BerriAI/litellm/pull/15546)
    - Feat: UI - Allow setting Provider Discounts on UI - [PR #15550](https://github.com/BerriAI/litellm/pull/15550)

- **Budgets**
    - Fix: improve budget clarity - [PR #15682](https://github.com/BerriAI/litellm/pull/15682)

---

## Performance / Loadbalancing / Reliability improvements

- **Router Optimizations**
    - Perf(router): use shallow copy instead of deepcopy for model aliases - 10-100x faster than deepcopy on nested dict structures - [PR #15576](https://github.com/BerriAI/litellm/pull/15576)
    - Perf(router): optimize string concatenation in hash generation - Improves time complexity from O(n²) to O(n) - [PR #15575](https://github.com/BerriAI/litellm/pull/15575)
    - Perf(router): optimize model lookups with O(1) data structures - Replace O(n) scans with index map lookups - [PR #15578](https://github.com/BerriAI/litellm/pull/15578)
    - Perf(router): optimize model lookups with O(1) index maps - Use model_id_to_deployment_index_map and model_name_to_deployment_indices for instant lookups - [PR #15574](https://github.com/BerriAI/litellm/pull/15574)
    - Perf(router): optimize timing functions in completion hot path - Use time.perf_counter() for duration measurements and time.monotonic() for timeout calculations, providing 30-40% faster timing calls - [PR #15617](https://github.com/BerriAI/litellm/pull/15617)

- **SSL/TLS Performance**
    - Feat(ssl): add configurable ECDH curve for TLS performance - Configure via ssl_ecdh_curve setting to disable PQC on OpenSSL 3.x for better performance - [PR #15617](https://github.com/BerriAI/litellm/pull/15617)

- **Token Counter**
    - Fix(token-counter): extract model_info from deployment for custom_tokenizer - [PR #15680](https://github.com/BerriAI/litellm/pull/15680)

- **Performance Metrics**
    - Add: perf summary - [PR #15458](https://github.com/BerriAI/litellm/pull/15458)

- **CI/CD**
    - Fix: CI/CD - Missing env key & Linter type error - [PR #15606](https://github.com/BerriAI/litellm/pull/15606)

---

## Documentation Updates

- **Provider Documentation**
    - Litellm docs 10 11 2025 - [PR #15457](https://github.com/BerriAI/litellm/pull/15457)
    - Docs: add ecs deployment guide - [PR #15468](https://github.com/BerriAI/litellm/pull/15468)
    - Docs: Update benchmark results - [PR #15461](https://github.com/BerriAI/litellm/pull/15461)
    - Fix: add missing context to benchmark docs - [PR #15688](https://github.com/BerriAI/litellm/pull/15688)

- **General**
    - Fixed a few typos - [PR #15267](https://github.com/BerriAI/litellm/pull/15267)

---

## New Contributors

* @jlan-nl made their first contribution in [PR #15374](https://github.com/BerriAI/litellm/pull/15374)
* @ImadSaddik made their first contribution in [PR #15267](https://github.com/BerriAI/litellm/pull/15267)
* @huangyafei made their first contribution in [PR #15472](https://github.com/BerriAI/litellm/pull/15472)
* @mubashir1osmani made their first contribution in [PR #15468](https://github.com/BerriAI/litellm/pull/15468)
* @kowyo made their first contribution in [PR #15465](https://github.com/BerriAI/litellm/pull/15465)
* @dhruvyad made their first contribution in [PR #15448](https://github.com/BerriAI/litellm/pull/15448)
* @davizucon made their first contribution in [PR #15544](https://github.com/BerriAI/litellm/pull/15544)
* @FelipeRodriguesGare made their first contribution in [PR #15540](https://github.com/BerriAI/litellm/pull/15540)
* @ndrsfel made their first contribution in [PR #15557](https://github.com/BerriAI/litellm/pull/15557)
* @shinharaguchi made their first contribution in [PR #15598](https://github.com/BerriAI/litellm/pull/15598)
* @TensorNull made their first contribution in [PR #15591](https://github.com/BerriAI/litellm/pull/15591)
* @TeddyAmkie made their first contribution in [PR #15583](https://github.com/BerriAI/litellm/pull/15583)
* @aniketmaurya made their first contribution in [PR #15580](https://github.com/BerriAI/litellm/pull/15580)
* @eddierichter-amd made their first contribution in [PR #15554](https://github.com/BerriAI/litellm/pull/15554)
* @konekohana made their first contribution in [PR #15535](https://github.com/BerriAI/litellm/pull/15535)
* @Classic298 made their first contribution in [PR #15495](https://github.com/BerriAI/litellm/pull/15495)
* @afogel made their first contribution in [PR #15599](https://github.com/BerriAI/litellm/pull/15599)
* @orolega made their first contribution in [PR #15633](https://github.com/BerriAI/litellm/pull/15633)
* @LucasSugi made their first contribution in [PR #15634](https://github.com/BerriAI/litellm/pull/15634)
* @uc4w6c made their first contribution in [PR #15619](https://github.com/BerriAI/litellm/pull/15619)
* @Sameerlite made their first contribution in [PR #15658](https://github.com/BerriAI/litellm/pull/15658)
* @yuneng-jiang made their first contribution in [PR #15672](https://github.com/BerriAI/litellm/pull/15672)
* @Nikro made their first contribution in [PR #15680](https://github.com/BerriAI/litellm/pull/15680)

---

## Full Changelog

**[View complete changelog on GitHub](https://github.com/BerriAI/litellm/compare/v1.78.0-stable...v1.78.4-stable)**

