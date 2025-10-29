---
title: "v1.77.5-stable - MCP OAuth 2.0 Support"
slug: "v1-77-5"
date: 2025-09-29T10:00:00
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
ghcr.io/berriai/litellm:v1.77.5-stable
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.77.5
```

</TabItem>
</Tabs>

---

## Key Highlights

- **MCP OAuth 2.0 Support** - Enhanced authentication for Model Context Protocol integrations
- **Scheduled Key Rotations** - Automated key rotation capabilities for enhanced security
- **New Gemini 2.5 Flash & Flash-lite Models** - Latest September 2025 preview models with improved pricing and features
- **Performance Improvements** - 54% RPS improvement

---

### Performance Improvements - 54% RPS Improvement

<Image img={require('../../img/release_notes/perf_77_5.png')}  style={{ width: '800px', height: 'auto' }} />

<br/>

This release brings a 54% RPS improvement (1,040 → 1,602 RPS, aggregated) per instance. 

The improvement comes from fixing O(n²) inefficiencies in the LiteLLM Router, primarily caused by repeated use of `in` statements inside loops over large arrays. 

Tests were run with a database-only setup (no cache hits).

#### Test Setup

All benchmarks were executed using Locust with 1,000 concurrent users and a ramp-up of 500. The environment was configured to stress the routing layer and eliminate caching as a variable.

**System Specs**

- **CPU:** 8 vCPUs
- **Memory:** 32 GB RAM

**Configuration (config.yaml)**

View the complete configuration: [gist.github.com/AlexsanderHamir/config.yaml](https://gist.github.com/AlexsanderHamir/53f7d554a5d2afcf2c4edb5b6be68ff4)

**Load Script (no_cache_hits.py)**

View the complete load testing script: [gist.github.com/AlexsanderHamir/no_cache_hits.py](https://gist.github.com/AlexsanderHamir/42c33d7a4dc7a57f56a78b560dee3a42)

---


## New Models / Updated Models

#### New Model Support

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| Gemini | `gemini-2.5-flash-preview-09-2025` | 1M | $0.30 | $2.50 | Chat, reasoning, vision, audio |
| Gemini | `gemini-2.5-flash-lite-preview-09-2025` | 1M | $0.10 | $0.40 | Chat, reasoning, vision, audio |
| Gemini | `gemini-flash-latest` | 1M | $0.30 | $2.50 | Chat, reasoning, vision, audio |
| Gemini | `gemini-flash-lite-latest` | 1M | $0.10 | $0.40 | Chat, reasoning, vision, audio |
| DeepSeek | `deepseek-chat` | 131K | $0.60 | $1.70 | Chat, function calling, caching |
| DeepSeek | `deepseek-reasoner` | 131K | $0.60 | $1.70 | Chat, reasoning |
| Bedrock | `deepseek.v3-v1:0` | 164K | $0.58 | $1.68 | Chat, reasoning, function calling |
| Azure | `azure/gpt-5-codex` | 272K | $1.25 | $10.00 | Responses API, reasoning, vision |
| OpenAI | `gpt-5-codex` | 272K | $1.25 | $10.00 | Responses API, reasoning, vision |
| SambaNova | `sambanova/DeepSeek-V3.1` | 33K | $3.00 | $4.50 | Chat, reasoning, function calling |
| SambaNova | `sambanova/gpt-oss-120b` | 131K | $3.00 | $4.50 | Chat, reasoning, function calling |
| Bedrock | `qwen.qwen3-coder-480b-a35b-v1:0` | 262K | $0.22 | $1.80 | Chat, reasoning, function calling |
| Bedrock | `qwen.qwen3-235b-a22b-2507-v1:0` | 262K | $0.22 | $0.88 | Chat, reasoning, function calling |
| Bedrock | `qwen.qwen3-coder-30b-a3b-v1:0` | 262K | $0.15 | $0.60 | Chat, reasoning, function calling |
| Bedrock | `qwen.qwen3-32b-v1:0` | 131K | $0.15 | $0.60 | Chat, reasoning, function calling |
| Vertex AI | `vertex_ai/qwen/qwen3-next-80b-a3b-instruct-maas` | 262K | $0.15 | $1.20 | Chat, function calling |
| Vertex AI | `vertex_ai/qwen/qwen3-next-80b-a3b-thinking-maas` | 262K | $0.15 | $1.20 | Chat, function calling |
| Vertex AI | `vertex_ai/deepseek-ai/deepseek-v3.1-maas` | 164K | $1.35 | $5.40 | Chat, reasoning, function calling |
| OpenRouter | `openrouter/x-ai/grok-4-fast:free` | 2M | $0.00 | $0.00 | Chat, reasoning, function calling |
| XAI | `xai/grok-4-fast-reasoning` | 2M | $0.20 | $0.50 | Chat, reasoning, function calling |
| XAI | `xai/grok-4-fast-non-reasoning` | 2M | $0.20 | $0.50 | Chat, function calling |

#### Features

- **[Gemini](../../docs/providers/gemini)**
    - Added Gemini 2.5 Flash and Flash-lite preview models (September 2025 release) with improved pricing - [PR #14948](https://github.com/BerriAI/litellm/pull/14948)
    - Added new Anthropic web fetch tool support - [PR #14951](https://github.com/BerriAI/litellm/pull/14951)
- **[XAI](../../docs/providers/xai)**
    - Add xai/grok-4-fast models - [PR #14833](https://github.com/BerriAI/litellm/pull/14833)
- **[Anthropic](../../docs/providers/anthropic)**
    - Updated Claude Sonnet 4 configs to reflect million-token context window pricing - [PR #14639](https://github.com/BerriAI/litellm/pull/14639)
    - Added supported text field to anthropic citation response - [PR #14164](https://github.com/BerriAI/litellm/pull/14164)
- **[Bedrock](../../docs/providers/bedrock)**
    - Added support for Qwen models family & Deepseek 3.1 to Amazon Bedrock - [PR #14845](https://github.com/BerriAI/litellm/pull/14845)
    - Support requestMetadata in Bedrock Converse API - [PR #14570](https://github.com/BerriAI/litellm/pull/14570)
- **[Vertex AI](../../docs/providers/vertex)**
    - Added vertex_ai/qwen models and azure/gpt-5-codex - [PR #14844](https://github.com/BerriAI/litellm/pull/14844)
    - Update vertex ai qwen model pricing - [PR #14828](https://github.com/BerriAI/litellm/pull/14828)
    - Vertex AI Context Caching: use Vertex ai API v1 instead of v1beta1 and accept 'cachedContent' param - [PR #14831](https://github.com/BerriAI/litellm/pull/14831)
- **[SambaNova](../../docs/providers/sambanova)**
    - Add sambanova deepseek v3.1 and gpt-oss-120b - [PR #14866](https://github.com/BerriAI/litellm/pull/14866)
- **[OpenAI](../../docs/providers/openai)**
    - Fix inconsistent token configs for gpt-5 models - [PR #14942](https://github.com/BerriAI/litellm/pull/14942)
    - GPT-3.5-Turbo price updated - [PR #14858](https://github.com/BerriAI/litellm/pull/14858)
- **[OpenRouter](../../docs/providers/openrouter)**
    - Add gpt-5 and gpt-5-codex to OpenRouter cost map - [PR #14879](https://github.com/BerriAI/litellm/pull/14879)
- **[VLLM](../../docs/providers/vllm)**
    - Fix vllm passthrough - [PR #14778](https://github.com/BerriAI/litellm/pull/14778)
- **[Flux](../../docs/image_generation)**
    - Support flux image edit - [PR #14790](https://github.com/BerriAI/litellm/pull/14790)

### Bug Fixes

- **[Anthropic](../../docs/providers/anthropic)**
    - Fix: Support claude code auth via subscription (anthropic) - [PR #14821](https://github.com/BerriAI/litellm/pull/14821)
    - Fix Anthropic streaming IDs - [PR #14965](https://github.com/BerriAI/litellm/pull/14965)
    - Revert incorrect changes to sonnet-4 max output tokens - [PR #14933](https://github.com/BerriAI/litellm/pull/14933)
- **[OpenAI](../../docs/providers/openai)**
    - Fix a bug where openai image edit silently ignores multiple images - [PR #14893](https://github.com/BerriAI/litellm/pull/14893)
- **[VLLM](../../docs/providers/vllm)**
    - Fix: vLLM provider's rerank endpoint from /v1/rerank to /rerank - [PR #14938](https://github.com/BerriAI/litellm/pull/14938)

#### New Provider Support

- **[W&B Inference](../../docs/providers/wandb)**
    - Add W&B Inference to LiteLLM - [PR #14416](https://github.com/BerriAI/litellm/pull/14416)

---

## LLM API Endpoints

#### Features

- **General**
    - Add SDK support for additional headers - [PR #14761](https://github.com/BerriAI/litellm/pull/14761)
    - Add shared_session parameter for aiohttp ClientSession reuse - [PR #14721](https://github.com/BerriAI/litellm/pull/14721)

#### Bugs

- **General**
    - Fix: Streaming tool call index assignment for multiple tool calls - [PR #14587](https://github.com/BerriAI/litellm/pull/14587)
    - Fix load credentials in token counter proxy - [PR #14808](https://github.com/BerriAI/litellm/pull/14808)

---

## Management Endpoints / UI

#### Features

- **Proxy CLI Auth** 
    - Allow re-using cli auth token - [PR #14780](https://github.com/BerriAI/litellm/pull/14780)
    - Create a python method to login using litellm proxy - [PR #14782](https://github.com/BerriAI/litellm/pull/14782)
    - Fixes for LiteLLM Proxy CLI to Auth to Gateway - [PR #14836](https://github.com/BerriAI/litellm/pull/14836)
    
**Virtual Keys**    
    - Initial support for scheduled key rotations - [PR #14877](https://github.com/BerriAI/litellm/pull/14877)
    - Allow scheduling key rotations when creating virtual keys - [PR #14960](https://github.com/BerriAI/litellm/pull/14960)

**Models + Endpoints**
    - Fix: added Oracle to provider's list - [PR #14835](https://github.com/BerriAI/litellm/pull/14835)


#### Bugs

- **SSO** - Fix: SSO "Clear" button writes empty values instead of removing SSO config - [PR #14826](https://github.com/BerriAI/litellm/pull/14826)
- **Admin Settings** - Remove useful links from admin settings - [PR #14918](https://github.com/BerriAI/litellm/pull/14918)
- **Management Routes** - Add /user/list to management routes - [PR #14868](https://github.com/BerriAI/litellm/pull/14868)
---

## Logging / Guardrail / Prompt Management Integrations

#### Features

- **[DataDog](../../docs/proxy/logging#datadog)**
    - Logging - `datadog` callback Log message content w/o sending to datadog - [PR #14909](https://github.com/BerriAI/litellm/pull/14909)
- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Adding langfuse usage details for cached tokens - [PR #10955](https://github.com/BerriAI/litellm/pull/10955)
- **[Opik](../../docs/proxy/logging#opik)**
    - Improve opik integration code - [PR #14888](https://github.com/BerriAI/litellm/pull/14888)
- **[SQS](../../docs/proxy/logging#sqs)**
    - Error logging support for SQS Logger - [PR #14974](https://github.com/BerriAI/litellm/pull/14974)

#### Guardrails

- **LakeraAI v2 Guardrail** - Ensure exception is raised correctly - [PR #14867](https://github.com/BerriAI/litellm/pull/14867)
- **Presidio Guardrail** - Support custom entity types in Presidio guardrail with Union[PiiEntityType, str] - [PR #14899](https://github.com/BerriAI/litellm/pull/14899)
- **Noma Guardrail** - Add noma guardrail provider to ui - [PR #14415](https://github.com/BerriAI/litellm/pull/14415)

#### Prompt Management

- **BitBucket Integration** - Add BitBucket Integration for Prompt Management - [PR #14882](https://github.com/BerriAI/litellm/pull/14882)

---

## Spend Tracking, Budgets and Rate Limiting

- **Service Tier Pricing** - Add service_tier based pricing support for openai (BOTH Service & Priority Support) - [PR #14796](https://github.com/BerriAI/litellm/pull/14796)
- **Cost Tracking** - Show input, output, tool call cost breakdown in StandardLoggingPayload - [PR #14921](https://github.com/BerriAI/litellm/pull/14921)
- **Parallel Request Limiter v3** 
    - Ensure Lua scripts can execute on redis cluster - [PR #14968](https://github.com/BerriAI/litellm/pull/14968)
    - Fix: get metadata info from both metadata and litellm_metadata fields - [PR #14783](https://github.com/BerriAI/litellm/pull/14783)
- **Priority Reservation** - Fix: Priority Reservation: keys without priority metadata receive higher priority than keys with explicit priority configurations - [PR #14832](https://github.com/BerriAI/litellm/pull/14832)

---

## MCP Gateway

- **MCP Configuration** - Enable custom fields in mcp_info configuration - [PR #14794](https://github.com/BerriAI/litellm/pull/14794)
- **MCP Tools** - Remove server_name prefix from list_tools - [PR #14720](https://github.com/BerriAI/litellm/pull/14720)
- **OAuth Flow** - Initial commit for v2 oauth flow - [PR #14964](https://github.com/BerriAI/litellm/pull/14964)

---

## Performance / Loadbalancing / Reliability improvements

- **Memory Leak Fix** - Fix InMemoryCache unbounded growth when TTLs are set - [PR #14869](https://github.com/BerriAI/litellm/pull/14869)
- **Cache Performance** - Fix: cache root cause - [PR #14827](https://github.com/BerriAI/litellm/pull/14827)
- **Concurrency Fix** - Fix concurrency/scaling when many Python threads do streaming using *sync* completions - [PR #14816](https://github.com/BerriAI/litellm/pull/14816)
- **Performance Optimization** - Fix: reduce get_deployment cost to O(1) - [PR #14967](https://github.com/BerriAI/litellm/pull/14967)
- **Performance Optimization** - Fix: remove slow string operation - [PR #14955](https://github.com/BerriAI/litellm/pull/14955)
- **DB Connection Management** - Fix: DB connection state retries - [PR #14925](https://github.com/BerriAI/litellm/pull/14925)



---

## Documentation Updates

- **Provider Documentation** - Fix docs for provider_specific_params.md - [PR #14787](https://github.com/BerriAI/litellm/pull/14787)
- **Model References** - Update model references from gemini-pro to gemini-2.5-pro - [PR #14775](https://github.com/BerriAI/litellm/pull/14775)
- **Letta Guide** - Add Letta Guide documentation - [PR #14798](https://github.com/BerriAI/litellm/pull/14798)
- **README** - Make the README document clearer - [PR #14860](https://github.com/BerriAI/litellm/pull/14860)
- **Session Management** - Update docs for session management availability - [PR #14914](https://github.com/BerriAI/litellm/pull/14914)
- **Cost Documentation** - Add documentation for additional cost-related keys in custom pricing - [PR #14949](https://github.com/BerriAI/litellm/pull/14949)
- **Azure Passthrough** - Add azure passthrough documentation - [PR #14958](https://github.com/BerriAI/litellm/pull/14958)
- **General Documentation** - Doc updates sept 2025 - [PR #14769](https://github.com/BerriAI/litellm/pull/14769)
    - Clarified bridging between endpoints and mode in docs.
    - Added Vertex AI Gemini API configuration as an alternative in relevant guides.
    Linked AWS authentication info in the Bedrock guardrails documentation.
    - Added Cancel Response API usage with code snippets
    - Clarified that SSO (Single Sign-On) is free for up to 5 users:
    - Alphabetized sidebar, leaving quick start / intros at top of categories
    - Documented max_connections under cache_params.
    - Clarified IAM AssumeRole Policy requirements.
    - Added transform utilities example to Getting Started (showing request transformation).
    - Added references to models.litellm.ai as the full models list in various docs.
    - Added a code snippet for async_post_call_success_hook.
    - Removed broken links to callbacks management guide. - Reformatted and linked cookbooks + other relevant docs
- **Documentation Corrections** - Corrected docs updates sept 2025 - [PR #14916](https://github.com/BerriAI/litellm/pull/14916)

---

## New Contributors

* @uzaxirr made their first contribution in [PR #14761](https://github.com/BerriAI/litellm/pull/14761)
* @xprilion made their first contribution in [PR #14416](https://github.com/BerriAI/litellm/pull/14416)
* @CH-GAGANRAJ made their first contribution in [PR #14779](https://github.com/BerriAI/litellm/pull/14779)
* @otaviofbrito made their first contribution in [PR #14778](https://github.com/BerriAI/litellm/pull/14778)
* @danielmklein made their first contribution in [PR #14639](https://github.com/BerriAI/litellm/pull/14639)
* @Jetemple made their first contribution in [PR #14826](https://github.com/BerriAI/litellm/pull/14826)
* @akshoop made their first contribution in [PR #14818](https://github.com/BerriAI/litellm/pull/14818)
* @hazyone made their first contribution in [PR #14821](https://github.com/BerriAI/litellm/pull/14821)
* @leventov made their first contribution in [PR #14816](https://github.com/BerriAI/litellm/pull/14816)
* @fabriciojoc made their first contribution in [PR #10955](https://github.com/BerriAI/litellm/pull/10955)
* @onlylonly made their first contribution in [PR #14845](https://github.com/BerriAI/litellm/pull/14845)
* @Copilot made their first contribution in [PR #14869](https://github.com/BerriAI/litellm/pull/14869)
* @arsh72 made their first contribution in [PR #14899](https://github.com/BerriAI/litellm/pull/14899)
* @berri-teddy made their first contribution in [PR #14914](https://github.com/BerriAI/litellm/pull/14914)
* @vpbill made their first contribution in [PR #14415](https://github.com/BerriAI/litellm/pull/14415)
* @kgritesh made their first contribution in [PR #14893](https://github.com/BerriAI/litellm/pull/14893)
* @oytunkutrup1 made their first contribution in [PR #14858](https://github.com/BerriAI/litellm/pull/14858)
* @nherment made their first contribution in [PR #14933](https://github.com/BerriAI/litellm/pull/14933)
* @deepanshululla made their first contribution in [PR #14974](https://github.com/BerriAI/litellm/pull/14974)
* @TeddyAmkie made their first contribution in [PR #14758](https://github.com/BerriAI/litellm/pull/14758)
* @SmartManoj made their first contribution in [PR #14775](https://github.com/BerriAI/litellm/pull/14775)
* @uc4w6c made their first contribution in [PR #14720](https://github.com/BerriAI/litellm/pull/14720)
* @luizrennocosta made their first contribution in [PR #14783](https://github.com/BerriAI/litellm/pull/14783)
* @AlexsanderHamir made their first contribution in [PR #14827](https://github.com/BerriAI/litellm/pull/14827)
* @dharamendrak made their first contribution in [PR #14721](https://github.com/BerriAI/litellm/pull/14721)
* @TomeHirata made their first contribution in [PR #14164](https://github.com/BerriAI/litellm/pull/14164)
* @mrFranklin made their first contribution in [PR #14860](https://github.com/BerriAI/litellm/pull/14860)
* @luisfucros made their first contribution in [PR #14866](https://github.com/BerriAI/litellm/pull/14866)
* @huangyafei made their first contribution in [PR #14879](https://github.com/BerriAI/litellm/pull/14879)
* @thiswillbeyourgithub made their first contribution in [PR #14949](https://github.com/BerriAI/litellm/pull/14949)
* @Maximgitman made their first contribution in [PR #14965](https://github.com/BerriAI/litellm/pull/14965)
* @subnet-dev made their first contribution in [PR #14938](https://github.com/BerriAI/litellm/pull/14938)
* @22mSqRi made their first contribution in [PR #14972](https://github.com/BerriAI/litellm/pull/14972)

---

## **[Full Changelog](https://github.com/BerriAI/litellm/compare/v1.77.3.rc.1...v1.77.5.rc.1)**
