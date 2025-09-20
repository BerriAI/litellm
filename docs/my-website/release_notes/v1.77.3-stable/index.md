---
title: "[Preview] v1.77.3-stable - Priority Based Rate Limiting"
slug: "v1-77-3"
date: 2025-09-21T10:00:00
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
ghcr.io/berriai/litellm:main-v1.77.3-stable
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.77.3
```

</TabItem>
</Tabs>

---

## Key Highlights

- **+550 RPS Performance Improvements** - Optimizations in request handling and object initialization
- **Priority Based Rate Limiting** - Improved rate limiting for high-traffic scenarios


## New Models / Updated Models

#### New Model Support

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| SambaNova | `sambanova/deepseek-v3.1` | 128K | $0.90 | $0.90 | Chat completions |
| SambaNova | `sambanova/gpt-oss-120b` | 128K | $0.72 | $0.72 | Chat completions |
| OVHCloud | Various models | Varies | Contact provider | Contact provider | Chat completions |
| CompactifAI | Various models | Varies | Contact provider | Contact provider | Chat completions |
| TwelveLabs | `twelvelabs/marengo-embed-2.7` | 32K | $0.12 | $0.00 | Embeddings |


---

## LLM API Endpoints

#### New Features

- **[/responses](../../docs/response_api)** API cancel endpoint support - [PR #14594](https://github.com/BerriAI/litellm/pull/14594)
- **OpenAI & Azure Cancel Endpoint** support for batch operations - [PR #14561](https://github.com/BerriAI/litellm/pull/14561)
- **AWS Bedrock CountTokens API** implementation - [PR #14557](https://github.com/BerriAI/litellm/pull/14557)
- **Gemini Batch API** support - [PR #14733](https://github.com/BerriAI/litellm/pull/14733)

#### Provider Updates

- **[OVHCloud AI Endpoints](../../docs/providers/ovhcloud)** - New provider support - [PR #14494](https://github.com/BerriAI/litellm/pull/14494)
- **[CompactifAI](../../docs/providers/compactifai)** - New provider integration - [PR #14532](https://github.com/BerriAI/litellm/pull/14532)
- **[Cohere](../../docs/providers/cohere)** - Handle Generate API deprecation, default to chat endpoints - [PR #14676](https://github.com/BerriAI/litellm/pull/14676)
- **[Bedrock](../../docs/providers/bedrock)**
  - Cross-region inference profile cost calculation fix - [PR #14566](https://github.com/BerriAI/litellm/pull/14566)
  - AWS external ID parameter support - [PR #14582](https://github.com/BerriAI/litellm/pull/14582)
  - Empty arguments handling in tool calls - [PR #14583](https://github.com/BerriAI/litellm/pull/14583)
  - Titan V2 encoding_format parameter support - [PR #14687](https://github.com/BerriAI/litellm/pull/14687)
- **[Vertex AI](../../docs/providers/vertex)**
  - Gemini labels field provider-aware filtering - [PR #14563](https://github.com/BerriAI/litellm/pull/14563)
  - Avoid deepcopy crash with non-pickleables - [PR #14418](https://github.com/BerriAI/litellm/pull/14418)
- **[Volcengine](../../docs/providers/volcengine)**
  - Fixed thinking parameters when disabled - [PR #14569](https://github.com/BerriAI/litellm/pull/14569)

#### Bug Fixes

- **Grok Code Models** - Fix unsupported stop parameter - [PR #14565](https://github.com/BerriAI/litellm/pull/14565)
- **Completion Chat ID** fix - [PR #14548](https://github.com/BerriAI/litellm/pull/14548)
- **Gemini API Base** update - [PR #14604](https://github.com/BerriAI/litellm/pull/14604)
- **Gemini 2.5 Flash Image Preview** model routing fix - [PR #14715](https://github.com/BerriAI/litellm/pull/14715)
- **Rate Limiter** AttributeError fix - [PR #14609](https://github.com/BerriAI/litellm/pull/14609)

---

## Management Endpoints / UI

#### Features

- **Team Member Service Account Keys** - Allow team members to view keys they create - [PR #14619](https://github.com/BerriAI/litellm/pull/14619)
- **Default Budget for JWT Teams** - Auto-assign budgets to generated teams - [PR #14514](https://github.com/BerriAI/litellm/pull/14514)
- **Langsmith Sampling Rate** - Team-level tracing configuration - [PR #14740](https://github.com/BerriAI/litellm/pull/14740)
- **SSO Access Control Groups** - Enhanced token info endpoint integration - [PR #14738](https://github.com/BerriAI/litellm/pull/14738)
- **Health Test Connect Protection** - Restrict access based on model creation permissions - [PR #14650](https://github.com/BerriAI/litellm/pull/14650)

#### Bug Fixes

- **SCIM v2** - Fix group PUSH and PUT operations for non-existent members - [PR #14581](https://github.com/BerriAI/litellm/pull/14581)
- **Guardrail View/Edit/Delete** behavior fixes - [PR #14622](https://github.com/BerriAI/litellm/pull/14622)
- **In-Memory Guardrail** update failures - [PR #14653](https://github.com/BerriAI/litellm/pull/14653)

---

## Performance Improvements

- **+500 RPS Performance Boost** when sending the `user` field - [PR #14616](https://github.com/BerriAI/litellm/pull/14616)
- **+50 RPS** by removing iscoroutine from hot path - [PR #14649](https://github.com/BerriAI/litellm/pull/14649)
- **7% reduction** in __init__ overhead - [PR #14689](https://github.com/BerriAI/litellm/pull/14689)
- **Generic Object Pool** implementation for better resource management - [PR #14702](https://github.com/BerriAI/litellm/pull/14702)

---

## Logging / Guardrail Integrations

#### New Integrations

- **[PostHog Observability](../../docs/observability/posthog)** - Complete observability integration - [PR #14610](https://github.com/BerriAI/litellm/pull/14610)
- **Langfuse Logging** for Responses API - [PR #14597](https://github.com/BerriAI/litellm/pull/14597)
- **DataDog Spend Metrics** - Enhanced spend tracking - [PR #14555](https://github.com/BerriAI/litellm/pull/14555)
- **DataDog Stream Support** - is_streamed_request parameter - [PR #14673](https://github.com/BerriAI/litellm/pull/14673)

#### Guardrail Features

- **Tool Permission Guardrail** - Fine-grained tool access control - [PR #14519](https://github.com/BerriAI/litellm/pull/14519)
- **Bedrock Guardrails** - Selective guarding support with runtime endpoint configuration - [PR #14575](https://github.com/BerriAI/litellm/pull/14575), [PR #14650](https://github.com/BerriAI/litellm/pull/14650)
- **Amazon Bedrock Guardrail Info View** - Enhanced logging visualization - [PR #14696](https://github.com/BerriAI/litellm/pull/14696)
- **Default Last Message** in guardrails - [PR #14640](https://github.com/BerriAI/litellm/pull/14640)

#### Bug Fixes

- **DataDog Tool Calls** - Fixed metadata passing - [PR #14531](https://github.com/BerriAI/litellm/pull/14531)
- **Prometheus Multi-Worker** support - [PR #14530](https://github.com/BerriAI/litellm/pull/14530)
- **User Email Labels** in Prometheus monitoring - [PR #14520](https://github.com/BerriAI/litellm/pull/14520)
- **Bedrock Guardrail Silent Failure** correction - [PR #14707](https://github.com/BerriAI/litellm/pull/14707)
- **Opik Timezone Issue** fix - [PR #14708](https://github.com/BerriAI/litellm/pull/14708)

---

## Performance / Loadbalancing / Reliability improvements

#### Rate Limiting & Caching

- **Cache Key Collision Fix** - Resolved soft budget alert cache issues - [PR #14491](https://github.com/BerriAI/litellm/pull/14491)
- **Dynamic Rate Limiter v3** - Priority routing improvements - [PR #14734](https://github.com/BerriAI/litellm/pull/14734)
- **Enhanced Rate Limit Errors** - More detailed error messages - [PR #14736](https://github.com/BerriAI/litellm/pull/14736)

#### Cost Tracking

- **Responses API Cost Calculation** fix - [PR #14675](https://github.com/BerriAI/litellm/pull/14675)
- **Anthropic Cache Token Pricing** - Separate 1-hour vs 5-minute cache creation costs - [PR #14620](https://github.com/BerriAI/litellm/pull/14620), [PR #14652](https://github.com/BerriAI/litellm/pull/14652)
- **Indochina Time Timezone** support for budget resets - [PR #14666](https://github.com/BerriAI/litellm/pull/14666)

---

## General Proxy Improvements

#### MCP (Model Context Protocol)

- **MCP Server Alias Parsing** - Multi-part URL path support - [PR #14558](https://github.com/BerriAI/litellm/pull/14558)
- **MCP Filter Recomputation** - After server deletion - [PR #14542](https://github.com/BerriAI/litellm/pull/14542)
- **MCP Gateway Tools List** improvements - [PR #14695](https://github.com/BerriAI/litellm/pull/14695)

#### Storage & Configuration

- **S3 Endpoint URL** - Fixed 404 errors - [PR #14559](https://github.com/BerriAI/litellm/pull/14559)
- **Response API Cold Storage** - Improved handling and configuration - [PR #14534](https://github.com/BerriAI/litellm/pull/14534)
- **Middle-Truncation** for spend log payloads - [PR #14637](https://github.com/BerriAI/litellm/pull/14637)

#### Batch Processing

- **Bedrock Retrieve Endpoint** - Batch support - [PR #14618](https://github.com/BerriAI/litellm/pull/14618)
- **Bedrock Twelve Labs Embedding** provider support - [PR #14697](https://github.com/BerriAI/litellm/pull/14697)

#### Security

- **Security Update** - Bump aiohttp==3.12.14, fix CVE-2025-53643 - [PR #14638](https://github.com/BerriAI/litellm/pull/14638)

---

## New Contributors

* @luisfucros made their first contribution in [PR #14500](https://github.com/BerriAI/litellm/pull/14500)
* @hanakannzashi made their first contribution in [PR #14548](https://github.com/BerriAI/litellm/pull/14548)
* @eliasto made their first contribution in [PR #14494](https://github.com/BerriAI/litellm/pull/14494)
* @Rasmusafj made their first contribution in [PR #14491](https://github.com/BerriAI/litellm/pull/14491)
* @LingXuanYin made their first contribution in [PR #14569](https://github.com/BerriAI/litellm/pull/14569)
* @ronaldpereira made their first contribution in [PR #14613](https://github.com/BerriAI/litellm/pull/14613)
* @hula-la made their first contribution in [PR #14534](https://github.com/BerriAI/litellm/pull/14534)
* @carlos-marchal-ph made their first contribution in [PR #14610](https://github.com/BerriAI/litellm/pull/14610)
* @akraines made their first contribution in [PR #14637](https://github.com/BerriAI/litellm/pull/14637)
* @mrFranklin made their first contribution in [PR #14708](https://github.com/BerriAI/litellm/pull/14708)
* @tcx4c70 made their first contribution in [PR #14675](https://github.com/BerriAI/litellm/pull/14675)
* @michaeltansg made their first contribution in [PR #14666](https://github.com/BerriAI/litellm/pull/14666)
* @tosi29 made their first contribution in [PR #14725](https://github.com/BerriAI/litellm/pull/14725)
* @gmdfalk made their first contribution in [PR #14735](https://github.com/BerriAI/litellm/pull/14735)
* @FelipeRodriguesGare made their first contribution in [PR #14733](https://github.com/BerriAI/litellm/pull/14733)
* @mritunjaysharma394 made their first contribution in [PR #14678](https://github.com/BerriAI/litellm/pull/14678)

---

## **[Full Changelog](https://github.com/BerriAI/litellm/compare/v1.77.2.rc.1...v1.77.3.rc.1)**
