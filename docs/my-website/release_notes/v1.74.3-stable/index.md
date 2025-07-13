---
title: "[PRE-RELEASE] v1.74.3-stable"
slug: "v1-74-3-stable"
date: 2025-07-12T10:00:00
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
ghcr.io/berriai/litellm:v1.74.3.rc
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.74.3.rc  
```

</TabItem>
</Tabs>

---


## New Models / Updated Models

#### Pricing / Context Window Updates

| Provider    | Model                                  | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Type |
| ----------- | -------------------------------------- | -------------- | ------------------- | -------------------- | ---- |
| Xai | `xai/grok-4` | 256k | $3.00 | $15.00 | New |
| Xai | `xai/grok-4-0709` | 256k | $3.00 | $15.00 | New |
| Xai | `xai/grok-4-latest` | 256k | $3.00 | $15.00 | New |
| Mistral | `mistral/devstral-small-2507` | 128k | $0.1 | $0.3 | New |
| Mistral | `mistral/devstral-medium-2507` | 128k | $0.4 | $2 | New |
| Azure OpenAI | `azure/o3-deep-research` | 200k | $10 | $40 | New |


#### Features
- **XInference**
    - Image generation API support - [PR](https://github.com/BerriAI/litellm/pull/12439)
- **Bedrock**
    - API Key Auth support for AWS Bedrock API - [PR](https://github.com/BerriAI/litellm/pull/12495)
- **🆕 Dashscope**
    - New integration from Alibaba (enables qwen usage) - [PR](https://github.com/BerriAI/litellm/pull/12361)
- **🆕 Bytez**
    - New /chat/completion integration - [PR](https://github.com/BerriAI/litellm/pull/12121)

#### Bugs
- **Github Copilot**
    - Fix API base url for Github Copilot - [PR](https://github.com/BerriAI/litellm/pull/12418)
- **Bedrock**
    - Ensure supported bedrock/converse/ params = bedrock/ params - [PR](https://github.com/BerriAI/litellm/pull/12466)
    - Fix cache token cost calculation - [PR](https://github.com/BerriAI/litellm/pull/12488)
- **XAI**
    - ensure finish_reason includes tool calls when xai responses with tool calls - [PR](https://github.com/BerriAI/litellm/pull/12545)

---

## LLM API Endpoints

#### Features
- **[/completions](../../docs/text_completion)**
    - Return ‘reasoning_content’ on streaming - [PR](https://github.com/BerriAI/litellm/pull/12377)
- **[/chat/completions](../../docs/completion/input)** 
    - Add 'thinking blocks' to stream chunk builder - [PR](https://github.com/BerriAI/litellm/pull/12395)
- **[/v1/messages](../../docs/anthropic_unified)**
    - Fallbacks support - [PR](https://github.com/BerriAI/litellm/pull/12440)
    - tool call handling for non-anthropic models (/v1/messages to /chat/completion bridge) - [PR](https://github.com/BerriAI/litellm/pull/12473)

---

## [MCP Gateway](../../docs/mcp)

[NEEDS IMAGE]

#### Features
- **[Cost Tracking](../../docs/mcp#-mcp-cost-tracking)**
    - Add Cost Tracking - [PR](https://github.com/BerriAI/litellm/pull/12385)
    - Add usage tracking - [PR](https://github.com/BerriAI/litellm/pull/12397)
    - Add custom cost configuration for each MCP tool - [PR](https://github.com/BerriAI/litellm/pull/12499)
    - Add support for editing MCP cost per tool - [PR](https://github.com/BerriAI/litellm/pull/12501)
    - Allow using custom post call MCP hook for cost tracking - [PR](https://github.com/BerriAI/litellm/pull/12469)
- **[Auth](../../docs/mcp#using-your-mcp-with-client-side-credentials)**
    - Allow customizing what client side auth header to use - [PR](https://github.com/BerriAI/litellm/pull/12460)
    - Raises error when MCP server header is malformed in the request - [PR](https://github.com/BerriAI/litellm/pull/12494)
- **[MCP Server](../../docs/mcp#adding-your-mcp)**
    - Allow using stdio MCPs with LiteLLM (enables using Circle CI MCP w/ LiteLLM) - [PR](https://github.com/BerriAI/litellm/pull/12530), [Get Started](../../docs/mcp#adding-a-stdio-mcp-server)

#### Bugs
- **General**
    - Fix task group is not initialized error - [PR](https://github.com/BerriAI/litellm/pull/12411) s/o [@juancarlosm](https://github.com/juancarlosm)
- **[MCP Server](../../docs/mcp#adding-your-mcp)**
    - Fix mcp tool separator to work with Claude code - [PR](https://github.com/BerriAI/litellm/pull/12430), [Get Started](../../docs/mcp#adding-your-mcp)
    - Add validation to mcp server name to not allow "-" (enables namespaces to work) - [PR](https://github.com/BerriAI/litellm/pull/12515)


---

## Management Endpoints / UI


[NEEDS IMAGE]

#### Features
- **Model Hub**
    - new model hub table view - [PR](https://github.com/BerriAI/litellm/pull/12468)
    - new /public/model_hub endpoint - [PR](https://github.com/BerriAI/litellm/pull/12468)
    - Make Model Hub OSS - [PR](https://github.com/BerriAI/litellm/pull/12553)
    - New ‘make public’ modal flow for showing proxy models on public model hub - [PR](https://github.com/BerriAI/litellm/pull/12555)
- **MCP**
    - support for internal users to use and manage MCP servers - [PR](https://github.com/BerriAI/litellm/pull/12458)
    - Adds UI support to add MCP access groups (similar to namespaces) - [PR](https://github.com/BerriAI/litellm/pull/12470)
    - MCP Tool Testing Playground - [PR](https://github.com/BerriAI/litellm/pull/12520)
    - Show cost config on root of MCP settings - [PR](https://github.com/BerriAI/litellm/pull/12526)
- **Test Key**
    - Stick sessions - [PR](https://github.com/BerriAI/litellm/pull/12365)
    - MCP Access Groups - allow mcp access groups - [PR](https://github.com/BerriAI/litellm/pull/12529)
- **Usage**
    - Truncate long labels and improve tooltip in Top API Keys chart - [PR](https://github.com/BerriAI/litellm/pull/12371)
    - Improve Chart Readability for Tag Usage - [PR](https://github.com/BerriAI/litellm/pull/12378)
- **Teams**
    - Prevent navigation reset after team member operations - [PR](https://github.com/BerriAI/litellm/pull/12424)
    - Team Members - reset budget, if duration set - [PR](https://github.com/BerriAI/litellm/pull/12534)
    - Use central team member budget when max_budget_in_team set on UI - [PR](https://github.com/BerriAI/litellm/pull/12533)
- **SSO**
    - Allow users to run a custom sso login handler - [PR](https://github.com/BerriAI/litellm/pull/12465)
- **Navbar**
    - improve user dropdown UI with premium badge and cleaner layout - [PR](https://github.com/BerriAI/litellm/pull/12502)
- **General**
    - Consistent layout for Create and Back buttons on all the pages - [PR](https://github.com/BerriAI/litellm/pull/12542)
    - Align Show Password with Checkbox - [PR](https://github.com/BerriAI/litellm/pull/12538)
    - Prevent writing default user setting updates to yaml (causes error in non-root env) - [PR](https://github.com/BerriAI/litellm/pull/12533)

#### Bugs
- **Model Hub**
    - fix duplicates in /model_group/info - [PR](https://github.com/BerriAI/litellm/pull/12468)
- **MCP**
    - Fix UI not syncing MCP access groups properly with object permissions - [PR](https://github.com/BerriAI/litellm/pull/12523)

---

## Logging / Guardrail Integrations

#### Features
- **[Langfuse](../../docs/observability/langfuse_integration)**
    - Version bump - [PR](https://github.com/BerriAI/litellm/pull/12376)
    - LANGFUSE_TRACING_ENVIRONMENT support - [PR](https://github.com/BerriAI/litellm/pull/12376), [NEEDS DOCS]
- **[Bedrock Guardrails](../../docs/proxy/guardrails/bedrock)**
    - Raise Bedrock output text on 'BLOCKED' actions from guardrail - [PR](https://github.com/BerriAI/litellm/pull/12435)
- **[OTEL](../../docs/observability/opentelemetry_integration)**
    - `OTEL_RESOURCE_ATTRIBUTES` support - [PR](https://github.com/BerriAI/litellm/pull/12468)
- **[Guardrails AI](../../docs/proxy/guardrails/guardrails_ai)**
    - pre-call + logging only guardrail (pii detection/competitor names) support - [PR](https://github.com/BerriAI/litellm/pull/12506)
- **[Guardrails](../../docs/proxy/guardrails/quick_start)**
    - [Enterprise] Support tag based mode for guardrails - [PR](https://github.com/BerriAI/litellm/pull/12508), [NEEDS DOCS]
- **[OpenAI Moderations API](../../docs/proxy/guardrails/openai_moderation)**
    - New guardrail integration - [PR](https://github.com/BerriAI/litellm/pull/12519)
- **[Prometheus](../../docs/proxy/prometheus)**
    - support tag based metrics (enables prometheus metrics for measuring roo-code/cline/claude code engagement) - [PR](https://github.com/BerriAI/litellm/pull/12534), [NEEDS DOCS]

#### Bugs
- **[Prometheus](../../docs/proxy/prometheus)**
    - Remove experimental `_by_tag` metrics (fixes cardinality issue) - [PR](https://github.com/BerriAI/litellm/pull/12395)
- **[Datadog LLM Observability Logger]** [NEEDS DOCS]
    - push total_cost - [PR](https://github.com/BerriAI/litellm/pull/12467)
- **[Slack Alerting](../../docs/proxy/alerting)**
    - Fix slack alerting for outage and region outage alerts - [PR](https://github.com/BerriAI/litellm/pull/12464), [Get Started](../../docs/proxy/alerting#region-outage-alerting--enterprise-feature)

---

## Performance / Loadbalancing / Reliability improvements

#### Bugs
- **[Responses API Bridge](../../docs/response_api#calling-non-responses-api-endpoints-responses-to-chatcompletions-bridge)**
    - add image support for Responses API when falling back on Chat Completions - [PR](https://github.com/BerriAI/litellm/pull/12204) s/o [@ryan-castner](https://github.com/ryan-castner)
- **aiohttp**
    - Properly close aiohttp client sessions to prevent resource leaks - [PR](https://github.com/BerriAI/litellm/pull/12251)
- **Router**
    - don't add invalid deployment to router pattern match - [PR](https://github.com/BerriAI/litellm/pull/12459)


---

## General Proxy Improvements

- **Startup**
  - Add new banner on startup - [PR](https://github.com/BerriAI/litellm/pull/12328)
- **Dependencies**
  - Update pydantic version - [PR](https://github.com/BerriAI/litellm/pull/12213)


---

## New Contributors
* @wildcard made their first contribution in https://github.com/BerriAI/litellm/pull/12157
* @colesmcintosh made their first contribution in https://github.com/BerriAI/litellm/pull/12168
* @seyeong-han made their first contribution in https://github.com/BerriAI/litellm/pull/11946
* @dinggh made their first contribution in https://github.com/BerriAI/litellm/pull/12162
* @raz-alon made their first contribution in https://github.com/BerriAI/litellm/pull/11432
* @tofarr made their first contribution in https://github.com/BerriAI/litellm/pull/12200
* @szafranek made their first contribution in https://github.com/BerriAI/litellm/pull/12179
* @SamBoyd made their first contribution in https://github.com/BerriAI/litellm/pull/12147
* @lizzij made their first contribution in https://github.com/BerriAI/litellm/pull/12219
* @cipri-tom made their first contribution in https://github.com/BerriAI/litellm/pull/12201
* @zsimjee made their first contribution in https://github.com/BerriAI/litellm/pull/12185
* @jroberts2600 made their first contribution in https://github.com/BerriAI/litellm/pull/12175
* @njbrake made their first contribution in https://github.com/BerriAI/litellm/pull/12202
* @NANDINI-star made their first contribution in https://github.com/BerriAI/litellm/pull/12244
* @utsumi-fj made their first contribution in https://github.com/BerriAI/litellm/pull/12230
* @dcieslak19973 made their first contribution in https://github.com/BerriAI/litellm/pull/12283
* @hanouticelina made their first contribution in https://github.com/BerriAI/litellm/pull/12286
* @lowjiansheng made their first contribution in https://github.com/BerriAI/litellm/pull/11999
* @JoostvDoorn made their first contribution in https://github.com/BerriAI/litellm/pull/12281
* @takashiishida made their first contribution in https://github.com/BerriAI/litellm/pull/12239

## **[Git Diff](https://github.com/BerriAI/litellm/compare/v1.73.6-stable...v1.74.0-stable)**

