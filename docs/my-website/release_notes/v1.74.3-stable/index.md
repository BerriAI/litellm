---
title: "v1.74.3-stable"
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
docker.litellm.ai/berriai/litellm:v1.74.3-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.74.3.post1
```

</TabItem>
</Tabs>

---

## Key Highlights 

- **MCP: Model Access Groups** - Add mcp servers to access groups, for easily managing access to users and teams.
- **MCP: Tool Cost Tracking** - Set prices for each MCP tool. 
- **Model Hub v2** - New OSS Model Hub for telling developers what models are available on the proxy.
- **Bytez** - New LLM API Provider.
- **Dashscope API** - Call Alibaba's qwen models via new Dashscope API Provider.

---

## MCP Gateway: Model Access Groups

<Image 
  img={require('../../img/release_notes/mcp_access_groups.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

<br/>

v1.74.3-stable adds support for adding MCP servers to access groups, this makes it **easier for Proxy Admins** to manage access to MCP servers across users and teams.

For **developers**, this means you can now connect to multiple MCP servers by passing the access group name in the `x-mcp-servers` header.

Read more [here](https://docs.litellm.ai/docs/mcp#grouping-mcps-access-groups)

---

## MCP Gateway: Tool Cost Tracking

<Image 
  img={require('../../img/release_notes/mcp_tool_cost_tracking.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

<br/>

This release adds cost tracking for MCP tool calls. This is great for **Proxy Admins** giving MCP access to developers as you can now attribute MCP tool call costs to specific LiteLLM keys and teams.

You can set:
- **Uniform server cost**: Set a uniform cost for all tools from a server
- **Individual tool cost**: Define individual costs for specific tools (e.g., search_tool costs $10, get_weather costs $5).
- **Dynamic costs**: For use cases where you want to set costs based on the MCP's response, you can write a custom post mcp call hook to parse responses and set costs dynamically.

[Get started](https://docs.litellm.ai/docs/mcp#mcp-cost-tracking)

---

## Model Hub v2

<Image 
  img={require('../../img/release_notes/model_hub_v2.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

<br/>

v1.74.3-stable introduces a new OSS Model Hub for telling developers what models are available on the proxy.

This is great for **Proxy Admins** as you can now tell developers what models are available on the proxy.

This improves on the previous model hub by enabling:
- The ability to show **Developers** models, even if they don't have a LiteLLM key.
- The ability for **Proxy Admins** to select specific models to be public on the model hub.
- Improved search and filtering capabilities:
    - search for models by partial name (e.g. `xai grok-4`)
    - filter by provider and feature (e.g. 'vision' models)
    - sort by cost (e.g. cheapest vision model from OpenAI)

[Get started](../../docs/proxy/model_hub)

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
- **[Xinference](../../docs/providers/xinference)**
    - Image generation API support - [PR](https://github.com/BerriAI/litellm/pull/12439)
- **[Bedrock](../../docs/providers/bedrock)**
    - API Key Auth support for AWS Bedrock API - [PR](https://github.com/BerriAI/litellm/pull/12495)
- **[🆕 Dashscope](../../docs/providers/dashscope)**
    - New integration from Alibaba (enables qwen usage) - [PR](https://github.com/BerriAI/litellm/pull/12361)
- **[🆕 Bytez](../../docs/providers/bytez)**
    - New /chat/completion integration - [PR](https://github.com/BerriAI/litellm/pull/12121)

#### Bugs
- **[Github Copilot](../../docs/providers/github_copilot)**
    - Fix API base url for Github Copilot - [PR](https://github.com/BerriAI/litellm/pull/12418)
- **[Bedrock](../../docs/providers/bedrock)**
    - Ensure supported bedrock/converse/ params = bedrock/ params - [PR](https://github.com/BerriAI/litellm/pull/12466)
    - Fix cache token cost calculation - [PR](https://github.com/BerriAI/litellm/pull/12488)
- **[XAI](../../docs/providers/xai)**
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

<Image 
  img={require('../../img/release_notes/mcp_tool_cost_tracking.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

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


<Image 
  img={require('../../img/release_notes/model_hub_v2.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

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
    - LANGFUSE_TRACING_ENVIRONMENT support - [PR](https://github.com/BerriAI/litellm/pull/12376)
- **[Bedrock Guardrails](../../docs/proxy/guardrails/bedrock)**
    - Raise Bedrock output text on 'BLOCKED' actions from guardrail - [PR](https://github.com/BerriAI/litellm/pull/12435)
- **[OTEL](../../docs/observability/opentelemetry_integration)**
    - `OTEL_RESOURCE_ATTRIBUTES` support - [PR](https://github.com/BerriAI/litellm/pull/12468)
- **[Guardrails AI](../../docs/proxy/guardrails/guardrails_ai)**
    - pre-call + logging only guardrail (pii detection/competitor names) support - [PR](https://github.com/BerriAI/litellm/pull/12506)
- **[Guardrails](../../docs/proxy/guardrails/quick_start)**
    - [Enterprise] Support tag based mode for guardrails - [PR](https://github.com/BerriAI/litellm/pull/12508), [Get Started](../../docs/proxy/guardrails/quick_start#-tag-based-guardrail-modes)
- **[OpenAI Moderations API](../../docs/proxy/guardrails/openai_moderation)**
    - New guardrail integration - [PR](https://github.com/BerriAI/litellm/pull/12519)
- **[Prometheus](../../docs/proxy/prometheus)**
    - support tag based metrics (enables prometheus metrics for measuring roo-code/cline/claude code engagement) - [PR](https://github.com/BerriAI/litellm/pull/12534), [Get Started](../../docs/proxy/prometheus#custom-tags)
- **[Datadog LLM Observability](../../docs/observability/datadog)**
    - Added `total_cost` field to track costs in DataDog LLM observability metrics - [PR](https://github.com/BerriAI/litellm/pull/12467)

#### Bugs
- **[Prometheus](../../docs/proxy/prometheus)**
    - Remove experimental `_by_tag` metrics (fixes cardinality issue) - [PR](https://github.com/BerriAI/litellm/pull/12395)
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

#### Bugs
- **S3**
  - s3 config.yaml file - ensure yaml safe load is used - [PR](https://github.com/BerriAI/litellm/pull/12373)
- **Audit Logs**
  - Add audit logs for model updates - [PR](https://github.com/BerriAI/litellm/pull/12396)
- **Startup**
  - Multiple API Keys Created on Startup when max_budget is enabled - [PR](https://github.com/BerriAI/litellm/pull/12436)
- **Auth**
  - Resolve model group alias on Auth (if user has access to underlying model, allow alias request to work) - [PR](https://github.com/BerriAI/litellm/pull/12440)
- **config.yaml**
  - fix parsing environment_variables from config.yaml - [PR](https://github.com/BerriAI/litellm/pull/12482)
- **Security**
  - Log hashed jwt w/ prefix instead of actual value - [PR](https://github.com/BerriAI/litellm/pull/12524)

#### Features
- **MCP**
    - Bump mcp version on docker img - [PR](https://github.com/BerriAI/litellm/pull/12362)
- **Request Headers**
    - Forward ‘anthropic-beta’ header when forward_client_headers_to_llm_api is true - [PR](https://github.com/BerriAI/litellm/pull/12462)

---

## New Contributors
* @kanaka made their first contribution in https://github.com/BerriAI/litellm/pull/12418
* @juancarlosm made their first contribution in https://github.com/BerriAI/litellm/pull/12411
* @DmitriyAlergant made their first contribution in https://github.com/BerriAI/litellm/pull/12356
* @Rayshard made their first contribution in https://github.com/BerriAI/litellm/pull/12487
* @minghao51 made their first contribution in https://github.com/BerriAI/litellm/pull/12361
* @jdietzsch91 made their first contribution in https://github.com/BerriAI/litellm/pull/12488
* @iwinux made their first contribution in https://github.com/BerriAI/litellm/pull/12473
* @andresC98 made their first contribution in https://github.com/BerriAI/litellm/pull/12413
* @EmaSuriano made their first contribution in https://github.com/BerriAI/litellm/pull/12509
* @strawgate made their first contribution in https://github.com/BerriAI/litellm/pull/12528
* @inf3rnus made their first contribution in https://github.com/BerriAI/litellm/pull/12121

## **[Git Diff](https://github.com/BerriAI/litellm/compare/v1.74.0-stable...v1.74.3-stable)**

