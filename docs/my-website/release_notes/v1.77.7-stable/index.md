---
title: "v1.77.7-stable - 2.9x Lower Median Latency"
slug: "v1-77-7"
date: 2025-10-04T10:00:00
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
ghcr.io/berriai/litellm:v1.77.7.rc.1
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.77.7.rc.1
```

</TabItem>
</Tabs>

---

## Key Highlights

- **Dynamic Rate Limiter v3** - Automatically maximizes throughput when capacity is available (< 80% saturation) by allowing lower-priority requests to use unused capacity, then switches to fair priority-based allocation under high load (≥ 80%) to prevent blocking
- **Major Performance Improvements** - 2.9x lower median latency at 1,000 concurrent users.
- **Claude Sonnet 4.5** - Support for Anthropic's new Claude Sonnet 4.5 model family with 200K+ context and tiered pricing
- **MCP Gateway Enhancements** - Fine-grained tool control, server permissions, and forwardable headers
- **AMD Lemonade & Nvidia NIM** - New provider support for AMD Lemonade and Nvidia NIM Rerank
- **GitLab Prompt Management** - GitLab-based prompt management integration

### Performance - 2.9x Lower Median Latency

<Image img={require('../../img/release_notes/perf_77_7.png')}  style={{ width: '800px', height: 'auto' }} />

<br/>

This update removes LiteLLM router inefficiencies, reducing complexity from O(M×N) to O(1). Previously, it built a new array and ran repeated checks like data["model"] in llm_router.get_model_ids(). Now, a direct ID-to-deployment map eliminates redundant allocations and scans.

As a result, performance improved across all latency percentiles:

- **Median latency:** 320 ms → **110 ms** (−65.6%)
- **p95 latency:** 850 ms → **440 ms** (−48.2%)
- **p99 latency:** 1,400 ms → **810 ms** (−42.1%)
- **Average latency:** 864 ms → **310 ms** (−64%)


#### Test Setup

**Locust**

- **Concurrent users:** 1,000
- **Ramp-up:** 500

**System Specs**

- **CPU:** 4 vCPUs
- **Memory:** 8 GB RAM
- **LiteLLM Workers:** 4
- **Instances**: 4

**Configuration (config.yaml)**

View the complete configuration: [gist.github.com/AlexsanderHamir/config.yaml](https://gist.github.com/AlexsanderHamir/53f7d554a5d2afcf2c4edb5b6be68ff4)

**Load Script (no_cache_hits.py)**

View the complete load testing script: [gist.github.com/AlexsanderHamir/no_cache_hits.py](https://gist.github.com/AlexsanderHamir/42c33d7a4dc7a57f56a78b560dee3a42)

### MCP OAuth 2.0 Support

<Image img={require('../../img/mcp_updates.jpg')} style={{ width: '800px', height: 'auto' }} />

<br/>

This release adds support for OAuth 2.0 Client Credentials for MCP servers. This is great for **Internal Dev Tools** use-cases, as it enables your users to call MCP servers, with their own credentials. E.g. Allowing your developers to call the Github MCP, with their own credentials.

[Set it up today on Claude Code](../../docs/tutorials/claude_responses_api#connecting-mcp-servers)

### Scheduled Key Rotations

<Image img={require('../../img/release_notes/schedule_key_rotations.png')}  style={{ width: '800px', height: 'auto' }} />

<br/>

This release brings support for scheduling virtual key rotations on LiteLLM AI Gateway. 
 
From this release you can enforce Virtual Keys to rotate on a schedule of your choice e.g every 15 days/30 days/60 days etc.
 
This is great for Proxy Admins who need to enforce security policies for production workloads. 

[Get Started](../../docs/proxy/virtual_keys#scheduled-key-rotations)


## New Models / Updated Models

#### New Model Support

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| Anthropic | `claude-sonnet-4-5` | 200K | $3.00 | $15.00 | Chat, reasoning, vision, function calling, prompt caching |
| Anthropic | `claude-sonnet-4-5-20250929` | 200K | $3.00 | $15.00 | Chat, reasoning, vision, function calling, prompt caching |
| Bedrock | `eu.anthropic.claude-sonnet-4-5-20250929-v1:0` | 200K | $3.00 | $15.00 | Chat, reasoning, vision, function calling, prompt caching |
| Azure AI | `azure_ai/grok-4` | 131K | $5.50 | $27.50 | Chat, reasoning, function calling, web search |
| Azure AI | `azure_ai/grok-4-fast-reasoning` | 131K | $0.43 | $1.73 | Chat, reasoning, function calling, web search |
| Azure AI | `azure_ai/grok-4-fast-non-reasoning` | 131K | $0.43 | $1.73 | Chat, function calling, web search |
| Azure AI | `azure_ai/grok-code-fast-1` | 131K | $3.50 | $17.50 | Chat, function calling, web search |
| Groq | `groq/moonshotai/kimi-k2-instruct-0905` | Context varies | Pricing varies | Pricing varies | Chat, function calling |
| Ollama | Ollama Cloud models | Varies | Free | Free | Self-hosted models via Ollama Cloud |

#### Features

- **[Anthropic](../../docs/providers/anthropic)**
    - Add new claude-sonnet-4-5 model family with tiered pricing above 200K tokens - [PR #15041](https://github.com/BerriAI/litellm/pull/15041)
    - Add anthropic/claude-sonnet-4-5 to model price json with prompt caching support - [PR #15049](https://github.com/BerriAI/litellm/pull/15049)
    - Add 200K prices for Sonnet 4.5 - [PR #15140](https://github.com/BerriAI/litellm/pull/15140)
    - Add cost tracking for /v1/messages in streaming response - [PR #15102](https://github.com/BerriAI/litellm/pull/15102)
    - Add /v1/messages/count_tokens to Anthropic routes for non-admin user access - [PR #15034](https://github.com/BerriAI/litellm/pull/15034)
- **[Gemini](../../docs/providers/gemini)**
    - Ignore type param for gemini tools - [PR #15022](https://github.com/BerriAI/litellm/pull/15022)
- **[Vertex AI](../../docs/providers/vertex)**
    - Add LiteLLM Overhead metric for VertexAI - [PR #15040](https://github.com/BerriAI/litellm/pull/15040)
    - Support googlemap grounding in vertex ai - [PR #15179](https://github.com/BerriAI/litellm/pull/15179)
- **[Azure](../../docs/providers/azure)**
    - Add azure_ai grok-4 model family - [PR #15137](https://github.com/BerriAI/litellm/pull/15137)
    - Use the `extra_query` parameter for GET requests in Azure Batch - [PR #14997](https://github.com/BerriAI/litellm/pull/14997)
    - Use extra_query for download results (Batch API) - [PR #15025](https://github.com/BerriAI/litellm/pull/15025)
    - Add support for Azure AD token-based authorization - [PR #14813](https://github.com/BerriAI/litellm/pull/14813)
- **[Ollama](../../docs/providers/ollama)**
    - Add ollama cloud models - [PR #15008](https://github.com/BerriAI/litellm/pull/15008)
- **[Groq](../../docs/providers/groq)**
    - Add groq/moonshotai/kimi-k2-instruct-0905 - [PR #15079](https://github.com/BerriAI/litellm/pull/15079)
- **[OpenAI](../../docs/providers/openai)**
    - Add support for GPT 5 codex models - [PR #14841](https://github.com/BerriAI/litellm/pull/14841)
- **[DeepInfra](../../docs/providers/deepinfra)**
    - Update DeepInfra model data refresh with latest pricing - [PR #14939](https://github.com/BerriAI/litellm/pull/14939)
- **[Bedrock](../../docs/providers/bedrock)**
    - Add JP Cross-Region Inference - [PR #15188](https://github.com/BerriAI/litellm/pull/15188)
    - Add "eu.anthropic.claude-sonnet-4-5-20250929-v1:0" - [PR #15181](https://github.com/BerriAI/litellm/pull/15181)
    - Add twelvelabs bedrock Async Invoke Support - [PR #14871](https://github.com/BerriAI/litellm/pull/14871)
- **[Nvidia NIM](../../docs/providers/nvidia_nim)**
    - Add Nvidia NIM Rerank Support - [PR #15152](https://github.com/BerriAI/litellm/pull/15152)

### Bug Fixes

- **[VLLM](../../docs/providers/vllm)**
    - Fix response_format bug in hosted vllm audio_transcription - [PR #15010](https://github.com/BerriAI/litellm/pull/15010)
    - Fix passthrough of atranscription into kwargs going to upstream provider - [PR #15005](https://github.com/BerriAI/litellm/pull/15005)
- **[OCI](../../docs/providers/oci)**
    - Fix OCI Generative AI Integration when using Proxy - [PR #15072](https://github.com/BerriAI/litellm/pull/15072)
- **General**
    - Fix: Authorization header to use correct "Bearer" capitalization - [PR #14764](https://github.com/BerriAI/litellm/pull/14764)
    - Bug fix: gpt-5-chat-latest has incorrect max_input_tokens value - [PR #15116](https://github.com/BerriAI/litellm/pull/15116)
    - Update request handling for original exceptions - [PR #15013](https://github.com/BerriAI/litellm/pull/15013)

#### New Provider Support

- **[AMD Lemonade](../../docs/providers/lemonade)**
    - Add AMD Lemonade provider support - [PR #14840](https://github.com/BerriAI/litellm/pull/14840)

---

## LLM API Endpoints

#### Features

- **[Responses API](../../docs/response_api)**
    - Return Cost for Responses API Streaming requests - [PR #15053](https://github.com/BerriAI/litellm/pull/15053)

- **[/generateContent](../../docs/providers/gemini)**
    - Add full support for native Gemini API translation - [PR #15029](https://github.com/BerriAI/litellm/pull/15029)

- **Passthrough Gemini Routes**
    - Add Gemini generateContent passthrough cost tracking - [PR #15014](https://github.com/BerriAI/litellm/pull/15014)
    - Add streamGenerateContent cost tracking in passthrough - [PR #15199](https://github.com/BerriAI/litellm/pull/15199)

- **Passthrough Vertex AI Routes**
    - Add cost tracking for Vertex AI Passthrough `/predict` endpoint - [PR #15019](https://github.com/BerriAI/litellm/pull/15019)
    - Add cost tracking for Vertex AI Live API WebSocket Passthrough - [PR #14956](https://github.com/BerriAI/litellm/pull/14956)

- **General**
    - Preserve Whitespace Characters in Model Response Streams - [PR #15160](https://github.com/BerriAI/litellm/pull/15160)
    - Add provider name to payload specification - [PR #15130](https://github.com/BerriAI/litellm/pull/15130)
    - Ensure query params are forwarded from origin url to downstream request - [PR #15087](https://github.com/BerriAI/litellm/pull/15087)

---

## Management Endpoints / UI

#### Features

- **Virtual Keys**
    - Ensure LLM_API_KEYs can access pass through routes - [PR #15115](https://github.com/BerriAI/litellm/pull/15115)
    - Support 'guaranteed_throughput' when setting limits on keys belonging to a team - [PR #15120](https://github.com/BerriAI/litellm/pull/15120)
    
- **Models + Endpoints**
    - Ensure OCI secret fields not shared on /models and /v1/models endpoints - [PR #15085](https://github.com/BerriAI/litellm/pull/15085)
    - Add snowflake on UI - [PR #15083](https://github.com/BerriAI/litellm/pull/15083)
    - Make UI theme settings publicly accessible for custom branding - [PR #15074](https://github.com/BerriAI/litellm/pull/15074)
    
- **Admin Settings**
    - Ensure OTEL settings are saved in DB after set on UI - [PR #15118](https://github.com/BerriAI/litellm/pull/15118)
    - Top api key tags - [PR #15151](https://github.com/BerriAI/litellm/pull/15151), [PR #15156](https://github.com/BerriAI/litellm/pull/15156)

- **MCP**
    - show health status of MCP servers - [PR #15185](https://github.com/BerriAI/litellm/pull/15185)
    - allow setting extra headers on the UI - [PR #15185](https://github.com/BerriAI/litellm/pull/15185)
    - allow editing allowed tools on the UI - [PR #15185](https://github.com/BerriAI/litellm/pull/15185)

### Bug Fixes

- **Virtual Keys**
    - (security) prevent user key from updating other user keys - [PR #15201](https://github.com/BerriAI/litellm/pull/15201)
    - (security) don't return all keys with blank key alias on /v2/key/info - [PR #15201](https://github.com/BerriAI/litellm/pull/15201)
    - Fix Session Token Cookie Infinite Logout Loop - [PR #15146](https://github.com/BerriAI/litellm/pull/15146)

- **Models + Endpoints**
    - Make UI theme settings publicly accessible for custom branding - [PR #15074](https://github.com/BerriAI/litellm/pull/15074)

- **Teams**
    - fix failed copy to clipboard for http ui - [PR #15195](https://github.com/BerriAI/litellm/pull/15195)

- **Logs**
    - fix logs page render logs on filter lookup - [PR #15195](https://github.com/BerriAI/litellm/pull/15195)
    - fix lookup list of end users (migrate to more efficient /customers/list lookup) - [PR #15195](https://github.com/BerriAI/litellm/pull/15195)

- **Test key**
    - update selected model on key change - [PR #15197](https://github.com/BerriAI/litellm/pull/15197)

- **Dashboard**
    - Fix LiteLLM model name fallback in dashboard overview - [PR #14998](https://github.com/BerriAI/litellm/pull/14998)


---

## Logging / Guardrail / Prompt Management Integrations

#### Features

- **[OpenTelemetry](../../docs/observability/otel)**
    - Use generation_name for span naming in logging method - [PR #14799](https://github.com/BerriAI/litellm/pull/14799)
- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Handle non-serializable objects in Langfuse logging - [PR #15148](https://github.com/BerriAI/litellm/pull/15148)
    - Set usage_details.total in langfuse integration - [PR #15015](https://github.com/BerriAI/litellm/pull/15015)
- **[Prometheus](../../docs/proxy/prometheus)**
    - support custom metadata labels on key/team - [PR #15094](https://github.com/BerriAI/litellm/pull/15094)


#### Guardrails

- **[Javelin](../../docs/proxy/guardrails)**
    - Add Javelin standalone guardrails integration for LiteLLM Proxy - [PR #14983](https://github.com/BerriAI/litellm/pull/14983)
    - Add logging for important status fields in guardrails - [PR #15090](https://github.com/BerriAI/litellm/pull/15090)
    - Don't run post_call guardrail if no text returned from Bedrock - [PR #15106](https://github.com/BerriAI/litellm/pull/15106)

#### Prompt Management

- **[GitLab](../../docs/proxy/prompt_management)**
    - GitLab based Prompt manager - [PR #14988](https://github.com/BerriAI/litellm/pull/14988)

---

## Spend Tracking, Budgets and Rate Limiting

- **Cost Tracking** 
    - Proxy: end user cost tracking in the responses API - [PR #15124](https://github.com/BerriAI/litellm/pull/15124)
- **Parallel Request Limiter v3** 
    - Use well known redis cluster hashing algorithm - [PR #15052](https://github.com/BerriAI/litellm/pull/15052)
    - Fixes to dynamic rate limiter v3 - add saturation detection - [PR #15119](https://github.com/BerriAI/litellm/pull/15119)
    - Dynamic Rate Limiter v3 - fixes for detecting saturation + fixes for post saturation behavior - [PR #15192](https://github.com/BerriAI/litellm/pull/15192)
- **Teams** 
    - Add model specific tpm/rpm limits to teams on LiteLLM - [PR #15044](https://github.com/BerriAI/litellm/pull/15044)

---

## MCP Gateway

- **Server Configuration** 
    - Specify forwardable headers, specify allowed/disallowed tools for MCP servers - [PR #15002](https://github.com/BerriAI/litellm/pull/15002)
    - Enforce server permissions on call tools - [PR #15044](https://github.com/BerriAI/litellm/pull/15044)
    - MCP Gateway Fine-grained Tools Addition - [PR #15153](https://github.com/BerriAI/litellm/pull/15153)
- **Bug Fixes** 
    - Remove servername prefix mcp tools tests - [PR #14986](https://github.com/BerriAI/litellm/pull/14986)
    - Resolve regression with duplicate Mcp-Protocol-Version header - [PR #15050](https://github.com/BerriAI/litellm/pull/15050)
    - Fix test_mcp_server.py - [PR #15183](https://github.com/BerriAI/litellm/pull/15183)

---

## Performance / Loadbalancing / Reliability improvements

- **Router Optimizations**
    - **+62.5% P99 Latency Improvement** - Remove router inefficiencies (from O(M*N) to O(1)) - [PR #15046](https://github.com/BerriAI/litellm/pull/15046)
    - Remove hasattr checks in Router - [PR #15082](https://github.com/BerriAI/litellm/pull/15082)
    - Remove Double Lookups - [PR #15084](https://github.com/BerriAI/litellm/pull/15084)
    - Optimize _filter_cooldown_deployments from O(n×m + k×n) to O(n) - [PR #15091](https://github.com/BerriAI/litellm/pull/15091)
    - Optimize unhealthy deployment filtering in retry path (O(n*m) → O(n+m)) - [PR #15110](https://github.com/BerriAI/litellm/pull/15110)
- **Cache Optimizations**
    - Reduce complexity of InMemoryCache.evict_cache from O(n*log(n)) to O(log(n)) - [PR #15000](https://github.com/BerriAI/litellm/pull/15000)
    - Avoiding expensive operations when cache isn't available - [PR #15182](https://github.com/BerriAI/litellm/pull/15182)
- **Worker Management**
    - Add proxy CLI option to recycle workers after N requests - [PR #15007](https://github.com/BerriAI/litellm/pull/15007)
- **Metrics & Monitoring**
    - LiteLLM Overhead metric tracking - Add support for tracking litellm overhead on cache hits - [PR #15045](https://github.com/BerriAI/litellm/pull/15045)

---

## Documentation Updates

- **Provider Documentation** 
    - Update litellm docs from latest release - [PR #15004](https://github.com/BerriAI/litellm/pull/15004)
    - Add missing api_key parameter - [PR #15058](https://github.com/BerriAI/litellm/pull/15058)
- **General Documentation** 
    - Use docker compose instead of docker-compose - [PR #15024](https://github.com/BerriAI/litellm/pull/15024)
    - Add railtracks to projects that are using litellm - [PR #15144](https://github.com/BerriAI/litellm/pull/15144)
    - Perf: Last week improvement - [PR #15193](https://github.com/BerriAI/litellm/pull/15193)
    - Sync models GitHub documentation with Loom video and cross-reference - [PR #15191](https://github.com/BerriAI/litellm/pull/15191)

---

## Security Fixes

- **JWT Token Security** - Don't log JWT SSO token on .info() log - [PR #15145](https://github.com/BerriAI/litellm/pull/15145)

---

## New Contributors

* @herve-ves made their first contribution in [PR #14998](https://github.com/BerriAI/litellm/pull/14998)
* @wenxi-onyx made their first contribution in [PR #15008](https://github.com/BerriAI/litellm/pull/15008)
* @jpetrucciani made their first contribution in [PR #15005](https://github.com/BerriAI/litellm/pull/15005)
* @abhijitjavelin made their first contribution in [PR #14983](https://github.com/BerriAI/litellm/pull/14983)
* @ZeroClover made their first contribution in [PR #15039](https://github.com/BerriAI/litellm/pull/15039)
* @cedarm made their first contribution in [PR #15043](https://github.com/BerriAI/litellm/pull/15043)
* @Isydmr made their first contribution in [PR #15025](https://github.com/BerriAI/litellm/pull/15025)
* @serializer made their first contribution in [PR #15013](https://github.com/BerriAI/litellm/pull/15013)
* @eddierichter-amd made their first contribution in [PR #14840](https://github.com/BerriAI/litellm/pull/14840)
* @malags made their first contribution in [PR #15000](https://github.com/BerriAI/litellm/pull/15000)
* @henryhwang made their first contribution in [PR #15029](https://github.com/BerriAI/litellm/pull/15029)
* @plafleur made their first contribution in [PR #15111](https://github.com/BerriAI/litellm/pull/15111)
* @tyler-liner made their first contribution in [PR #14799](https://github.com/BerriAI/litellm/pull/14799)
* @Amir-R25 made their first contribution in [PR #15144](https://github.com/BerriAI/litellm/pull/15144)
* @georg-wolflein made their first contribution in [PR #15124](https://github.com/BerriAI/litellm/pull/15124)
* @niharm made their first contribution in [PR #15140](https://github.com/BerriAI/litellm/pull/15140)
* @anthony-liner made their first contribution in [PR #15015](https://github.com/BerriAI/litellm/pull/15015)
* @rishiganesh2002 made their first contribution in [PR #15153](https://github.com/BerriAI/litellm/pull/15153)
* @danielaskdd made their first contribution in [PR #15160](https://github.com/BerriAI/litellm/pull/15160)
* @JVenberg made their first contribution in [PR #15146](https://github.com/BerriAI/litellm/pull/15146)
* @speglich made their first contribution in [PR #15072](https://github.com/BerriAI/litellm/pull/15072)
* @daily-kim made their first contribution in [PR #14764](https://github.com/BerriAI/litellm/pull/14764)

---

## **[Full Changelog](https://github.com/BerriAI/litellm/compare/v1.77.5.rc.4...v1.77.7.rc.1)**
