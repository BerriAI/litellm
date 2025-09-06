---
title: "v1.74.7-stable"
slug: "v1-74-7"
date: 2025-07-19T10:00:00
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
ghcr.io/berriai/litellm:v1.74.7-stable.patch.1
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.74.7.post2
```

</TabItem>
</Tabs>

---

## Key Highlights 


- **Vector Stores** - Support for Vertex RAG Engine, PG Vector, OpenAI & Azure OpenAI Vector Stores.
- **Bulk Editing Users** - Bulk editing users on the UI.
- **Health Check Improvements** - Prevent unnecessary pod restarts during high traffic.
- **New LLM Providers** - Added Moonshot AI and Vercel v0 provider support.

---

## Vector Stores API

<Image img={require('../../img/release_notes/vector_stores.png')} />


This release introduces support for using VertexAI RAG Engine, PG Vector, Bedrock Knowledge Bases, and OpenAI Vector Stores with LiteLLM.

This is ideal for use cases requiring external knowledge sources with LLMs.

This brings the following benefits for LiteLLM users:

**Proxy Admin Benefits:**
- Fine-grained access control: determine which Keys and Teams can access specific Vector Stores
- Complete usage tracking and monitoring across all vector store operations

**Developer Benefits:**
- Simple, unified interface for querying vector stores and using them with LLM API requests
- Consistent API experience across all supported vector store providers 



[Get started](../../docs/completion/knowledgebase)


---

## Bulk Editing Users

<Image img={require('../../img/bulk_edit_graphic.png')} />

v1.74.7-stable introduces Bulk Editing Users on the UI. This is useful for:
- granting all existing users to a default team (useful for controlling access / tracking spend by team)
- controlling personal model access for existing users

[Read more](https://docs.litellm.ai/docs/proxy/ui/bulk_edit_users)

---

## Health Check Server

<Image alt="Separate Health App Architecture" img={require('../../img/separate_health_app_architecture.png')} style={{ borderRadius: '8px', marginBottom: '1em', maxWidth: '100%' }} />

This release brings reliability improvements that prevent unnecessary pod restarts during high traffic. Previously, when the main LiteLLM app was busy serving traffic, health endpoints would timeout even when pods were healthy. 
 
Starting with this release, you can run health endpoints on an isolated process with a dedicated port. This ensures liveness and readiness probes remain responsive even when the main LiteLLM app is under heavy load.

[Read More](https://docs.litellm.ai/docs/proxy/prod#10-use-a-separate-health-check-app)


---

## New Models / Updated Models

#### Pricing / Context Window Updates

| Provider    | Model                                  | Context Window | Input ($/1M tokens) | Output ($/1M tokens) |
| ----------- | -------------------------------------- | -------------- | ------------------- | -------------------- |
| Azure AI | `azure_ai/grok-3` | 131k | $3.30 | $16.50 |
| Azure AI | `azure_ai/global/grok-3` | 131k | $3.00 | $15.00 |
| Azure AI | `azure_ai/global/grok-3-mini` | 131k | $0.25 | $1.27 |
| Azure AI | `azure_ai/grok-3-mini` | 131k | $0.275 | $1.38 |
| Azure AI | `azure_ai/jais-30b-chat` | 8k | $3200 | $9710 |
| Groq | `groq/moonshotai-kimi-k2-instruct` | 131k | $1.00 | $3.00 |
| AI21 | `jamba-large-1.7` | 256k | $2.00 | $8.00 |
| AI21 | `jamba-mini-1.7` | 256k | $0.20 | $0.40 |
| Together.ai | `together_ai/moonshotai/Kimi-K2-Instruct` | 131k | $1.00 | $3.00 |
| v0 | `v0/v0-1.0-md` | 128k | $3.00 | $15.00 |
| v0 | `v0/v0-1.5-md` | 128k | $3.00 | $15.00 |
| v0 | `v0/v0-1.5-lg` | 512k | $15.00 | $75.00 |
| Moonshot | `moonshot/moonshot-v1-8k` | 8k | $0.20 | $2.00 |
| Moonshot | `moonshot/moonshot-v1-32k` | 32k | $1.00 | $3.00 |
| Moonshot | `moonshot/moonshot-v1-128k` | 131k | $2.00 | $5.00 |
| Moonshot | `moonshot/moonshot-v1-auto` | 131k | $2.00 | $5.00 |
| Moonshot | `moonshot/kimi-k2-0711-preview` | 131k | $0.60 | $2.50 |
| Moonshot | `moonshot/moonshot-v1-32k-0430` | 32k | $1.00 | $3.00 |
| Moonshot | `moonshot/moonshot-v1-128k-0430` | 131k | $2.00 | $5.00 |
| Moonshot | `moonshot/moonshot-v1-8k-0430` | 8k | $0.20 | $2.00 |
| Moonshot | `moonshot/kimi-latest` | 131k | $2.00 | $5.00 |
| Moonshot | `moonshot/kimi-latest-8k` | 8k | $0.20 | $2.00 |
| Moonshot | `moonshot/kimi-latest-32k` | 32k | $1.00 | $3.00 |
| Moonshot | `moonshot/kimi-latest-128k` | 131k | $2.00 | $5.00 |
| Moonshot | `moonshot/kimi-thinking-preview` | 131k | $30.00 | $30.00 |
| Moonshot | `moonshot/moonshot-v1-8k-vision-preview` | 8k | $0.20 | $2.00 |
| Moonshot | `moonshot/moonshot-v1-32k-vision-preview` | 32k | $1.00 | $3.00 |
| Moonshot | `moonshot/moonshot-v1-128k-vision-preview` | 131k | $2.00 | $5.00 |


#### Features

- **[🆕 Moonshot API (Kimi)](../../docs/providers/moonshot)**
    - New LLM API integration for accessing Kimi models - [PR #12592](https://github.com/BerriAI/litellm/pull/12592), [Get Started](../../docs/providers/moonshot)
- **[🆕 v0 Provider](../../docs/providers/v0)**
    - New provider integration for v0.dev - [PR #12751](https://github.com/BerriAI/litellm/pull/12751), [Get Started](../../docs/providers/v0)
- **[OpenAI](../../docs/providers/openai)**
    - Use OpenAI DeepResearch models with `litellm.completion` (`/chat/completions`) - [PR #12627](https://github.com/BerriAI/litellm/pull/12627) **DOC NEEDED**
- **[Azure OpenAI](../../docs/providers/azure_openai)**
    - Use Azure OpenAI DeepResearch models with `litellm.completion` (`/chat/completions`) - [PR #12627](https://github.com/BerriAI/litellm/pull/12627) **DOC NEEDED**
    - Added `response_format` support for openai gpt-4.1 models - [PR #12745](https://github.com/BerriAI/litellm/pull/12745)
- **[Anthropic](../../docs/providers/anthropic)**
    - Tool cache control support - [PR #12668](https://github.com/BerriAI/litellm/pull/12668)
- **[Bedrock](../../docs/providers/bedrock)**
    - Claude 4 /invoke route support - [PR #12599](https://github.com/BerriAI/litellm/pull/12599), [Get Started](../../docs/providers/bedrock)
    - Application inference profile tool choice support - [PR #12599](https://github.com/BerriAI/litellm/pull/12599)
- **[Gemini](../../docs/providers/gemini)**
    - Custom TTL support for context caching - [PR #12541](https://github.com/BerriAI/litellm/pull/12541)
    - Fix implicit caching cost calculation for Gemini 2.x models - [PR #12585](https://github.com/BerriAI/litellm/pull/12585)
- **[VertexAI](../../docs/providers/vertex)**
    - Added Vertex AI RAG Engine support (use with OpenAI compatible `/vector_stores` API) - [PR #12752](https://github.com/BerriAI/litellm/pull/12595), [Get Started](../../docs/completion/knowledgebase)
- **[vLLM](../../docs/providers/vllm)**
    - Added support for using Rerank endpoints with vLLM - [PR #12738](https://github.com/BerriAI/litellm/pull/12738), [Get Started](../../docs/providers/vllm#rerank)
- **[AI21](../../docs/providers/ai21)**
    - Added ai21/jamba-1.7 model family pricing - [PR #12593](https://github.com/BerriAI/litellm/pull/12593), [Get Started](../../docs/providers/ai21)
- **[Together.ai](../../docs/providers/together_ai)**
    - [New Model] add together_ai/moonshotai/Kimi-K2-Instruct - [PR #12645](https://github.com/BerriAI/litellm/pull/12645), [Get Started](../../docs/providers/together_ai)
- **[Groq](../../docs/providers/groq)**
    - Add groq/moonshotai-kimi-k2-instruct model configuration - [PR #12648](https://github.com/BerriAI/litellm/pull/12648), [Get Started](../../docs/providers/groq)
- **[Github Copilot](../../docs/providers/github_copilot)**
    - Change System prompts to assistant prompts for GH Copilot - [PR #12742](https://github.com/BerriAI/litellm/pull/12742), [Get Started](../../docs/providers/github_copilot)


#### Bugs
- **[Anthropic](../../docs/providers/anthropic)**
    - Fix streaming + response_format + tools bug - [PR #12463](https://github.com/BerriAI/litellm/pull/12463)
- **[XAI](../../docs/providers/xai)**
    - grok-4 does not support the `stop` param - [PR #12646](https://github.com/BerriAI/litellm/pull/12646)
- **[AWS](../../docs/providers/bedrock)**
    - Role chaining with web authentication for AWS Bedrock - [PR #12607](https://github.com/BerriAI/litellm/pull/12607)
- **[VertexAI](../../docs/providers/vertex)**
    - Add project_id to cached credentials - [PR #12661](https://github.com/BerriAI/litellm/pull/12661)
- **[Bedrock](../../docs/providers/bedrock)**
    - Fix bedrock nova micro and nova lite context window info in [PR #12619](https://github.com/BerriAI/litellm/pull/12619)

---

## LLM API Endpoints

#### Features
- **[/chat/completions](../../docs/completion/input)** 
    - Include tool calls in output of trim_messages - [PR #11517](https://github.com/BerriAI/litellm/pull/11517)
- **[/v1/vector_stores](../../docs/vector_stores/search)**
    - New OpenAI-compatible vector store endpoints - [PR #12699](https://github.com/BerriAI/litellm/pull/12699), [Get Started](../../docs/vector_stores/search)
    - Vector store search endpoint - [PR #12749](https://github.com/BerriAI/litellm/pull/12749), [Get Started](../../docs/vector_stores/search)
    - Support for using PG Vector as a vector store - [PR #12667](https://github.com/BerriAI/litellm/pull/12667), [Get Started](../../docs/completion/knowledgebase)
- **[/streamGenerateContent](../../docs/generateContent)**
    - Non-gemini model support - [PR #12647](https://github.com/BerriAI/litellm/pull/12647)

#### Bugs
- **[/vector_stores](../../docs/vector_stores/search)**
    - Knowledge Base Call returning error when passing as `tools` - [PR #12628](https://github.com/BerriAI/litellm/pull/12628)

---

## [MCP Gateway](../../docs/mcp)

#### Features
- **[Access Groups](../../docs/mcp#grouping-mcps-access-groups)**
    - Allow MCP access groups to be added via litellm proxy config.yaml - [PR #12654](https://github.com/BerriAI/litellm/pull/12654)
    - List tools from access list for keys - [PR #12657](https://github.com/BerriAI/litellm/pull/12657)
- **[Namespacing](../../docs/mcp#mcp-namespacing)**
    - URL-based namespacing for better segregation - [PR #12658](https://github.com/BerriAI/litellm/pull/12658)
    - Make MCP_TOOL_PREFIX_SEPARATOR configurable from env - [PR #12603](https://github.com/BerriAI/litellm/pull/12603)
- **[Gateway Features](../../docs/mcp#mcp-gateway-features)**
    - Allow using MCPs with all LLM APIs (VertexAI, Gemini, Groq, etc.) when using /responses - [PR #12546](https://github.com/BerriAI/litellm/pull/12546)

#### Bugs
    - Fix to update object permission on update/delete key/team - [PR #12701](https://github.com/BerriAI/litellm/pull/12701)
    - Include /mcp in list of available routes on proxy - [PR #12612](https://github.com/BerriAI/litellm/pull/12612)

---

## Management Endpoints / UI

#### Features
- **Keys**
    - Regenerate Key State Management improvements - [PR #12729](https://github.com/BerriAI/litellm/pull/12729)
- **Models**
    - Wildcard model filter support - [PR #12597](https://github.com/BerriAI/litellm/pull/12597)
    - Fixes for handling team only models on UI - [PR #12632](https://github.com/BerriAI/litellm/pull/12632)
- **Usage Page**
    - Fix Y-axis labels overlap on Spend per Tag chart - [PR #12754](https://github.com/BerriAI/litellm/pull/12754)
- **Teams**
    - Allow setting custom key duration + show key creation stats - [PR #12722](https://github.com/BerriAI/litellm/pull/12722)
    - Enable team admins to update member roles - [PR #12629](https://github.com/BerriAI/litellm/pull/12629)
- **Users**
    - New `/user/bulk_update` endpoint - [PR #12720](https://github.com/BerriAI/litellm/pull/12720)
- **Logs Page**
    - Add `end_user` filter on UI Logs Page - [PR #12663](https://github.com/BerriAI/litellm/pull/12663)
- **MCP Servers**
    - Copy MCP Server name functionality - [PR #12760](https://github.com/BerriAI/litellm/pull/12760)
- **Vector Stores**
    - UI support for clicking into Vector Stores - [PR #12741](https://github.com/BerriAI/litellm/pull/12741)
    - Allow adding Vertex RAG Engine, OpenAI, Azure through UI - [PR #12752](https://github.com/BerriAI/litellm/pull/12752)
- **General**
    - Add Copy-on-Click for all IDs (Key, Team, Organization, MCP Server) - [PR #12615](https://github.com/BerriAI/litellm/pull/12615)
- **[SCIM](../../docs/proxy/scim)**
    - Add GET /ServiceProviderConfig endpoint - [PR #12664](https://github.com/BerriAI/litellm/pull/12664)

#### Bugs
- **Teams**
    - Ensure user id correctly added when creating new teams - [PR #12719](https://github.com/BerriAI/litellm/pull/12719)
    - Fixes for handling team-only models on UI - [PR #12632](https://github.com/BerriAI/litellm/pull/12632)

---

## Logging / Guardrail Integrations

#### Features
- **[Google Cloud Model Armor](../../docs/proxy/guardrails/google_cloud_model_armor)**
    - New guardrails integration - [PR #12492](https://github.com/BerriAI/litellm/pull/12492)
- **[Bedrock Guardrails](../../docs/proxy/guardrails/bedrock)**
    - Allow disabling exception on 'BLOCKED' action - [PR #12693](https://github.com/BerriAI/litellm/pull/12693)
- **[Guardrails AI](../../docs/proxy/guardrails/guardrails_ai)**
    - Support `llmOutput` based guardrails as pre-call hooks - [PR #12674](https://github.com/BerriAI/litellm/pull/12674)
- **[DataDog LLM Observability](../../docs/proxy/logging#datadog)**
    - Add support for tracking the correct span type based on LLM Endpoint used - [PR #12652](https://github.com/BerriAI/litellm/pull/12652)
- **[Custom Logging](../../docs/proxy/logging)**
    - Allow reading custom logger python scripts from S3 or GCS Bucket - [PR #12623](https://github.com/BerriAI/litellm/pull/12623)

#### Bugs
- **[General Logging](../../docs/proxy/logging)**
    - StandardLoggingPayload on cache_hits should track custom llm provider - [PR #12652](https://github.com/BerriAI/litellm/pull/12652)
- **[S3 Buckets](../../docs/proxy/logging#s3-buckets)**
    - S3 v2 log uploader crashes when using with guardrails - [PR #12733](https://github.com/BerriAI/litellm/pull/12733)

---

## Performance / Loadbalancing / Reliability improvements

#### Features
- **Health Checks**
    - Separate health app for liveness probes - [PR #12669](https://github.com/BerriAI/litellm/pull/12669)
    - Health check app on separate port - [PR #12718](https://github.com/BerriAI/litellm/pull/12718)
- **Caching**
    - Add Azure Blob cache support - [PR #12587](https://github.com/BerriAI/litellm/pull/12587)
- **Router**
    - Handle ZeroDivisionError with zero completion tokens in lowest_latency strategy - [PR #12734](https://github.com/BerriAI/litellm/pull/12734)

#### Bugs
- **Database**
    - Use upsert for managed object table to avoid UniqueViolationError - [PR #11795](https://github.com/BerriAI/litellm/pull/11795)
    - Refactor to support use_prisma_migrate for helm hook - [PR #12600](https://github.com/BerriAI/litellm/pull/12600)
- **Cache**
    - Fix: redis caching for embedding response models - [PR #12750](https://github.com/BerriAI/litellm/pull/12750)

---

## Helm Chart

- DB Migration Hook: refactor to support use_prisma_migrate - for helm hook [PR](https://github.com/BerriAI/litellm/pull/12600)
- Add envVars and extraEnvVars support to Helm migrations job - [PR #12591](https://github.com/BerriAI/litellm/pull/12591)

## General Proxy Improvements

#### Features
- **Control Plane + Data Plane Architecture**
    - Control Plane + Data Plane support - [PR #12601](https://github.com/BerriAI/litellm/pull/12601)
- **Proxy CLI**
    - Add "keys import" command to CLI - [PR #12620](https://github.com/BerriAI/litellm/pull/12620)
- **Swagger Documentation**
    - Add swagger docs for LiteLLM /chat/completions, /embeddings, /responses - [PR #12618](https://github.com/BerriAI/litellm/pull/12618)
- **Dependencies**
    - Loosen rich version from ==13.7.1 to >=13.7.1 - [PR #12704](https://github.com/BerriAI/litellm/pull/12704)


#### Bugs

- Verbose log is enabled by default fix - [PR #12596](https://github.com/BerriAI/litellm/pull/12596)

- Add support for disabling callbacks in request body - [PR #12762](https://github.com/BerriAI/litellm/pull/12762)
- Handle circular references in spend tracking metadata JSON serialization - [PR #12643](https://github.com/BerriAI/litellm/pull/12643)

---

## New Contributors
* @AntonioKL made their first contribution in https://github.com/BerriAI/litellm/pull/12591
* @marcelodiaz558 made their first contribution in https://github.com/BerriAI/litellm/pull/12541
* @dmcaulay made their first contribution in https://github.com/BerriAI/litellm/pull/12463
* @demoray made their first contribution in https://github.com/BerriAI/litellm/pull/12587
* @staeiou made their first contribution in https://github.com/BerriAI/litellm/pull/12631
* @stefanc-ai2 made their first contribution in https://github.com/BerriAI/litellm/pull/12622
* @RichardoC made their first contribution in https://github.com/BerriAI/litellm/pull/12607
* @yeahyung made their first contribution in https://github.com/BerriAI/litellm/pull/11795
* @mnguyen96 made their first contribution in https://github.com/BerriAI/litellm/pull/12619
* @rgambee made their first contribution in https://github.com/BerriAI/litellm/pull/11517
* @jvanmelckebeke made their first contribution in https://github.com/BerriAI/litellm/pull/12725
* @jlaurendi made their first contribution in https://github.com/BerriAI/litellm/pull/12704
* @doublerr made their first contribution in https://github.com/BerriAI/litellm/pull/12661

## **[Full Changelog](https://github.com/BerriAI/litellm/compare/v1.74.3-stable...v1.74.7-stable)**
