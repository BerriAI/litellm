---
title: "v1.73.6-stable"
slug: "v1-73-6-stable"
date: 2025-06-28T10:00:00
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
docker.litellm.ai/berriai/litellm:v1.73.6-stable.patch.1
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.73.6.post1
```

</TabItem>
</Tabs>

---

## Key Highlights 


### Claude on gemini-cli


<Image img={require('../../img/release_notes/gemini_cli.png')} />

<br/>

This release brings support for using gemini-cli with LiteLLM. 

You can use claude-sonnet-4, gemini-2.5-flash (Vertex AI & Google AI Studio), gpt-4.1 and any LiteLLM supported model on gemini-cli.

When you use gemini-cli with LiteLLM you get the following benefits:

**Developer Benefits:**
- Universal Model Access: Use any LiteLLM supported model (Anthropic, OpenAI, Vertex AI, Bedrock, etc.) through the gemini-cli interface.
- Higher Rate Limits & Reliability: Load balance across multiple models and providers to avoid hitting individual provider limits, with fallbacks to ensure you get responses even if one provider fails.

**Proxy Admin Benefits:**
- Centralized Management: Control access to all models through a single LiteLLM proxy instance without giving your developers API Keys to each provider.
- Budget Controls: Set spending limits and track costs across all gemini-cli usage.

[Get Started](../../docs/tutorials/litellm_gemini_cli)

<br/>

### Batch API Cost Tracking

<Image img={require('../../img/release_notes/batch_api_cost_tracking.jpg')}/>

<br/>

v1.73.6 brings cost tracking for [LiteLLM Managed Batch API](../../docs/proxy/managed_batches) calls to LiteLLM. Previously, this was not being done for Batch API calls using LiteLLM Managed Files. Now, LiteLLM will store the status of each batch call in the DB and poll incomplete batch jobs in the background, emitting a spend log for cost tracking once the batch is complete.

There is no new flag / change needed on your end. Over the next few weeks we hope to extend this to cover batch cost tracking for the Anthropic passthrough as well. 


[Get Started](../../docs/proxy/managed_batches)

---

## New Models / Updated Models

### Pricing / Context Window Updates

| Provider    | Model                                  | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Type |
| ----------- | -------------------------------------- | -------------- | ------------------- | -------------------- | ---- |
| Azure OpenAI | `azure/o3-pro` | 200k | $20.00 | $80.00 | New |
| OpenRouter | `openrouter/mistralai/mistral-small-3.2-24b-instruct` | 32k | $0.1 | $0.3 | New |
| OpenAI | `o3-deep-research` | 200k | $10.00 | $40.00 | New |
| OpenAI | `o3-deep-research-2025-06-26` | 200k | $10.00 | $40.00 | New |
| OpenAI | `o4-mini-deep-research` | 200k | $2.00 | $8.00 | New |
| OpenAI | `o4-mini-deep-research-2025-06-26` | 200k | $2.00 | $8.00 | New |
| Deepseek | `deepseek-r1` | 65k | $0.55 | $2.19 | New |
| Deepseek | `deepseek-v3` | 65k | $0.27 | $0.07 | New |


### Updated Models
#### Bugs
    - **[Sambanova](../../docs/providers/sambanova)**
        - Handle float timestamps - [PR](https://github.com/BerriAI/litellm/pull/11971) s/o [@neubig](https://github.com/neubig)
    - **[Azure](../../docs/providers/azure)**
        - support Azure Authentication method (azure ad token, api keys) on Responses API - [PR](https://github.com/BerriAI/litellm/pull/11941) s/o [@hsuyuming](https://github.com/hsuyuming)
        - Map ‘image_url’ str as nested dict - [PR](https://github.com/BerriAI/litellm/pull/12075) s/o [@davis-featherstone](https://github.com/davis-featherstone)
    - **[Watsonx](../../docs/providers/watsonx)**
        - Set ‘model’ field to None when model is part of a custom deployment - fixes error raised by WatsonX in those cases - [PR](https://github.com/BerriAI/litellm/pull/11854) s/o [@cbjuan](https://github.com/cbjuan)
    - **[Perplexity](../../docs/providers/perplexity)**
        - Support web_search_options - [PR](https://github.com/BerriAI/litellm/pull/11983)
        - Support citation token and search queries cost calculation - [PR](https://github.com/BerriAI/litellm/pull/11938)
    - **[Anthropic](../../docs/providers/anthropic)**
        - Null value in usage block handling - [PR](https://github.com/BerriAI/litellm/pull/12068)
    - **Gemini ([Google AI Studio](../../docs/providers/gemini) + [VertexAI](../../docs/providers/vertex))**
        - Only use accepted format values (enum and datetime) - else gemini raises errors - [PR](https://github.com/BerriAI/litellm/pull/11989) 
        - Cache tools if passed alongside cached content (else gemini raises an error) - [PR](https://github.com/BerriAI/litellm/pull/11989)
        - Json schema translation improvement: Fix unpack_def handling of nested $ref inside anyof items - [PR](https://github.com/BerriAI/litellm/pull/11964)
    - **[Mistral](../../docs/providers/mistral)**
        - Fix thinking prompt to match hugging face recommendation - [PR](https://github.com/BerriAI/litellm/pull/12007)
        - Add `supports_response_schema: true` for all mistral models except codestral-mamba - [PR](https://github.com/BerriAI/litellm/pull/12024)
    - **[Ollama](../../docs/providers/ollama)**
        - Fix unnecessary await on embedding calls - [PR](https://github.com/BerriAI/litellm/pull/12024)
#### Features
    - **[Azure OpenAI](../../docs/providers/azure)**
        - Check if o-series model supports reasoning effort (enables drop_params to work for o1 models) 
        - Assistant + tool use cost tracking - [PR](https://github.com/BerriAI/litellm/pull/12045)
    - **[Nvidia Nim](../../docs/providers/nvidia_nim)**
        - Add ‘response_format’ param support - [PR](https://github.com/BerriAI/litellm/pull/12003) @shagunb-acn 
    - **[ElevenLabs](../../docs/providers/elevenlabs)**
        - New STT provider - [PR](https://github.com/BerriAI/litellm/pull/12119)

---
## LLM API Endpoints

#### Features
    - [**/mcp**](../../docs/mcp)
        - Send appropriate auth string value to `/tool/call` endpoint with `x-mcp-auth` - [PR](https://github.com/BerriAI/litellm/pull/11968) s/o [@wagnerjt](https://github.com/wagnerjt)
    - [**/v1/messages**](../../docs/anthropic_unified)
        - [Custom LLM](../../docs/providers/custom_llm_server#anthropic-v1messages) support - [PR](https://github.com/BerriAI/litellm/pull/12016)
    - [**/chat/completions**](../../docs/completion/input)
        - Azure Responses API via chat completion support - [PR](https://github.com/BerriAI/litellm/pull/12016)
    - [**/responses**](../../docs/response_api)
        - Add reasoning content support for non-openai providers - [PR](https://github.com/BerriAI/litellm/pull/12055)
    - **[NEW] /generateContent**
        - New endpoints for gemini cli support - [PR](https://github.com/BerriAI/litellm/pull/12040)
        - Support calling Google AI Studio / VertexAI Gemini models in their native format - [PR](https://github.com/BerriAI/litellm/pull/12046)
        - Add logging + cost tracking for stream + non-stream vertex/google ai studio routes - [PR](https://github.com/BerriAI/litellm/pull/12058)
        - Add Bridge from generateContent to /chat/completions - [PR](https://github.com/BerriAI/litellm/pull/12081)
    - [**/batches**](../../docs/batches)
        - Filter deployments to only those where managed file was written to - [PR](https://github.com/BerriAI/litellm/pull/12048)
        - Save all model / file id mappings in db (previously it was just the first one) - enables ‘true’ loadbalancing - [PR](https://github.com/BerriAI/litellm/pull/12048)
        - Support List Batches with target model name specified - [PR](https://github.com/BerriAI/litellm/pull/12049)

---
## Spend Tracking / Budget Improvements

#### Features
    - [**Passthrough**](../../docs/pass_through)
        - [Bedrock](../../docs/pass_through/bedrock) - cost tracking (`/invoke` + `/converse` routes) on streaming + non-streaming - [PR](https://github.com/BerriAI/litellm/pull/12123)
        - [VertexAI](../../docs/pass_through/vertex_ai) - anthropic cost calculation support - [PR](https://github.com/BerriAI/litellm/pull/11992)
    - [**Batches**](../../docs/batches)
        - Background job for cost tracking LiteLLM Managed batches - [PR](https://github.com/BerriAI/litellm/pull/12125)

---
## Management Endpoints / UI

#### Bugs
    - **General UI**
        - Fix today selector date mutation in dashboard components - [PR](https://github.com/BerriAI/litellm/pull/12042)
    - **Usage**
        - Aggregate usage data across all pages of paginated endpoint - [PR](https://github.com/BerriAI/litellm/pull/12033)
    - **Teams**
        - De-duplicate models in team settings dropdown - [PR](https://github.com/BerriAI/litellm/pull/12074)
    - **Models**
        - Preserve public model name when selecting ‘test connect’ with azure model (previously would reset) - [PR](https://github.com/BerriAI/litellm/pull/11713)
    - **Invitation Links**
        - Ensure Invite links email contain the correct invite id when using tf provider - [PR](https://github.com/BerriAI/litellm/pull/12130)
#### Features
    - **Models**
        - Add ‘last success’ column to health check table - [PR](https://github.com/BerriAI/litellm/pull/11903)
    - **MCP**
        - New UI component to support auth types: api key, bearer token, basic auth - [PR](https://github.com/BerriAI/litellm/pull/11968) s/o [@wagnerjt](https://github.com/wagnerjt)
        - Ensure internal users can access /mcp and /mcp/ routes - [PR](https://github.com/BerriAI/litellm/pull/12106)
    - **SCIM**
        - Ensure default_internal_user_params are applied for new users - [PR](https://github.com/BerriAI/litellm/pull/12015)
    - **Team**
        - Support default key expiry for team member keys - [PR](https://github.com/BerriAI/litellm/pull/12023)
        - Expand team member add check to cover user email - [PR](https://github.com/BerriAI/litellm/pull/12082)
    - **UI**
        - Restrict UI access by SSO group - [PR](https://github.com/BerriAI/litellm/pull/12023)
    - **Keys**
        - Add new new_key param for regenerating key - [PR](https://github.com/BerriAI/litellm/pull/12087)
    - **Test Keys**
        - New ‘get code’ button for getting runnable python code snippet based on ui configuration - [PR](https://github.com/BerriAI/litellm/pull/11629)

--- 

## Logging / Guardrail Integrations

#### Bugs
    - **Braintrust**
        - Adds model to metadata to enable braintrust cost estimation - [PR](https://github.com/BerriAI/litellm/pull/12022)
#### Features
    - **Callbacks**
        - (Enterprise) - disable logging callbacks in request headers - [PR](https://github.com/BerriAI/litellm/pull/11985)
        - Add List Callbacks API Endpoint - [PR](https://github.com/BerriAI/litellm/pull/11987)
    - **Bedrock Guardrail**
        - Don't raise exception on intervene action - [PR](https://github.com/BerriAI/litellm/pull/11875)
        - Ensure PII Masking is applied on response streaming or non streaming content when using post call - [PR](https://github.com/BerriAI/litellm/pull/12086)
    - **[NEW] Palo Alto Networks Prisma AIRS Guardrail**
        - [PR](https://github.com/BerriAI/litellm/pull/12116)
    - **ElasticSearch**
        - New Elasticsearch Logging Tutorial - [PR](https://github.com/BerriAI/litellm/pull/11761)
    - **Message Redaction**
        - Preserve usage / model information  for Embedding redaction - [PR](https://github.com/BerriAI/litellm/pull/12088)

---

## Performance / Loadbalancing / Reliability improvements

#### Bugs
    - **Team-only models**
        - Filter team-only models from routing logic for non-team calls
    - **Context Window Exceeded error**
        - Catch anthropic exceptions - [PR](https://github.com/BerriAI/litellm/pull/12113)
#### Features
    - **Router**
        - allow using dynamic cooldown time for a specific deployment - [PR](https://github.com/BerriAI/litellm/pull/12037)
        - handle cooldown_time = 0 for deployments - [PR](https://github.com/BerriAI/litellm/pull/12108)
    - **Redis**
        - Add better debugging to see what variables are set - [PR](https://github.com/BerriAI/litellm/pull/12073)

---

## General Proxy Improvements

#### Bugs
    - **aiohttp**
        - Check HTTP_PROXY vars in networking requests
        - Allow using HTTP_ Proxy settings with trust_env

#### Features
    - **Docs**
        - Add recommended spec - [PR](https://github.com/BerriAI/litellm/pull/11980)
    - **Swagger**
        - Introduce new environment variable NO_REDOC to opt-out Redoc - [PR](https://github.com/BerriAI/litellm/pull/12092)


---

## New Contributors
* @mukesh-dream11 made their first contribution in https://github.com/BerriAI/litellm/pull/11969
* @cbjuan made their first contribution in https://github.com/BerriAI/litellm/pull/11854
* @ryan-castner made their first contribution in https://github.com/BerriAI/litellm/pull/12055
* @davis-featherstone made their first contribution in https://github.com/BerriAI/litellm/pull/12075
* @Gum-Joe made their first contribution in https://github.com/BerriAI/litellm/pull/12068
* @jroberts2600 made their first contribution in https://github.com/BerriAI/litellm/pull/12116
* @ohmeow made their first contribution in https://github.com/BerriAI/litellm/pull/12022
* @amarrella made their first contribution in https://github.com/BerriAI/litellm/pull/11942
* @zhangyoufu made their first contribution in https://github.com/BerriAI/litellm/pull/12092
* @bougou made their first contribution in https://github.com/BerriAI/litellm/pull/12088
* @codeugar made their first contribution in https://github.com/BerriAI/litellm/pull/11972
* @glgh made their first contribution in https://github.com/BerriAI/litellm/pull/12133

## **[Git Diff](https://github.com/BerriAI/litellm/compare/v1.73.0-stable...v1.73.6.rc-draft)**
