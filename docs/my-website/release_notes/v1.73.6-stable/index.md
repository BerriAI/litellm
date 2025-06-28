---
title: "[PRE-RELEASE] v1.73.6-stable"
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


:::warning

## Known Issues

The `non-root` docker image has a known issue around the UI not loading. If you use the `non-root` docker image we recommend waiting before upgrading to this version. We will post a patch fix for this.

:::

## Deploy this version

This release is not out yet. The pre-release will be live on Sunday and the stable release will be live on Wednesday.


---


## New Models / Updated Models

### Updated Models
#### Bugs
    - **Sambanova**
        - Handle float timestamps - [PR](https://github.com/BerriAI/litellm/pull/11971) s/o @neubig
    - **Azure**
        - support Azure Authentication method (azure ad token, api keys) - [PR](https://github.com/BerriAI/litellm/pull/11941) @hsuyuming
        - Map ‘image_url’ str as nested dict - [PR](https://github.com/BerriAI/litellm/pull/12075) s/o davis-featherstone
    - **Watsonx**
        - Set ‘model’ field to None when model is part of a custom deployment - fixes error raised by WatsonX in those cases - [PR](https://github.com/BerriAI/litellm/pull/11854) s/o @cbjuan
    - **Perplexity**
        - Support web_search_options - [PR](https://github.com/BerriAI/litellm/pull/11983)
        - Support citation token and search queries cost calculation - [PR](https://github.com/BerriAI/litellm/pull/11938)
    - **Anthropic**
        - Null value in usage block handling - [PR](https://github.com/BerriAI/litellm/pull/12068)
#### Features
    - **Azure OpenAI**
        - Check if o-series model supports reasoning effort (enables drop_params to work for o1 models) 
        - Add o3-pro model pricing 
        - Assistant + tool use cost tracking - [PR](https://github.com/BerriAI/litellm/pull/12045)
    - **OpenRouter**
        - Add Mistral 3.2 24B to model mapping
    - **Gemini (Google AI Studio + VertexAI)**
        - Only use accepted format values (enum and datetime) - else gemini raises errors - [PR](https://github.com/BerriAI/litellm/pull/11989) 
        - Cache tools if passed alongside cached content (else gemini raises an error) - [PR](https://github.com/BerriAI/litellm/pull/11989)
        - Json schema translation improvement: Fix unpack_def handling of nested $ref inside anyof items - [PR](https://github.com/BerriAI/litellm/pull/11964)
    - **NVIDIA Nim**
        - Add ‘response_format’ param support - [PR](https://github.com/BerriAI/litellm/pull/12003) @shagunb-acn 
    - **Mistral**
        - Fix thinking prompt to match hugging face recommendation - [PR](https://github.com/BerriAI/litellm/pull/12007)
        - Add `supports_response_schema: true` for all mistral models except codestral-mamba - [PR](https://github.com/BerriAI/litellm/pull/12024)
    - **Ollama**
        - Fix unnecessary await on embedding calls - [PR](https://github.com/BerriAI/litellm/pull/12024)
    - **OpenAI**
        - New o3 and o4-mini deep research models - [PR](https://github.com/BerriAI/litellm/pull/12109)
    - **ElevenLabs**
        - New STT provider - [PR](https://github.com/BerriAI/litellm/pull/12119)
    - **Deepseek**
        - Add deepseek-r1 + deepseek-v3 cost tracking - [PR](https://github.com/BerriAI/litellm/pull/11972)

---
## LLM API Endpoints

### Features
    - **MCP**
        - Send appropriate auth string value to `/tool/call` endpoint with `x-mcp-auth` - [PR](https://github.com/BerriAI/litellm/pull/11968) s/o @wagnerjt
    - **/v1/messages**
        - Custom LLM support - [PR](https://github.com/BerriAI/litellm/pull/12016)
    - **/chat/completions**
        - Azure Responses API via chat completion support - [PR](https://github.com/BerriAI/litellm/pull/12016)
    - **/responses**
        - Add reasoning content support for non-openai providers - [PR](https://github.com/BerriAI/litellm/pull/12055)
    - **[NEW] /generateContent**
        1. New endpoints for gemini cli support https://github.com/BerriAI/litellm/pull/12040
        2. Support calling Google AI Studio / VertexAI Gemini models in their native format - https://github.com/BerriAI/litellm/pull/12046
        3. Add logging + cost tracking for stream + non-stream vertex/google ai studio routes - https://github.com/BerriAI/litellm/pull/12058
        4. Add Bridge from generateContent to /chat/completions - https://github.com/BerriAI/litellm/pull/12081
    - **/batches**
        - Filter deployments to only those where managed file was written to - [PR](https://github.com/BerriAI/litellm/pull/12048)
        - Save all model / file id mappings in db (previously it was just the first one) - enables ‘true’ loadbalancing - [PR](https://github.com/BerriAI/litellm/pull/12048)
        - Support List Batches with target model name specified - [PR](https://github.com/BerriAI/litellm/pull/12049)

---
## Spend Tracking / Budget Improvements

### Features
    - **Passthrough**
        - Bedrock cost tracking (`/invoke` + `/converse` routes) on streaming + non-streaming - [PR](https://github.com/BerriAI/litellm/pull/12123)
        - VertexAI - anthropic cost calculation support - [PR](https://github.com/BerriAI/litellm/pull/11992)
    - **Batches**
        - Background job for cost tracking LiteLLM Managed batches - [PR](https://github.com/BerriAI/litellm/pull/12125)

---
## Management Endpoints / UI

### Bugs
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
### Features
    - **Models**
        - Add ‘last success’ column to health check table - [PR](https://github.com/BerriAI/litellm/pull/11903)
    - **MCP**
        - New UI component to support auth types: api key, bearer token, basic auth - [PR](https://github.com/BerriAI/litellm/pull/11968) s/o @wagnerjt
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

### Bugs
    - **Braintrust**
        - Adds model to metadata to enable braintrust cost estimation - [PR](https://github.com/BerriAI/litellm/pull/12022)
### Features
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

### Bugs
    - **Team-only models**
        - Filter team-only models from routing logic for non-team calls
    - **Context Window Exceeded error**
        - Catch anthropic exceptions - [PR](https://github.com/BerriAI/litellm/pull/12113)
### Features
    - **Router**
        - allow using dynamic cooldown time for a specific deployment - [PR](https://github.com/BerriAI/litellm/pull/12037)
        - handle cooldown_time = 0 for deployments - [PR](https://github.com/BerriAI/litellm/pull/12108)
    - **Redis**
        - Add better debugging to see what variables are set - [PR](https://github.com/BerriAI/litellm/pull/12073)

---

## General Proxy Improvements

### Bugs
    - **aiohttp**
        - Check HTTP_PROXY vars in networking requests
        - Allow using HTTP_ Proxy settings with trust_env

### Features
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
