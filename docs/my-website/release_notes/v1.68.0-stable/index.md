---
title: v1.68.0-stable
slug: v1.68.0-stable
date: 2025-05-03T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
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
docker run
-e STORE_MODEL_IN_DB=True
-p 4000:4000
docker.litellm.ai/berriai/litellm:main-v1.68.0-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.68.0.post1
```
</TabItem>
</Tabs>

## Key Highlights

LiteLLM v1.68.0-stable will be live soon. Here are the key highlights of this release:

- **Bedrock Knowledge Base**: You can now call query your Bedrock Knowledge Base with all LiteLLM models via `/chat/completion` or `/responses` API.
- **Rate Limits**: This release brings accurate rate limiting across multiple instances, reducing spillover to at most 10 additional requests in high traffic. 
- **Meta Llama API**: Added support for Meta Llama API [Get Started](https://docs.litellm.ai/docs/providers/meta_llama)
- **LlamaFile**: Added support for LlamaFile [Get Started](https://docs.litellm.ai/docs/providers/llamafile)

## Bedrock Knowledge Base (Vector Store)

<Image img={require('../../img/release_notes/bedrock_kb.png')}/>
<br/>

This release adds support for Bedrock vector stores (knowledge bases) in LiteLLM. With this update, you can:

- Use Bedrock vector stores in the OpenAI /chat/completions spec with all LiteLLM supported models. 
- View all available vector stores through the LiteLLM UI or API.
- Configure vector stores to be always active for specific models.
- Track vector store usage in LiteLLM Logs.

For the next release we plan on allowing you to set key, user, team, org permissions for vector stores. 

[Read more here](https://docs.litellm.ai/docs/completion/knowledgebase)

## Rate Limiting

<Image img={require('../../img/multi_instance_rate_limiting.png')}/>
<br/>


This release brings accurate multi-instance rate limiting across keys/users/teams. Outlining key engineering changes below:

- **Change**: Instances now increment cache value instead of setting it. To avoid calling Redis on each request, this is synced every 0.01s.
- **Accuracy**: In testing, we saw a maximum spill over from expected of 10 requests, in high traffic (100 RPS, 3 instances), vs. current 189 request spillover
- **Performance**: Our load tests show this to reduce median response time by 100ms in high traffic 

This is currently behind a feature flag, and we plan to have this be the default by next week. To enable this today, just add this environment variable:

```
export LITELLM_RATE_LIMIT_ACCURACY=true
```

[Read more here](../../docs/proxy/users#beta-multi-instance-rate-limiting) 



## New Models / Updated Models
- **Gemini ([VertexAI](https://docs.litellm.ai/docs/providers/vertex#usage-with-litellm-proxy-server) + [Google AI Studio](https://docs.litellm.ai/docs/providers/gemini))**
    - Handle more json schema - openapi schema conversion edge cases [PR](https://github.com/BerriAI/litellm/pull/10351)
    - Tool calls - return ‘finish_reason=“tool_calls”’ on gemini tool calling response [PR](https://github.com/BerriAI/litellm/pull/10485)
- **[VertexAI](../../docs/providers/vertex#metallama-api)**
    - Meta/llama-4 model support [PR](https://github.com/BerriAI/litellm/pull/10492)
    - Meta/llama3 - handle tool call result in content [PR](https://github.com/BerriAI/litellm/pull/10492)
    - Meta/* - return ‘finish_reason=“tool_calls”’ on tool calling response [PR](https://github.com/BerriAI/litellm/pull/10492)
- **[Bedrock](../../docs/providers/bedrock#litellm-proxy-usage)**
    - [Image Generation](../../docs/providers/bedrock#image-generation) - Support new ‘stable-image-core’ models - [PR](https://github.com/BerriAI/litellm/pull/10351)
    - [Knowledge Bases](../../docs/completion/knowledgebase) - support using Bedrock knowledge bases with `/chat/completions` [PR](https://github.com/BerriAI/litellm/pull/10413)
    - [Anthropic](../../docs/providers/bedrock#litellm-proxy-usage) - add ‘supports_pdf_input’ for claude-3.7-bedrock models [PR](https://github.com/BerriAI/litellm/pull/9917), [Get Started](../../docs/completion/document_understanding#checking-if-a-model-supports-pdf-input)
- **[OpenAI](../../docs/providers/openai)**
    - Support OPENAI_BASE_URL in addition to OPENAI_API_BASE [PR](https://github.com/BerriAI/litellm/pull/10423)
    - Correctly re-raise 504 timeout errors [PR](https://github.com/BerriAI/litellm/pull/10462)
    - Native Gpt-4o-mini-tts support [PR](https://github.com/BerriAI/litellm/pull/10462)
- 🆕 **[Meta Llama API](../../docs/providers/meta_llama)** provider [PR](https://github.com/BerriAI/litellm/pull/10451)
- 🆕 **[LlamaFile](../../docs/providers/llamafile)** provider [PR](https://github.com/BerriAI/litellm/pull/10482)

## LLM API Endpoints
- **[Response API](../../docs/response_api)** 
    - Fix for handling multi turn sessions [PR](https://github.com/BerriAI/litellm/pull/10415)
- **[Embeddings](../../docs/embedding/supported_embedding)**
    - Caching fixes - [PR](https://github.com/BerriAI/litellm/pull/10424)
        - handle str -> list cache
        - Return usage tokens for cache hit 
        - Combine usage tokens on partial cache hits 
- 🆕 **[Vector Stores](../../docs/completion/knowledgebase)**
    - Allow defining Vector Store Configs - [PR](https://github.com/BerriAI/litellm/pull/10448)
    - New StandardLoggingPayload field for requests made when a vector store is used - [PR](https://github.com/BerriAI/litellm/pull/10509)
    - Show Vector Store / KB Request on LiteLLM Logs Page  - [PR](https://github.com/BerriAI/litellm/pull/10514)
    - Allow using vector store in OpenAI API spec with tools - [PR](https://github.com/BerriAI/litellm/pull/10516)
- **[MCP](../../docs/mcp)**
    - Ensure Non-Admin virtual keys can access /mcp routes - [PR](https://github.com/BerriAI/litellm/pull/10473)
      
      **Note:** Currently, all Virtual Keys are able to access the MCP endpoints. We are working on a feature to allow restricting MCP access by keys/teams/users/orgs. Follow [here](https://github.com/BerriAI/litellm/discussions/9891) for updates.
- **Moderations**
    - Add logging callback support for `/moderations` API - [PR](https://github.com/BerriAI/litellm/pull/10390)


## Spend Tracking / Budget Improvements
- **[OpenAI](../../docs/providers/openai)**
    - [computer-use-preview](../../docs/providers/openai/responses_api#computer-use) cost tracking / pricing [PR](https://github.com/BerriAI/litellm/pull/10422)
    - [gpt-4o-mini-tts](../../docs/providers/openai/text_to_speech) input cost tracking - [PR](https://github.com/BerriAI/litellm/pull/10462)
- **[Fireworks AI](../../docs/providers/fireworks_ai)** - pricing updates - new `0-4b` model pricing tier + llama4 model pricing
- **[Budgets](../../docs/proxy/users#set-budgets)**
    - [Budget resets](../../docs/proxy/users#reset-budgets) now happen as start of day/week/month - [PR](https://github.com/BerriAI/litellm/pull/10333)
    - Trigger [Soft Budget Alerts](../../docs/proxy/alerting#soft-budget-alerts-for-virtual-keys) When Key Crosses Threshold - [PR](https://github.com/BerriAI/litellm/pull/10491)
- **[Token Counting](../../docs/completion/token_usage#3-token_counter)**
    - Rewrite of token_counter() function to handle to prevent undercounting tokens - [PR](https://github.com/BerriAI/litellm/pull/10409)


## Management Endpoints / UI
- **Virtual Keys**
    - Fix filtering on key alias - [PR](https://github.com/BerriAI/litellm/pull/10455)
    - Support global filtering on keys - [PR](https://github.com/BerriAI/litellm/pull/10455)
    - Pagination - fix clicking on next/back buttons on table - [PR](https://github.com/BerriAI/litellm/pull/10528)
- **Models**
    - Triton - Support adding model/provider on UI - [PR](https://github.com/BerriAI/litellm/pull/10456)
    - VertexAI - Fix adding vertex models with reusable credentials - [PR](https://github.com/BerriAI/litellm/pull/10528)
    - LLM Credentials - show existing credentials for easy editing - [PR](https://github.com/BerriAI/litellm/pull/10519)
- **Teams**
    - Allow reassigning team to other org - [PR](https://github.com/BerriAI/litellm/pull/10527)
- **Organizations**
    - Fix showing org budget on table - [PR](https://github.com/BerriAI/litellm/pull/10528)



## Logging / Guardrail Integrations
- **[Langsmith](../../docs/observability/langsmith_integration)**
    - Respect [langsmith_batch_size](../../docs/observability/langsmith_integration#local-testing---control-batch-size) param - [PR](https://github.com/BerriAI/litellm/pull/10411)

## Performance / Loadbalancing / Reliability improvements
- **[Redis](../../docs/proxy/caching)**
    - Ensure all redis queues are periodically flushed, this fixes an issue where redis queue size was growing indefinitely when request tags were used - [PR](https://github.com/BerriAI/litellm/pull/10393)
- **[Rate Limits](../../docs/proxy/users#set-rate-limit)**
    - [Multi-instance rate limiting](../../docs/proxy/users#beta-multi-instance-rate-limiting) support across keys/teams/users/customers - [PR](https://github.com/BerriAI/litellm/pull/10458), [PR](https://github.com/BerriAI/litellm/pull/10497), [PR](https://github.com/BerriAI/litellm/pull/10500)
- **[Azure OpenAI OIDC](../../docs/providers/azure#entra-id---use-azure_ad_token)**
    - allow using litellm defined params for [OIDC Auth](../../docs/providers/azure#entra-id---use-azure_ad_token) - [PR](https://github.com/BerriAI/litellm/pull/10394)


## General Proxy Improvements
- **Security**
    - Allow [blocking web crawlers](../../docs/proxy/enterprise#blocking-web-crawlers) - [PR](https://github.com/BerriAI/litellm/pull/10420)
- **Auth**
    - Support [`x-litellm-api-key` header param by default](../../docs/pass_through/vertex_ai#use-with-virtual-keys), this fixes an issue from the prior release where `x-litellm-api-key` was not being used on vertex ai passthrough requests - [PR](https://github.com/BerriAI/litellm/pull/10392)
    - Allow key at max budget to call non-llm api endpoints - [PR](https://github.com/BerriAI/litellm/pull/10392)
- 🆕 **[Python Client Library](../../docs/proxy/management_cli) for LiteLLM Proxy management endpoints**
    - Initial PR - [PR](https://github.com/BerriAI/litellm/pull/10445)
    - Support for doing HTTP requests - [PR](https://github.com/BerriAI/litellm/pull/10452)
- **Dependencies**
    - Don’t require uvloop for windows - [PR](https://github.com/BerriAI/litellm/pull/10483)
