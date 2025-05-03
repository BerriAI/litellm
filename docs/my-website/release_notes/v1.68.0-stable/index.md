---
title: v1.68.0-stable - Improved User Management
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

tags: ["responses_api", "ui_improvements", "security", "session_management"]
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
ghcr.io/berriai/litellm:main-v1.67.4-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.67.4.post1
```
</TabItem>
</Tabs>

## New Models / Updated Models
- **Gemini (VertexAI + Google AI Studio)**
    - Handle more json schema - openapi schema conversion edge cases - https://github.com/BerriAI/litellm/pull/10351
    - Tool calls - return ‘finish_reason=“tool_calls”’ on gemini tool calling response - https://github.com/BerriAI/litellm/pull/10485
- **VertexAI**
    - Meta/llama-4 model support - https://github.com/BerriAI/litellm/pull/10492
    - Meta/llama3 - handle tool call result in content - https://github.com/BerriAI/litellm/pull/10492
    - Meta/* - return ‘finish_reason=“tool_calls”’ on tool calling response - https://github.com/BerriAI/litellm/pull/10492
- **Bedrock**
    - Image Generation - Support new ‘stable-image-core’ models - https://github.com/BerriAI/litellm/pull/10351
    - Knowledge Bases - support using Bedrock knowledge bases with `/chat/completions` https://github.com/BerriAI/litellm/pull/10413
    - Anthropic - add ‘supports_pdf_input’ for claude-3.7-bedroc models 
- **OpenAI**
    - Support OPENAI_BASE_URL in addition to OPENAI_API_BASE - https://github.com/BerriAI/litellm/pull/10423
    - Correctly re-raise 504 timeout errors - https://github.com/BerriAI/litellm/pull/10462
    - Native Gpt-4o-mini-tts support - https://github.com/BerriAI/litellm/pull/10462
- 🆕 **LlamaFile** provider [Docs](../../docs/providers/llamafile), [PR](https://github.com/BerriAI/litellm/pull/10482)



## LLM API Endpoints
- **Response API** 
    - Fix for handling multi turn sessions https://github.com/BerriAI/litellm/pull/10415
- **Embeddings**
    - Caching fixes - https://github.com/BerriAI/litellm/pull/10424
        - handle str -> list cache
        - Return usage tokens for cache hit 
        - Combine usage tokens on partial cache hits 
- 🆕 **Vector Stores**
    - Allow defining Vector Store Configs - https://github.com/BerriAI/litellm/pull/10448
    - Log StandardLoggingVectorStoreRequest for requests made when a vector store - https://github.com/BerriAI/litellm/pull/10509
    - Show Vector Store / KB Request on LiteLLM Logs Page  - https://github.com/BerriAI/litellm/pull/10514
    - Allow using vector store in OpenAI API spec with tools - https://github.com/BerriAI/litellm/pull/10516
- **MCP**
    - Ensure Non-Admin virtual keys can access /mcp routes - https://github.com/BerriAI/litellm/pull/10473



## Spend Tracking / Budget Improvements
- **OpenAI**
    - computer-use-preview cost tracking / pricing https://github.com/BerriAI/litellm/pull/10422
    - Gpt-4o-mini-tts input cost tracking - https://github.com/BerriAI/litellm/pull/10462
- **Fireworks AI** - pricing updates - new `0-4b` model pricing tier + llama4 model pricing
- **Budgets**
    - Budget resets now happen as start of day/week/month - https://github.com/BerriAI/litellm/pull/10333
    - Trigger Soft Budget Webhooks When Key Crosses Threshold - https://github.com/BerriAI/litellm/pull/10491
- **Token Counting**
    - Rewrite of token_counter() function to handle to prevent undercounting tokens - https://github.com/BerriAI/litellm/pull/10409


## Management Endpoints / UI
- **Models**
    - Triton - Support adding model/provider on UI - https://github.com/BerriAI/litellm/pull/10456
    - VertexAI - Fix adding vertex models with reusable credentials - https://github.com/BerriAI/litellm/pull/10528
    - LLM Credentials - show existing credentials for easy editing - https://github.com/BerriAI/litellm/pull/10519
- **Virtual Keys**
    - Fix filtering on key alias - https://github.com/BerriAI/litellm/pull/10455
    - Support global filtering on keys - https://github.com/BerriAI/litellm/pull/10455
    - Pagination - fix clicking on next/back buttons on table - https://github.com/BerriAI/litellm/pull/10528
- **Organizations**
    - Fix showing org budget on table - https://github.com/BerriAI/litellm/pull/10528
- **Teams**
    - Allow reassigning team to other org - https://github.com/BerriAI/litellm/pull/10527


## Logging / Guardrail Integrations
- Add logging callback support for `/moderations` API https://github.com/BerriAI/litellm/pull/10390
- Langsmith - respect langsmith_batch_size param https://github.com/BerriAI/litellm/pull/10411

## Performance / Loadbalancing / Reliability improvements
- **Redis**
    - Ensure all redis queues are periodically flushed https://github.com/BerriAI/litellm/pull/10393
- **Rate Limits**
    - Multi-instance rate limiting support across keys/teams/users/customers - https://github.com/BerriAI/litellm/pull/10458, https://github.com/BerriAI/litellm/pull/10497, 
- **Azure OpenAI OIDC**
    - allow using litellm defined params for OIDC Auth - https://github.com/BerriAI/litellm/pull/10394


## General Proxy Improvements
- **Security**
    - Allow blocking web crawlers - https://github.com/BerriAI/litellm/pull/10420
- **Auth**
    - Support `x-litellm-api-key` param by default https://github.com/BerriAI/litellm/pull/10392
    - allow key at max budget to call non-llm api endpoints https://github.com/BerriAI/litellm/pull/10392
- **New Python Client Library for LiteLLM Proxy management endpoints**
    - Initial PR - https://github.com/BerriAI/litellm/pull/10445
    - Support for doing HTTP requests - https://github.com/BerriAI/litellm/pull/10452
- **Windows**
    - Don’t require uvloop for windows - https://github.com/BerriAI/litellm/pull/10483
