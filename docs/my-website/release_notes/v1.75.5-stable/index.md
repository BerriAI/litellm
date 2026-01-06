---
title: "v1.75.5-stable - Redis latency improvements"
slug: "v1-75-5"
date: 2025-08-10T10:00:00
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
docker.litellm.ai/berriai/litellm:v1.75.5-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.75.5.post2
```

</TabItem>
</Tabs>

---

## Key Highlights

- **Redis - Latency Improvements** - Reduces P99 latency by 50% with Redis enabled. 
- **Responses API Session Management** - Support for managing responses API sessions with images.
- **Oracle Cloud Infrastructure** - New LLM provider for calling models on Oracle Cloud Infrastructure.
- **Digital Ocean's Gradient AI** - New LLM provider for calling models on Digital Ocean's Gradient AI platform.

---

### Risk of Upgrade

If you build the proxy from the pip package, you should hold off on upgrading. This version makes `prisma migrate deploy` our default for managing the DB. This is safer, as it doesn't reset the DB, but it requires a manual `prisma generate` step. 

Users of our Docker image, are **not** affected by this change. 

---

## Redis Latency Improvements

<Image 
  img={require('../../img/release_notes/faster_caching_calls.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

<br/>

This release adds in-memory caching for Redis requests, enabling faster response times in high-traffic. Now, LiteLLM instances will check their in-memory cache for a cache hit, before checking Redis. This reduces caching-related latency from 100ms for LLM API calls to sub-1ms, on cache hits. 

---

## Responses API Session Management w/ Images

<Image 
  img={require('../../img/release_notes/responses_api_session_mgt_images.jpg')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

<br/>

LiteLLM now supports session management for Responses API requests with images. This is great for use-cases like chatbots, that are using the Responses API to track the state of a conversation. LiteLLM session management works across **ALL** LLM API's (including Anthropic, Bedrock, OpenAI, etc). LiteLLM session management works by storing the request and response content in an s3 bucket, you can specify. 

---


## New Models / Updated Models

#### New Model Support

| Provider    | Model                                  | Context Window | Input ($/1M tokens) | Output ($/1M tokens) |
| ----------- | -------------------------------------- | -------------- | ------------------- | -------------------- | 
| Bedrock | `bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0` | 200k | $15 | $75 |
| Bedrock | `bedrock/openai.gpt-oss-20b-1:0` | 200k | 0.07 | 0.3 |
| Bedrock | `bedrock/openai.gpt-oss-120b-1:0` | 200k | 0.15 | 0.6 |
| Fireworks AI | `fireworks_ai/accounts/fireworks/models/glm-4p5` | 128k | 0.55 | 2.19 |
| Fireworks AI | `fireworks_ai/accounts/fireworks/models/glm-4p5-air` | 128k | 0.22 | 0.88 |
| Fireworks AI | `fireworks_ai/accounts/fireworks/models/gpt-oss-120b` | 131072 | 0.15 | 0.6 |
| Fireworks AI | `fireworks_ai/accounts/fireworks/models/gpt-oss-20b` | 131072 | 0.05 | 0.2 |
| Groq | `groq/openai/gpt-oss-20b` | 131072 | 0.1 | 0.5 |
| Groq | `groq/openai/gpt-oss-120b` | 131072 | 0.15 | 0.75 |
| OpenAI | `openai/gpt-5` | 400k | 1.25 | 10 | 
| OpenAI | `openai/gpt-5-2025-08-07` | 400k | 1.25 | 10 | 
| OpenAI | `openai/gpt-5-mini` | 400k | 0.25 | 2 |
| OpenAI | `openai/gpt-5-mini-2025-08-07` | 400k | 0.25 | 2 | 
| OpenAI | `openai/gpt-5-nano` | 400k | 0.05 | 0.4 | 
| OpenAI | `openai/gpt-5-nano-2025-08-07` | 400k | 0.05 | 0.4 | 
| OpenAI | `openai/gpt-5-chat` | 400k | 1.25 | 10 | 
| OpenAI | `openai/gpt-5-chat-latest` | 400k | 1.25 | 10 | 
| Azure | `azure/gpt-5` | 400k | 1.25 | 10 | 
| Azure | `azure/gpt-5-2025-08-07` | 400k | 1.25 | 10 | 
| Azure | `azure/gpt-5-mini` | 400k | 0.25 | 2 | 
| Azure | `azure/gpt-5-mini-2025-08-07` | 400k | 0.25 | 2 | 
| Azure | `azure/gpt-5-nano-2025-08-07` | 400k | 0.05 | 0.4 | 
| Azure | `azure/gpt-5-nano` | 400k | 0.05 | 0.4 | 
| Azure | `azure/gpt-5-chat` | 400k | 1.25 | 10 | 
| Azure | `azure/gpt-5-chat-latest` | 400k | 1.25 | 10 | 

#### Features

- **[OCI](../../docs/providers/oci)**
    - New LLM provider - [PR #13206](https://github.com/BerriAI/litellm/pull/13206)
- **[JinaAI](../../docs/providers/jina_ai)**
    - support multimodal embedding models - [PR #13181](https://github.com/BerriAI/litellm/pull/13181)
- **GPT-5 ([OpenAI](../../docs/providers/openai)/[Azure](../../docs/providers/azure))**
    - Support drop_params for temperature - [PR #13390](https://github.com/BerriAI/litellm/pull/13390)
    - Map max_tokens to max_completion_tokens - [PR #13390](https://github.com/BerriAI/litellm/pull/13390)
- **[Anthropic](../../docs/providers/anthropic)**
    - Add claude-opus-4-1 on model cost map - [PR #13384](https://github.com/BerriAI/litellm/pull/13384)
- **[OpenRouter](../../docs/providers/openrouter)**
    - Add gpt-oss to model cost map - [PR #13442](https://github.com/BerriAI/litellm/pull/13442)
- **[Cerebras](../../docs/providers/cerebras)**
    - Add gpt-oss to model cost map - [PR #13442](https://github.com/BerriAI/litellm/pull/13442)
- **[Azure](../../docs/providers/azure)**
    - Support drop params for ‘temperature’ on o-series models - [PR #13353](https://github.com/BerriAI/litellm/pull/13353)
- **[GradientAI](../../docs/providers/gradient_ai)**
    - New LLM Provider - [PR #12169](https://github.com/BerriAI/litellm/pull/12169)

#### Bugs

- **[OpenAI](../../docs/providers/openai)**
    - Add ‘service_tier’ and ‘safety_identifier’ as supported responses api params - [PR #13258](https://github.com/BerriAI/litellm/pull/13258)
    - Correct pricing for web search on 4o-mini - [PR #13269](https://github.com/BerriAI/litellm/pull/13269)
- **[Mistral](../../docs/providers/mistral)**
    - Handle $id and $schema fields when calling mistral - [PR #13389](https://github.com/BerriAI/litellm/pull/13389)
---

## LLM API Endpoints

#### Features

- `/responses` 
    - Responses API Session Handling w/ support for images - [PR #13347](https://github.com/BerriAI/litellm/pull/13347)
    - failed if input containing ResponseReasoningItem - [PR #13465](https://github.com/BerriAI/litellm/pull/13465)
    - Support custom tools - [PR #13418](https://github.com/BerriAI/litellm/pull/13418)

#### Bugs

- `/chat/completions` 
    - Fix completion_token_details usage object missing ‘text’ tokens - [PR #13234](https://github.com/BerriAI/litellm/pull/13234)
    - (SDK) handle tool being a pydantic object - [PR #13274](https://github.com/BerriAI/litellm/pull/13274)
    - include cost in streaming usage object - [PR #13418](https://github.com/BerriAI/litellm/pull/13418)
    - Exclude none fields on /chat/completion - allows usage with n8n - [PR #13320](https://github.com/BerriAI/litellm/pull/13320)
- `/responses` 
    - Transform function call in response for non-openai models (gemini/anthropic) - [PR #13260](https://github.com/BerriAI/litellm/pull/13260)
    - Fix unsupported operand error with model groups - [PR #13293](https://github.com/BerriAI/litellm/pull/13293)
    - Responses api session management for streaming responses - [PR #13396](https://github.com/BerriAI/litellm/pull/13396)
- `/v1/messages`
    - Added litellm claude code count tokens - [PR #13261](https://github.com/BerriAI/litellm/pull/13261)
- `/vector_stores`
    - Fix create/search vector store errors - [PR #13285](https://github.com/BerriAI/litellm/pull/13285)
---

## [MCP Gateway](../../docs/mcp)

#### Features

- Add route check for internal users - [PR #13350](https://github.com/BerriAI/litellm/pull/13350)
- MCP Guardrails - docs - [PR #13392](https://github.com/BerriAI/litellm/pull/13392)


#### Bugs

- Fix auth on UI for bearer token servers - [PR #13312](https://github.com/BerriAI/litellm/pull/13312)
- allow access group on mcp tool retrieval - [PR #13425](https://github.com/BerriAI/litellm/pull/13425)


---

## Management Endpoints / UI

#### Features

- **Teams**
    - Add team deletion check for teams with keys - [PR #12953](https://github.com/BerriAI/litellm/pull/12953)
- **Models**
    - Add ability to set model alias per key/team - [PR #13276](https://github.com/BerriAI/litellm/pull/13276)
    - New button to reload model pricing from model cost map - [PR #13464](https://github.com/BerriAI/litellm/pull/13464), [PR #13470](https://github.com/BerriAI/litellm/pull/13470)
- **Keys**
    - Make ‘team’ field required when creating service account keys - [PR #13302](https://github.com/BerriAI/litellm/pull/13302)
    - Gray out key-based logging settings for non-enterprise users - prevents confusion on if ‘logging’ all up is supported - [PR #13431](https://github.com/BerriAI/litellm/pull/13431)
- **Navbar**
    - Add logo customization for LiteLLM admin UI - [PR #12958](https://github.com/BerriAI/litellm/pull/12958)
- **Logs**
    - Add token breakdowns on logs + session page - [PR #13357](https://github.com/BerriAI/litellm/pull/13357)
- **Usage**
    - Ensure Usage Page loads after the DB has large entries - [PR #13400](https://github.com/BerriAI/litellm/pull/13400)
- **Test Key Page**
    - allow uploading images for /chat/completions and /responses - [PR #13445](https://github.com/BerriAI/litellm/pull/13445)
- **MCP**
    - Add auth tokens to local storage auth - [PR #13473](https://github.com/BerriAI/litellm/pull/13473)

#### Bugs

- **Custom Root Path**
    - Fix login route when SSO is enabled - [PR #13267](https://github.com/BerriAI/litellm/pull/13267)
- **Customers/End-users**
    - Allow calling /v1/models when end user over budget - allows model listing to work on OpenWebUI when customer over budget - [PR #13320](https://github.com/BerriAI/litellm/pull/13320)
- **Teams**
    - Remove user - team membership, when user removed from team - [PR #13433](https://github.com/BerriAI/litellm/pull/13433)
- **Errors**
    - Bubble up network errors to user for Logging and Alerts page - [PR #13427](https://github.com/BerriAI/litellm/pull/13427)
- **Model Hub**
    - Show pricing for azure models, when base model is set - [PR #13418](https://github.com/BerriAI/litellm/pull/13418)
---

## Logging / Guardrail Integrations

#### Features

- **Bedrock Guardrails**
    - Redacted sensitive information in bedrock guardrails error message - [PR #13356](https://github.com/BerriAI/litellm/pull/13356)
- **Standard Logging Payload**
    - Fix ‘can’t register atextexit’ bug - [PR #13436](https://github.com/BerriAI/litellm/pull/13436)

#### Bugs

- **Braintrust**
    - Allow setting of braintrust callback base url - [PR #13368](https://github.com/BerriAI/litellm/pull/13368)
- **OTEL**
    - Track pre_call hook latency  - [PR #13362](https://github.com/BerriAI/litellm/pull/13362)

---

## Performance / Loadbalancing / Reliability improvements

#### Features

- **Team-BYOK models**
    - Add wildcard model support - [PR #13278](https://github.com/BerriAI/litellm/pull/13278)
- **Caching**
    - GCP IAM auth support for caching - [PR #13275](https://github.com/BerriAI/litellm/pull/13275)
- **Latency**
    - reduce p99 latency w/ redis enabled by 50% - only updates model usage if tpm/rpm limits set - [PR #13362](https://github.com/BerriAI/litellm/pull/13362)

---

## General Proxy Improvements

#### Features

- **Models**
    - Support /v1/models/\{model_id\} retrieval - [PR #13268](https://github.com/BerriAI/litellm/pull/13268)
- **Multi-instance**
    - Ensure disable_llm_api_endpoints works - [PR #13278](https://github.com/BerriAI/litellm/pull/13278)
- **Logs**
    - Add apscheduler log suppress - [PR #13299](https://github.com/BerriAI/litellm/pull/13299)
- **Helm**
    - Add labels to migrations job template - [PR #13343](https://github.com/BerriAI/litellm/pull/13343) s/o [@unique-jakub](https://github.com/unique-jakub)

#### Bugs

- **Non-root image**
    - Fix non-root image for migration - [PR #13379](https://github.com/BerriAI/litellm/pull/13379)
- **Get Routes**
    - Load get routes when using fastapi-offline - [PR #13466](https://github.com/BerriAI/litellm/pull/13466)
- **Health checks**
    - Generate unique trace IDs for Langfuse health checks - [PR #13468](https://github.com/BerriAI/litellm/pull/13468)
- **Swagger**
    - Allow using Swagger for /chat/completions - [PR #13469](https://github.com/BerriAI/litellm/pull/13469)
- **Auth**
    - Fix JWTs access not working with model access groups - [PR #13474](https://github.com/BerriAI/litellm/pull/13474)
    
---

## New Contributors

* @bbartels made their first contribution in https://github.com/BerriAI/litellm/pull/13244
* @breno-aumo made their first contribution in https://github.com/BerriAI/litellm/pull/13206
* @pascalwhoop made their first contribution in https://github.com/BerriAI/litellm/pull/13122
* @ZPerling made their first contribution in https://github.com/BerriAI/litellm/pull/13045
* @zjx20 made their first contribution in https://github.com/BerriAI/litellm/pull/13181
* @edwarddamato made their first contribution in https://github.com/BerriAI/litellm/pull/13368
* @msannan2 made their first contribution in https://github.com/BerriAI/litellm/pull/12169


## **[Full Changelog](https://github.com/BerriAI/litellm/compare/v1.74.15-stable...v1.75.5-stable.rc-draft)**