---
title: "v1.72.0-stable"
slug: "v1-72-0-stable"
date: 2025-05-31T10:00:00
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
docker.litellm.ai/berriai/litellm:main-v1.72.0-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.72.0
```
</TabItem>
</Tabs>


## Key Highlights

LiteLLM v1.72.0-stable.rc is live now. Here are the key highlights of this release:

- **Vector Store Permissions**: Control Vector Store access at the Key, Team, and Organization level.
- **Rate Limiting Sliding Window support**: Improved accuracy for Key/Team/User rate limits with request tracking across minutes.
- **Aiohttp Transport used by default**: Aiohttp transport is now the default transport for LiteLLM networking requests. This gives users 2x higher RPS per instance with a 40ms median latency overhead.
- **Bedrock Agents**: Call Bedrock Agents with `/chat/completions`, `/response` endpoints.
- **Anthropic File API**: Upload and analyze CSV files with Claude-4 on Anthropic via LiteLLM.
- **Prometheus**: End users (`end_user`) will no longer be tracked by default on Prometheus. Tracking end_users on prometheus is now opt-in. This is done to prevent the response from `/metrics` from  becoming too large. [Read More](../../docs/proxy/prometheus#tracking-end_user-on-prometheus)


---

## Vector Store Permissions

This release brings support for managing permissions for vector stores by Keys, Teams, Organizations (entities) on LiteLLM. When a request attempts to query a vector store, LiteLLM will block it if the requesting entity lacks the proper permissions.

This is great for use cases that require access to restricted data that you don't want everyone to use. 

Over the next week we plan on adding permission management for MCP Servers.

---
## Aiohttp Transport used by default

Aiohttp transport is now the default transport for LiteLLM networking requests. This gives users 2x higher RPS per instance with a 40ms median latency overhead. This has been live on LiteLLM Cloud for a week + gone through alpha users testing for a week.


If you encounter any issues, you can disable using the aiohttp transport in the following ways:

**On LiteLLM Proxy**

Set the `DISABLE_AIOHTTP_TRANSPORT=True` in the environment variables. 

```yaml showLineNumbers title="Environment Variable"
export DISABLE_AIOHTTP_TRANSPORT="True"
```

**On LiteLLM Python SDK**

Set the `disable_aiohttp_transport=True` to disable aiohttp transport. 

```python showLineNumbers title="Python SDK"
import litellm

litellm.disable_aiohttp_transport = True # default is False, enable this to disable aiohttp transport
result = litellm.completion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello, world!"}],
)
print(result)
```

---


## New Models / Updated Models

- **[Bedrock](../../docs/providers/bedrock)**
    - Video support for Bedrock Converse - [PR](https://github.com/BerriAI/litellm/pull/11166)
    - InvokeAgents support as /chat/completions route - [PR](https://github.com/BerriAI/litellm/pull/11239), [Get Started](../../docs/providers/bedrock_agents)
    - AI21 Jamba models compatibility fixes - [PR](https://github.com/BerriAI/litellm/pull/11233)
    - Fixed duplicate maxTokens parameter for Claude with thinking - [PR](https://github.com/BerriAI/litellm/pull/11181)
- **[Gemini (Google AI Studio + Vertex AI)](https://docs.litellm.ai/docs/providers/gemini)**
    - Parallel tool calling support with `parallel_tool_calls` parameter - [PR](https://github.com/BerriAI/litellm/pull/11125)
    - All Gemini models now support parallel function calling - [PR](https://github.com/BerriAI/litellm/pull/11225)
- **[VertexAI](../../docs/providers/vertex)**
    - codeExecution tool support and anyOf handling - [PR](https://github.com/BerriAI/litellm/pull/11195)
    - Vertex AI Anthropic support on /v1/messages - [PR](https://github.com/BerriAI/litellm/pull/11246)
    - Thinking, global regions, and parallel tool calling improvements - [PR](https://github.com/BerriAI/litellm/pull/11194)
    - Web Search Support [PR](https://github.com/BerriAI/litellm/commit/06484f6e5a7a2f4e45c490266782ed28b51b7db6)
- **[Anthropic](../../docs/providers/anthropic)**
    - Thinking blocks on streaming support - [PR](https://github.com/BerriAI/litellm/pull/11194)
    - Files API with form-data support on passthrough - [PR](https://github.com/BerriAI/litellm/pull/11256)
    - File ID support on /chat/completion - [PR](https://github.com/BerriAI/litellm/pull/11256)
- **[xAI](../../docs/providers/xai)**
    - Web Search Support [PR](https://github.com/BerriAI/litellm/commit/06484f6e5a7a2f4e45c490266782ed28b51b7db6)
- **[Google AI Studio](../../docs/providers/gemini)**
    - Web Search Support [PR](https://github.com/BerriAI/litellm/commit/06484f6e5a7a2f4e45c490266782ed28b51b7db6)
- **[Mistral](../../docs/providers/mistral)**
    - Updated mistral-medium prices and context sizes - [PR](https://github.com/BerriAI/litellm/pull/10729)
- **[Ollama](../../docs/providers/ollama)**
    - Tool calls parsing on streaming - [PR](https://github.com/BerriAI/litellm/pull/11171)
- **[Cohere](../../docs/providers/cohere)**
    - Swapped Cohere and Cohere Chat provider positioning - [PR](https://github.com/BerriAI/litellm/pull/11173)
- **[Nebius AI Studio](../../docs/providers/nebius)**
    - New provider integration - [PR](https://github.com/BerriAI/litellm/pull/11143)

## LLM API Endpoints

- **[Image Edits API](../../docs/image_generation)**
    - Azure support for /v1/images/edits - [PR](https://github.com/BerriAI/litellm/pull/11160)
    - Cost tracking for image edits endpoint (OpenAI, Azure) - [PR](https://github.com/BerriAI/litellm/pull/11186)
- **[Completions API](../../docs/completion/chat)**
    - Codestral latency overhead tracking on /v1/completions - [PR](https://github.com/BerriAI/litellm/pull/10879)
- **[Audio Transcriptions API](../../docs/audio/speech)**
    - GPT-4o mini audio preview pricing without date - [PR](https://github.com/BerriAI/litellm/pull/11207)
    - Non-default params support for audio transcription - [PR](https://github.com/BerriAI/litellm/pull/11212)
- **[Responses API](../../docs/response_api)**
    - Session management fixes for using Non-OpenAI models - [PR](https://github.com/BerriAI/litellm/pull/11254)

## Management Endpoints / UI

- **Vector Stores**
    - Permission management for LiteLLM Keys, Teams, and Organizations - [PR](https://github.com/BerriAI/litellm/pull/11213)
    - UI display of vector store permissions - [PR](https://github.com/BerriAI/litellm/pull/11277)
    - Vector store access controls enforcement - [PR](https://github.com/BerriAI/litellm/pull/11281)
    - Object permissions fixes and QA improvements - [PR](https://github.com/BerriAI/litellm/pull/11291)
- **Teams**
    - "All proxy models" display when no models selected - [PR](https://github.com/BerriAI/litellm/pull/11187)
    - Removed redundant teamInfo call, using existing teamsList - [PR](https://github.com/BerriAI/litellm/pull/11051)
    - Improved model tags display on Keys, Teams and Org pages - [PR](https://github.com/BerriAI/litellm/pull/11022)
- **SSO/SCIM**
    - Bug fixes for showing SCIM token on UI - [PR](https://github.com/BerriAI/litellm/pull/11220)
- **General UI**
    - Fix "UI Session Expired. Logging out" - [PR](https://github.com/BerriAI/litellm/pull/11279)
    - Support for forwarding /sso/key/generate to server root path URL - [PR](https://github.com/BerriAI/litellm/pull/11165)


## Logging / Guardrails Integrations

#### Logging
- **[Prometheus](../../docs/proxy/prometheus)**
    - End users will no longer be tracked by default on Prometheus. Tracking end_users on prometheus is now opt-in. [PR](https://github.com/BerriAI/litellm/pull/11192)
- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Performance improvements: Fixed "Max langfuse clients reached" issue - [PR](https://github.com/BerriAI/litellm/pull/11285)
- **[Helicone](../../docs/observability/helicone_integration)**
    - Base URL support - [PR](https://github.com/BerriAI/litellm/pull/11211)
- **[Sentry](../../docs/proxy/logging#sentry)**
    - Added sentry sample rate configuration - [PR](https://github.com/BerriAI/litellm/pull/10283)

#### Guardrails
- **[Bedrock Guardrails](../../docs/proxy/guardrails/bedrock)**
    - Streaming support for bedrock post guard - [PR](https://github.com/BerriAI/litellm/pull/11247)
    - Auth parameter persistence fixes - [PR](https://github.com/BerriAI/litellm/pull/11270)
- **[Pangea Guardrails](../../docs/proxy/guardrails/pangea)**
    - Added Pangea provider to Guardrails hook - [PR](https://github.com/BerriAI/litellm/pull/10775)


## Performance / Reliability Improvements
- **aiohttp Transport**
    - Handling for aiohttp.ClientPayloadError - [PR](https://github.com/BerriAI/litellm/pull/11162)
    - SSL verification settings support - [PR](https://github.com/BerriAI/litellm/pull/11162)
    - Rollback to httpx==0.27.0 for stability - [PR](https://github.com/BerriAI/litellm/pull/11146)
- **Request Limiting**
    - Sliding window logic for parallel request limiter v2 - [PR](https://github.com/BerriAI/litellm/pull/11283)


## Bug Fixes

- **LLM API Fixes**
    - Added missing request_kwargs to get_available_deployment call - [PR](https://github.com/BerriAI/litellm/pull/11202)
    - Fixed calling Azure O-series models - [PR](https://github.com/BerriAI/litellm/pull/11212)
    - Support for dropping non-OpenAI params via additional_drop_params - [PR](https://github.com/BerriAI/litellm/pull/11246)
    - Fixed frequency_penalty to repeat_penalty parameter mapping - [PR](https://github.com/BerriAI/litellm/pull/11284)
    - Fix for embedding cache hits on string input - [PR](https://github.com/BerriAI/litellm/pull/11211)
- **General**
    - OIDC provider improvements and audience bug fix - [PR](https://github.com/BerriAI/litellm/pull/10054)
    - Removed AzureCredentialType restriction on AZURE_CREDENTIAL - [PR](https://github.com/BerriAI/litellm/pull/11272)
    - Prevention of sensitive key leakage to Langfuse - [PR](https://github.com/BerriAI/litellm/pull/11165)
    - Fixed healthcheck test using curl when curl not in image - [PR](https://github.com/BerriAI/litellm/pull/9737)

## New Contributors
* [@agajdosi](https://github.com/agajdosi) made their first contribution in [#9737](https://github.com/BerriAI/litellm/pull/9737)
* [@ketangangal](https://github.com/ketangangal) made their first contribution in [#11161](https://github.com/BerriAI/litellm/pull/11161)
* [@Aktsvigun](https://github.com/Aktsvigun) made their first contribution in [#11143](https://github.com/BerriAI/litellm/pull/11143)
* [@ryanmeans](https://github.com/ryanmeans) made their first contribution in [#10775](https://github.com/BerriAI/litellm/pull/10775)
* [@nikoizs](https://github.com/nikoizs) made their first contribution in [#10054](https://github.com/BerriAI/litellm/pull/10054)
* [@Nitro963](https://github.com/Nitro963) made their first contribution in [#11202](https://github.com/BerriAI/litellm/pull/11202)
* [@Jacobh2](https://github.com/Jacobh2) made their first contribution in [#11207](https://github.com/BerriAI/litellm/pull/11207)
* [@regismesquita](https://github.com/regismesquita) made their first contribution in [#10729](https://github.com/BerriAI/litellm/pull/10729)
* [@Vinnie-Singleton-NN](https://github.com/Vinnie-Singleton-NN) made their first contribution in [#10283](https://github.com/BerriAI/litellm/pull/10283)
* [@trashhalo](https://github.com/trashhalo) made their first contribution in [#11219](https://github.com/BerriAI/litellm/pull/11219)
* [@VigneshwarRajasekaran](https://github.com/VigneshwarRajasekaran) made their first contribution in [#11223](https://github.com/BerriAI/litellm/pull/11223)
* [@AnilAren](https://github.com/AnilAren) made their first contribution in [#11233](https://github.com/BerriAI/litellm/pull/11233)
* [@fadil4u](https://github.com/fadil4u) made their first contribution in [#11242](https://github.com/BerriAI/litellm/pull/11242)
* [@whitfin](https://github.com/whitfin) made their first contribution in [#11279](https://github.com/BerriAI/litellm/pull/11279)
* [@hcoona](https://github.com/hcoona) made their first contribution in [#11272](https://github.com/BerriAI/litellm/pull/11272)
* [@keyute](https://github.com/keyute) made their first contribution in [#11173](https://github.com/BerriAI/litellm/pull/11173)
* [@emmanuel-ferdman](https://github.com/emmanuel-ferdman) made their first contribution in [#11230](https://github.com/BerriAI/litellm/pull/11230)

## Demo Instance

Here's a Demo Instance to test changes:

- Instance: https://demo.litellm.ai/
- Login Credentials:
    - Username: admin
    - Password: sk-1234

## [Git Diff](https://github.com/BerriAI/litellm/releases)
