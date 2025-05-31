---
title: v1.72.0-stable
slug: v1.72.0-stable
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
ghcr.io/berriai/litellm:main-v1.72.0-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.72.0
```
</TabItem>
</Tabs>

## Key Highlights

LiteLLM v1.72.0-stable is live now. Here are the key highlights of this release:

- **Vector Store Permissions**: Complete access control system for vector stores across Keys, Teams, and Organizations.
- **MCP Server Integration**: Native support for MCP servers with easy configuration through the LiteLLM UI.
- **Enhanced Provider Support**: New Nebius AI Studio integration, improved Gemini parallel tool calling, and Azure image edits support.

## Vector Store Permissions

This release introduces comprehensive vector store access controls, allowing administrators to manage permissions at the Key, Team, and Organization levels. This provides:

- **Granular Access Control**: Control which vector stores users can access based on their permissions
- **Team-based Management**: Assign vector store permissions to entire teams  
- **Organization-wide Policies**: Set organization-level vector store access rules
- **UI Integration**: Full management interface for viewing and editing vector store permissions

Vector store permissions are now enforced across all LiteLLM authentication mechanisms, ensuring secure access to your vector data.

## MCP Server Integration

This release brings native Model Context Protocol (MCP) server support to LiteLLM:

- **Well-known MCP Servers**: Pre-configured access to popular MCP servers
- **UI Configuration**: Add and manage MCP servers directly through the LiteLLM dashboard  
- **Automatic Initialization**: MCP servers are automatically initialized when the MCP package is available
- **Seamless Integration**: Works with existing LiteLLM authentication and routing

## New Models / Updated Models

- **[Gemini](https://docs.litellm.ai/docs/providers/gemini)**
    - Parallel tool calling support with `parallel_tool_calls` parameter - [PR](https://github.com/BerriAI/litellm/pull/11125)
    - All Gemini models now support parallel function calling - [PR](https://github.com/BerriAI/litellm/pull/11225)
    - Stream thinking as reasoning_content support - [PR](https://github.com/BerriAI/litellm/pull/11290)
- **[Nebius AI Studio](../../docs/providers/nebius)**
    - New provider integration - [PR](https://github.com/BerriAI/litellm/pull/11143)
- **[Bedrock](../../docs/providers/bedrock)**
    - Video support for Bedrock Converse - [PR](https://github.com/BerriAI/litellm/pull/11166)
    - InvokeAgents support as /chat/completions route - [PR](https://github.com/BerriAI/litellm/pull/11239)
    - AI21 Jamba models compatibility fixes - [PR](https://github.com/BerriAI/litellm/pull/11233)
    - Fixed duplicate maxTokens parameter for Claude with thinking - [PR](https://github.com/BerriAI/litellm/pull/11181)
- **[VertexAI](../../docs/providers/vertex)**
    - codeExecution tool support and anyOf handling - [PR](https://github.com/BerriAI/litellm/pull/11195)
    - Anthropic support on /v1/messages - [PR](https://github.com/BerriAI/litellm/pull/11246)
    - Thinking, global regions, and parallel tool calling improvements - [PR](https://github.com/BerriAI/litellm/pull/11194)
- **[Anthropic](../../docs/providers/anthropic)**
    - Thinking blocks on streaming support - [PR](https://github.com/BerriAI/litellm/pull/11194)
    - Files API with form-data support on passthrough - [PR](https://github.com/BerriAI/litellm/pull/11256)
    - File ID support on /chat/completion - [PR](https://github.com/BerriAI/litellm/pull/11256)
- **[Mistral](../../docs/providers/mistral)**
    - Updated mistral-medium prices and context sizes - [PR](https://github.com/BerriAI/litellm/pull/10729)
- **[Ollama](../../docs/providers/ollama)**
    - Tool calls parsing on streaming - [PR](https://github.com/BerriAI/litellm/pull/11171)
- **[Cohere](../../docs/providers/cohere)**
    - Swapped Cohere and Cohere Chat provider positioning - [PR](https://github.com/BerriAI/litellm/pull/11173)

## LLM API Endpoints

- **[Image Edits](../../docs/image_generation)**
    - Azure support for /v1/images/edits - [PR](https://github.com/BerriAI/litellm/pull/11160)
    - Cost tracking for image edits endpoint (OpenAI, Azure) - [PR](https://github.com/BerriAI/litellm/pull/11186)
- **[Completions](../../docs/completion/chat)**
    - Codestral latency overhead tracking on /v1/completions - [PR](https://github.com/BerriAI/litellm/pull/10879)
    - 'contains' support for ChatCompletionDeltaToolCall - [PR](https://github.com/BerriAI/litellm/pull/10879)
- **[Audio](../../docs/audio/speech)**
    - GPT-4o mini audio preview pricing without date - [PR](https://github.com/BerriAI/litellm/pull/11207)
    - Non-default params support for audio transcription - [PR](https://github.com/BerriAI/litellm/pull/11212)
- **[Responses API](../../docs/response_api)**
    - Session management improvements - [PR](https://github.com/BerriAI/litellm/pull/11254)

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
    - Fixed duplicate object_permission field in LiteLLM_TeamTable - [PR](https://github.com/BerriAI/litellm/pull/11219)
- **SCIM**
    - Security fixes for SCIM token exposure on UI - [PR](https://github.com/BerriAI/litellm/pull/11220)
- **Session Management**
    - Streamlined session expiration in UI - [PR](https://github.com/BerriAI/litellm/pull/11279)
- **Dependencies**
    - Updated Next.js from 14.2.26 to 15.2.4 - [PR](https://github.com/BerriAI/litellm/pull/11216)

## Authentication & Security

- **[Azure OIDC](../../docs/proxy/token_auth)**
    - OIDC provider improvements and audience bug fix - [PR](https://github.com/BerriAI/litellm/pull/10054)
- **Secret Managers**
    - Removed AzureCredentialType restriction on AZURE_CREDENTIAL - [PR](https://github.com/BerriAI/litellm/pull/11272)
- **SSO/Key Management**
    - Support for forwarding /sso/key/generate to server root path URL - [PR](https://github.com/BerriAI/litellm/pull/11165)
    - Prevention of sensitive key leakage to Langfuse - [PR](https://github.com/BerriAI/litellm/pull/11165)

## Logging / Alerting Integrations

- **[Prometheus](../../docs/proxy/prometheus)**
    - Option to disable end_user tracking by default - [PR](https://github.com/BerriAI/litellm/pull/11192)
    - Flag to enable end_user tracking on Prometheus - [PR](https://github.com/BerriAI/litellm/pull/11192)
- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Fixed "Max langfuse clients reached" performance issue - [PR](https://github.com/BerriAI/litellm/pull/11285)
    - Debug fixes for langfuse client management - [PR](https://github.com/BerriAI/litellm/pull/11221)
- **[Helicone](../../docs/proxy/logging)**
    - Base URL support - [PR](https://github.com/BerriAI/litellm/pull/11211)
    - Fix for embedding cache hits on string input - [PR](https://github.com/BerriAI/litellm/pull/11211)
- **[Pangea](../../docs/proxy/guardrails)**
    - Added Pangea provider to Guardrails hook - [PR](https://github.com/BerriAI/litellm/pull/10775)
- **[Sentry](../../docs/proxy/logging)**
    - Added sentry sample rate configuration - [PR](https://github.com/BerriAI/litellm/pull/10283)

## Performance / Reliability Improvements

- **aiohttp Transport**
    - Handling for aiohttp.ClientPayloadError - [PR](https://github.com/BerriAI/litellm/pull/11162)
    - SSL verification settings support - [PR](https://github.com/BerriAI/litellm/pull/11162)
    - Rollback to httpx==0.27.0 for stability - [PR](https://github.com/BerriAI/litellm/pull/11146)
- **Request Limiting**
    - Sliding window logic for parallel request limiter v2 - [PR](https://github.com/BerriAI/litellm/pull/11283)
- **Docker**
    - Fixed healthcheck test using curl when curl not in image - [PR](https://github.com/BerriAI/litellm/pull/9737)
- **Parameter Handling**
    - Support for dropping non-OpenAI params via additional_drop_params - [PR](https://github.com/BerriAI/litellm/pull/11246)
    - Fixed frequency_penalty to repeat_penalty parameter mapping - [PR](https://github.com/BerriAI/litellm/pull/11284)
- **Timeout Management**
    - Increased timeout configurations - [PR](https://github.com/BerriAI/litellm/pull/11288)

## Guardrails & Security

- **Bedrock Guardrails**
    - Streaming support for bedrock post guard - [PR](https://github.com/BerriAI/litellm/pull/11247)
    - Auth parameter persistence fixes - [PR](https://github.com/BerriAI/litellm/pull/11270)

## Bug Fixes

This release includes numerous bug fixes to improve stability and reliability:

- **Model Deployment Fixes**
    - Added missing request_kwargs to get_available_deployment call - [PR](https://github.com/BerriAI/litellm/pull/11202)
    - Fixed calling Azure O-series models - [PR](https://github.com/BerriAI/litellm/pull/11212)
    - Fixed deprecation_date value for Llama Groq models - [PR](https://github.com/BerriAI/litellm/pull/11151)

- **Documentation & Examples**
    - Updated Azure OpenAI documentation - [PR](https://github.com/BerriAI/litellm/pull/11161)
    - Fixed syntax error in Python example code - [PR](https://github.com/BerriAI/litellm/pull/11242)
    - Updated Docker quick start guide to use gpt-4o instead of gpt-3.5-turbo - [PR](https://github.com/BerriAI/litellm/pull/11223)
    - Fixed LiteLLM CLA reference - [PR](https://github.com/BerriAI/litellm/pull/11230)

- **GitHub Actions & Testing**
    - Fixed GitHub action testing issues - [PR](https://github.com/BerriAI/litellm/pull/11163)

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
