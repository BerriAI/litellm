---
title: "v1.80.8-stable - Introducing A2A Agent Gateway"
slug: "v1-80-8"
date: 2025-12-06T10:00:00
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
docker.litellm.ai/berriai/litellm:v1.80.8-stable
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.80.8
```

</TabItem>
</Tabs>

---

## Key Highlights

- **Agent Gateway (A2A)** - [Invoke agents through the AI Gateway with request/response logging and access controls](../../docs/a2a)
- **Guardrails API v2** - [Generic Guardrail API with streaming support, structured messages, and tool call checks](../../docs/adding_provider/generic_guardrail_api)
- **Customer (End User) Usage UI** - [Track and visualize end-user spend directly in the dashboard](../../docs/proxy/customer_usage)
- **vLLM Batch + Files API** - [Support for batch and files API with vLLM deployments](../../docs/batches)
- **Dynamic Rate Limiting on Teams** - [Enable dynamic rate limits and priority reservation on team-level](../../docs/proxy/team_budgets)
- **Google Cloud Chirp3 HD** - [New text-to-speech provider with Chirp3 HD voices](../../docs/text_to_speech)

---

### Agent Gateway (A2A)

<Image 
  img={require('../../img/a2a_gateway.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

<br/>

This release introduces **A2A Agent Gateway** for LiteLLM, allowing you to invoke and manage A2A agents with the same controls you have for LLM APIs.

As a **LiteLLM Gateway Admin**, you can now do the following:
    - **Request/Response Logging** - Every agent invocation is logged to the Logs page with full request and response tracking.
    - **Access Control** - Control which Team/Key can access which agents.

As a developer, you can continue using the A2A SDK, all you need to do is point you `A2AClient` to the LiteLLM proxy URL and your API key.

**Works with the A2A SDK:**

```python
from a2a.client import A2AClient

client = A2AClient(
    base_url="http://localhost:4000",  # Your LiteLLM proxy
    api_key="sk-1234"                   # LiteLLM API key
)

response = client.send_message(
    agent_id="my-agent",
    message="What's the status of my order?"
)
```

Get started with Agent Gateway here: [Agent Gateway Documentation](../../docs/a2a)

---

### Customer (End User) Usage UI

<Image
img={require('../../img/customer_usage.png')}
style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

Users can now filter usage statistics by customers, providing the same granular filtering capabilities available for teams and organizations.

**Details:**

- Filter usage analytics, spend logs, and activity metrics by customer ID
- View customer-level breakdowns alongside existing team and user-level filters
- Consistent filtering experience across all usage and analytics views

---

## New Providers and Endpoints

### New Providers (5 new providers)

| Provider | Supported LiteLLM Endpoints | Description |
| -------- | ------------------- | ----------- |
| **[Z.AI (Zhipu AI)](../../docs/providers/zai)** | `/v1/chat/completions`, `/v1/responses`, `/v1/messages` | Built-in support for Zhipu AI GLM models |
| **[RAGFlow](../../docs/providers/ragflow)** | `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/vector_stores` | RAG-based chat completions with vector store support |
| **[PublicAI](../../docs/providers/publicai)** | `/v1/chat/completions`, `/v1/responses`, `/v1/messages` | OpenAI-compatible provider via JSON config |
| **[Google Cloud Chirp3 HD](../../docs/text_to_speech)** | `/v1/audio/speech`, `/v1/audio/speech/stream` | Text-to-speech with Google Cloud Chirp3 HD voices |

### New LLM API Endpoints (2 new endpoints)

| Endpoint | Method | Description | Documentation |
| -------- | ------ | ----------- | ------------- |
| `/v1/agents/invoke` | POST | Invoke A2A agents through the AI Gateway | [Agent Gateway](../../docs/a2a) |
| `/cursor/chat/completions` | POST | Cursor BYOK endpoint - accepts Responses API input, returns Chat Completions output | [Cursor Integration](../../docs/tutorials/cursor_integration) |

---

## New Models / Updated Models

#### New Model Support (33 new models)

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| OpenAI | `gpt-5.1-codex-max` | 400K | $1.25 | $10.00 | Reasoning, vision, PDF input, responses API |
| Azure | `azure/gpt-5.1-codex-max` | 400K | $1.25 | $10.00 | Reasoning, vision, PDF input, responses API |
| Anthropic | `claude-opus-4-5` | 200K | $5.00 | $25.00 | Computer use, reasoning, vision |
| Bedrock | `global.anthropic.claude-opus-4-5-20251101-v1:0` | 200K | $5.00 | $25.00 | Computer use, reasoning, vision |
| Bedrock | `amazon.nova-2-lite-v1:0` | 1M | $0.30 | $2.50 | Reasoning, vision, video, PDF input |
| Bedrock | `amazon.titan-image-generator-v2:0` | - | - | $0.008/image | Image generation |
| Fireworks | `fireworks_ai/deepseek-v3p2` | 164K | $1.20 | $1.20 | Function calling, response schema |
| Fireworks | `fireworks_ai/kimi-k2-instruct-0905` | 262K | $0.60 | $2.50 | Function calling, response schema |
| DeepSeek | `deepseek/deepseek-v3.2` | 164K | $0.28 | $0.40 | Reasoning, function calling |
| Mistral | `mistral/mistral-large-3` | 256K | $0.50 | $1.50 | Function calling, vision |
| Azure AI | `azure_ai/mistral-large-3` | 256K | $0.50 | $1.50 | Function calling, vision |
| Moonshot | `moonshot/kimi-k2-0905-preview` | 262K | $0.60 | $2.50 | Function calling, web search |
| Moonshot | `moonshot/kimi-k2-turbo-preview` | 262K | $1.15 | $8.00 | Function calling, web search |
| Moonshot | `moonshot/kimi-k2-thinking-turbo` | 262K | $1.15 | $8.00 | Function calling, web search |
| OpenRouter | `openrouter/deepseek/deepseek-v3.2` | 164K | $0.28 | $0.40 | Reasoning, function calling |
| Databricks | `databricks/databricks-claude-haiku-4-5` | 200K | $1.00 | $5.00 | Reasoning, function calling |
| Databricks | `databricks/databricks-claude-opus-4` | 200K | $15.00 | $75.00 | Reasoning, function calling |
| Databricks | `databricks/databricks-claude-opus-4-1` | 200K | $15.00 | $75.00 | Reasoning, function calling |
| Databricks | `databricks/databricks-claude-opus-4-5` | 200K | $5.00 | $25.00 | Reasoning, function calling |
| Databricks | `databricks/databricks-claude-sonnet-4` | 200K | $3.00 | $15.00 | Reasoning, function calling |
| Databricks | `databricks/databricks-claude-sonnet-4-1` | 200K | $3.00 | $15.00 | Reasoning, function calling |
| Databricks | `databricks/databricks-gemini-2-5-flash` | 1M | $0.30 | $2.50 | Function calling |
| Databricks | `databricks/databricks-gemini-2-5-pro` | 1M | $1.25 | $10.00 | Function calling |
| Databricks | `databricks/databricks-gpt-5` | 400K | $1.25 | $10.00 | Function calling |
| Databricks | `databricks/databricks-gpt-5-1` | 400K | $1.25 | $10.00 | Function calling |
| Databricks | `databricks/databricks-gpt-5-mini` | 400K | $0.25 | $2.00 | Function calling |
| Databricks | `databricks/databricks-gpt-5-nano` | 400K | $0.05 | $0.40 | Function calling |
| Vertex AI | `vertex_ai/chirp` | - | $30.00/1M chars | - | Text-to-speech (Chirp3 HD) |
| Z.AI | `zai/glm-4.6` | 200K | $0.60 | $2.20 | Function calling |
| Z.AI | `zai/glm-4.5` | 128K | $0.60 | $2.20 | Function calling |
| Z.AI | `zai/glm-4.5v` | 128K | $0.60 | $1.80 | Function calling, vision |
| Z.AI | `zai/glm-4.5-flash` | 128K | Free | Free | Function calling |
| Vertex AI | `vertex_ai/bge-large-en-v1.5` | - | - | - | BGE Embeddings |

#### Features

- **[OpenAI](../../docs/providers/openai)**
    - Add `gpt-5.1-codex-max` model pricing and configuration - [PR #17541](https://github.com/BerriAI/litellm/pull/17541)
    - Add xhigh reasoning effort for gpt-5.1-codex-max - [PR #17585](https://github.com/BerriAI/litellm/pull/17585)
    - Add clear error message for empty LLM endpoint responses - [PR #17445](https://github.com/BerriAI/litellm/pull/17445)

- **[Azure OpenAI](../../docs/providers/azure/azure)**
    - Allow reasoning_effort='none' for Azure gpt-5.1 models - [PR #17311](https://github.com/BerriAI/litellm/pull/17311)

- **[Anthropic](../../docs/providers/anthropic)**
    - Add `claude-opus-4-5` alias to pricing data - [PR #17313](https://github.com/BerriAI/litellm/pull/17313)
    - Parse `<budget:thinking>` blocks for opus 4.5 - [PR #17534](https://github.com/BerriAI/litellm/pull/17534)
    - Update new Anthropic features as reviewed - [PR #17142](https://github.com/BerriAI/litellm/pull/17142)
    - Skip empty text blocks in Anthropic system messages - [PR #17442](https://github.com/BerriAI/litellm/pull/17442)

- **[Bedrock](../../docs/providers/bedrock)**
    - Add Nova embedding support - [PR #17253](https://github.com/BerriAI/litellm/pull/17253)
    - Add support for Bedrock Qwen 2 imported model - [PR #17461](https://github.com/BerriAI/litellm/pull/17461)
    - Bedrock OpenAI model support - [PR #17368](https://github.com/BerriAI/litellm/pull/17368)
    - Add support for file content download for Bedrock batches - [PR #17470](https://github.com/BerriAI/litellm/pull/17470)
    - Make streaming chunk size configurable in Bedrock API - [PR #17357](https://github.com/BerriAI/litellm/pull/17357)
    - Add experimental latest-user filtering for Bedrock - [PR #17282](https://github.com/BerriAI/litellm/pull/17282)
    - Handle Cohere v4 embed response dictionary format - [PR #17220](https://github.com/BerriAI/litellm/pull/17220)
    - Remove not compatible beta header from Bedrock - [PR #17301](https://github.com/BerriAI/litellm/pull/17301)
    - Add model price and details for Global Opus 4.5 Bedrock endpoint - [PR #17380](https://github.com/BerriAI/litellm/pull/17380)

- **[Gemini (Google AI Studio + Vertex AI)](../../docs/providers/gemini)**
    - Add better handling in image generation for Gemini models - [PR #17292](https://github.com/BerriAI/litellm/pull/17292)
    - Fix reasoning_content showing duplicate content in streaming responses - [PR #17266](https://github.com/BerriAI/litellm/pull/17266)
    - Handle partial JSON chunks after first valid chunk - [PR #17496](https://github.com/BerriAI/litellm/pull/17496)
    - Fix Gemini 3 last chunk thinking block - [PR #17403](https://github.com/BerriAI/litellm/pull/17403)
    - Fix Gemini image_tokens treated as text tokens in cost calculation - [PR #17554](https://github.com/BerriAI/litellm/pull/17554)
    - Make sure that media resolution is only for Gemini 3 model - [PR #17137](https://github.com/BerriAI/litellm/pull/17137)

- **[Vertex AI](../../docs/providers/vertex)**
    - Add Google Cloud Chirp3 HD support on /speech - [PR #17391](https://github.com/BerriAI/litellm/pull/17391)
    - Add BGE Embeddings support - [PR #17362](https://github.com/BerriAI/litellm/pull/17362)
    - Handle global location for Vertex AI image generation endpoint - [PR #17255](https://github.com/BerriAI/litellm/pull/17255)
    - Add Google Private API Endpoint to Vertex AI fields - [PR #17382](https://github.com/BerriAI/litellm/pull/17382)

- **[Z.AI (Zhipu AI)](../../docs/providers/zai)**
    - Add Z.AI as built-in provider - [PR #17307](https://github.com/BerriAI/litellm/pull/17307)

- **[GitHub Copilot](../../docs/providers/github_copilot)**
    - Add Embedding API support - [PR #17278](https://github.com/BerriAI/litellm/pull/17278)
    - Preserve encrypted_content in reasoning items for multi-turn conversations - [PR #17130](https://github.com/BerriAI/litellm/pull/17130)

- **[Databricks](../../docs/providers/databricks)**
    - Update Databricks model pricing and add new models - [PR #17277](https://github.com/BerriAI/litellm/pull/17277)

- **[OVHcloud](../../docs/providers/ovhcloud)**
    - Add support of audio transcription for OVHcloud - [PR #17305](https://github.com/BerriAI/litellm/pull/17305)

- **[Mistral](../../docs/providers/mistral)**
    - Add Mistral Large 3 model support - [PR #17547](https://github.com/BerriAI/litellm/pull/17547)

- **[Moonshot](../../docs/providers/moonshot)**
    - Fix missing Moonshot turbo models and fix incorrect pricing - [PR #17432](https://github.com/BerriAI/litellm/pull/17432)

- **[Together AI](../../docs/providers/togetherai)**
    - Add context window exception mapping for Together AI - [PR #17284](https://github.com/BerriAI/litellm/pull/17284)

- **[WatsonX](../../docs/providers/watsonx/index)**
    - Allow passing zen_api_key dynamically - [PR #16655](https://github.com/BerriAI/litellm/pull/16655)
    - Fix Watsonx Audio Transcription API - [PR #17326](https://github.com/BerriAI/litellm/pull/17326)
    - Fix audio transcriptions, don't force content type in request headers - [PR #17546](https://github.com/BerriAI/litellm/pull/17546)

- **[Fireworks AI](../../docs/providers/fireworks_ai)**
    - Add new model `fireworks_ai/kimi-k2-instruct-0905` - [PR #17328](https://github.com/BerriAI/litellm/pull/17328)
    - Add `fireworks/deepseek-v3p2` - [PR #17395](https://github.com/BerriAI/litellm/pull/17395)

- **[DeepSeek](../../docs/providers/deepseek)**
    - Support Deepseek 3.2 with Reasoning - [PR #17384](https://github.com/BerriAI/litellm/pull/17384)

- **[Nova Lite 2](../../docs/providers/bedrock)**
    - Add Nova Lite 2 reasoning support with reasoningConfig - [PR #17371](https://github.com/BerriAI/litellm/pull/17371)

- **[Ollama](../../docs/providers/ollama)**
    - Fix auth not working with ollama.com - [PR #17191](https://github.com/BerriAI/litellm/pull/17191)

- **[Groq](../../docs/providers/groq)**
    - Fix supports_response_schema before using json_tool_call workaround - [PR #17438](https://github.com/BerriAI/litellm/pull/17438)

- **[vLLM](../../docs/providers/vllm)**
    - Fix empty response + vLLM streaming - [PR #17516](https://github.com/BerriAI/litellm/pull/17516)

- **[Azure AI](../../docs/providers/azure_ai)**
    - Migrate Anthropic provider to Azure AI - [PR #17202](https://github.com/BerriAI/litellm/pull/17202)
    - Fix GA path for Azure OpenAI realtime models - [PR #17260](https://github.com/BerriAI/litellm/pull/17260)

- **[Bedrock TwelveLabs](../../docs/providers/bedrock#twelvelabs-pegasus---video-understanding)**
    - Add support for TwelveLabs Pegasus video understanding - [PR #17193](https://github.com/BerriAI/litellm/pull/17193)

### Bug Fixes

- **[Bedrock](../../docs/providers/bedrock)**
    - Fix extra_headers in messages API bedrock invoke - [PR #17271](https://github.com/BerriAI/litellm/pull/17271)
    - Fix Bedrock models in model map - [PR #17419](https://github.com/BerriAI/litellm/pull/17419)
    - Make Bedrock converse messages respect modify_params as expected - [PR #17427](https://github.com/BerriAI/litellm/pull/17427)
    - Fix Anthropic beta headers for Bedrock imported Qwen models - [PR #17467](https://github.com/BerriAI/litellm/pull/17467)
    - Preserve usage from JSON response for OpenAI provider in Bedrock - [PR #17589](https://github.com/BerriAI/litellm/pull/17589)

- **[SambaNova](../../docs/providers/sambanova)**
    - Fix acompletion throws error with SambaNova models - [PR #17217](https://github.com/BerriAI/litellm/pull/17217)

- **General**
    - Fix AttributeError when metadata is null in request body - [PR #17306](https://github.com/BerriAI/litellm/pull/17306)
    - Fix 500 error for malformed request - [PR #17291](https://github.com/BerriAI/litellm/pull/17291)
    - Respect custom LLM provider in header - [PR #17290](https://github.com/BerriAI/litellm/pull/17290)
    - Replace deprecated .dict() with .model_dump() in streaming_handler - [PR #17359](https://github.com/BerriAI/litellm/pull/17359)

---

## LLM API Endpoints

#### Features

- **[Responses API](../../docs/response_api)**
    - Add cost tracking for responses API - [PR #17258](https://github.com/BerriAI/litellm/pull/17258)
    - Map output_tokens_details of responses API to completion_tokens_details - [PR #17458](https://github.com/BerriAI/litellm/pull/17458)
    - Add image generation support for Responses API - [PR #16586](https://github.com/BerriAI/litellm/pull/16586)

- **[Batch API](../../docs/batches)**
    - Add vLLM batch+files API support - [PR #15823](https://github.com/BerriAI/litellm/pull/15823)
    - Fix optional parameter default value - [PR #17434](https://github.com/BerriAI/litellm/pull/17434)
    - Add status parameter as optional for FileObject - [PR #17431](https://github.com/BerriAI/litellm/pull/17431)

- **[Video Generation API](../../docs/videos)**
    - Add passthrough cost tracking for Veo - [PR #17296](https://github.com/BerriAI/litellm/pull/17296)

- **[OCR API](../../docs/ocr)**
    - Add missing OCR and aOCR to CallTypes enum - [PR #17435](https://github.com/BerriAI/litellm/pull/17435)

- **General**
    - Support routing to only websearch supported deployments - [PR #17500](https://github.com/BerriAI/litellm/pull/17500)

#### Bugs

- **General**
    - Fix streaming error validation - [PR #17242](https://github.com/BerriAI/litellm/pull/17242)
    - Add length validation for empty tool_calls in delta - [PR #17523](https://github.com/BerriAI/litellm/pull/17523)

---

## Management Endpoints / UI

#### Features

- **New Login Page**
    - New Login Page UI - [PR #17443](https://github.com/BerriAI/litellm/pull/17443)
    - Refactor /login route - [PR #17379](https://github.com/BerriAI/litellm/pull/17379)
    - Add auto_redirect_to_sso to UI Config - [PR #17399](https://github.com/BerriAI/litellm/pull/17399)
    - Add Auto Redirect to SSO to New Login Page - [PR #17451](https://github.com/BerriAI/litellm/pull/17451)

- **Customer (End User) Usage**
    - Customer (end user) Usage feature - [PR #17498](https://github.com/BerriAI/litellm/pull/17498)
    - Customer Usage UI - [PR #17506](https://github.com/BerriAI/litellm/pull/17506)
    - Add Info Banner for Customer Usage - [PR #17598](https://github.com/BerriAI/litellm/pull/17598)

- **Virtual Keys**
    - Standardize API Key vs Virtual Key in UI - [PR #17325](https://github.com/BerriAI/litellm/pull/17325)
    - Add User Alias Column to Internal User Table - [PR #17321](https://github.com/BerriAI/litellm/pull/17321)
    - Delete Credential Enhancements - [PR #17317](https://github.com/BerriAI/litellm/pull/17317)

- **Models + Endpoints**
    - Show all credential values on Edit Credential Modal - [PR #17397](https://github.com/BerriAI/litellm/pull/17397)
    - Change Edit Team Models Shown to Match Create Team - [PR #17394](https://github.com/BerriAI/litellm/pull/17394)
    - Support Images in Compare UI - [PR #17562](https://github.com/BerriAI/litellm/pull/17562)

- **Callbacks**
    - Show all callbacks on UI - [PR #16335](https://github.com/BerriAI/litellm/pull/16335)
    - Credentials to use React Query - [PR #17465](https://github.com/BerriAI/litellm/pull/17465)

- **Management Routes**
    - Allow admin viewer to access global tag usage - [PR #17501](https://github.com/BerriAI/litellm/pull/17501)
    - Allow wildcard routes for nonproxy admin (SCIM) - [PR #17178](https://github.com/BerriAI/litellm/pull/17178)
    - Return 404 when a user is not found on /user/info - [PR #16850](https://github.com/BerriAI/litellm/pull/16850)

- **OCI Configuration**
    - Enable Oracle Cloud Infrastructure configuration via UI - [PR #17159](https://github.com/BerriAI/litellm/pull/17159)

#### Bugs

- **UI Fixes**
    - Fix Request and Response Panel JSONViewer - [PR #17233](https://github.com/BerriAI/litellm/pull/17233)
    - Adding Button Loading States to Edit Settings - [PR #17236](https://github.com/BerriAI/litellm/pull/17236)
    - Fix Various Text, button state, and test changes - [PR #17237](https://github.com/BerriAI/litellm/pull/17237)
    - Fix Fallbacks Immediately Deleting before API resolves - [PR #17238](https://github.com/BerriAI/litellm/pull/17238)
    - Remove Feature Flags - [PR #17240](https://github.com/BerriAI/litellm/pull/17240)
    - Fix metadata tags and model name display in UI for Azure passthrough - [PR #17258](https://github.com/BerriAI/litellm/pull/17258)
    - Change labeling around Vertex Fields - [PR #17383](https://github.com/BerriAI/litellm/pull/17383)
    - Remove second scrollbar when sidebar is expanded + tooltip z index - [PR #17436](https://github.com/BerriAI/litellm/pull/17436)
    - Fix Select in Edit Membership Modal - [PR #17524](https://github.com/BerriAI/litellm/pull/17524)
    - Change useAuthorized Hook to redirect to new Login Page - [PR #17553](https://github.com/BerriAI/litellm/pull/17553)

- **SSO**
    - Fix the generic SSO provider - [PR #17227](https://github.com/BerriAI/litellm/pull/17227)
    - Clear SSO integration for all users - [PR #17287](https://github.com/BerriAI/litellm/pull/17287)
    - Fix SSO users not added to Entra synced team - [PR #17331](https://github.com/BerriAI/litellm/pull/17331)

- **Auth / JWT**
    - JWT Auth - Allow using regular OIDC flow with user info endpoints - [PR #17324](https://github.com/BerriAI/litellm/pull/17324)
    - Fix litellm user auth not passing issue - [PR #17342](https://github.com/BerriAI/litellm/pull/17342)
    - Add other routes in JWT auth - [PR #17345](https://github.com/BerriAI/litellm/pull/17345)
    - Fix new org team validate against org - [PR #17333](https://github.com/BerriAI/litellm/pull/17333)
    - Fix litellm_enterprise ensure imported routes exist - [PR #17337](https://github.com/BerriAI/litellm/pull/17337)
    - Use organization.members instead of deprecated organization field - [PR #17557](https://github.com/BerriAI/litellm/pull/17557)

- **Organizations/Teams**
    - Fix organization max budget not enforced - [PR #17334](https://github.com/BerriAI/litellm/pull/17334)
    - Fix budget update to allow null max_budget - [PR #17545](https://github.com/BerriAI/litellm/pull/17545)

---

## AI Integrations (2 new integrations)

### Logging (1 new integration)

#### New Integration

- **[Weave](../../docs/proxy/logging)**
    - Basic Weave OTEL integration - [PR #17439](https://github.com/BerriAI/litellm/pull/17439)

#### Improvements & Fixes

- **[DataDog](../../docs/proxy/logging#datadog)**
    - Fix Datadog callback regression when ddtrace is installed - [PR #17393](https://github.com/BerriAI/litellm/pull/17393)

- **[Arize Phoenix](../../docs/observability/arize_integration)**
    - Fix clean arize-phoenix traces - [PR #16611](https://github.com/BerriAI/litellm/pull/16611)

- **[MLflow](../../docs/proxy/logging#mlflow)**
    - Fix MLflow streaming spans for Anthropic passthrough - [PR #17288](https://github.com/BerriAI/litellm/pull/17288)

- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Fix Langfuse logger test mock setup - [PR #17591](https://github.com/BerriAI/litellm/pull/17591)

- **General**
    - Improve PII anonymization handling in logging callbacks - [PR #17207](https://github.com/BerriAI/litellm/pull/17207)

### Guardrails (1 new integration)

#### New Integration

- **[Generic Guardrail API](../../docs/adding_provider/generic_guardrail_api)**
    - Generic Guardrail API - allows guardrail providers to add INSTANT support for LiteLLM w/out PR to repo - [PR #17175](https://github.com/BerriAI/litellm/pull/17175)
    - Guardrails API V2 - user api key metadata, session id, specify input type (request/response), image support - [PR #17338](https://github.com/BerriAI/litellm/pull/17338)
    - Guardrails API - add streaming support - [PR #17400](https://github.com/BerriAI/litellm/pull/17400)
    - Guardrails API - support tool call checks on OpenAI `/chat/completions`, OpenAI `/responses`, Anthropic `/v1/messages` - [PR #17459](https://github.com/BerriAI/litellm/pull/17459)
    - Guardrails API - new `structured_messages` param - [PR #17518](https://github.com/BerriAI/litellm/pull/17518)
    - Correctly map a v1/messages call to the anthropic unified guardrail - [PR #17424](https://github.com/BerriAI/litellm/pull/17424)
    - Support during_call event type for unified guardrails - [PR #17514](https://github.com/BerriAI/litellm/pull/17514)

#### Improvements & Fixes

- **[Noma Guardrail](../../docs/proxy/guardrails/noma_security)**
    - Refactor Noma guardrail to use shared Responses transformation and include system instructions - [PR #17315](https://github.com/BerriAI/litellm/pull/17315)

- **[Presidio](../../docs/proxy/guardrails/pii_masking_v2)**
    - Handle empty content and error dict responses in guardrails - [PR #17489](https://github.com/BerriAI/litellm/pull/17489)
    - Fix Presidio guardrail test TypeError and license base64 decoding error - [PR #17538](https://github.com/BerriAI/litellm/pull/17538)

- **[Tool Permissions](../../docs/proxy/guardrails/tool_permission)**
    - Add regex-based tool_name/tool_type matching for tool-permission - [PR #17164](https://github.com/BerriAI/litellm/pull/17164)
    - Add images for tool permission guardrail documentation - [PR #17322](https://github.com/BerriAI/litellm/pull/17322)

- **[AIM Guardrails](../../docs/proxy/guardrails/aim_security)**
    - Fix AIM guardrail tests - [PR #17499](https://github.com/BerriAI/litellm/pull/17499)

- **[Bedrock Guardrails](../../docs/proxy/guardrails/bedrock)**
    - Fix Bedrock Guardrail indent and import - [PR #17378](https://github.com/BerriAI/litellm/pull/17378)

- **General Guardrails**
    - Mask all matching keywords in content filter - [PR #17521](https://github.com/BerriAI/litellm/pull/17521)
    - Ensure guardrail metadata is preserved in request_data - [PR #17593](https://github.com/BerriAI/litellm/pull/17593)
    - Fix apply_guardrail method and improve test isolation - [PR #17555](https://github.com/BerriAI/litellm/pull/17555)

### Secret Managers

- **[CyberArk](../../docs/secret_managers/cyberark)**
    - Allow setting SSL verify to false - [PR #17433](https://github.com/BerriAI/litellm/pull/17433)

- **General**
    - Make email and secret manager operations independent in key management hooks - [PR #17551](https://github.com/BerriAI/litellm/pull/17551)

---

## Spend Tracking, Budgets and Rate Limiting

- **Rate Limiting**
    - Parallel Request Limiter with /messages - [PR #17426](https://github.com/BerriAI/litellm/pull/17426)
    - Allow using dynamic rate limit/priority reservation on teams - [PR #17061](https://github.com/BerriAI/litellm/pull/17061)
    - Dynamic Rate Limiter - Fix token count increases/decreases by 1 instead of actual count + Redis TTL - [PR #17558](https://github.com/BerriAI/litellm/pull/17558)

- **Spend Logs**
    - Deprecate `spend/logs` & add `spend/logs/v2` - [PR #17167](https://github.com/BerriAI/litellm/pull/17167)
    - Optimize SpendLogs queries to use timestamp filtering for index usage - [PR #17504](https://github.com/BerriAI/litellm/pull/17504)

- **Enforce User Param**
    - Enforce support of enforce_user_param to OpenAI post endpoints - [PR #17407](https://github.com/BerriAI/litellm/pull/17407)

---

## MCP Gateway

- **MCP Configuration**
    - Remove URL format validation for MCP server endpoints - [PR #17270](https://github.com/BerriAI/litellm/pull/17270)
    - Add stack trace to MCP error message - [PR #17269](https://github.com/BerriAI/litellm/pull/17269)

- **MCP Tool Results**
    - Preserve tool metadata in CallToolResult - [PR #17561](https://github.com/BerriAI/litellm/pull/17561)

---

## Agent Gateway (A2A)

- **Agent Invocation**
    - Allow invoking agents through AI Gateway - [PR #17440](https://github.com/BerriAI/litellm/pull/17440)
    - Allow tracking request/response in "Logs" Page - [PR #17449](https://github.com/BerriAI/litellm/pull/17449)

- **Agent Access Control**
    - Enforce Allowed agents by key, team + add agent access groups on backend - [PR #17502](https://github.com/BerriAI/litellm/pull/17502)

- **Agent Gateway UI**
    - Allow testing agents on UI - [PR #17455](https://github.com/BerriAI/litellm/pull/17455)
    - Set allowed agents by key, team - [PR #17511](https://github.com/BerriAI/litellm/pull/17511)

---

## Performance / Loadbalancing / Reliability improvements

- **Audio/Speech Performance**
    - Fix `/audio/speech` performance by using `shared_sessions` - [PR #16739](https://github.com/BerriAI/litellm/pull/16739)

- **Memory Optimization**
    - Prevent memory leak in aiohttp connection pooling - [PR #17388](https://github.com/BerriAI/litellm/pull/17388)
    - Lazy-load utils to reduce memory + import time - [PR #17171](https://github.com/BerriAI/litellm/pull/17171)

- **Database**
    - Update default database connection number - [PR #17353](https://github.com/BerriAI/litellm/pull/17353)
    - Update default proxy_batch_write_at number - [PR #17355](https://github.com/BerriAI/litellm/pull/17355)
    - Add background health checks to db - [PR #17528](https://github.com/BerriAI/litellm/pull/17528)

- **Proxy Caching**
    - Fix proxy caching between requests in aiohttp transport - [PR #17122](https://github.com/BerriAI/litellm/pull/17122)

- **Session Management**
    - Fix session consistency, move Lasso API version away from source code - [PR #17316](https://github.com/BerriAI/litellm/pull/17316)
    - Conditionally pass enable_cleanup_closed to aiohttp TCPConnector - [PR #17367](https://github.com/BerriAI/litellm/pull/17367)

- **Vector Store**
    - Fix vector store configuration synchronization failure - [PR #17525](https://github.com/BerriAI/litellm/pull/17525)

---

## Documentation Updates

- **Provider Documentation**
    - Add Azure AI Foundry documentation for Claude models - [PR #17104](https://github.com/BerriAI/litellm/pull/17104)
    - Document responses and embedding API for GitHub Copilot - [PR #17456](https://github.com/BerriAI/litellm/pull/17456)
    - Add gpt-5.1-codex-max to OpenAI provider documentation - [PR #17602](https://github.com/BerriAI/litellm/pull/17602)
    - Update Instructions For Phoenix Integration - [PR #17373](https://github.com/BerriAI/litellm/pull/17373)

- **Guides**
    - Add guide on how to debug gateway error vs provider error - [PR #17387](https://github.com/BerriAI/litellm/pull/17387)
    - Agent Gateway documentation - [PR #17454](https://github.com/BerriAI/litellm/pull/17454)
    - A2A Permission management documentation - [PR #17515](https://github.com/BerriAI/litellm/pull/17515)
    - Update docs to link agent hub - [PR #17462](https://github.com/BerriAI/litellm/pull/17462)

- **Projects**
    - Add Google ADK and Harbor to projects - [PR #17352](https://github.com/BerriAI/litellm/pull/17352)
    - Add Microsoft Agent Lightning to projects - [PR #17422](https://github.com/BerriAI/litellm/pull/17422)

- **Cleanup**
    - Cleanup: Remove orphan docs pages and Docusaurus template files - [PR #17356](https://github.com/BerriAI/litellm/pull/17356)
    - Remove `source .env` from docs - [PR #17466](https://github.com/BerriAI/litellm/pull/17466)

---

## Infrastructure / CI/CD

- **Helm Chart**
    - Add ingress-only labels - [PR #17348](https://github.com/BerriAI/litellm/pull/17348)

- **Docker**
    - Add retry logic to apk package installation in Dockerfile.non_root - [PR #17596](https://github.com/BerriAI/litellm/pull/17596)
    - Chainguard fixes - [PR #17406](https://github.com/BerriAI/litellm/pull/17406)

- **OpenAPI Schema**
    - Refactor add_schema_to_components to move definitions to components/schemas - [PR #17389](https://github.com/BerriAI/litellm/pull/17389)

- **Security**
    - Fix security vulnerability: update mdast-util-to-hast to 13.2.1 - [PR #17601](https://github.com/BerriAI/litellm/pull/17601)
    - Bump jws from 3.2.2 to 3.2.3 - [PR #17494](https://github.com/BerriAI/litellm/pull/17494)

---

## New Contributors

* @weichiet made their first contribution in [PR #17242](https://github.com/BerriAI/litellm/pull/17242)
* @AndyForest made their first contribution in [PR #17220](https://github.com/BerriAI/litellm/pull/17220)
* @omkar806 made their first contribution in [PR #17217](https://github.com/BerriAI/litellm/pull/17217)
* @v0rtex20k made their first contribution in [PR #17178](https://github.com/BerriAI/litellm/pull/17178)
* @hxomer made their first contribution in [PR #17207](https://github.com/BerriAI/litellm/pull/17207)
* @orgersh92 made their first contribution in [PR #17316](https://github.com/BerriAI/litellm/pull/17316)
* @dannykopping made their first contribution in [PR #17313](https://github.com/BerriAI/litellm/pull/17313)
* @rioiart made their first contribution in [PR #17333](https://github.com/BerriAI/litellm/pull/17333)
* @codgician made their first contribution in [PR #17278](https://github.com/BerriAI/litellm/pull/17278)
* @epistoteles made their first contribution in [PR #17277](https://github.com/BerriAI/litellm/pull/17277)
* @kothamah made their first contribution in [PR #17368](https://github.com/BerriAI/litellm/pull/17368)
* @flozonn made their first contribution in [PR #17371](https://github.com/BerriAI/litellm/pull/17371)
* @richardmcsong made their first contribution in [PR #17389](https://github.com/BerriAI/litellm/pull/17389)
* @matt-greathouse made their first contribution in [PR #17384](https://github.com/BerriAI/litellm/pull/17384)
* @mossbanay made their first contribution in [PR #17380](https://github.com/BerriAI/litellm/pull/17380)
* @mhielpos-asapp made their first contribution in [PR #17376](https://github.com/BerriAI/litellm/pull/17376)
* @Joilence made their first contribution in [PR #17367](https://github.com/BerriAI/litellm/pull/17367)
* @deepaktammali made their first contribution in [PR #17357](https://github.com/BerriAI/litellm/pull/17357)
* @axiomofjoy made their first contribution in [PR #16611](https://github.com/BerriAI/litellm/pull/16611)
* @DevajMody made their first contribution in [PR #17445](https://github.com/BerriAI/litellm/pull/17445)
* @andrewtruong made their first contribution in [PR #17439](https://github.com/BerriAI/litellm/pull/17439)
* @AnasAbdelR made their first contribution in [PR #17490](https://github.com/BerriAI/litellm/pull/17490)
* @dominicfeliton made their first contribution in [PR #17516](https://github.com/BerriAI/litellm/pull/17516)
* @kristianmitk made their first contribution in [PR #17504](https://github.com/BerriAI/litellm/pull/17504)
* @rgshr made their first contribution in [PR #17130](https://github.com/BerriAI/litellm/pull/17130)
* @dominicfallows made their first contribution in [PR #17489](https://github.com/BerriAI/litellm/pull/17489)
* @irfansofyana made their first contribution in [PR #17467](https://github.com/BerriAI/litellm/pull/17467)
* @GusBricker made their first contribution in [PR #17191](https://github.com/BerriAI/litellm/pull/17191)
* @OlivverX made their first contribution in [PR #17255](https://github.com/BerriAI/litellm/pull/17255)
* @withsmilo made their first contribution in [PR #17585](https://github.com/BerriAI/litellm/pull/17585)

---

## Full Changelog

**[View complete changelog on GitHub](https://github.com/BerriAI/litellm/compare/v1.80.7-nightly...v1.80.8)**

