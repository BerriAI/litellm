---
title: "v1.80.5-stable - Gemini 3.0 Support"
slug: "v1-80-5"
date: 2025-11-22T10:00:00
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
docker.litellm.ai/berriai/litellm:v1.80.5-stable
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.80.5
```

</TabItem>
</Tabs>

---

## Key Highlights

- **Gemini 3** - [Day-0 support for Gemini 3 models with thought signatures](../../blog/gemini_3)
- **Prompt Management** - [Full prompt versioning support with UI for editing, testing, and version history](../../docs/proxy/litellm_prompt_management)
- **MCP Hub** - [Publish and discover MCP servers within your organization](../../docs/proxy/ai_hub#mcp-servers)
- **Model Compare UI** - [Side-by-side model comparison interface for testing](../../docs/proxy/model_compare_ui)
- **Batch API Spend Tracking** - [Granular spend tracking with custom metadata for batch and file creation requests](../../docs/proxy/cost_tracking#-custom-spend-log-metadata)
- **AWS IAM Secret Manager** - [IAM role authentication support for AWS Secret Manager](../../docs/secret_managers/aws_secret_manager#iam-role-assumption)
- **Logging Callback Controls** - [Admin-level controls to prevent callers from disabling logging callbacks in compliance environments](../../docs/proxy/dynamic_logging#disabling-dynamic-callback-management-enterprise)
- **Proxy CLI JWT Authentication** - [Enable developers to authenticate to LiteLLM AI Gateway using the Proxy CLI](../../docs/proxy/cli_sso)
- **Batch API Routing** - [Route batch operations to different provider accounts using model-specific credentials from your config.yaml](../../docs/batches#multi-account--model-based-routing)

---

### Prompt Management

<Image 
  img={require('../../img/prompt_history.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

<br/>
<br/>

This release introduces **LiteLLM Prompt Studio** - a comprehensive prompt management solution built directly into the LiteLLM UI. Create, test, and version your prompts without leaving your browser.

You can now do the following on LiteLLM Prompt Studio:

- **Create & Test Prompts**: Build prompts with developer messages (system instructions) and test them in real-time with an interactive chat interface
- **Dynamic Variables**: Use `{{variable_name}}` syntax to create reusable prompt templates with automatic variable detection
- **Version Control**: Automatic versioning for every prompt update with complete version history tracking and rollback capabilities
- **Prompt Studio**: Edit prompts in a dedicated studio environment with live testing and preview

**API Integration:**

Use your prompts in any application with simple API calls:

```python
response = client.chat.completions.create(
    model="gpt-4",
    extra_body={
        "prompt_id": "your-prompt-id",
        "prompt_version": 2,  # Optional: specify version
        "prompt_variables": {"name": "value"}  # Optional: pass variables
    }
)
```

Get started here: [LiteLLM Prompt Management Documentation](../../docs/proxy/litellm_prompt_management)

---

### Performance – `/realtime` 182× Lower p99 Latency

This update reduces `/realtime` latency by removing redundant encodings on the hot path, reusing shared SSL contexts, and caching formatting strings that were being regenerated twice per request despite rarely changing.

#### Results

| Metric          | Before    | After     | Improvement                |
| --------------- | --------- | --------- | -------------------------- |
| Median latency  | 2,200 ms  | **59 ms** | **−97% (~37× faster)**     |
| p95 latency     | 8,500 ms  | **67 ms** | **−99% (~127× faster)**    |
| p99 latency     | 18,000 ms | **99 ms** | **−99% (~182× faster)**    |
| Average latency | 3,214 ms  | **63 ms** | **−98% (~51× faster)**     |
| RPS             | 165       | **1,207** | **+631% (~7.3× increase)** |


#### Test Setup

| Category | Specification |
|----------|---------------|
| **Load Testing** | Locust: 1,000 concurrent users, 500 ramp-up |
| **System** | 4 vCPUs, 8 GB RAM, 4 workers, 4 instances |
| **Database** | PostgreSQL (Redis unused) |
| **Configuration** | [config.yaml](https://gist.github.com/AlexsanderHamir/420fb44c31c00b4f17a99588637f01ec) |
| **Load Script** | [no_cache_hits.py](https://gist.github.com/AlexsanderHamir/73b83ada21d9b84d4fe09665cf1745f5) |

---

### Model Compare UI

New interactive playground UI enables side-by-side comparison of multiple LLM models, making it easy to evaluate and compare model responses.

**Features:**
- Compare responses from multiple models in real-time
- Side-by-side view with synchronized scrolling
- Support for all LiteLLM-supported models
- Cost tracking per model
- Response time comparison
- Pre-configured prompts for quick and easy testing

**Details:**

- **Parameterization**: Configure API keys, endpoints, models, and model parameters, as well as interaction types (chat completions, embeddings, etc.)

- **Model Comparison**: Compare up to 3 different models simultaneously with side-by-side response views

- **Comparison Metrics**: View detailed comparison information including:

  - Time To First Token
  - Input / Output / Reasoning Tokens
  - Total Latency
  - Cost (if enabled in config)

- **Safety Filters**: Configure and test guardrails (safety filters) directly in the playground interface

[Get Started with Model Compare](../../docs/proxy/model_compare_ui)

## New Providers and Endpoints

### New Providers

| Provider | Supported Endpoints | Description |
| -------- | ------------------- | ----------- |
| **[Docker Model Runner](../../docs/providers/docker_model_runner)** | `/v1/chat/completions` | Run LLM models in Docker containers |

---

## New Models / Updated Models

#### New Model Support

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| Azure | `azure/gpt-5.1` | 272K | $1.38 | $11.00 | Reasoning, vision, PDF input, responses API |
| Azure | `azure/gpt-5.1-2025-11-13` | 272K | $1.38 | $11.00 | Reasoning, vision, PDF input, responses API |
| Azure | `azure/gpt-5.1-codex` | 272K | $1.38 | $11.00 | Responses API, reasoning, vision |
| Azure | `azure/gpt-5.1-codex-2025-11-13` | 272K | $1.38 | $11.00 | Responses API, reasoning, vision |
| Azure | `azure/gpt-5.1-codex-mini` | 272K | $0.275 | $2.20 | Responses API, reasoning, vision |
| Azure | `azure/gpt-5.1-codex-mini-2025-11-13` | 272K | $0.275 | $2.20 | Responses API, reasoning, vision |
| Azure EU | `azure/eu/gpt-5-2025-08-07` | 272K | $1.375 | $11.00 | Reasoning, vision, PDF input |
| Azure EU | `azure/eu/gpt-5-mini-2025-08-07` | 272K | $0.275 | $2.20 | Reasoning, vision, PDF input |
| Azure EU | `azure/eu/gpt-5-nano-2025-08-07` | 272K | $0.055 | $0.44 | Reasoning, vision, PDF input |
| Azure EU | `azure/eu/gpt-5.1` | 272K | $1.38 | $11.00 | Reasoning, vision, PDF input, responses API |
| Azure EU | `azure/eu/gpt-5.1-codex` | 272K | $1.38 | $11.00 | Responses API, reasoning, vision |
| Azure EU | `azure/eu/gpt-5.1-codex-mini` | 272K | $0.275 | $2.20 | Responses API, reasoning, vision |
| Gemini | `gemini-3-pro-preview` | 2M | $1.25 | $5.00 | Reasoning, vision, function calling |
| Gemini | `gemini-3-pro-image` | 2M | $1.25 | $5.00 | Image generation, reasoning |
| OpenRouter | `openrouter/deepseek/deepseek-v3p1-terminus` | 164K | $0.20 | $0.40 | Function calling, reasoning |
| OpenRouter | `openrouter/moonshot/kimi-k2-instruct` | 262K | $0.60 | $2.50 | Function calling, web search |
| OpenRouter | `openrouter/gemini/gemini-3-pro-preview` | 2M | $1.25 | $5.00 | Reasoning, vision, function calling |
| XAI | `xai/grok-4.1-fast` | 2M | $0.20 | $0.50 | Reasoning, function calling |
| Together AI | `together_ai/z-ai/glm-4.6` | 203K | $0.40 | $1.75 | Function calling, reasoning |
| Cerebras | `cerebras/gpt-oss-120b` | 131K | $0.60 | $0.60 | Function calling |
| Bedrock | `anthropic.claude-sonnet-4-5-20250929-v1:0` | 200K | $3.00 | $15.00 | Computer use, reasoning, vision |

#### Features

- **[Gemini (Google AI Studio + Vertex AI)](../../docs/providers/gemini)**
    - Add Day 0 gemini-3-pro-preview support - [PR #16719](https://github.com/BerriAI/litellm/pull/16719)
    - Add support for Gemini 3 Pro Image model - [PR #16938](https://github.com/BerriAI/litellm/pull/16938)
    - Add reasoning_content to streaming responses with tools enabled - [PR #16854](https://github.com/BerriAI/litellm/pull/16854)
    - Add includeThoughts=True for Gemini 3 reasoning_effort - [PR #16838](https://github.com/BerriAI/litellm/pull/16838)
    - Support thought signatures for Gemini 3 in responses API - [PR #16872](https://github.com/BerriAI/litellm/pull/16872)
    - Correct wrong system message handling for gemma - [PR #16767](https://github.com/BerriAI/litellm/pull/16767)
    - Gemini 3 Pro Image: capture image_tokens and support cost_per_output_image - [PR #16912](https://github.com/BerriAI/litellm/pull/16912)
    - Fix missing costs for gemini-2.5-flash-image - [PR #16882](https://github.com/BerriAI/litellm/pull/16882)
    - Gemini 3 thought signatures in tool call id - [PR #16895](https://github.com/BerriAI/litellm/pull/16895)

- **[Azure](../../docs/providers/azure)**
    - Add azure gpt-5.1 models - [PR #16817](https://github.com/BerriAI/litellm/pull/16817)
    - Add Azure models 2025 11 to cost maps - [PR #16762](https://github.com/BerriAI/litellm/pull/16762)
    - Update Azure Pricing - [PR #16371](https://github.com/BerriAI/litellm/pull/16371)
    - Add SSML Support for Azure Text-to-Speech (AVA) - [PR #16747](https://github.com/BerriAI/litellm/pull/16747)

- **[OpenAI](../../docs/providers/openai)**
    - Support GPT-5.1 reasoning.effort='none' in proxy - [PR #16745](https://github.com/BerriAI/litellm/pull/16745)
    - Add gpt-5.1-codex and gpt-5.1-codex-mini models to documentation - [PR #16735](https://github.com/BerriAI/litellm/pull/16735)
    - Inherit BaseVideoConfig to enable async content response for OpenAI video - [PR #16708](https://github.com/BerriAI/litellm/pull/16708)

- **[Anthropic](../../docs/providers/anthropic)**
    - Add support for `strict` parameter in Anthropic tool schemas - [PR #16725](https://github.com/BerriAI/litellm/pull/16725)
    - Add image as url support to anthropic - [PR #16868](https://github.com/BerriAI/litellm/pull/16868)
    - Add thought signature support to v1/messages api - [PR #16812](https://github.com/BerriAI/litellm/pull/16812)
    - Anthropic - support Structured Outputs `output_format` for Claude 4.5 sonnet and Opus 4.1 - [PR #16949](https://github.com/BerriAI/litellm/pull/16949)

- **[Bedrock](../../docs/providers/bedrock)**
    - Haiku 4.5 correct Bedrock configs - [PR #16732](https://github.com/BerriAI/litellm/pull/16732)
    - Ensure consistent chunk IDs in Bedrock streaming responses - [PR #16596](https://github.com/BerriAI/litellm/pull/16596)
    - Add Claude 4.5 to US Gov Cloud - [PR #16957](https://github.com/BerriAI/litellm/pull/16957)
    - Fix images being dropped from tool results for bedrock - [PR #16492](https://github.com/BerriAI/litellm/pull/16492)

- **[Vertex AI](../../docs/providers/vertex)**
    - Add Vertex AI Image Edit Support - [PR #16828](https://github.com/BerriAI/litellm/pull/16828)
    - Update veo 3 pricing and add prod models - [PR #16781](https://github.com/BerriAI/litellm/pull/16781)
    - Fix Video download for veo3 - [PR #16875](https://github.com/BerriAI/litellm/pull/16875)

- **[Snowflake](../../docs/providers/snowflake)**
    - Snowflake provider support: added embeddings, PAT, account_id - [PR #15727](https://github.com/BerriAI/litellm/pull/15727)

- **[OCI](../../docs/providers/oci)**
    - Add oci_endpoint_id Parameter for OCI Dedicated Endpoints - [PR #16723](https://github.com/BerriAI/litellm/pull/16723)

- **[XAI](../../docs/providers/xai)**
    - Add support for Grok 4.1 Fast models - [PR #16936](https://github.com/BerriAI/litellm/pull/16936)

- **[Together AI](../../docs/providers/togetherai)**
    - Add GLM 4.6 from together.ai - [PR #16942](https://github.com/BerriAI/litellm/pull/16942)

- **[Cerebras](../../docs/providers/cerebras)**
    - Fix Cerebras GPT-OSS-120B model name - [PR #16939](https://github.com/BerriAI/litellm/pull/16939)

### Bug Fixes

- **[OpenAI](../../docs/providers/openai)**
    - Fix for 16863 - openai conversion from responses to completions - [PR #16864](https://github.com/BerriAI/litellm/pull/16864)
    - Revert "Make all gpt-5 and reasoning models to responses by default" - [PR #16849](https://github.com/BerriAI/litellm/pull/16849)

- **General**
    - Get custom_llm_provider from query param - [PR #16731](https://github.com/BerriAI/litellm/pull/16731)
    - Fix optional param mapping - [PR #16852](https://github.com/BerriAI/litellm/pull/16852)
    - Add None check for litellm_params - [PR #16754](https://github.com/BerriAI/litellm/pull/16754)

---

## LLM API Endpoints

#### Features

- **[Responses API](../../docs/response_api)**
    - Add Responses API support for gpt-5.1-codex model - [PR #16845](https://github.com/BerriAI/litellm/pull/16845)
    - Add managed files support for responses API - [PR #16733](https://github.com/BerriAI/litellm/pull/16733)
    - Add extra_body support for response supported api params from chat completion - [PR #16765](https://github.com/BerriAI/litellm/pull/16765)

- **[Batch API](../../docs/batches)**
    - Support /delete for files + support /cancel for batches - [PR #16387](https://github.com/BerriAI/litellm/pull/16387)
    - Add config based routing support for batches and files - [PR #16872](https://github.com/BerriAI/litellm/pull/16872)
    - Populate spend_logs_metadata in batch and files endpoints - [PR #16921](https://github.com/BerriAI/litellm/pull/16921)

- **[Search APIs](../../docs/search)**
    - Search APIs - error in firecrawl-search "Invalid request body" - [PR #16943](https://github.com/BerriAI/litellm/pull/16943)

- **[Vector Stores](../../docs/vector_stores)**
    - Fix vector store create issue - [PR #16804](https://github.com/BerriAI/litellm/pull/16804)
    - Team vector-store permissions now respected for key access - [PR #16639](https://github.com/BerriAI/litellm/pull/16639)

- **[Audio Transcription](../../docs/audio_transcription)**
    - Fix audio transcription cost tracking - [PR #16478](https://github.com/BerriAI/litellm/pull/16478)
    - Add missing shared_sessions to audio/transcriptions - [PR #16858](https://github.com/BerriAI/litellm/pull/16858)

- **[Video Generation API](../../docs/video_generation)**
    - Fix videos tagging - [PR #16770](https://github.com/BerriAI/litellm/pull/16770)

#### Bugs

- **General**
    - Responses API cost tracking with custom deployment names - [PR #16778](https://github.com/BerriAI/litellm/pull/16778)
    - Trim logged response strings in spend-logs - [PR #16654](https://github.com/BerriAI/litellm/pull/16654)

---

## Management Endpoints / UI

#### Features

- **Proxy CLI Auth**
    - Allow using JWTs for signing in with Proxy CLI - [PR #16756](https://github.com/BerriAI/litellm/pull/16756)

- **Virtual Keys**
    - Fix Key Model Alias Not Working - [PR #16896](https://github.com/BerriAI/litellm/pull/16896)

- **Models + Endpoints**
    - Add additional model settings to chat models in test key - [PR #16793](https://github.com/BerriAI/litellm/pull/16793)
    - Deactivate delete button on model table for config models - [PR #16787](https://github.com/BerriAI/litellm/pull/16787)
    - Change Public Model Hub to use proxyBaseUrl - [PR #16892](https://github.com/BerriAI/litellm/pull/16892)
    - Add JSON Viewer to request/response panel - [PR #16687](https://github.com/BerriAI/litellm/pull/16687)
    - Standarize icon images - [PR #16837](https://github.com/BerriAI/litellm/pull/16837)

- **Teams**
    - Teams table empty state - [PR #16738](https://github.com/BerriAI/litellm/pull/16738)

- **Fallbacks**
    - Fallbacks icon button tooltips and delete with friction - [PR #16737](https://github.com/BerriAI/litellm/pull/16737)

- **MCP Servers**
    - Delete user and MCP Server Modal, MCP Table Tooltips - [PR #16751](https://github.com/BerriAI/litellm/pull/16751)

- **Callbacks**
    - Expose backend endpoint for callbacks settings - [PR #16698](https://github.com/BerriAI/litellm/pull/16698)
    - Edit add callbacks route to use data from backend - [PR #16699](https://github.com/BerriAI/litellm/pull/16699)

- **Usage & Analytics**
    - Allow partial matches for user ID in User Table - [PR #16952](https://github.com/BerriAI/litellm/pull/16952)

- **General UI**
    - Allow setting base_url in API reference docs - [PR #16674](https://github.com/BerriAI/litellm/pull/16674)
    - Change /public fields to honor server root path - [PR #16930](https://github.com/BerriAI/litellm/pull/16930)
    - Correct ui build - [PR #16702](https://github.com/BerriAI/litellm/pull/16702)
    - Enable automatic dark/light mode based on system preference - [PR #16748](https://github.com/BerriAI/litellm/pull/16748)

#### Bugs

- **UI Fixes**
    - Fix flaky tests due to antd Notification Manager - [PR #16740](https://github.com/BerriAI/litellm/pull/16740)
    - Fix UI MCP Tool Test Regression - [PR #16695](https://github.com/BerriAI/litellm/pull/16695)
    - Fix edit logging settings not appearing - [PR #16798](https://github.com/BerriAI/litellm/pull/16798)
    - Add css to truncate long request ids in request viewer - [PR #16665](https://github.com/BerriAI/litellm/pull/16665)
    - Remove azure/ prefix in Placeholder for Azure in Add Model - [PR #16597](https://github.com/BerriAI/litellm/pull/16597)
    - Remove UI Session Token from user/info return - [PR #16851](https://github.com/BerriAI/litellm/pull/16851)
    - Remove console logs and errors from model tab - [PR #16455](https://github.com/BerriAI/litellm/pull/16455)
    - Change Bulk Invite User Roles to Match Backend - [PR #16906](https://github.com/BerriAI/litellm/pull/16906)
    - Mock Tremor's Tooltip to Fix Flaky UI Tests - [PR #16786](https://github.com/BerriAI/litellm/pull/16786)
    - Fix e2e ui playwright test - [PR #16799](https://github.com/BerriAI/litellm/pull/16799)
    - Fix Tests in CI/CD - [PR #16972](https://github.com/BerriAI/litellm/pull/16972)

- **SSO**
    - Ensure `role` from SSO provider is used when a user is inserted onto LiteLLM - [PR #16794](https://github.com/BerriAI/litellm/pull/16794)
    - Docs - SSO - Manage User Roles via Azure App Roles - [PR #16796](https://github.com/BerriAI/litellm/pull/16796)

- **Auth**
    - Ensure Team Tags works when using JWT Auth - [PR #16797](https://github.com/BerriAI/litellm/pull/16797)
    - Fix key never expires - [PR #16692](https://github.com/BerriAI/litellm/pull/16692)

- **Swagger UI**
    - Fixes Swagger UI resolver errors for chat completion endpoints caused by Pydantic v2 `$defs` not being properly exposed in the OpenAPI schema - [PR #16784](https://github.com/BerriAI/litellm/pull/16784)

---

## AI Integrations

### Logging

- **[Arize Phoenix](../../docs/observability/arize_phoenix)**
    - Fix arize phoenix logging - [PR #16301](https://github.com/BerriAI/litellm/pull/16301)
    - Arize Phoenix - root span logging - [PR #16949](https://github.com/BerriAI/litellm/pull/16949)

- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Filter secret fields form Langfuse - [PR #16842](https://github.com/BerriAI/litellm/pull/16842)

- **General**
    - Exclude litellm_credential_name from Sensitive Data Masker (Updated) - [PR #16958](https://github.com/BerriAI/litellm/pull/16958)
    - Allow admins to disable, dynamic callback controls - [PR #16750](https://github.com/BerriAI/litellm/pull/16750)

### Guardrails

- **[IBM Guardrails](../../docs/proxy/guardrails)**
    - Fix IBM Guardrails optional params, add extra_headers field - [PR #16771](https://github.com/BerriAI/litellm/pull/16771)

- **[Noma Guardrail](../../docs/proxy/guardrails)**
    - Use LiteLLM key alias as fallback Noma applicationId in NomaGuardrail - [PR #16832](https://github.com/BerriAI/litellm/pull/16832)
    - Allow custom violation message for tool-permission guardrail - [PR #16916](https://github.com/BerriAI/litellm/pull/16916)

- **[Grayswan Guardrail](../../docs/proxy/guardrails)**
    - Grayswan guardrail passthrough on flagged - [PR #16891](https://github.com/BerriAI/litellm/pull/16891)

- **General Guardrails**
    - Fix prompt injection not working - [PR #16701](https://github.com/BerriAI/litellm/pull/16701)

### Prompt Management

- **[Prompt Management](../../docs/proxy/prompt_management)**
    - Allow specifying just prompt_id in a request to a model - [PR #16834](https://github.com/BerriAI/litellm/pull/16834)
    - Add support for versioning prompts - [PR #16836](https://github.com/BerriAI/litellm/pull/16836)
    - Allow storing prompt version in DB - [PR #16848](https://github.com/BerriAI/litellm/pull/16848)
    - Add UI for editing the prompts - [PR #16853](https://github.com/BerriAI/litellm/pull/16853)
    - Allow testing prompts with Chat UI - [PR #16898](https://github.com/BerriAI/litellm/pull/16898)
    - Allow viewing version history - [PR #16901](https://github.com/BerriAI/litellm/pull/16901)
    - Allow specifying prompt version in code - [PR #16929](https://github.com/BerriAI/litellm/pull/16929)
    - UI, allow seeing model, prompt id for Prompt - [PR #16932](https://github.com/BerriAI/litellm/pull/16932)
    - Show "get code" section for prompt management + minor polish of showing version history - [PR #16941](https://github.com/BerriAI/litellm/pull/16941)

### Secret Managers

- **[AWS Secrets Manager](../../docs/secret_managers)**
    - Adds IAM role assumption support for AWS Secret Manager - [PR #16887](https://github.com/BerriAI/litellm/pull/16887)

---

## MCP Gateway

- **MCP Hub** - Publish/discover MCP Servers within a company - [PR #16857](https://github.com/BerriAI/litellm/pull/16857)
- **MCP Resources** - MCP resources support - [PR #16800](https://github.com/BerriAI/litellm/pull/16800)
- **MCP OAuth** - Docs - mcp oauth flow details - [PR #16742](https://github.com/BerriAI/litellm/pull/16742)
- **MCP Lifecycle** - Drop MCPClient.connect and use run_with_session lifecycle - [PR #16696](https://github.com/BerriAI/litellm/pull/16696)
- **MCP Server IDs** - Add mcp server ids - [PR #16904](https://github.com/BerriAI/litellm/pull/16904)
- **MCP URL Format** - Fix mcp url format - [PR #16940](https://github.com/BerriAI/litellm/pull/16940)


---

## Performance / Loadbalancing / Reliability improvements

- **Realtime Endpoint Performance** - Fix bottlenecks degrading realtime endpoint performance - [PR #16670](https://github.com/BerriAI/litellm/pull/16670)
- **SSL Context Caching** - Cache SSL contexts to prevent excessive memory allocation - [PR #16955](https://github.com/BerriAI/litellm/pull/16955)
- **Cache Optimization** - Fix cache cooldown key generation - [PR #16954](https://github.com/BerriAI/litellm/pull/16954)
- **Router Cache** - Fix routing for requests with same cacheable prefix but different user messages - [PR #16951](https://github.com/BerriAI/litellm/pull/16951)
- **Redis Event Loop** - Fix redis event loop closed at first call - [PR #16913](https://github.com/BerriAI/litellm/pull/16913)
- **Dependency Management** - Upgrade pydantic to version 2.11.0 - [PR #16909](https://github.com/BerriAI/litellm/pull/16909)

---

## Documentation Updates

- **Provider Documentation**
    - Add missing details to benchmark comparison - [PR #16690](https://github.com/BerriAI/litellm/pull/16690)
    - Fix anthropic pass-through endpoint - [PR #16883](https://github.com/BerriAI/litellm/pull/16883)
    - Cleanup repo and improve AI docs - [PR #16775](https://github.com/BerriAI/litellm/pull/16775)

- **API Documentation**
    - Add docs related to openai metadata - [PR #16872](https://github.com/BerriAI/litellm/pull/16872)
    - Update docs with all supported endpoints and cost tracking - [PR #16872](https://github.com/BerriAI/litellm/pull/16872)

- **General Documentation**
    - Add mini-swe-agent to Projects built on LiteLLM - [PR #16971](https://github.com/BerriAI/litellm/pull/16971)

---

## Infrastructure / CI/CD

- **UI Testing**
    - Break e2e_ui_testing into build, unit, and e2e steps - [PR #16783](https://github.com/BerriAI/litellm/pull/16783)
    - Building UI for Testing - [PR #16968](https://github.com/BerriAI/litellm/pull/16968)
    - CI/CD Fixes - [PR #16937](https://github.com/BerriAI/litellm/pull/16937)

- **Dependency Management**
    - Bump js-yaml from 3.14.1 to 3.14.2 in /tests/proxy_admin_ui_tests/ui_unit_tests - [PR #16755](https://github.com/BerriAI/litellm/pull/16755)
    - Bump js-yaml from 3.14.1 to 3.14.2 - [PR #16802](https://github.com/BerriAI/litellm/pull/16802)

- **Migration**
    - Migration job labels - [PR #16831](https://github.com/BerriAI/litellm/pull/16831)

- **Config**
    - This yaml actually works - [PR #16757](https://github.com/BerriAI/litellm/pull/16757)

- **Release Notes**
    - Add perf improvements on embeddings to release notes - [PR #16697](https://github.com/BerriAI/litellm/pull/16697)
    - Docs - v1.80.0 - [PR #16694](https://github.com/BerriAI/litellm/pull/16694)

- **Investigation**
    - Investigate issue root cause - [PR #16859](https://github.com/BerriAI/litellm/pull/16859)

---

## New Contributors

* @mattmorgis made their first contribution in [PR #16371](https://github.com/BerriAI/litellm/pull/16371)
* @mmandic-coatue made their first contribution in [PR #16732](https://github.com/BerriAI/litellm/pull/16732)
* @Bradley-Butcher made their first contribution in [PR #16725](https://github.com/BerriAI/litellm/pull/16725)
* @BenjaminLevy made their first contribution in [PR #16757](https://github.com/BerriAI/litellm/pull/16757)
* @CatBraaain made their first contribution in [PR #16767](https://github.com/BerriAI/litellm/pull/16767)
* @tushar8408 made their first contribution in [PR #16831](https://github.com/BerriAI/litellm/pull/16831)
* @nbsp1221 made their first contribution in [PR #16845](https://github.com/BerriAI/litellm/pull/16845)
* @idola9 made their first contribution in [PR #16832](https://github.com/BerriAI/litellm/pull/16832)
* @nkukard made their first contribution in [PR #16864](https://github.com/BerriAI/litellm/pull/16864)
* @alhuang10 made their first contribution in [PR #16852](https://github.com/BerriAI/litellm/pull/16852)
* @sebslight made their first contribution in [PR #16838](https://github.com/BerriAI/litellm/pull/16838)
* @TsurumaruTsuyoshi made their first contribution in [PR #16905](https://github.com/BerriAI/litellm/pull/16905)
* @cyberjunk made their first contribution in [PR #16492](https://github.com/BerriAI/litellm/pull/16492)
* @colinlin-stripe made their first contribution in [PR #16895](https://github.com/BerriAI/litellm/pull/16895)
* @sureshdsk made their first contribution in [PR #16883](https://github.com/BerriAI/litellm/pull/16883)
* @eiliyaabedini made their first contribution in [PR #16875](https://github.com/BerriAI/litellm/pull/16875)
* @justin-tahara made their first contribution in [PR #16957](https://github.com/BerriAI/litellm/pull/16957)
* @wangsoft made their first contribution in [PR #16913](https://github.com/BerriAI/litellm/pull/16913)
* @dsduenas made their first contribution in [PR #16891](https://github.com/BerriAI/litellm/pull/16891)

---

## Known Issues
* `/audit` and `/user/available_users` routes return 404. Fixed in [PR #17337](https://github.com/BerriAI/litellm/pull/17337)

---

## Full Changelog

**[View complete changelog on GitHub](https://github.com/BerriAI/litellm/compare/v1.80.0-nightly...v1.80.5.rc.2)**
