---
title: "[Pre-Release] v1.74.0-stable"
slug: "v1-74-0-stable"
date: 2025-07-05T10:00:00
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

:::info

This is a pre-release version of v1.74.0. The stable version will be released on July 9th, 2025.

:::

## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:v1.74.0.rc
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.74.0.post1
```

</TabItem>
</Tabs>

---

## Key Highlights 


---

## New Models / Updated Models

### New Providers
- **GitHub Copilot**: New LLM API provider support

### Updated Models
#### Bugs
    - **[Azure](../../docs/providers/azure)**
        - Support Azure Content Safety Guardrails in LiteLLM proxy
        - Fix responses API bridge - respect responses/ format
        - Add Azure AI Cohere rerank v3.5 support
    - **[Gemini](../../docs/providers/gemini)**
        - Fix gemini tool call sequence
        - Handle kwargs + litellm params containing stream in generate content bridge
        - Fix custom api_base path preservation (with revert for further improvements)
    - **[Mistral](../../docs/providers/mistral)**
        - Fix transform_response handling for empty string content
        - Turn Mistral to use llm_http_handler for improved performance
    - **[Vertex AI](../../docs/providers/vertex)**
        - Add size parameter support for Vertex AI image generation
        - Fix Vertex Anthropic models usage with gemini-cli
    - **[Bedrock](../../docs/providers/bedrock)**
        - Fix bedrock guardrails post_call for streaming responses
        - Support optional args for bedrock in factory.py
    - **[Ollama](../../docs/providers/ollama)**
        - Fix default parameters for ollama-chat
    - **[VLLM](../../docs/providers/vllm)**
        - Add 'audio_url' message type support
    - **[Hugging Face](../../docs/providers/huggingface)**
        - Fix Hugging Face tests and integration

#### Features
    - **Custom LLM Providers**
        - Pass through extra_ properties on "custom" llm provider
    - **Anthropic**
        - Fix user_id validation logic
        - Support anthropic_messages call type with max tokens check
    - **Prompt Management**
        - Langfuse prompt_version support for better prompt management

---

## LLM API Endpoints

#### Features
    - [**/v1/messages**](../../docs/anthropic_unified)
        - Remove hardcoded model name on streaming
        - Support for non-anthropic models (gemini/openai/etc.) token usage returned when calling /v1/messages
        - Fix using /messages with lowest latency routing
    - [**/generateContent**](../../docs/providers/gemini)
        - Allow passing litellm_params when using generateContent API endpoint
        - Support for OpenAI models - only pass supported params
    - [**/responses**](../../docs/response_api)
        - Fix responses API - resolve 'got multiple values for keyword argument litellm_trace_id'
        - Support for Azure responses API bridge
    - **Tool Choice**
        - Support Cursor IDE tool_choice format `{"type": "auto"}`
    - **Streaming**
        - Store finish reason, even if is_finished flag is set
        - Fix streaming cost tracking with prompt caching for VertexAI Anthropic

#### Bugs
    - **LlamaAPI**
        - Fix Error code: 307 for LlamaAPI Streaming Chat
    - **Custom Headers**
        - Enable setting custom header tags
    - **Cost Calculation**
        - Fix allow strings in calculate cost function

---

## Spend Tracking / Budget Improvements

#### Features
    - **Batches**
        - Support batch retrieve with target model Query Param
        - Add failure logging support for s3 logger
    - **DeepEval**
        - Fix DeepEval logging format for failure events
    - **Arize**
        - Add Arize Team Based Logging capabilities

---

## Management Endpoints / UI

#### Bugs
    - **Teams**
        - Prevent team model reset on model add
        - Return team-only models on /v2/model/info
        - Render team member budget correctly
    - **User Roles**
        - Correctly display 'Internal Viewer' user role
    - **UI Rendering**
        - Fix rendering UI on non-root images
    - **Callback Management**
        - Handle proxy internal callbacks in callback management test

#### Features
    - **Team Management**
        - Allow viewing/editing team based callbacks
        - Add team specific logging callbacks support
    - **Budget Display**
        - Comma separated spend and budget display
    - **Proxy CLI**
        - Add litellm-proxy cli login for starting to use litellm proxy
    - **Callback UI**
        - Add logos to callback list for better visualization

---

## Logging / Guardrail Integrations

#### Features
    - **AWS SQS**
        - New AWS SQS Logging Integration
    - **Azure Content Safety**
        - Add Azure Content Safety Guardrails to LiteLLM proxy
        - Add azure content safety guardrails to the UI
    - **Customizable Email Templates**
        - Support customizable email template - subject and signature
    - **JSON Logging**
        - Initialize JSON logging for all loggers when JSON_LOGS=True
    - **Sentry Integration**
        - Add Sentry scrubbing for better error tracking
    - **Message Redaction**
        - Ensure message redaction works for responses API logging

---

## MCP (Model Context Protocol) Improvements

#### Features
    - **URL Handling**
        - Add changes to MCP URL wrapping
        - Add MCP URL masking on frontend
    - **Tool Management**
        - Add error handling for MCP tools not found or invalid server
        - Add MCP tool prefix functionality
        - Segregate MCP tools on connections using headers
    - **Security**
        - Add MCP servers header to the scope of header
        - Fix SSL certificate error for MCP connections

---

## Performance / Loadbalancing / Reliability improvements

#### Features
    - **SDK Performance**
        - 2 second faster Python SDK import times
        - Reduce python sdk import time by additional 0.3s
    - **Prometheus Metrics**
        - Add better error validation when users configure prometheus metrics and labels to control cardinality
    - **OpenMeter Integration**
        - Fix OpenMeter integration error handling

---

## General Proxy Improvements

#### Bugs
    - **Configuration Handling**
        - Handle empty config.yaml
        - Fix gemini /models - replace models/ as expected, instead of using 'strip'
        - Fix flaky test_keys_delete_error_handling test
    - **Custom CA Bundle**
        - Fix custom ca bundle support in aiohttp transport
    - **Pydantic**
        - Update pydantic version

#### Features
    - **Documentation**
        - Update management_cli.md
        - Use the -d flag in docs instead of -D
        - Update Vertex Model Garden doc to use SDK for deploy + chat completion
        - Fix config file description in k8s deployment
        - Improve readme: replace claude-3-sonnet because it will be retired soon
    - **Startup Banner**
        - Add new banner on startup
    - **Test Infrastructure**
        - Move panw prisma airs test file location per feedback

---

## New Contributors
* @wildcard made their first contribution in https://github.com/BerriAI/litellm/pull/12157
* @colesmcintosh made their first contribution in https://github.com/BerriAI/litellm/pull/12168
* @szafranek made their first contribution in https://github.com/BerriAI/litellm/pull/12179
* @seyeong-han made their first contribution in https://github.com/BerriAI/litellm/pull/11946
* @dinggh made their first contribution in https://github.com/BerriAI/litellm/pull/12162
* @raz-alon made their first contribution in https://github.com/BerriAI/litellm/pull/11432
* @tofarr made their first contribution in https://github.com/BerriAI/litellm/pull/12200
* @lizzij made their first contribution in https://github.com/BerriAI/litellm/pull/12219
* @cipri-tom made their first contribution in https://github.com/BerriAI/litellm/pull/12201
* @zsimjee made their first contribution in https://github.com/BerriAI/litellm/pull/12185
* @jroberts2600 made their first contribution in https://github.com/BerriAI/litellm/pull/12175
* @SamBoyd made their first contribution in https://github.com/BerriAI/litellm/pull/12147
* @njbrake made their first contribution in https://github.com/BerriAI/litellm/pull/12202
* @NANDINI-star made their first contribution in https://github.com/BerriAI/litellm/pull/12244
* @utsumi-fj made their first contribution in https://github.com/BerriAI/litellm/pull/12230
* @dcieslak19973 made their first contribution in https://github.com/BerriAI/litellm/pull/12283
* @hanouticelina made their first contribution in https://github.com/BerriAI/litellm/pull/12286
* @takashiishida made their first contribution in https://github.com/BerriAI/litellm/pull/12239
* @lowjiansheng made their first contribution in https://github.com/BerriAI/litellm/pull/11999
* @JoostvDoorn made their first contribution in https://github.com/BerriAI/litellm/pull/12281

## **[Git Diff](https://github.com/BerriAI/litellm/compare/v1.73.6-stable...v1.74.0-stable)**
