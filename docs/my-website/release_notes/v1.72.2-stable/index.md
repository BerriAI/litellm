---
title: "v1.72.2-stable"
slug: "v1-72-2-stable"
date: 2025-06-07T10:00:00
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
docker run
-e STORE_MODEL_IN_DB=True
-p 4000:4000
docker.litellm.ai/berriai/litellm:main-v1.72.2-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.72.2.post1
```

</TabItem>
</Tabs>

## TLDR

* **Why Upgrade**
    - Performance Improvements for /v1/messages: For this endpoint LiteLLM Proxy overhead is now down to 50ms at 250 RPS. 
    - Accurate Rate Limiting: Multi-instance rate limiting now tracks rate limits across keys, models, teams, and users with 0 spillover.
    - Audit Logs on UI: Track when Keys, Teams, and Models were deleted by viewing Audit Logs on the LiteLLM UI.
    - /v1/messages all models support: You can now use all LiteLLM models (`gpt-4.1`, `o1-pro`, `gemini-2.5-pro`) with /v1/messages API. 
    - [Anthropic MCP](../../docs/providers/anthropic#mcp-tool-calling): Use remote MCP Servers with Anthropic Models. 
* **Who Should Read**
    - Teams using `/v1/messages` API (Claude Code)
    - Proxy Admins using LiteLLM Virtual Keys and setting rate limits
* **Risk of Upgrade**
    - **Medium**
        - Upgraded `ddtrace==3.8.0`, if you use DataDog tracing this is a medium level risk. We recommend monitoring logs for any issues.



---

## `/v1/messages` Performance Improvements

<Image 
  img={require('../../img/release_notes/v1_messages_perf.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

This release brings significant performance improvements to the /v1/messages API on LiteLLM. 

For this endpoint LiteLLM Proxy overhead latency is now down to 50ms, and each instance can handle 250 RPS. We validated these improvements through load testing with payloads containing over 1,000 streaming chunks.

This is great for real time use cases with large requests (eg. multi turn conversations, Claude Code, etc.). 

## Multi-Instance Rate Limiting Improvements

<Image 
  img={require('../../img/release_notes/multi_instance_rate_limits_v3.jpg')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

LiteLLM now accurately tracks rate limits across keys, models, teams, and users with 0 spillover.

This is a significant improvement over the previous version, which faced issues with leakage and spillover in high traffic, multi-instance setups.

**Key Changes:**
- Redis is now part of the rate limit check, instead of being a background sync. This ensures accuracy and reduces read/write operations during low activity.
- LiteLLM now uses Lua scripts to ensure all checks are atomic.
- In-memory caching uses Redis values. This prevents drift, and reduces Redis queries once objects are over their limit.

These changes are currently behind the feature flag - `EXPERIMENTAL_ENABLE_MULTI_INSTANCE_RATE_LIMITING=True`. We plan to GA this in our next release - subject to feedback.

## Audit Logs on UI

<Image 
  img={require('../../img/release_notes/ui_audit_log.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

This release introduces support for viewing audit logs in the UI. As a Proxy Admin, you can now check if and when a key was deleted, along with who performed the action.

LiteLLM tracks changes to the following entities and actions: 

- **Entities:** Keys, Teams, Users, Models
- **Actions:** Create, Update, Delete, Regenerate



## New Models / Updated Models

**Newly Added Models**

| Provider    | Model                                  | Context Window | Input ($/1M tokens) | Output ($/1M tokens) |
| ----------- | -------------------------------------- | -------------- | ------------------- | -------------------- |
| Anthropic   | `claude-4-opus-20250514`               | 200K           | $15.00              | $75.00               |
| Anthropic   | `claude-4-sonnet-20250514`             | 200K           | $3.00               | $15.00               |
| VertexAI, Google AI Studio      | `gemini-2.5-pro-preview-06-05`         | 1M             | $1.25               | $10.00               |
| OpenAI      | `codex-mini-latest`                    | 200K           | $1.50               | $6.00                |
| Cerebras    | `qwen-3-32b`                           | 128K           | $0.40               | $0.80                |
| SambaNova   | `DeepSeek-R1`                          | 32K            | $5.00               | $7.00                |
| SambaNova   | `DeepSeek-R1-Distill-Llama-70B`       | 131K           | $0.70               | $1.40                |



### Model Updates

- **[Anthropic](../../docs/providers/anthropic)**
    - Cost tracking added for new Claude models - [PR](https://github.com/BerriAI/litellm/pull/11339)
        - `claude-4-opus-20250514`
        - `claude-4-sonnet-20250514`
    - Support for MCP tool calling with Anthropic models - [PR](https://github.com/BerriAI/litellm/pull/11474)
- **[Google AI Studio](../../docs/providers/gemini)**
    - Google Gemini 2.5 Pro Preview 06-05 support - [PR](https://github.com/BerriAI/litellm/pull/11447)
    - Gemini streaming thinking content parsing with `reasoning_content` - [PR](https://github.com/BerriAI/litellm/pull/11298)
    - Support for no reasoning option for Gemini models - [PR](https://github.com/BerriAI/litellm/pull/11393)
    - URL context support for Gemini models - [PR](https://github.com/BerriAI/litellm/pull/11351)
    - Gemini embeddings-001 model prices and context window - [PR](https://github.com/BerriAI/litellm/pull/11332)
- **[OpenAI](../../docs/providers/openai)**
    - Cost tracking for `codex-mini-latest` - [PR](https://github.com/BerriAI/litellm/pull/11492)
- **[Vertex AI](../../docs/providers/vertex)**
    - Cache token tracking on streaming calls - [PR](https://github.com/BerriAI/litellm/pull/11387)
    - Return response_id matching upstream response ID for stream and non-stream - [PR](https://github.com/BerriAI/litellm/pull/11456)
- **[Cerebras](../../docs/providers/cerebras)**
    - Cerebras/qwen-3-32b model pricing and context window - [PR](https://github.com/BerriAI/litellm/pull/11373)
- **[HuggingFace](../../docs/providers/huggingface)**
    - Fixed embeddings using non-default `input_type` - [PR](https://github.com/BerriAI/litellm/pull/11452)
- **[DataRobot](../../docs/providers/datarobot)**
    - New provider integration for enterprise AI workflows - [PR](https://github.com/BerriAI/litellm/pull/10385)
- **[DeepSeek](../../docs/providers/together_ai)**
    - DeepSeek R1 family model configuration via Together AI - [PR](https://github.com/BerriAI/litellm/pull/11394)
    - DeepSeek R1 pricing and context window configuration - [PR](https://github.com/BerriAI/litellm/pull/11339)

---

## LLM API Endpoints

- **[Images API](../../docs/image_generation)**
    - Azure endpoint support for image endpoints - [PR](https://github.com/BerriAI/litellm/pull/11482)
- **[Anthropic Messages API](../../docs/completion/chat)**
    - Support for ALL LiteLLM Providers (OpenAI, Azure, Bedrock, Vertex, DeepSeek, etc.) on /v1/messages API Spec - [PR](https://github.com/BerriAI/litellm/pull/11502)
    - Performance improvements for /v1/messages route - [PR](https://github.com/BerriAI/litellm/pull/11421)
    - Return streaming usage statistics when using LiteLLM with Bedrock models - [PR](https://github.com/BerriAI/litellm/pull/11469)
- **[Embeddings API](../../docs/embedding/supported_embedding)**
    - Provider-specific optional params handling for embedding calls - [PR](https://github.com/BerriAI/litellm/pull/11346)
    - Proper Sagemaker request attribute usage for embeddings - [PR](https://github.com/BerriAI/litellm/pull/11362)
- **[Rerank API](../../docs/rerank/supported_rerank)**
    - New HuggingFace rerank provider support - [PR](https://github.com/BerriAI/litellm/pull/11438), [Guide](../../docs/providers/huggingface_rerank)

---

## Spend Tracking

- Added token tracking for anthropic batch calls via /anthropic passthrough route- [PR](https://github.com/BerriAI/litellm/pull/11388)

---

## Management Endpoints / UI


- **SSO/Authentication**
    - SSO configuration endpoints and UI integration with persistent settings - [PR](https://github.com/BerriAI/litellm/pull/11417)
    - Update proxy admin ID role in DB + Handle SSO redirects with custom root path - [PR](https://github.com/BerriAI/litellm/pull/11384)
    - Support returning virtual key in custom auth - [PR](https://github.com/BerriAI/litellm/pull/11346)
    - User ID validation to ensure it is not an email or phone number - [PR](https://github.com/BerriAI/litellm/pull/10102)
- **Teams**
    - Fixed Create/Update team member API 500 error - [PR](https://github.com/BerriAI/litellm/pull/10479)
    - Enterprise feature gating for RegenerateKeyModal in KeyInfoView - [PR](https://github.com/BerriAI/litellm/pull/11400)
- **SCIM**
    - Fixed SCIM running patch operation case sensitivity - [PR](https://github.com/BerriAI/litellm/pull/11335)
- **General**
    - Converted action buttons to sticky footer action buttons - [PR](https://github.com/BerriAI/litellm/pull/11293)
    - Custom Server Root Path - support for serving UI on a custom root path - [Guide](../../docs/proxy/custom_root_ui)
---

## Logging / Guardrails Integrations

#### Logging
- **[S3](../../docs/proxy/logging#s3)**
    - Async + Batched S3 Logging for improved performance - [PR](https://github.com/BerriAI/litellm/pull/11340)
- **[DataDog](../../docs/observability/datadog_integration)**
    - Add instrumentation for streaming chunks - [PR](https://github.com/BerriAI/litellm/pull/11338)
    - Add DD profiler to monitor Python profile of LiteLLM CPU% - [PR](https://github.com/BerriAI/litellm/pull/11375)
    - Bump DD trace version - [PR](https://github.com/BerriAI/litellm/pull/11426)
- **[Prometheus](../../docs/proxy/prometheus)**
    - Pass custom metadata labels in litellm_total_token metrics - [PR](https://github.com/BerriAI/litellm/pull/11414)
- **[GCS](../../docs/proxy/logging#google-cloud-storage)**
    - Update GCSBucketBase to handle GSM project ID if passed - [PR](https://github.com/BerriAI/litellm/pull/11409)

#### Guardrails
- **[Presidio](../../docs/proxy/guardrails/presidio)**
    - Add presidio_language yaml configuration support for guardrails - [PR](https://github.com/BerriAI/litellm/pull/11331)

---

## Performance / Reliability Improvements

- **Performance Optimizations**
    - Don't run auth on /health/liveliness endpoints - [PR](https://github.com/BerriAI/litellm/pull/11378)
    - Don't create 1 task for every hanging request alert - [PR](https://github.com/BerriAI/litellm/pull/11385)
    - Add debugging endpoint to track active /asyncio-tasks - [PR](https://github.com/BerriAI/litellm/pull/11382)
    - Make batch size for maximum retention in spend logs controllable - [PR](https://github.com/BerriAI/litellm/pull/11459)
    - Expose flag to disable token counter - [PR](https://github.com/BerriAI/litellm/pull/11344)
    - Support pipeline redis lpop for older redis versions - [PR](https://github.com/BerriAI/litellm/pull/11425)
---

## Bug Fixes

- **LLM API Fixes**
    - **Anthropic**: Fix regression when passing file url's to the 'file_id' parameter - [PR](https://github.com/BerriAI/litellm/pull/11387)
    - **Vertex AI**: Fix Vertex AI any_of issues for Description and Default. - [PR](https://github.com/BerriAI/litellm/issues/11383) 
    - Fix transcription model name mapping - [PR](https://github.com/BerriAI/litellm/pull/11333)
    - **Image Generation**: Fix None values in usage field for gpt-image-1 model responses - [PR](https://github.com/BerriAI/litellm/pull/11448)
    - **Responses API**: Fix _transform_responses_api_content_to_chat_completion_content doesn't support file content type - [PR](https://github.com/BerriAI/litellm/pull/11494)
    - **Fireworks AI**: Fix rate limit exception mapping - detect "rate limit" text in error messages - [PR](https://github.com/BerriAI/litellm/pull/11455)
- **Spend Tracking/Budgets**
    - Respect user_header_name property for budget selection and user identification - [PR](https://github.com/BerriAI/litellm/pull/11419)
- **MCP Server**
    - Remove duplicate server_id MCP config servers - [PR](https://github.com/BerriAI/litellm/pull/11327)
- **Function Calling**
    - supports_function_calling works with llm_proxy models - [PR](https://github.com/BerriAI/litellm/pull/11381)
- **Knowledge Base**
    - Fixed Knowledge Base Call returning error - [PR](https://github.com/BerriAI/litellm/pull/11467)

---

## New Contributors
* [@mjnitz02](https://github.com/mjnitz02) made their first contribution in [#10385](https://github.com/BerriAI/litellm/pull/10385)
* [@hagan](https://github.com/hagan) made their first contribution in [#10479](https://github.com/BerriAI/litellm/pull/10479)
* [@wwells](https://github.com/wwells) made their first contribution in [#11409](https://github.com/BerriAI/litellm/pull/11409)
* [@likweitan](https://github.com/likweitan) made their first contribution in [#11400](https://github.com/BerriAI/litellm/pull/11400)
* [@raz-alon](https://github.com/raz-alon) made their first contribution in [#10102](https://github.com/BerriAI/litellm/pull/10102)
* [@jtsai-quid](https://github.com/jtsai-quid) made their first contribution in [#11394](https://github.com/BerriAI/litellm/pull/11394)
* [@tmbo](https://github.com/tmbo) made their first contribution in [#11362](https://github.com/BerriAI/litellm/pull/11362)
* [@wangsha](https://github.com/wangsha) made their first contribution in [#11351](https://github.com/BerriAI/litellm/pull/11351)
* [@seankwalker](https://github.com/seankwalker) made their first contribution in [#11452](https://github.com/BerriAI/litellm/pull/11452)
* [@pazevedo-hyland](https://github.com/pazevedo-hyland) made their first contribution in [#11381](https://github.com/BerriAI/litellm/pull/11381)
* [@cainiaoit](https://github.com/cainiaoit) made their first contribution in [#11438](https://github.com/BerriAI/litellm/pull/11438)
* [@vuanhtu52](https://github.com/vuanhtu52) made their first contribution in [#11508](https://github.com/BerriAI/litellm/pull/11508)

---

## Demo Instance

Here's a Demo Instance to test changes:

- Instance: https://demo.litellm.ai/
- Login Credentials:
    - Username: admin
    - Password: sk-1234

## [Git Diff](https://github.com/BerriAI/litellm/releases/tag/v1.72.2-stable)
