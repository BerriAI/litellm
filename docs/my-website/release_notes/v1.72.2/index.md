---
title: "[Pre Release] v1.72.2-stable"
slug: "v1-72-2-stable"
date: 2025-06-07T10:00:00
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


:::info

The release candidate is live now.

The production release will be live on Wednesday.

:::


## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run
-e STORE_MODEL_IN_DB=True
-p 4000:4000
ghcr.io/berriai/litellm:main-v1.72.2.rc
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.72.2
```
</TabItem>
</Tabs>

## TLDR

* **Why upgrade?**
    - /v1/messages API performance improvements (lower latency, higher RPS)
    - Multi instance rate limiting support
    - Full Claude-4 cost tracking & Gemini 2.5 Pro preview
* **Who must read?**
    - Teams using `/v1/messages` API (Claude Code), LiteLLM Rate Limiting
* **Risk level**
    - **Medium**
        - Upgraded `ddtrace==3.8.0`, if you use DataDog tracing this is a medium level risk. We recommend monitoring logs for any issues.



---

## New Models / Updated Models

- **[Anthropic](../../docs/providers/anthropic)**
    - Cost tracking added for new Claude models - [PR](https://github.com/BerriAI/litellm/pull/11339)
        - `claude-4-opus-20250514`
        - `claude-4-sonnet-20250514`
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
    - New HuggingFace rerank provider support - [PR](https://github.com/BerriAI/litellm/pull/11438)
- **[DataRobot](../../docs/providers/datarobot)**
    - New provider integration for enterprise AI workflows - [PR](https://github.com/BerriAI/litellm/pull/10385)
- **[DeepSeek](../../docs/providers/together_ai)**
    - DeepSeek R1 family model configuration via Together AI - [PR](https://github.com/BerriAI/litellm/pull/11394)
    - DeepSeek R1 pricing and context window configuration - [PR](https://github.com/BerriAI/litellm/pull/11339)

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



## Spend Tracking

- Added token tracking for anthropic batch calls via /anthropic passthrough route- [PR](https://github.com/BerriAI/litellm/pull/11388)

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
    - Custom Server Root Path improvements - don't require reserving `/litellm` route - [PR](https://github.com/BerriAI/litellm/pull/11460)

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


## Performance / Reliability Improvements

- **Performance Optimizations**
    - Don't run auth on /health/liveliness endpoints - [PR](https://github.com/BerriAI/litellm/pull/11378)
    - Don't create 1 task for every hanging request alert - [PR](https://github.com/BerriAI/litellm/pull/11385)
    - Add debugging endpoint to track active /asyncio-tasks - [PR](https://github.com/BerriAI/litellm/pull/11382)
    - Make batch size for maximum retention in spend logs controllable - [PR](https://github.com/BerriAI/litellm/pull/11459)
    - Expose flag to disable token counter - [PR](https://github.com/BerriAI/litellm/pull/11344)
    - Support pipeline redis lpop for older redis versions - [PR](https://github.com/BerriAI/litellm/pull/11425)

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

## Demo Instance

Here's a Demo Instance to test changes:

- Instance: https://demo.litellm.ai/
- Login Credentials:
    - Username: admin
    - Password: sk-1234

## [Git Diff](https://github.com/BerriAI/litellm/releases)
