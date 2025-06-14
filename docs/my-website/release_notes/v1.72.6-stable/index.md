---
title: "[PRE-RELEASE] v1.72.6-stable"
slug: "v1-72-6-stable"
date: 2025-06-14T10:00:00
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

:::info

This version is not out yet. 

:::


## TLDR

* **Why Upgrade**
* **Who Should Read**
* **Risk of Upgrade**



---

## Key Highlights

---


## New / Updated Models

### Pricing / Context Window Updates

| Provider    | Model                                  | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Type |
| ----------- | -------------------------------------- | -------------- | ------------------- | -------------------- | -------------------- |
| VertexAI   | `vertex_ai/claude-opus-4`               | 200K           | $15.00              | $75.00               | New |
| OpenAI   | `gpt-4o-audio-preview-2025-06-03`             | 128k           | $2.5 (text), $40 (audio)              | $10 (text), $80 (audio)               | New |
| OpenAI | `o3-pro` | 200k | 20 | 80 | New |
| OpenAI | `o3-pro-2025-06-10` | 200k | 20 | 80 | New |
| OpenAI | `o3` | 200k | 2 | 8 | Updated |
| OpenAI | `o3-2025-04-16` | 200k | 2 | 8 | Updated |
| Azure | `azure/gpt-4o-mini-transcribe` | 16k | 1.25 (text), 3 (audio) | 5 (text) | New |
| Mistral | `mistral/magistral-medium-latest` | 40k | 2 | 5 | New |
| Mistral | `mistral/magistral-small-latest` | 40k | 0.5 | 1.5 | New |

- Deepgram: `nova-3` cost per second pricing is [now supported](https://github.com/BerriAI/litellm/pull/11634).

### Updated Models
- **Bugs**
    - **Watsonx**
        - Ignore space id on Watsonx deployments (throws json errors) - [PR](https://github.com/BerriAI/litellm/pull/11527)
    - **Ollama**
        - Set tool call id for streaming calls - [PR](https://github.com/BerriAI/litellm/pull/11528)
    - **Gemini (VertexAI + Google AI Studio)**
        - Fix tool call indexes - [PR](https://github.com/BerriAI/litellm/pull/11558)
        - Handle empty string for arguments in function calls - [PR](https://github.com/BerriAI/litellm/pull/11601)
        - Add audio/ogg mime type support when inferring from file url’s - [PR](https://github.com/BerriAI/litellm/pull/11635)
    - **Custom LLM**
        - Fix passing api_base, api_key, litellm_params_dict to custom_llm embedding methods - [PR](https://github.com/BerriAI/litellm/pull/11450) s/o [ElefHead](https://github.com/ElefHead)
    - **Huggingface**
        - Add /chat/completions to endpoint url when missing - [PR](https://github.com/BerriAI/litellm/pull/11630)
    - **Deepgram**
        - Support async httpx calls - [PR](https://github.com/BerriAI/litellm/pull/11641)
    - **Anthropic**
        - Append prefix (if set) to assistant content start - [PR](https://github.com/BerriAI/litellm/pull/11719)
- **Features**
    - **VertexAI**
        - Support vertex credentials set via env var on passthrough - [PR](https://github.com/BerriAI/litellm/pull/11527)
        - Support for choosing ‘global’ region when model is only available there - [PR](https://github.com/BerriAI/litellm/pull/11566)
        - Anthropic passthrough cost calculation + token tracking - [PR](https://github.com/BerriAI/litellm/pull/11611)
        - Support ‘global’ vertex region on passthrough - [PR](https://github.com/BerriAI/litellm/pull/11661)
    - **Anthropic**
        - ‘none’ tool choice param support - [PR](https://github.com/BerriAI/litellm/pull/11695)
    - **Perplexity**
        - Add ‘reasoning_effort’ support - [PR](https://github.com/BerriAI/litellm/pull/11562)
    - **Mistral**
        - Add mistral reasoning support - [PR](https://github.com/BerriAI/litellm/pull/11642)
    - **SGLang**
        - Map context window exceeded error for proper handling - [PR](https://github.com/BerriAI/litellm/pull/11575/)
    - **Deepgram**
        - Provider specific params support - [PR](https://github.com/BerriAI/litellm/pull/11638)
    - **Azure**
        - Return content safety filter results - [PR](https://github.com/BerriAI/litellm/pull/11655)
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
