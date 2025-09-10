---
title: "v1.76.3-stable - Performance, Video Generation & CloudZero Integration"
slug: "v1-76-3"
date: 2025-09-06T10:00:00
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

:::warning

This release has a known issue where startup is leading to Out of Memory errors when deploying on Kubernetes. We recommend waiting before upgrading to this version.

:::


## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:v1.76.3
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.76.3
```

</TabItem>
</Tabs>

---

## Key Highlights

- **Major Performance Improvements** +400 RPS when using correct amount of workers + CPU cores combination
- **Video Generation Support** - Added Google AI Studio  and Vertex AI Veo Video Generation through LiteLLM Pass through routes
- **CloudZero Integration** - New cost tracking integration for exporting LiteLLM Usage and Spend data to CloudZero. 

## Major Changes 
- **Performance Optimization**: LiteLLM Proxy now achieves +400 RPS when using correct amount of CPU cores - [PR #14153](https://github.com/BerriAI/litellm/pull/14153), [PR #14242](https://github.com/BerriAI/litellm/pull/14242)
  
  By default, LiteLLM will now use `num_workers = os.cpu_count()` to achieve optimal performance. 
  
  **Override Options:**
  
  Set environment variable:
  ```bash
  DEFAULT_NUM_WORKERS_LITELLM_PROXY=1
  ```
  
  Or start LiteLLM Proxy with:
  ```bash
  litellm --num_workers 1
  ```

- **Security Fix**: Fixed memory_usage_in_mem_cache cache endpoint vulnerability - [PR #14229](https://github.com/BerriAI/litellm/pull/14229)

---

## Performance Improvements

This release includes significant performance optimizations. On our internal benchmarks we saw 1 instance get +400 RPS when using correct amount of  workers + CPU cores combination.

- **+400 RPS Performance Boost** - LiteLLM Proxy now uses correct amount of CPU cores for optimal performance - [PR #14153](https://github.com/BerriAI/litellm/pull/14153)
- **Default CPU Workers** - Changed DEFAULT_NUM_WORKERS_LITELLM_PROXY default to number of CPUs - [PR #14242](https://github.com/BerriAI/litellm/pull/14242)


---

## New Models / Updated Models

#### New Model Support

| Provider    | Model                                  | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| ----------- | -------------------------------------- | -------------- | ------------------- | -------------------- | -------- |
| OpenRouter | `openrouter/openai/gpt-4.1` | 1M | $2.00 | $8.00 | Chat completions with vision |
| OpenRouter | `openrouter/openai/gpt-4.1-mini` | 1M | $0.40 | $1.60 | Efficient chat completions |
| OpenRouter | `openrouter/openai/gpt-4.1-nano` | 1M | $0.10 | $0.40 | Ultra-efficient chat |
| Vertex AI | `vertex_ai/openai/gpt-oss-20b-maas` | 131K | $0.075 | $0.30 | Reasoning support |
| Vertex AI | `vertex_ai/openai/gpt-oss-120b-maas` | 131K | $0.15 | $0.60 | Advanced reasoning |
| Gemini | `gemini/veo-3.0-generate-preview` | 1K | - | $0.75/sec | Video generation |
| Gemini | `gemini/veo-3.0-fast-generate-preview` | 1K | - | $0.40/sec | Fast video generation |
| Gemini | `gemini/veo-2.0-generate-001` | 1K | - | $0.35/sec | Video generation |
| Volcengine | `doubao-embedding-large` | 4K | Free | Free | 2048-dim embeddings |
| Together AI | `together_ai/deepseek-ai/DeepSeek-V3.1` | 128K | $0.60 | $1.70 | Reasoning support |

#### Features

- **[Google Gemini](../../docs/providers/gemini)**
    - Added 'thoughtSignature' support via 'thinking_blocks' - [PR #14122](https://github.com/BerriAI/litellm/pull/14122)
    - Added support for reasoning_effort='minimal' for Gemini models - [PR #14262](https://github.com/BerriAI/litellm/pull/14262)
- **[OpenRouter](../../docs/providers/openrouter)**
    - Added GPT-4.1 model family - [PR #14101](https://github.com/BerriAI/litellm/pull/14101)
- **[Groq](../../docs/providers/groq)**
    - Added support for reasoning_effort parameter - [PR #14207](https://github.com/BerriAI/litellm/pull/14207)
- **[X.AI](../../docs/providers/xai)**
    - Fixed XAI cost calculation - [PR #14127](https://github.com/BerriAI/litellm/pull/14127)
- **[Vertex AI](../../docs/providers/vertex)**
    - Added support for GPT-OSS models on Vertex AI - [PR #14184](https://github.com/BerriAI/litellm/pull/14184)
    - Added additionalProperties to Vertex AI Schema definition - [PR #14252](https://github.com/BerriAI/litellm/pull/14252)
- **[VLLM](../../docs/providers/vllm)**
    - Handle output parsing responses API output - [PR #14121](https://github.com/BerriAI/litellm/pull/14121)
- **[Ollama](../../docs/providers/ollama)**
    - Added unified 'thinking' param support via `reasoning_content` - [PR #14121](https://github.com/BerriAI/litellm/pull/14121)
- **[Anthropic](../../docs/providers/anthropic)**
    - Added supported text field to anthropic citation response - [PR #14126](https://github.com/BerriAI/litellm/pull/14126)
- **[OCI Provider](../../docs/providers/oci)**
    - Handle assistant messages with both content and tool_calls - [PR #14171](https://github.com/BerriAI/litellm/pull/14171)
- **[Bedrock](../../docs/providers/bedrock)**
    - Fixed structure output - [PR #14130](https://github.com/BerriAI/litellm/pull/14130)
    - Added initial support for Bedrock Batches API - [PR #14190](https://github.com/BerriAI/litellm/pull/14190)
- **[Databricks](../../docs/providers/databricks)**
    - Added support for anthropic citation API in Databricks - [PR #14077](https://github.com/BerriAI/litellm/pull/14077)

### Bug Fixes
- **[Google Gemini (Google AI Studio + Vertex AI)](../../docs/providers/gemini)**
    - Fixed Gemini 2.5 Pro schema validation with OpenAI-style type arrays in tools - [PR #14154](https://github.com/BerriAI/litellm/pull/14154)
    - Fixed Gemini Tool Calling empty enum property - [PR #14155](https://github.com/BerriAI/litellm/pull/14155)

#### New Provider Support

- **[Volcengine](../../docs/providers/volcengine)**
    - Added Volcengine embedding module with handler and transformation logic - [PR #14028](https://github.com/BerriAI/litellm/pull/14028)

---

## LLM API Endpoints

#### Features

- **[Images API](../../docs/image_generation)**
    - Added pass through image generation and image editing on OpenAI - [PR #14292](https://github.com/BerriAI/litellm/pull/14292)
    - Support extra_body parameter for image generation - [PR #14211](https://github.com/BerriAI/litellm/pull/14211)
- **[Responses API](../../docs/response_api)**
    - Fixed response API for reasoning item in input for litellm proxy - [PR #14200](https://github.com/BerriAI/litellm/pull/14200)
    - Added structured output for SDK - [PR #14206](https://github.com/BerriAI/litellm/pull/14206)
- **[Bedrock Passthrough](../../docs/pass_through/bedrock)**
    - Support AWS_BEDROCK_RUNTIME_ENDPOINT on bedrock passthrough - [PR #14156](https://github.com/BerriAI/litellm/pull/14156)
- **[Google AI Studio Passthrough](../../docs/pass_through/google_ai_studio)**
    - Allow using Veo Video Generation through LiteLLM Pass through routes - [PR #14228](https://github.com/BerriAI/litellm/pull/14228)
- **General**
    - Added support for safety_identifier parameter in chat.completions.create - [PR #14174](https://github.com/BerriAI/litellm/pull/14174)
    - Fixed misclassified 500 error on invalid image_url in /chat/completions request - [PR #14149](https://github.com/BerriAI/litellm/pull/14149)
    - Fixed token count error for Gemini CLI - [PR #14133](https://github.com/BerriAI/litellm/pull/14133)

#### Bugs

- **General**
    - Remove "/" or ":" from model name when being used as h11 header name - [PR #14191](https://github.com/BerriAI/litellm/pull/14191)
    - Bug fix for openai.gpt-oss when using reasoning_effort parameter - [PR #14300](https://github.com/BerriAI/litellm/pull/14300)

---

## Spend Tracking, Budgets and Rate Limiting

### Features
    - Added header support for spend_logs_metadata - [PR #14186](https://github.com/BerriAI/litellm/pull/14186)
    - Litellm passthrough cost tracking for chat completion - [PR #14256](https://github.com/BerriAI/litellm/pull/14256)

### Bug Fixes
    - Fixed TPM Rate Limit Bug - [PR #14237](https://github.com/BerriAI/litellm/pull/14237)
    - Fixed Key Budget not resets at expectable times - [PR #14241](https://github.com/BerriAI/litellm/pull/14241)



## Management Endpoints / UI

#### Features

- **UI Improvements**
    - Logs page screen size fixed - [PR #14135](https://github.com/BerriAI/litellm/pull/14135)
    - Create Organization Tooltip added on Success - [PR #14132](https://github.com/BerriAI/litellm/pull/14132)
    - Back to Keys should say Back to Logs - [PR #14134](https://github.com/BerriAI/litellm/pull/14134)
    - Add client side pagination on All Models table - [PR #14136](https://github.com/BerriAI/litellm/pull/14136)
    - Model Filters UI improvement - [PR #14131](https://github.com/BerriAI/litellm/pull/14131)
    - Remove table filter on user info page - [PR #14169](https://github.com/BerriAI/litellm/pull/14169)
    - Team name badge added on the User Details - [PR #14003](https://github.com/BerriAI/litellm/pull/14003)
    - Fix: Log page parameter passing error - [PR #14193](https://github.com/BerriAI/litellm/pull/14193)
- **Authentication & Authorization**
    - Support for ES256/ES384/ES512 and EdDSA JWT verification - [PR #14118](https://github.com/BerriAI/litellm/pull/14118)
    - Ensure `team_id` is a required field for generating service account keys - [PR #14270](https://github.com/BerriAI/litellm/pull/14270)

#### Bugs

- **General**
    - Validate store model in db setting - [PR #14269](https://github.com/BerriAI/litellm/pull/14269)

---

## Logging / Guardrail Integrations

#### Features

- **[Datadog](../../docs/proxy/logging#datadog)**
    - Ensure `apm_id` is set on DD LLM Observability traces - [PR #14272](https://github.com/BerriAI/litellm/pull/14272)
- **[Braintrust](../../docs/proxy/logging#braintrust)**
    - Fix logging when OTEL is enabled - [PR #14122](https://github.com/BerriAI/litellm/pull/14122)
- **[OTEL](../../docs/proxy/logging#otel)**
    - Optional Metrics and Logs following semantic conventions - [PR #14179](https://github.com/BerriAI/litellm/pull/14179)
- **[Slack Alerting](../../docs/proxy/alerting)**
    - Added alert type to alert message to slack for easier handling - [PR #14176](https://github.com/BerriAI/litellm/pull/14176)

#### Guardrails
    - Added guardrail to the Anthropic API endpoint - [PR #14107](https://github.com/BerriAI/litellm/pull/14107)

#### New Integration

- **[CloudZero](../../docs/proxy/cost_tracking)**
    - LiteLLM x CloudZero Integration for Cost Tracking - [PR #14296](https://github.com/BerriAI/litellm/pull/14296)

---

## Performance / Loadbalancing / Reliability improvements

#### Features

- **Performance**
    - LiteLLM Proxy: +400 RPS when using correct amount of CPU cores - [PR #14153](https://github.com/BerriAI/litellm/pull/14153)
    - Allow using `x-litellm-stream-timeout` header for stream timeout in requests - [PR #14147](https://github.com/BerriAI/litellm/pull/14147)
    - Change DEFAULT_NUM_WORKERS_LITELLM_PROXY default to number CPUs - [PR #14242](https://github.com/BerriAI/litellm/pull/14242)
- **Monitoring**
    - Added Prometheus missing metrics - [PR #14139](https://github.com/BerriAI/litellm/pull/14139)
- **Timeout**
    - **Stream Timeout Control** - Allow using `x-litellm-stream-timeout` header for stream timeout in requests - [PR #14147](https://github.com/BerriAI/litellm/pull/14147)
- **Routing**
    - Fixed x-litellm-tags not routing with Responses API - [PR #14289](https://github.com/BerriAI/litellm/pull/14289)

#### Bugs

- **Security**
    - Fixed memory_usage_in_mem_cache cache endpoint vulnerability - [PR #14229](https://github.com/BerriAI/litellm/pull/14229)

---

## General Proxy Improvements

#### Features

- **SCIM Support**
    - Added better SCIM debugging - [PR #14221](https://github.com/BerriAI/litellm/pull/14221)
    - Bug fixes for handling SCIM Group Memberships - [PR #14226](https://github.com/BerriAI/litellm/pull/14226)
- **Kubernetes**
    - Added optional PodDisruptionBudget for litellm proxy - [PR #14093](https://github.com/BerriAI/litellm/pull/14093)
- **Error Handling**
    - Add model to azure error message - [PR #14294](https://github.com/BerriAI/litellm/pull/14294)

---

## New Contributors
* @iabhi4 made their first contribution in [PR #14093](https://github.com/BerriAI/litellm/pull/14093)
* @zainhas made their first contribution in [PR #14087](https://github.com/BerriAI/litellm/pull/14087)
* @LifeDJIK made their first contribution in [PR #14146](https://github.com/BerriAI/litellm/pull/14146)
* @retanoj made their first contribution in [PR #14133](https://github.com/BerriAI/litellm/pull/14133)
* @zhxlp made their first contribution in [PR #14193](https://github.com/BerriAI/litellm/pull/14193)
* @kayoch1n made their first contribution in [PR #14191](https://github.com/BerriAI/litellm/pull/14191)
* @kutsushitaneko made their first contribution in [PR #14171](https://github.com/BerriAI/litellm/pull/14171)
* @mjmendo made their first contribution in [PR #14176](https://github.com/BerriAI/litellm/pull/14176)
* @HarshavardhanK made their first contribution in [PR #14213](https://github.com/BerriAI/litellm/pull/14213)
* @eycjur made their first contribution in [PR #14207](https://github.com/BerriAI/litellm/pull/14207)
* @22mSqRi made their first contribution in [PR #14241](https://github.com/BerriAI/litellm/pull/14241)
* @onlylhf made their first contribution in [PR #14028](https://github.com/BerriAI/litellm/pull/14028)
* @btpemercier made their first contribution in [PR #11319](https://github.com/BerriAI/litellm/pull/11319)
* @tremlin made their first contribution in [PR #14287](https://github.com/BerriAI/litellm/pull/14287)
* @TobiMayr made their first contribution in [PR #14262](https://github.com/BerriAI/litellm/pull/14262)
* @Eitan1112 made their first contribution in [PR #14252](https://github.com/BerriAI/litellm/pull/14252)

---

## **[Full Changelog](https://github.com/BerriAI/litellm/compare/v1.76.1-nightly...v1.76.3-nightly)**
