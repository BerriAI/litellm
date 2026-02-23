---
title: "v1.76.0-stable - RPS Improvements"
slug: "v1-76-0"
date: 2025-08-23T10:00:00
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

LiteLLM is hiring a **Founding Backend Engineer**, in San Francisco. 

[Apply here](https://www.ycombinator.com/companies/litellm/jobs/6uvoBp3-founding-backend-engineer) if you're interested!
:::





## Deploy this version

:::info

This release is not live yet. 
:::


---

## New Models / Updated Models

#### Bugs
- **[OpenAI](../../docs/providers/openai)**
    - Gpt-5 chat: clarify does not support function calling [PR #13612](https://github.com/BerriAI/litellm/pull/13612), s/o  @[superpoussin22](https://github.com/superpoussin22)
- **[VertexAI](../../docs/providers/vertex)**
    - fix vertexai batch file format by @[thiagosalvatore](https://github.com/thiagosalvatore) in [PR #13576](https://github.com/BerriAI/litellm/pull/13576)
- **[LiteLLM Proxy](../../docs/providers/litellm_proxy)**
    - Add support for calling image_edits + image_generations via SDK to Proxy - [PR #13735](https://github.com/BerriAI/litellm/pull/13735)
- **[OpenRouter](../../docs/providers/openrouter)**
    - Fix max_output_tokens value for anthropic Claude 4 - [PR #13526](https://github.com/BerriAI/litellm/pull/13526)
- **[Gemini](../../docs/providers/gemini)**
    - Fix prompt caching cost calculation - [PR #13742](https://github.com/BerriAI/litellm/pull/13742)
- **[Azure](../../docs/providers/azure)**
    - Support `../openai/v1/respones` api base - [PR #13526](https://github.com/BerriAI/litellm/pull/13526)
    - Fix azure/gpt-5-chat max_input_tokens - [PR #13660](https://github.com/BerriAI/litellm/pull/13660)
- **[Groq](../../docs/providers/groq)**
    - streaming ASCII encoding issue - [PR #13675](https://github.com/BerriAI/litellm/pull/13675)
- **[Baseten](../../docs/providers/baseten)**
    - Refactored integration to use new openai-compatible endpoints - [PR #13783](https://github.com/BerriAI/litellm/pull/13783)
- **[Bedrock](../../docs/providers/bedrock)**
    - fix application inference profile for pass-through endpoints for bedrock - [PR #13881](https://github.com/BerriAI/litellm/pull/13881)
- **[DataRobot](../../docs/providers/datarobot)**
    - Updated URL handling for DataRobot provider URL - [PR #13880](https://github.com/BerriAI/litellm/pull/13880)

#### Features
- **[Together AI](../../docs/providers/together)**
    - Added Qwen3, Deepseek R1 0528 Throughput, GLM 4.5 and GPT-OSS models cost tracking - [PR #13637](https://github.com/BerriAI/litellm/pull/13637), s/o  @[Tasmay-Tibrewal](https://github.com/Tasmay-Tibrewal)
- **[Fireworks AI](../../docs/providers/fireworks_ai)**
    - add fireworks_ai/accounts/fireworks/models/deepseek-v3-0324 - [PR #13821](https://github.com/BerriAI/litellm/pull/13821)
- **[VertexAI](../../docs/providers/vertex)**
    - Add VertexAI qwen API Service - [PR #13828](https://github.com/BerriAI/litellm/pull/13828)
    - Add new VertexAI image models vertex_ai/imagen-4.0-generate-001, vertex_ai/imagen-4.0-ultra-generate-001, vertex_ai/imagen-4.0-fast-generate-001  - [PR #13874](https://github.com/BerriAI/litellm/pull/13874)
- **[Anthropic](../../docs/providers/anthropic)**
    - Add long context support w/ cost tracking - [PR #13759](https://github.com/BerriAI/litellm/pull/13759)
- **[DeepInfra](../../docs/providers/deepinfra)**
    - Add rerank endpoint support for deepinfra - [PR #13820](https://github.com/BerriAI/litellm/pull/13820)
    - Add new models for cost tracking - [PR #13883](https://github.com/BerriAI/litellm/pull/13883), s/o  @[Toy-97](https://github.com/Toy-97)
- **[Bedrock](../../docs/providers/bedrock)**
    - Add tool prompt caching on async calls - [PR #13803](https://github.com/BerriAI/litellm/pull/13803), s/o  @[UlookEE](https://github.com/UlookEE)
    - role chaining and session name with webauthentication for aws bedrock - [PR #13753](https://github.com/BerriAI/litellm/pull/13753), s/o @[RichardoC](https://github.com/RichardoC)
- **[Ollama](../../docs/providers/ollama)**
    - Handle Ollama null response when using tool calling with non-tool trained models - [PR #13902](https://github.com/BerriAI/litellm/pull/13902)
- **[OpenRouter](../../docs/providers/openrouter)**
    - Add deepseek/deepseek-chat-v3.1 support - [PR #13897](https://github.com/BerriAI/litellm/pull/13897)
- **[Mistral](../../docs/providers/mistral)**
    - Add support for calling mistral files via chat completions - [PR #13866](https://github.com/BerriAI/litellm/pull/13866), s/o  @[jinskjoy](https://github.com/jinskjoy)
    - Handle empty assistant content - [PR #13671](https://github.com/BerriAI/litellm/pull/13671)
    - Support new ‘thinking’ response block - [PR #13671](https://github.com/BerriAI/litellm/pull/13671)
- **[Databricks](../../docs/providers/databricks)**
    - remove deprecated dbrx models (dbrx-instruct, llama 3.1) - [PR #13843](https://github.com/BerriAI/litellm/pull/13843)
- **[AI/ML API](../../docs/providers/ai_ml_api)**
    - Image gen api support - [PR #13893](https://github.com/BerriAI/litellm/pull/13893)


## LLM API Endpoints
#### Bugs
- **[Responses API](../../docs/response_api)**
    - add default api version for openai responses api calls - [PR #13526](https://github.com/BerriAI/litellm/pull/13526)
    - support allowed_openai_params - [PR #13671](https://github.com/BerriAI/litellm/pull/13671)


## MCP Gateway
#### Bugs
- fix StreamableHTTPSessionManager .run() error - [PR #13666](https://github.com/BerriAI/litellm/pull/13666)

## Vector Stores 
#### Bugs
- **[Bedrock](../../docs/providers/bedrock)**
    - Using LiteLLM Managed Credentials for Query - [PR #13787](https://github.com/BerriAI/litellm/pull/13787)

## Management Endpoints / UI
#### Bugs
- **[Passthrough](../../docs/pass_through/intro)**
    - Fix query passthrough deletion - [PR #13622](https://github.com/BerriAI/litellm/pull/13622)

#### Features
- **Models**
    - Add Search Functionality for Public Model Names in Model Dashboard - [PR #13687](https://github.com/BerriAI/litellm/pull/13687)
    - Auto-Add `azure/` to deployment Name in UI - [PR #13685](https://github.com/BerriAI/litellm/pull/13685)
    - Models page row UI restructure - [PR #13771](https://github.com/BerriAI/litellm/pull/13771)
- **Notifications**
    - Add new notifications toast UI everywhere - [PR #13813](https://github.com/BerriAI/litellm/pull/13813)
- **Keys**
    - Fix key edit settings after regenerating a key - [PR #13815](https://github.com/BerriAI/litellm/pull/13815)
    - Require team_id when creating service account keys - [PR #13873](https://github.com/BerriAI/litellm/pull/13873)
    - Filter - show all options on filter option click - [PR #13858](https://github.com/BerriAI/litellm/pull/13858)
- **Usage**
    - Fix ‘Cannot read properties of undefined’ exception on user agent activity tab - [PR #13892](https://github.com/BerriAI/litellm/pull/13892)
- **SSO**
    - Free SSO usage for up to 5 users - [PR #13843](https://github.com/BerriAI/litellm/pull/13843)

## Logging / Guardrail Integrations
#### Bugs
- **[Bedrock Guardrails](../../docs/proxy/guardrails/bedrock)**
    - Add bedrock api key support - [PR #13835](https://github.com/BerriAI/litellm/pull/13835)
#### Features
- **[Datadog LLM Observability](../../docs/integrations/datadog)**
    - Add support for Failure Logging [PR #13726](https://github.com/BerriAI/litellm/pull/13726)
    - Add time to first token, litellm overhead, guardrail overhead latency metrics - [PR #13734](https://github.com/BerriAI/litellm/pull/13734)
    - Add support for tracing guardrail input/output - [PR #13767](https://github.com/BerriAI/litellm/pull/13767)
- **[Langfuse OTEL](../../docs/integrations/langfuse)**
    - Allow using Key/Team Based Logging - [PR #13791](https://github.com/BerriAI/litellm/pull/13791)
- **[AIM](../../docs/integrations/aim)**
    - Migrate to new firewall API - [PR #13748](https://github.com/BerriAI/litellm/pull/13748)
- **[OTEL](../../docs/observability/opentelemetry_integration)**
    - Add OTEL tracing for actual LLM API call - [PR #13836](https://github.com/BerriAI/litellm/pull/13836)
- **[MLFlow](../../docs/observability/mlflow_integration)**
    - Include predicted output in MLflow tracing - [PR #13795](https://github.com/BerriAI/litellm/pull/13795), s/o @TomeHirata  


## Performance / Loadbalancing / Reliability improvements
#### Bugs
- **[Cooldowns](../../docs/routing#how-cooldowns-work)**
    - don't return raw Azure Exceptions to client (can contain prompt leakage) - [PR #13529](https://github.com/BerriAI/litellm/pull/13529)
- **[Auto-router](../../docs/proxy/auto_routing)**
    - Ensures the relevant dependencies for auto router existing on LiteLLM Docker - [PR #13788](https://github.com/BerriAI/litellm/pull/13788)
- **Model Alias**
    - Fix calling key with access to model alias - [PR #13830](https://github.com/BerriAI/litellm/pull/13830)

#### Features
- **[S3 Caching](../../docs/proxy/caching)**
    - Use namespace as prefix for s3 cache - [PR #13704](https://github.com/BerriAI/litellm/pull/13704)
    - Async S3 Caching support (4x RPS improvement) - [PR #13852](https://github.com/BerriAI/litellm/pull/13852), s/o @[michal-otmianowski](https://github.com/michal-otmianowski)
- **Model Group header forwarding**
    - reuse same logic as global header forwarding - [PR #13741](https://github.com/BerriAI/litellm/pull/13741)
    - add support for hosted_vllm on UI - [PR #13885](https://github.com/BerriAI/litellm/pull/13885)
- **Performance**
    - Improve LiteLLM Python SDK RPS by +200 RPS (braintrust import + aiohttp transport fixes) - [PR #13839](https://github.com/BerriAI/litellm/pull/13839)
    - Use O(1) Set lookups for model routing - [PR #13879](https://github.com/BerriAI/litellm/pull/13879)
    - Reduce Significant CPU overhead from litellm_logging.py - [PR #13895](https://github.com/BerriAI/litellm/pull/13895)
    - Improvements for Async Success Handler (Logging Callbacks) - Approx +130 RPS - [PR #13905](https://github.com/BerriAI/litellm/pull/13905)


## General Proxy Improvements
#### Bugs

- **SDK**
    - Fix litellm compatibility with newest release of openAI (>v1.100.0) - [PR #13728](https://github.com/BerriAI/litellm/pull/13728)
- **Helm**
    - Add possibility to configure resources for migrations-job - [PR #13617](https://github.com/BerriAI/litellm/pull/13617)
    - Ensure Helm chart auto generated master keys follow sk-xxxx format - [PR #13871](https://github.com/BerriAI/litellm/pull/13871)
    - Enhance database configuration: add support for optional endpointKey - [PR #13763](https://github.com/BerriAI/litellm/pull/13763)
- **Rate Limits**
    - fixing descriptor/response size mismatch on parallel_request_limiter_v3 - [PR #13863](https://github.com/BerriAI/litellm/pull/13863), s/o  @[luizrennocosta](https://github.com/luizrennocosta)
- **Non-root**
    - fix permission access on prisma migrate in non-root image - [PR #13848](https://github.com/BerriAI/litellm/pull/13848), s/o @[Ithanil](https://github.com/Ithanil)