---
title: v1.71.1-stable - 97% lower median latency - aiohttp transport
slug: v1.71.1-stable
date: 2025-05-24T10:00:00
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
ghcr.io/berriai/litellm:main-v1.71.1-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.71.1
```
</TabItem>
</Tabs>

## Key Highlights

LiteLLM v1.71.1-stable is live now. Here are the key highlights of this release:



## Performance Improvements

This release includes significant performance enhancements:

- **aiohttp Transport**: 97% lower median latency (available as feature flag)


## New Models / Updated Models

- **[Gemini](../../docs/providers/google_ai_studio)**
    - New preview models added - [PR](https://github.com/BerriAI/litellm/pull/10991)
    - Additional model updates - [PR](https://github.com/BerriAI/litellm/pull/10998)
- **[Anthropic](../../docs/providers/anthropic)**
    - Claude-4 model family support - [PR](https://github.com/BerriAI/litellm/pull/11060)
    - Claude Sonnet 4 and Opus 4 reasoning_effort parameter support - [PR](https://github.com/BerriAI/litellm/pull/11114)
- **[xAI](../../docs/providers/xai)**
    - Grok-3 pricing information - [PR](https://github.com/BerriAI/litellm/pull/11028)
- **[LM Studio](../../docs/providers/lm_studio)**
    - Structured JSON schema outputs support - [PR](https://github.com/BerriAI/litellm/pull/10929)
- **[SambaNova](../../docs/providers/sambanova)**
    - Updated models and parameters - [PR](https://github.com/BerriAI/litellm/pull/10900)
- **[Databricks](../../docs/providers/databricks)**
    - Llama 4 Maverick model cost - [PR](https://github.com/BerriAI/litellm/pull/11008)
    - Claude 3.7 Sonnet output token cost correction - [PR](https://github.com/BerriAI/litellm/pull/11007)
- **[Azure](../../docs/providers/azure)**
    - Mistral Medium 25.05 support - [PR](https://github.com/BerriAI/litellm/pull/11063)
    - Certificate-based authentication support - [PR](https://github.com/BerriAI/litellm/pull/11069)
- **[Mistral](../../docs/providers/mistral)**
    - devstral-small-2505 model pricing and context window - [PR](https://github.com/BerriAI/litellm/pull/11103)
- **[Vertex AI](../../docs/providers/vertex)**
    - Global endpoints support - [PR](https://github.com/BerriAI/litellm/pull/10658)
    - authorized_user credentials type support - [PR](https://github.com/BerriAI/litellm/pull/10899)
- **[Ollama](../../docs/providers/ollama)**
    - Wildcard model support - [PR](https://github.com/BerriAI/litellm/pull/10982)
- **[CustomLLM](../../docs/providers/custom_llm)**
    - Embeddings support added - [PR](https://github.com/BerriAI/litellm/pull/10980)
- **[Featherless AI](../../docs/providers/featherless)**
    - Access to 4200+ models - [PR](https://github.com/BerriAI/litellm/pull/10596)

## LLM API Endpoints

- **[Image Edits](../../docs/image_generation)**
    - `/v1/images/edits` - Full support for image editing - [PR](https://github.com/BerriAI/litellm/pull/11020) [PR](https://github.com/BerriAI/litellm/pull/11123)
    - Content policy violation error mapping - [PR](https://github.com/BerriAI/litellm/pull/11113)
- **[Responses API](../../docs/response_api)**
    - MCP support for Responses API - [PR](https://github.com/BerriAI/litellm/pull/11029)
- **[Files API](../../docs/fine_tuning)**
    - LiteLLM Managed Files support for finetuning - [PR](https://github.com/BerriAI/litellm/pull/11039) [PR](https://github.com/BerriAI/litellm/pull/11040)
    - Validation for file operations (retrieve/list/delete) - [PR](https://github.com/BerriAI/litellm/pull/11081)

## Management Endpoints / UI

- **Teams**
    - Key and member count display - [PR](https://github.com/BerriAI/litellm/pull/10950)
    - Spend rounded to 4 decimal points - [PR](https://github.com/BerriAI/litellm/pull/11013)
    - Organization and team create buttons repositioned - [PR](https://github.com/BerriAI/litellm/pull/10948)
- **Keys**
    - Key reassignment and 'updated at' column - [PR](https://github.com/BerriAI/litellm/pull/10960)
    - Show model access groups during creation - [PR](https://github.com/BerriAI/litellm/pull/10965)
- **Logs**
    - Model filter on logs - [PR](https://github.com/BerriAI/litellm/pull/11048)
    - Passthrough endpoint error logs support - [PR](https://github.com/BerriAI/litellm/pull/10990)
- **Guardrails**
    - Config.yaml guardrails display - [PR](https://github.com/BerriAI/litellm/pull/10959)
- **Organizations/Users**
    - Spend rounded to 4 decimal points - [PR](https://github.com/BerriAI/litellm/pull/11023)
    - Show clear error when adding a user to a team - [PR](https://github.com/BerriAI/litellm/pull/10978)
- **[Audit Logs](../../docs/proxy/audit_logs)**
    - `/list` and `/info` endpoints exposed - [PR](https://github.com/BerriAI/litellm/pull/11102)

## Logging / Alerting Integrations

- **[Prometheus](../../docs/proxy/prometheus)**
    - Track `route` on proxy_* metrics - [PR](https://github.com/BerriAI/litellm/pull/10992)
- **[Langfuse](../../docs/proxy/langfuse)**
    - Support for `prompt_label` parameter - [PR](https://github.com/BerriAI/litellm/pull/11018)
    - Consistent modelParams logging - [PR](https://github.com/BerriAI/litellm/pull/11018)
- **[ConfidentAI](../../docs/observability/confidentai)**
    - Logging enabled for proxy and SDK - [PR](https://github.com/BerriAI/litellm/pull/10649)
- **[OpenTelemetry/Logfire](../../docs/observability/otel)**
    - Fixed proxy server initialization - [PR](https://github.com/BerriAI/litellm/pull/11091)

## Authentication & Security

- **[JWT Authentication](../../docs/proxy/jwt_auth)**
    - Default role configuration - [PR](https://github.com/BerriAI/litellm/pull/10995)
    - User to team mapping - [PR](https://github.com/BerriAI/litellm/pull/11108)
- **Custom Auth**
    - Switch between custom auth and API key auth - [PR](https://github.com/BerriAI/litellm/pull/11070)

## Performance / Reliability Improvements

- **[aiohttp Transport](../../docs/proxy/performance)**
    - 97% lower median latency (feature flagged) - [PR](https://github.com/BerriAI/litellm/pull/11097) [PR](https://github.com/BerriAI/litellm/pull/11132)
- **Background Health Checks**
    - Improved reliability - [PR](https://github.com/BerriAI/litellm/pull/10887)
- **Response Handling**
    - Better streaming status code detection - [PR](https://github.com/BerriAI/litellm/pull/10962)
    - Response ID propagation improvements - [PR](https://github.com/BerriAI/litellm/pull/11006)
- **Thread Management**
    - Removed error-creating threads for reliability - [PR](https://github.com/BerriAI/litellm/pull/11066)

## General Proxy Improvements

- **[Proxy CLI](../../docs/proxy/cli)**
    - Skip server startup flag - [PR](https://github.com/BerriAI/litellm/pull/10665)
    - Avoid DATABASE_URL override when provided - [PR](https://github.com/BerriAI/litellm/pull/11076)
- **Model Management**
    - Clear cache and reload after model updates - [PR](https://github.com/BerriAI/litellm/pull/10853)
    - Computer use support tracking - [PR](https://github.com/BerriAI/litellm/pull/10881)
- **Helm Chart**
    - LoadBalancer class support - [PR](https://github.com/BerriAI/litellm/pull/11064)

## Bug Fixes

This release includes numerous bug fixes to improve stability and reliability:

- **Provider Fixes**
    - Fixed Vertex AI quota_project_id parameter issue - [PR](https://github.com/BerriAI/litellm/pull/10915)
    - Fixed Cohere Rerank Provider - [PR](https://github.com/BerriAI/litellm/pull/10822)
    - Fixed Vertex AI credential refresh exceptions - [PR](https://github.com/BerriAI/litellm/pull/10969)
    - Fixed Anthropic streaming dict object handling - [PR](https://github.com/BerriAI/litellm/pull/11032)
    - Fixed OpenRouter stream usage ID issues - [PR](https://github.com/BerriAI/litellm/pull/11004)

- **Authentication & Users**
    - Fixed invitation email link generation - [PR](https://github.com/BerriAI/litellm/pull/10958)
    - Fixed JWT authentication default role - [PR](https://github.com/BerriAI/litellm/pull/10995)
    - Fixed user budget reset functionality - [PR](https://github.com/BerriAI/litellm/pull/10993)
    - Fixed SSO user compatibility and email validation - [PR](https://github.com/BerriAI/litellm/pull/11106)

- **Database & Infrastructure**
    - Fixed DB connection parameter handling - [PR](https://github.com/BerriAI/litellm/pull/10842)
    - Fixed invitation link Prisma queries - [PR](https://github.com/BerriAI/litellm/pull/11031)

- **UI & Display**
    - Fixed MCP tool rendering when no arguments required - [PR](https://github.com/BerriAI/litellm/pull/11012)
    - Fixed team model alias deletion - [PR](https://github.com/BerriAI/litellm/pull/11121)
    - Fixed team viewer permissions - [PR](https://github.com/BerriAI/litellm/pull/11127)

- **Model & Routing**
    - Fixed team model mapping in route requests - [PR](https://github.com/BerriAI/litellm/pull/11111)
    - Fixed standard optional parameter passing - [PR](https://github.com/BerriAI/litellm/pull/11124)
    - Fixed Claude 4 reasoning_effort parameter support - [PR](https://github.com/BerriAI/litellm/pull/11114)



## New Contributors
* [@DarinVerheijke](https://github.com/DarinVerheijke) made their first contribution in PR [#10596](https://github.com/BerriAI/litellm/pull/10596)
* [@estsauver](https://github.com/estsauver) made their first contribution in PR [#10929](https://github.com/BerriAI/litellm/pull/10929)
* [@mohittalele](https://github.com/mohittalele) made their first contribution in PR [#10665](https://github.com/BerriAI/litellm/pull/10665)
* [@pselden](https://github.com/pselden) made their first contribution in PR [#10899](https://github.com/BerriAI/litellm/pull/10899)
* [@unrealandychan](https://github.com/unrealandychan) made their first contribution in PR [#10842](https://github.com/BerriAI/litellm/pull/10842)
* [@dastaiger](https://github.com/dastaiger) made their first contribution in PR [#10946](https://github.com/BerriAI/litellm/pull/10946)
* [@slytechnical](https://github.com/slytechnical) made their first contribution in PR [#10881](https://github.com/BerriAI/litellm/pull/10881)
* [@daarko10](https://github.com/daarko10) made their first contribution in PR [#11006](https://github.com/BerriAI/litellm/pull/11006)
* [@sorenmat](https://github.com/sorenmat) made their first contribution in PR [#10658](https://github.com/BerriAI/litellm/pull/10658)
* [@matthid](https://github.com/matthid) made their first contribution in PR [#10982](https://github.com/BerriAI/litellm/pull/10982)
* [@jgowdy-godaddy](https://github.com/jgowdy-godaddy) made their first contribution in PR [#11032](https://github.com/BerriAI/litellm/pull/11032)
* [@bepotp](https://github.com/bepotp) made their first contribution in PR [#11008](https://github.com/BerriAI/litellm/pull/11008)
* [@jmorenoc-o](https://github.com/jmorenoc-o) made their first contribution in PR [#11031](https://github.com/BerriAI/litellm/pull/11031)
* [@martin-liu](https://github.com/martin-liu) made their first contribution in PR [#11076](https://github.com/BerriAI/litellm/pull/11076)
* [@gunjan-solanki](https://github.com/gunjan-solanki) made their first contribution in PR [#11064](https://github.com/BerriAI/litellm/pull/11064)
* [@tokoko](https://github.com/tokoko) made their first contribution in PR [#10980](https://github.com/BerriAI/litellm/pull/10980)
* [@spike-spiegel-21](https://github.com/spike-spiegel-21) made their first contribution in PR [#10649](https://github.com/BerriAI/litellm/pull/10649)
* [@kreatoo](https://github.com/kreatoo) made their first contribution in PR [#10927](https://github.com/BerriAI/litellm/pull/10927)
* [@baejooc](https://github.com/baejooc) made their first contribution in PR [#10887](https://github.com/BerriAI/litellm/pull/10887)
* [@keykbd](https://github.com/keykbd) made their first contribution in PR [#11114](https://github.com/BerriAI/litellm/pull/11114)
* [@dalssoft](https://github.com/dalssoft) made their first contribution in PR [#11088](https://github.com/BerriAI/litellm/pull/11088)
* [@jtong99](https://github.com/jtong99) made their first contribution in PR [#10853](https://github.com/BerriAI/litellm/pull/10853)

## Demo Instance

Here's a Demo Instance to test changes:

- Instance: https://demo.litellm.ai/
- Login Credentials:
    - Username: admin
    - Password: sk-1234

## [Git Diff](https://github.com/BerriAI/litellm/releases)
