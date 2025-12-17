---
title: v1.71.1-stable - 2x Higher Requests Per Second (RPS)
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
docker.litellm.ai/berriai/litellm:main-v1.71.1-stable
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

- **Performance improvements**: LiteLLM can now scale to 200 RPS per instance with a 74ms median response time. 
- **File Permissions**:  Control file access across OpenAI, Azure, VertexAI. 
- **MCP x OpenAI**: Use MCP servers with OpenAI Responses API.



## Performance Improvements

<Image img={require('../../img/perf_imp.png')}  style={{ width: '800px', height: 'auto' }} />

<br/>


This release brings aiohttp support for all LLM api providers. This means that LiteLLM can now scale to 200 RPS per instance with a 40ms median latency overhead. 

This change doubles the RPS LiteLLM can scale to at this latency overhead.

You can opt into this by enabling the flag below. (We expect to make this the default in 1 week.)


### Flag to enable

**On LiteLLM Proxy**

Set the `USE_AIOHTTP_TRANSPORT=True` in the environment variables. 

```yaml showLineNumbers title="Environment Variable"
export USE_AIOHTTP_TRANSPORT="True"
```

**On LiteLLM Python SDK**

Set the `use_aiohttp_transport=True` to enable aiohttp transport. 

```python showLineNumbers title="Python SDK"
import litellm

litellm.use_aiohttp_transport = True # default is False, enable this to use aiohttp transport
result = litellm.completion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello, world!"}],
)
print(result)
```

## File Permissions

<Image img={require('../../img/files_api_graphic.png')}  style={{ width: '800px', height: 'auto' }} />

<br/>

This release brings support for [File Permissions](../../docs/proxy/litellm_managed_files#file-permissions) and [Finetuning APIs](../../docs/proxy/managed_finetuning) to [LiteLLM Managed Files](../../docs/proxy/litellm_managed_files). This is great for: 

- **Proxy Admins**: as users can only view/edit/delete files they’ve created - even when using shared OpenAI/Azure/Vertex deployments.
- **Developers**: get a standard interface to use Files across Chat/Finetuning/Batch APIs.


## New Models / Updated Models

- **Gemini [VertexAI](https://docs.litellm.ai/docs/providers/vertex), [Google AI Studio](https://docs.litellm.ai/docs/providers/gemini)**
    - New gemini models - [PR 1](https://github.com/BerriAI/litellm/pull/10991), [PR 2](https://github.com/BerriAI/litellm/pull/10998)
        - `gemini-2.5-flash-preview-tts`
        - `gemini-2.0-flash-preview-image-generation`
        - `gemini/gemini-2.5-flash-preview-05-20`
        - `gemini-2.5-flash-preview-05-20`
- **[Anthropic](../../docs/providers/anthropic)**
    - Claude-4 model family support - [PR](https://github.com/BerriAI/litellm/pull/11060)
- **[Bedrock](../../docs/providers/bedrock)**
    - Claude-4 model family support - [PR](https://github.com/BerriAI/litellm/pull/11060)
    - Support for `reasoning_effort` and `thinking` parameters for Claude-4 - [PR](https://github.com/BerriAI/litellm/pull/11114)
- **[VertexAI](../../docs/providers/vertex)**
    - Claude-4 model family support - [PR](https://github.com/BerriAI/litellm/pull/11060)
    - Global endpoints support - [PR](https://github.com/BerriAI/litellm/pull/10658)
    - authorized_user credentials type support - [PR](https://github.com/BerriAI/litellm/pull/10899)
- **[xAI](../../docs/providers/xai)**
    - `xai/grok-3` pricing information - [PR](https://github.com/BerriAI/litellm/pull/11028)
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
- **[Ollama](../../docs/providers/ollama)**
    - Wildcard model support - [PR](https://github.com/BerriAI/litellm/pull/10982)
- **[CustomLLM](../../docs/providers/custom_llm_server)**
    - Embeddings support added - [PR](https://github.com/BerriAI/litellm/pull/10980)
- **[Featherless AI](../../docs/providers/featherless_ai)**
    - Access to 4200+ models - [PR](https://github.com/BerriAI/litellm/pull/10596)

## LLM API Endpoints

- **[Image Edits](../../docs/image_generation)**
    - `/v1/images/edits` - Support for /images/edits endpoint - [PR](https://github.com/BerriAI/litellm/pull/11020) [PR](https://github.com/BerriAI/litellm/pull/11123)
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
- **Audit Logs**
    - `/list` and `/info` endpoints for Audit Logs - [PR](https://github.com/BerriAI/litellm/pull/11102)

## Logging / Alerting Integrations

- **[Prometheus](../../docs/proxy/prometheus)**
    - Track `route` on proxy_* metrics - [PR](https://github.com/BerriAI/litellm/pull/10992)
- **[Langfuse](../../docs/proxy/logging#langfuse)**
    - Support for `prompt_label` parameter - [PR](https://github.com/BerriAI/litellm/pull/11018)
    - Consistent modelParams logging - [PR](https://github.com/BerriAI/litellm/pull/11018)
- **[DeepEval/ConfidentAI](../../docs/proxy/logging#deepeval)**
    - Logging enabled for proxy and SDK - [PR](https://github.com/BerriAI/litellm/pull/10649)
- **[Logfire](../../docs/proxy/logging)**
    - Fix otel proxy server initialization when using Logfire - [PR](https://github.com/BerriAI/litellm/pull/11091)

## Authentication & Security

- **[JWT Authentication](../../docs/proxy/token_auth)**
    - Support for applying default internal user parameters when upserting a user via JWT authentication - [PR](https://github.com/BerriAI/litellm/pull/10995)
    - Map a user to a team when upserting a user via JWT authentication - [PR](https://github.com/BerriAI/litellm/pull/11108)
- **Custom Auth**
    - Support for switching between custom auth and API key auth - [PR](https://github.com/BerriAI/litellm/pull/11070)

## Performance / Reliability Improvements

- **aiohttp Transport**
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

- **LLM Provider Fixes**
    - VertexAI: 
        - Fixed quota_project_id parameter issue - [PR](https://github.com/BerriAI/litellm/pull/10915)
        - Fixed credential refresh exceptions - [PR](https://github.com/BerriAI/litellm/pull/10969)
    - Cohere: 
        Fixes for adding Cohere models through LiteLLM UI - [PR](https://github.com/BerriAI/litellm/pull/10822)
    - Anthropic: 
        - Fixed streaming dict object handling for /v1/messages - [PR](https://github.com/BerriAI/litellm/pull/11032)
    - OpenRouter: 
        - Fixed stream usage ID issues - [PR](https://github.com/BerriAI/litellm/pull/11004)

- **Authentication & Users**
    - Fixed invitation email link generation - [PR](https://github.com/BerriAI/litellm/pull/10958) 
    - Fixed JWT authentication default role - [PR](https://github.com/BerriAI/litellm/pull/10995)
    - Fixed user budget reset functionality - [PR](https://github.com/BerriAI/litellm/pull/10993)
    - Fixed SSO user compatibility and email validation - [PR](https://github.com/BerriAI/litellm/pull/11106)

- **Database & Infrastructure**
    - Fixed DB connection parameter handling - [PR](https://github.com/BerriAI/litellm/pull/10842)
    - Fixed email invitation link  - [PR](https://github.com/BerriAI/litellm/pull/11031)

- **UI & Display**
    - Fixed MCP tool rendering when no arguments required - [PR](https://github.com/BerriAI/litellm/pull/11012)
    - Fixed team model alias deletion - [PR](https://github.com/BerriAI/litellm/pull/11121)
    - Fixed team viewer permissions - [PR](https://github.com/BerriAI/litellm/pull/11127)

- **Model & Routing**
    - Fixed team model mapping in route requests - [PR](https://github.com/BerriAI/litellm/pull/11111)
    - Fixed standard optional parameter passing - [PR](https://github.com/BerriAI/litellm/pull/11124)


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
