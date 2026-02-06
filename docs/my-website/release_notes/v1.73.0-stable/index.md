---
title: "v1.73.0-stable - Set default team for new users"
slug: "v1-73-0-stable"
date: 2025-06-21T10:00:00
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

## Known Issues

The `non-root` docker image has a known issue around the UI not loading. If you use the `non-root` docker image we recommend waiting before upgrading to this version. We will post a patch fix for this.

:::

## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
docker.litellm.ai/berriai/litellm:v1.73.0-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.73.0.post1
```

</TabItem>
</Tabs>


## TLDR


* **Why Upgrade**
    - User Management: Set default team for new users - enables giving all users $10 API keys for exploration.
    - Passthrough Endpoints v2: Enhanced support for subroutes and custom cost tracking for passthrough endpoints.
    - Health Check Dashboard: New frontend UI for monitoring model health and status.
* **Who Should Read**
    - Teams using **Passthrough Endpoints**
    - Teams using **User Management** on LiteLLM
    - Teams using **Health Check Dashboard** for models
    - Teams using **Claude Code** with LiteLLM
* **Risk of Upgrade**
    - **Low**
        - No major breaking changes to existing functionality.
- **Major Changes**
    - `User Agent` will be auto-tracked as a tag in LiteLLM UI Logs Page. This means for all LLM requests you will see a `User Agent` tag in the logs page.

---

## Key Highlights



### Set Default Team for New Users

<Image img={require('../../img/default_teams_product_ss.jpg')}/>

<br/>

v1.73.0 introduces the ability to assign new users to Default Teams. This makes it much easier to enable experimentation with LLMs within your company, while also **ensuring spend for exploration is tracked correctly.** 
 
What this means for **Proxy Admins**:
- Set a max budget per team member: This sets a max amount an individual can spend within a team. 
- Set a default team for new users: When a new user signs in via SSO / invitation link, they will be automatically added to this team. 

What this means for **Developers**: 
- View models across teams: You can now go to `Models + Endpoints` and view the models you have access to, across all teams you're a member of. 
- Safe create key modal: If you have no model access outside of a team (default behaviour), you are now nudged to select a team on the Create Key modal. This resolves a common confusion point for new users onboarding to the proxy. 

[Get Started](https://docs.litellm.ai/docs/tutorials/default_team_self_serve)


### Passthrough Endpoints v2

<Image img={require('../../img/release_notes/v2_pt.png')}/>


<br/>

This release brings support for adding billing and full URL forwarding for passthrough endpoints. 

Previously, you could only map simple endpoints, but now you can add just `/bria` and all subroutes automatically get forwarded - for example, `/bria/v1/text-to-image/base/model` and `/bria/v1/enhance_image` will both be forwarded to the target URL with the same path structure.

This means you as Proxy Admin can onboard third-party endpoints like Bria API and Mistral OCR, set a cost per request, and give your developers access to the complete API functionality.

[Learn more about Passthrough Endpoints](../../docs/proxy/pass_through)


### v2 Health Checks 

<Image img={require('../../img/release_notes/v2_health.png')}/>

<br/>

This release brings support for Proxy Admins to select which specific models to health check and see the health status as soon as its individual check completes, along with last check times.

This allows Proxy Admins to immediately identify which specific models are in a bad state and view the full error stack trace for faster troubleshooting.

---


## New / Updated Models

### Pricing / Context Window Updates

| Provider    | Model                                  | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Type |
| ----------- | -------------------------------------- | -------------- | ------------------- | -------------------- | ---- |
| Google VertexAI | `vertex_ai/imagen-4` | N/A | Image Generation | Image Generation | New |
| Google VertexAI | `vertex_ai/imagen-4-preview` | N/A | Image Generation | Image Generation | New |
| Gemini | `gemini-2.5-pro` | 2M | $1.25 | $5.00 | New |
| Gemini | `gemini-2.5-flash-lite` | 1M | $0.075 | $0.30 | New |
| OpenRouter | Various models | Updated | Updated | Updated | Updated |
| Azure | `azure/o3` | 200k | $2.00 | $8.00 | Updated |
| Azure | `azure/o3-pro` | 200k | $2.00 | $8.00 | Updated |
| Azure OpenAI | Azure Codex Models | Various | Various | Various | New |

### Updated Models

#### Features
- **[Azure](../../docs/providers/azure)**
    - Support for new /v1 preview Azure OpenAI API - [PR](https://github.com/BerriAI/litellm/pull/11934), [Get Started](../../docs/providers/azure/azure_responses#azure-codex-models)
    - Add Azure Codex Models support - [PR](https://github.com/BerriAI/litellm/pull/11934), [Get Started](../../docs/providers/azure/azure_responses#azure-codex-models)
    - Make Azure AD scope configurable - [PR](https://github.com/BerriAI/litellm/pull/11621)
    - Handle more GPT custom naming patterns - [PR](https://github.com/BerriAI/litellm/pull/11914)
    - Update o3 pricing to match OpenAI pricing - [PR](https://github.com/BerriAI/litellm/pull/11937)
- **[VertexAI](../../docs/providers/vertex)**
    - Add Vertex Imagen-4 models - [PR](https://github.com/BerriAI/litellm/pull/11767), [Get Started](../../docs/providers/vertex_image)
    - Anthropic streaming passthrough cost tracking - [PR](https://github.com/BerriAI/litellm/pull/11734)
- **[Gemini](../../docs/providers/gemini)**
    - Working Gemini TTS support via `/v1/speech` endpoint - [PR](https://github.com/BerriAI/litellm/pull/11832)
    - Fix gemini 2.5 flash config - [PR](https://github.com/BerriAI/litellm/pull/11830)
    - Add missing `flash-2.5-flash-lite` model and fix pricing - [PR](https://github.com/BerriAI/litellm/pull/11901)
    - Mark all gemini-2.5 models as supporting PDF input - [PR](https://github.com/BerriAI/litellm/pull/11907)
    - Add `gemini-2.5-pro` with reasoning support - [PR](https://github.com/BerriAI/litellm/pull/11927)
- **[AWS Bedrock](../../docs/providers/bedrock)**
    - AWS credentials no longer mandatory - [PR](https://github.com/BerriAI/litellm/pull/11765)
    - Add AWS Bedrock profiles for APAC region - [PR](https://github.com/BerriAI/litellm/pull/11883)
    - Fix AWS Bedrock Claude tool call index - [PR](https://github.com/BerriAI/litellm/pull/11842)
    - Handle base64 file data with `qs:..` prefix - [PR](https://github.com/BerriAI/litellm/pull/11908)
    - Add Mistral Small to BEDROCK_CONVERSE_MODELS - [PR](https://github.com/BerriAI/litellm/pull/11760)
- **[Mistral](../../docs/providers/mistral)**
    - Enhance Mistral API with parallel tool calls support - [PR](https://github.com/BerriAI/litellm/pull/11770)
- **[Meta Llama API](../../docs/providers/meta_llama)**
    - Enable tool calling for meta_llama models - [PR](https://github.com/BerriAI/litellm/pull/11895)
- **[Volcengine](../../docs/providers/volcengine)**
    - Add thinking parameter support - [PR](https://github.com/BerriAI/litellm/pull/11914)


#### Bugs

- **[VertexAI](../../docs/providers/vertex)**
    - Handle missing tokenCount in promptTokensDetails - [PR](https://github.com/BerriAI/litellm/pull/11896)
    - Fix vertex AI claude thinking params - [PR](https://github.com/BerriAI/litellm/pull/11796)
- **[Gemini](../../docs/providers/gemini)**
    - Fix web search error with responses API - [PR](https://github.com/BerriAI/litellm/pull/11894), [Get Started](../../docs/completion/web_search#responses-litellmresponses)
- **[Custom LLM](../../docs/providers/custom_llm_server)**
    - Set anthropic custom LLM provider property - [PR](https://github.com/BerriAI/litellm/pull/11907)
- **[Anthropic](../../docs/providers/anthropic)**
    - Bump anthropic package version - [PR](https://github.com/BerriAI/litellm/pull/11851)
- **[Ollama](../../docs/providers/ollama)**
    - Update ollama_embeddings to work on sync API - [PR](https://github.com/BerriAI/litellm/pull/11746)
    - Fix response_format not working - [PR](https://github.com/BerriAI/litellm/pull/11880)

---

## LLM API Endpoints

#### Features
- **[Responses API](../../docs/response_api)**
    - Day-0 support for OpenAI re-usable prompts Responses API - [PR](https://github.com/BerriAI/litellm/pull/11782), [Get Started](../../docs/providers/openai/responses_api#reusable-prompts)
    - Support passing image URLs in Completion-to-Responses bridge - [PR](https://github.com/BerriAI/litellm/pull/11833)
- **[MCP Gateway](../../docs/mcp)**
    - Add Allowed MCPs to Creating/Editing Organizations - [PR](https://github.com/BerriAI/litellm/pull/11893), [Get Started](../../docs/mcp#-mcp-permission-management)
    - Allow connecting to MCP with authentication headers - [PR](https://github.com/BerriAI/litellm/pull/11891), [Get Started](../../docs/mcp#using-your-mcp-with-client-side-credentials)
- **[Speech API](../../docs/speech)**
    - Working Gemini TTS support via OpenAI's `/v1/speech` endpoint - [PR](https://github.com/BerriAI/litellm/pull/11832)
- **[Passthrough Endpoints](../../docs/proxy/pass_through)**
    - Add support for subroutes for passthrough endpoints - [PR](https://github.com/BerriAI/litellm/pull/11827)
    - Support for setting custom cost per passthrough request - [PR](https://github.com/BerriAI/litellm/pull/11870)
    - Ensure "Request" is tracked for passthrough requests on LiteLLM Proxy - [PR](https://github.com/BerriAI/litellm/pull/11873)
    - Add V2 Passthrough endpoints on UI - [PR](https://github.com/BerriAI/litellm/pull/11905)
    - Move passthrough endpoints under Models + Endpoints in UI - [PR](https://github.com/BerriAI/litellm/pull/11871)
    - QA improvements for adding passthrough endpoints - [PR](https://github.com/BerriAI/litellm/pull/11909), [PR](https://github.com/BerriAI/litellm/pull/11939)
- **[Models API](../../docs/completion/model_alias)**
    - Allow `/models` to return correct models for custom wildcard prefixes - [PR](https://github.com/BerriAI/litellm/pull/11784)

#### Bugs

- **[Messages API](../../docs/anthropic_unified)**
    - Fix `/v1/messages` endpoint always using us-central1 with vertex_ai-anthropic models - [PR](https://github.com/BerriAI/litellm/pull/11831)
    - Fix model_group tracking for `/v1/messages` and `/moderations` - [PR](https://github.com/BerriAI/litellm/pull/11933)
    - Fix cost tracking and logging via `/v1/messages` API when using Claude Code - [PR](https://github.com/BerriAI/litellm/pull/11928)
- **[MCP Gateway](../../docs/mcp)**
    - Fix using MCPs defined on config.yaml - [PR](https://github.com/BerriAI/litellm/pull/11824)
- **[Chat Completion API](../../docs/completion/input)**
    - Allow dict for tool_choice argument in acompletion - [PR](https://github.com/BerriAI/litellm/pull/11860)
- **[Passthrough Endpoints](../../docs/pass_through/langfuse)**
    - Don't log request to Langfuse passthrough on Langfuse - [PR](https://github.com/BerriAI/litellm/pull/11768)

---

## Spend Tracking

#### Features
- **[User Agent Tracking](../../docs/proxy/cost_tracking)**
    - Automatically track spend by user agent (allows cost tracking for Claude Code) - [PR](https://github.com/BerriAI/litellm/pull/11781)
    - Add user agent tags in spend logs payload - [PR](https://github.com/BerriAI/litellm/pull/11872)
- **[Tag Management](../../docs/proxy/cost_tracking)**
    - Support adding public model names in tag management - [PR](https://github.com/BerriAI/litellm/pull/11908)

---

## Management Endpoints / UI

#### Features
- **Test Key Page**
    - Allow testing `/v1/messages` on the Test Key Page - [PR](https://github.com/BerriAI/litellm/pull/11930)
- **[SSO](../../docs/proxy/sso)**
    - Allow passing additional headers - [PR](https://github.com/BerriAI/litellm/pull/11781)
- **[JWT Auth](../../docs/proxy/jwt_auth)**
    - Correctly return user email - [PR](https://github.com/BerriAI/litellm/pull/11783)
- **[Model Management](../../docs/proxy/model_management)**
    - Allow editing model access group for existing model - [PR](https://github.com/BerriAI/litellm/pull/11783)
- **[Team Management](../../docs/proxy/team_management)**
    - Allow setting default team for new users - [PR](https://github.com/BerriAI/litellm/pull/11874), [PR](https://github.com/BerriAI/litellm/pull/11877)
    - Fix default team settings - [PR](https://github.com/BerriAI/litellm/pull/11887)
- **[SCIM](../../docs/proxy/scim)**
    - Add error handling for existing user on SCIM - [PR](https://github.com/BerriAI/litellm/pull/11862)
    - Add SCIM PATCH and PUT operations for users - [PR](https://github.com/BerriAI/litellm/pull/11863)
- **Health Check Dashboard**
    - Implement health check backend API and storage functionality - [PR](https://github.com/BerriAI/litellm/pull/11852)
    - Add LiteLLM_HealthCheckTable to database schema - [PR](https://github.com/BerriAI/litellm/pull/11677)
    - Implement health check frontend UI components and dashboard integration - [PR](https://github.com/BerriAI/litellm/pull/11679)
    - Add success modal for health check responses - [PR](https://github.com/BerriAI/litellm/pull/11899)
    - Fix clickable model ID in health check table - [PR](https://github.com/BerriAI/litellm/pull/11898)
    - Fix health check UI table design - [PR](https://github.com/BerriAI/litellm/pull/11897)

---

## Logging / Guardrails Integrations

#### Bugs
- **[Prometheus](../../docs/observability/prometheus)**
    - Fix bug for using prometheus metrics config - [PR](https://github.com/BerriAI/litellm/pull/11779)

---

## Security & Reliability

#### Security Fixes
- **[Documentation Security](../../docs)**
    - Security fixes for docs - [PR](https://github.com/BerriAI/litellm/pull/11776)
    - Add Trivy Security Scan for UI + Docs folder - remove all vulnerabilities - [PR](https://github.com/BerriAI/litellm/pull/11778)

#### Reliability Improvements
- **[Dependencies](../../docs)**
    - Fix aiohttp version requirement - [PR](https://github.com/BerriAI/litellm/pull/11777)
    - Bump next from 14.2.26 to 14.2.30 in UI dashboard - [PR](https://github.com/BerriAI/litellm/pull/11720)
- **[Networking](../../docs)**
    - Allow using CA Bundles - [PR](https://github.com/BerriAI/litellm/pull/11906)
    - Add workload identity federation between GCP and AWS - [PR](https://github.com/BerriAI/litellm/pull/10210)

---

## General Proxy Improvements

#### Features
- **[Deployment](../../docs/proxy/deploy)**
    - Add deployment annotations for Kubernetes - [PR](https://github.com/BerriAI/litellm/pull/11849)
    - Add ciphers in command and pass to hypercorn for proxy - [PR](https://github.com/BerriAI/litellm/pull/11916)
- **[Custom Root Path](../../docs/proxy/deploy)**
    - Fix loading UI on custom root path - [PR](https://github.com/BerriAI/litellm/pull/11912)
- **[SDK Improvements](../../docs/proxy/reliability)**
    - LiteLLM SDK / Proxy improvement (don't transform message client-side) - [PR](https://github.com/BerriAI/litellm/pull/11908)

#### Bugs
- **[Observability](../../docs/observability)**
    - Fix boto3 tracer wrapping for observability - [PR](https://github.com/BerriAI/litellm/pull/11869)


---

## New Contributors
* @kjoth made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11621)
* @shagunb-acn made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11760)
* @MadsRC made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11765)
* @Abiji-2020 made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11746)
* @salzubi401 made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11803)
* @orolega made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11826)
* @X4tar made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11796)
* @karen-veigas made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11858)
* @Shankyg made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11859)
* @pascallim made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/10210)
* @lgruen-vcgs made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11883)
* @rinormaloku made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11851)
* @InvisibleMan1306 made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11849)
* @ervwalter made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11937)
* @ThakeeNathees made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11880)
* @jnhyperion made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11842)
* @Jannchie made their first contribution in [PR](https://github.com/BerriAI/litellm/pull/11860)

---

## Demo Instance

Here's a Demo Instance to test changes:

- Instance: https://demo.litellm.ai/
- Login Credentials:
    - Username: admin
    - Password: sk-1234

## [Git Diff](https://github.com/BerriAI/litellm/compare/v1.72.6-stable...v1.73.0.rc)
