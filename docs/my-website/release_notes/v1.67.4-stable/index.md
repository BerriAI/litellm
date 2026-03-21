---
title: v1.67.4-stable - Improved User Management
slug: v1.67.4-stable
date: 2025-04-26T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg

tags: ["responses_api", "ui_improvements", "security", "session_management"]
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
docker.litellm.ai/berriai/litellm:main-v1.67.4-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.67.4.post1
```
</TabItem>
</Tabs>

## Key Highlights

- **Improved User Management**: This release enables search and filtering across users, keys, teams, and models.
- **Responses API Load Balancing**: Route requests across provider regions and ensure session continuity. 
- **UI Session Logs**: Group several requests to LiteLLM into a session. 

## Improved User Management

<Image img={require('../../img/release_notes/ui_search_users.png')}/>
<br/>

This release makes it easier to manage users and keys on LiteLLM. You can now search and filter across users, keys, teams, and models, and control user settings more easily.

New features include:

- Search for users by email, ID, role, or team.
- See all of a user's models, teams, and keys in one place.
- Change user roles and model access right from the Users Tab.

These changes help you spend less time on user setup and management on LiteLLM.

## Responses API Load Balancing

<Image img={require('../../img/release_notes/ui_responses_lb.png')}/>
<br/>

This release introduces load balancing for the Responses API, allowing you to route requests across provider regions and ensure session continuity. It works as follows:

- If a `previous_response_id` is provided, LiteLLM will route the request to the original deployment that generated the prior response — ensuring session continuity.
- If no `previous_response_id` is provided, LiteLLM will load-balance requests across your available deployments.

[Read more](https://docs.litellm.ai/docs/response_api#load-balancing-with-session-continuity)

## UI Session Logs

<Image img={require('../../img/ui_session_logs.png')}/>
<br/>

This release allow you to group requests to LiteLLM proxy into a session. If you specify a litellm_session_id in your request LiteLLM will automatically group all logs in the same session. This allows you to easily track usage and request content per session. 

[Read more](https://docs.litellm.ai/docs/proxy/ui_logs_sessions)

## New Models / Updated Models

- **OpenAI**
    1. Added `gpt-image-1` cost tracking [Get Started](https://docs.litellm.ai/docs/image_generation)
    2. Bug fix: added cost tracking for gpt-image-1 when quality is unspecified [PR](https://github.com/BerriAI/litellm/pull/10247)
- **Azure**
    1. Fixed timestamp granularities passing to whisper in Azure [Get Started](https://docs.litellm.ai/docs/audio_transcription)
    2. Added azure/gpt-image-1 pricing [Get Started](https://docs.litellm.ai/docs/image_generation), [PR](https://github.com/BerriAI/litellm/pull/10327)
    3. Added cost tracking for `azure/computer-use-preview`, `azure/gpt-4o-audio-preview-2024-12-17`, `azure/gpt-4o-mini-audio-preview-2024-12-17` [PR](https://github.com/BerriAI/litellm/pull/10178)
- **Bedrock**
    1. Added support for all compatible Bedrock parameters when model="arn:.." (Bedrock application inference profile models) [Get started](https://docs.litellm.ai/docs/providers/bedrock#bedrock-application-inference-profile), [PR](https://github.com/BerriAI/litellm/pull/10256)
    2. Fixed wrong system prompt transformation [PR](https://github.com/BerriAI/litellm/pull/10120)
- **VertexAI / Google AI Studio**
    1. Allow setting `budget_tokens=0` for `gemini-2.5-flash` [Get Started](https://docs.litellm.ai/docs/providers/gemini#usage---thinking--reasoning_content),[PR](https://github.com/BerriAI/litellm/pull/10198)
    2. Ensure returned `usage` includes thinking token usage [PR](https://github.com/BerriAI/litellm/pull/10198)
    3. Added cost tracking for `gemini-2.5-pro-preview-03-25` [PR](https://github.com/BerriAI/litellm/pull/10178)
- **Cohere**
    1. Added support for cohere command-a-03-2025 [Get Started](https://docs.litellm.ai/docs/providers/cohere), [PR](https://github.com/BerriAI/litellm/pull/10295)
- **SageMaker**
    1. Added support for max_completion_tokens parameter [Get Started](https://docs.litellm.ai/docs/providers/sagemaker), [PR](https://github.com/BerriAI/litellm/pull/10300)
- **Responses API**
    1. Added support for GET and DELETE operations - `/v1/responses/{response_id}` [Get Started](../../docs/response_api)
    2. Added session management support for all supported models [PR](https://github.com/BerriAI/litellm/pull/10321)
    3. Added routing affinity to maintain model consistency within sessions [Get Started](https://docs.litellm.ai/docs/response_api#load-balancing-with-routing-affinity), [PR](https://github.com/BerriAI/litellm/pull/10193)


## Spend Tracking Improvements

- **Bug Fix**: Fixed spend tracking bug, ensuring default litellm params aren't modified in memory [PR](https://github.com/BerriAI/litellm/pull/10167)
- **Deprecation Dates**: Added deprecation dates for Azure, VertexAI models [PR](https://github.com/BerriAI/litellm/pull/10308)

## Management Endpoints / UI

#### Users
- **Filtering and Searching**: 
  - Filter users by user_id, role, team, sso_id 
  - Search users by email

  <br/>

  <Image img={require('../../img/release_notes/user_filters.png')}/>

- **User Info Panel**: Added a new user information pane [PR](https://github.com/BerriAI/litellm/pull/10213)
  - View teams, keys, models associated with User 
  - Edit user role, model permissions 



#### Teams
- **Filtering and Searching**: 
    - Filter teams by Organization, Team ID [PR](https://github.com/BerriAI/litellm/pull/10324)
    - Search teams by Team Name [PR](https://github.com/BerriAI/litellm/pull/10324)

  <br/>

  <Image img={require('../../img/release_notes/team_filters.png')}/>



#### Keys
- **Key Management**: 
  - Support for cross-filtering and filtering by key hash [PR](https://github.com/BerriAI/litellm/pull/10322)
  - Fixed key alias reset when resetting filters [PR](https://github.com/BerriAI/litellm/pull/10099)
  - Fixed table rendering on key creation [PR](https://github.com/BerriAI/litellm/pull/10224)

#### UI Logs Page

- **Session Logs**: Added UI Session Logs [Get Started](https://docs.litellm.ai/docs/proxy/ui_logs_sessions)


#### UI Authentication & Security
- **Required Authentication**: Authentication now required for all dashboard pages [PR](https://github.com/BerriAI/litellm/pull/10229)
- **SSO Fixes**: Fixed SSO user login invalid token error [PR](https://github.com/BerriAI/litellm/pull/10298)
- [BETA] **Encrypted Tokens**: Moved UI to encrypted token usage [PR](https://github.com/BerriAI/litellm/pull/10302)
- **Token Expiry**: Support token refresh by re-routing to login page (fixes issue where expired token would show a blank page) [PR](https://github.com/BerriAI/litellm/pull/10250)

#### UI General fixes
- **Fixed UI Flicker**: Addressed UI flickering issues in Dashboard [PR](https://github.com/BerriAI/litellm/pull/10261)
- **Improved Terminology**: Better loading and no-data states on Keys and Tools pages [PR](https://github.com/BerriAI/litellm/pull/10253)
- **Azure Model Support**: Fixed editing Azure public model names and changing model names after creation [PR](https://github.com/BerriAI/litellm/pull/10249)
- **Team Model Selector**: Bug fix for team model selection [PR](https://github.com/BerriAI/litellm/pull/10171)


## Logging / Guardrail Integrations

- **Datadog**:
    1. Fixed Datadog LLM observability logging [Get Started](https://docs.litellm.ai/docs/proxy/logging#datadog), [PR](https://github.com/BerriAI/litellm/pull/10206)
- **Prometheus / Grafana**: 
    1. Enable datasource selection on LiteLLM Grafana Template [Get Started](https://docs.litellm.ai/docs/proxy/prometheus#-litellm-maintained-grafana-dashboards-), [PR](https://github.com/BerriAI/litellm/pull/10257)
- **AgentOps**: 
    1. Added AgentOps Integration [Get Started](https://docs.litellm.ai/docs/observability/agentops_integration), [PR](https://github.com/BerriAI/litellm/pull/9685)
- **Arize**: 
    1. Added missing attributes for Arize & Phoenix Integration [Get Started](https://docs.litellm.ai/docs/observability/arize_integration), [PR](https://github.com/BerriAI/litellm/pull/10215)


## General Proxy Improvements

- **Caching**: Fixed caching to account for `thinking` or `reasoning_effort` when calculating cache key [PR](https://github.com/BerriAI/litellm/pull/10140)
- **Model Groups**: Fixed handling for cases where user sets model_group inside model_info [PR](https://github.com/BerriAI/litellm/pull/10191)
- **Passthrough Endpoints**: Ensured `PassthroughStandardLoggingPayload` is logged with method, URL, request/response body [PR](https://github.com/BerriAI/litellm/pull/10194)
- **Fix SQL Injection**: Fixed potential SQL injection vulnerability in spend_management_endpoints.py [PR](https://github.com/BerriAI/litellm/pull/9878)



## Helm

- Fixed serviceAccountName on migration job [PR](https://github.com/BerriAI/litellm/pull/10258)

## Full Changelog

The complete list of changes can be found in the [GitHub release notes](https://github.com/BerriAI/litellm/compare/v1.67.0-stable...v1.67.4-stable).