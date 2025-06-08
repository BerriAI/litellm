---
title: v1.67.0-stable - SCIM Integration
slug: v1.67.0-stable
date: 2025-04-19T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg

tags: ["sso", "unified_file_id", "cost_tracking", "security"]
hide_table_of_contents: false
---
import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Key Highlights

- **SCIM Integration**: Enables identity providers (Okta, Azure AD, OneLogin, etc.) to automate user and team (group) provisioning, updates, and deprovisioning
- **Team and Tag based usage tracking**: You can now see usage and spend by team and tag at 1M+ spend logs.
- **Unified Responses API**: Support for calling Anthropic, Gemini, Groq, etc. via OpenAI's new Responses API.

Let's dive in.

## SCIM Integration

<Image img={require('../../img/scim_integration.png')}/>

This release adds SCIM support to LiteLLM. This allows your SSO provider (Okta, Azure AD, etc) to automatically create/delete users, teams, and memberships on LiteLLM. This means that when you remove a team on your SSO provider, your SSO provider will automatically delete the corresponding team on LiteLLM. 

[Read more](../../docs/tutorials/scim_litellm)
## Team and Tag based usage tracking

<Image img={require('../../img/release_notes/new_team_usage_highlight.jpg')}/>


This release improves team and tag based usage tracking at 1m+ spend logs, making it easy to monitor your LLM API Spend in production. This covers:

- View **daily spend** by teams + tags
- View **usage / spend by key**, within teams
- View **spend by multiple tags**
- Allow **internal users** to view spend of teams they're a member of

[Read more](#management-endpoints--ui)

## Unified Responses API

This release allows you to call Azure OpenAI, Anthropic, AWS Bedrock, and Google Vertex AI models via the POST /v1/responses endpoint on LiteLLM. This means you can now use popular tools like [OpenAI Codex](https://docs.litellm.ai/docs/tutorials/openai_codex) with your own models. 

<Image img={require('../../img/release_notes/unified_responses_api_rn.png')}/>


[Read more](https://docs.litellm.ai/docs/response_api)


## New Models / Updated Models

- **OpenAI**
    1. gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, o3, o3-mini, o4-mini pricing - [Get Started](../../docs/providers/openai#usage), [PR](https://github.com/BerriAI/litellm/pull/9990)
    2. o4 - correctly map o4 to openai o_series model
- **Azure AI**
    1. Phi-4 output cost per token fix - [PR](https://github.com/BerriAI/litellm/pull/9880)
    2. Responses API support [Get Started](../../docs/providers/azure#azure-responses-api),[PR](https://github.com/BerriAI/litellm/pull/10116)
- **Anthropic**
    1. redacted message thinking support - [Get Started](../../docs/providers/anthropic#usage---thinking--reasoning_content),[PR](https://github.com/BerriAI/litellm/pull/10129)
- **Cohere**
    1. `/v2/chat` Passthrough endpoint support w/ cost tracking - [Get Started](../../docs/pass_through/cohere), [PR](https://github.com/BerriAI/litellm/pull/9997)
- **Azure**
    1. Support azure tenant_id/client_id env vars - [Get Started](../../docs/providers/azure#entra-id---use-tenant_id-client_id-client_secret), [PR](https://github.com/BerriAI/litellm/pull/9993)
    2. Fix response_format check for 2025+ api versions - [PR](https://github.com/BerriAI/litellm/pull/9993)
    3. Add gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, o3, o3-mini, o4-mini pricing
- **VLLM**
    1. Files - Support 'file' message type for VLLM video url's - [Get Started](../../docs/providers/vllm#send-video-url-to-vllm), [PR](https://github.com/BerriAI/litellm/pull/10129)
    2. Passthrough - new `/vllm/` passthrough endpoint support [Get Started](../../docs/pass_through/vllm), [PR](https://github.com/BerriAI/litellm/pull/10002)
- **Mistral**
    1. new `/mistral` passthrough endpoint support [Get Started](../../docs/pass_through/mistral), [PR](https://github.com/BerriAI/litellm/pull/10002)
- **AWS**
    1. New mapped bedrock regions - [PR](https://github.com/BerriAI/litellm/pull/9430)
- **VertexAI / Google AI Studio**
    1. Gemini - Response format - Retain schema field ordering for google gemini and vertex by specifying propertyOrdering - [Get Started](../../docs/providers/vertex#json-schema), [PR](https://github.com/BerriAI/litellm/pull/9828)
    2. Gemini-2.5-flash - return reasoning content [Google AI Studio](../../docs/providers/gemini#usage---thinking--reasoning_content), [Vertex AI](../../docs/providers/vertex#thinking--reasoning_content)
    3. Gemini-2.5-flash - pricing + model information [PR](https://github.com/BerriAI/litellm/pull/10125)
    4. Passthrough - new `/vertex_ai/discovery` route - enables calling AgentBuilder API routes [Get Started](../../docs/pass_through/vertex_ai#supported-api-endpoints), [PR](https://github.com/BerriAI/litellm/pull/10084)
- **Fireworks AI**
    1. return tool calling responses in `tool_calls` field (fireworks incorrectly returns this as a json str in content) [PR](https://github.com/BerriAI/litellm/pull/10130)
- **Triton**
    1. Remove fixed remove bad_words / stop words from `/generate` call - [Get Started](../../docs/providers/triton-inference-server#triton-generate---chat-completion), [PR](https://github.com/BerriAI/litellm/pull/10163)
- **Other**
    1. Support for all litellm providers on Responses API (works with Codex) - [Get Started](../../docs/tutorials/openai_codex), [PR](https://github.com/BerriAI/litellm/pull/10132)
    2. Fix combining multiple tool calls in streaming response - [Get Started](../../docs/completion/stream#helper-function), [PR](https://github.com/BerriAI/litellm/pull/10040)


## Spend Tracking Improvements

- **Cost Control** - inject cache control points in prompt for cost reduction [Get Started](../../docs/tutorials/prompt_caching), [PR](https://github.com/BerriAI/litellm/pull/10000)
- **Spend Tags** - spend tags in headers - support x-litellm-tags even if tag based routing not enabled [Get Started](../../docs/proxy/request_headers#litellm-headers), [PR](https://github.com/BerriAI/litellm/pull/10000)
- **Gemini-2.5-flash** - support cost calculation for reasoning tokens [PR](https://github.com/BerriAI/litellm/pull/10141)

## Management Endpoints / UI
- **Users**
    1. Show created_at and updated_at on users page - [PR](https://github.com/BerriAI/litellm/pull/10033)
- **Virtual Keys**
    1. Filter by key alias - https://github.com/BerriAI/litellm/pull/10085
- **Usage Tab**

    1. Team based usage
        
        - New `LiteLLM_DailyTeamSpend` Table for aggregate team based usage logging - [PR](https://github.com/BerriAI/litellm/pull/10039)
        
        - New Team based usage dashboard + new `/team/daily/activity` API - [PR](https://github.com/BerriAI/litellm/pull/10081)
        - Return team alias on /team/daily/activity API - [PR](https://github.com/BerriAI/litellm/pull/10157)
        - allow internal user view spend for teams they belong to - [PR](https://github.com/BerriAI/litellm/pull/10157)
        - allow viewing top keys by team - [PR](https://github.com/BerriAI/litellm/pull/10157)

        <Image img={require('../../img/release_notes/new_team_usage.png')}/>

    2. Tag Based Usage
        - New `LiteLLM_DailyTagSpend` Table for aggregate tag based usage logging - [PR](https://github.com/BerriAI/litellm/pull/10071)
        - Restrict to only Proxy Admins - [PR](https://github.com/BerriAI/litellm/pull/10157)
        - allow viewing top keys by tag
        - Return tags passed in request (i.e. dynamic tags) on `/tag/list` API - [PR](https://github.com/BerriAI/litellm/pull/10157)
        <Image img={require('../../img/release_notes/new_tag_usage.png')}/>
    3. Track prompt caching metrics in daily user, team, tag tables - [PR](https://github.com/BerriAI/litellm/pull/10029)
    4. Show usage by key (on all up, team, and tag usage dashboards) - [PR](https://github.com/BerriAI/litellm/pull/10157)
    5. swap old usage with new usage tab
- **Models**
    1. Make columns resizable/hideable - [PR](https://github.com/BerriAI/litellm/pull/10119)
- **API Playground**
    1. Allow internal user to call api playground - [PR](https://github.com/BerriAI/litellm/pull/10157)
- **SCIM**
    1. Add LiteLLM SCIM Integration for Team and User management - [Get Started](../../docs/tutorials/scim_litellm), [PR](https://github.com/BerriAI/litellm/pull/10072)


## Logging / Guardrail Integrations
- **GCS**
    1. Fix gcs pub sub logging with env var GCS_PROJECT_ID - [Get Started](../../docs/observability/gcs_bucket_integration#usage), [PR](https://github.com/BerriAI/litellm/pull/10042)
- **AIM**
    1. Add litellm call id passing to Aim guardrails on pre and post-hooks calls - [Get Started](../../docs/proxy/guardrails/aim_security), [PR](https://github.com/BerriAI/litellm/pull/10021)
- **Azure blob storage**
    1. Ensure logging works in high throughput scenarios - [Get Started](../../docs/proxy/logging#azure-blob-storage), [PR](https://github.com/BerriAI/litellm/pull/9962)

## General Proxy Improvements

- **Support setting `litellm.modify_params` via env var** [PR](https://github.com/BerriAI/litellm/pull/9964)
- **Model Discovery** - Check provider’s `/models` endpoints when calling proxy’s `/v1/models` endpoint - [Get Started](../../docs/proxy/model_discovery), [PR](https://github.com/BerriAI/litellm/pull/9958)
- **`/utils/token_counter`** - fix retrieving custom tokenizer for db models - [Get Started](../../docs/proxy/configs#set-custom-tokenizer), [PR](https://github.com/BerriAI/litellm/pull/10047)
- **Prisma migrate** - handle existing columns in db table - [PR](https://github.com/BerriAI/litellm/pull/10138)

