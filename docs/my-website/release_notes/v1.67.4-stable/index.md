---
title: v1.67.4-stable
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

## Key Highlights


Let's dive in.

## Responses API


## New Models / Updated Models

- **OpenAI**
    1. Added `gpt-image-1` cost tracking [Get Started](https://docs.litellm.ai/docs/image_generation)
    2. Bug fix: added cost tracking for gpt-image-1 when quality is unspecified [PR](https://github.com/BerriAI/litellm/pull/10247)
- **Azure**
    1. Fixed timestamp granularities passing to whisper in Azure [Get Started](https://docs.litellm.ai/docs/audio_transcription)
    2. Added azure/gpt-image-1 pricing [Get Started](https://docs.litellm.ai/docs/image_generation), [PR](https://github.com/BerriAI/litellm/pull/10327)
- **Bedrock**
    1. Added support for all compatible Bedrock parameters when model="arn:.." (Bedrock application inference profile models) [Get started](https://docs.litellm.ai/docs/providers/bedrock#bedrock-application-inference-profile), [PR](https://github.com/BerriAI/litellm/pull/10256)
    2. Fixed wrong system prompt transformation [PR](https://github.com/BerriAI/litellm/pull/10120)
- **Gemini**
    1. Fixed passing back Gemini thinking content to API [PR](https://github.com/BerriAI/litellm/pull/10173)
    2. Various Gemini-2.5-flash improvements [PR](https://github.com/BerriAI/litellm/pull/10198)
- **Cohere**
    1. Added support for cohere command-a-03-2025 [PR](https://github.com/BerriAI/litellm/pull/10295)
- **SageMaker**
    1. Added support for max_completion_tokens parameter [PR](https://github.com/BerriAI/litellm/pull/10300)


## Spend Tracking Improvements

- **Bug Fix**: Fixed spend tracking bug, ensuring default litellm params aren't modified in memory [PR](https://github.com/BerriAI/litellm/pull/10167)
- **Model Pricing**: Model pricing updates for Azure & VertexAI [PR](https://github.com/BerriAI/litellm/pull/10178)
- **Updated Deprecation Dates**: Updated deprecation dates and prices [PR](https://github.com/BerriAI/litellm/pull/10308)

## Management Endpoints / UI

### User Management
- **User Info Panel**: Added a new user information pane [PR](https://github.com/BerriAI/litellm/pull/10213)
- **Global Sorting/Filtering**: 
  - Added global filtering to Users tab [PR](https://github.com/BerriAI/litellm/pull/10195)
  - Enabled global sorting to find users with highest spend [PR](https://github.com/BerriAI/litellm/pull/10211)
  - Support for filtering by user ID [PR](https://github.com/BerriAI/litellm/pull/10322)

### Teams & Keys Management
- **Team Filtering**: 
  - Added team-based filtering to the models page [PR](https://github.com/BerriAI/litellm/pull/10325)
  - Support for filtering by team ID and team name [PR](https://github.com/BerriAI/litellm/pull/10324)
- **Key Management**: 
  - Support for cross-filtering and filtering by key hash [PR](https://github.com/BerriAI/litellm/pull/10322)
  - Fixed key alias reset when resetting filters [PR](https://github.com/BerriAI/litellm/pull/10099)
  - Fixed table rendering on key creation [PR](https://github.com/BerriAI/litellm/pull/10224)

### Authentication & Security
- **Required Authentication**: Authentication now required for all dashboard pages [PR](https://github.com/BerriAI/litellm/pull/10229)
- **SSO Fixes**: Fixed SSO user login invalid token error [PR](https://github.com/BerriAI/litellm/pull/10298)
- **Encrypted Tokens**: Moved UI to encrypted token usage [PR](https://github.com/BerriAI/litellm/pull/10302)
- **Token Expiry**: Added token expiry logic to user dashboard [PR](https://github.com/BerriAI/litellm/pull/10250)

### UI Refinements
- **Fixed UI Flicker**: Addressed UI flickering issues in Dashboard [PR](https://github.com/BerriAI/litellm/pull/10261)
- **Improved Terminology**: Better loading and no-data states on Keys and Tools pages [PR](https://github.com/BerriAI/litellm/pull/10253)
- **Team Model Selector**: Bug fix for team model selection [PR](https://github.com/BerriAI/litellm/pull/10171)
- **Azure Model Support**: Fixed editing Azure public model names and changing model names after creation [PR](https://github.com/BerriAI/litellm/pull/10249)


## Logging / Guardrail Integrations

- **Passthrough Endpoints**: Ensured `PassthroughStandardLoggingPayload` is logged with method, URL, request/response body [PR](https://github.com/BerriAI/litellm/pull/10194)
- **Datadog**: Fixed Datadog LLM observability logging [PR](https://github.com/BerriAI/litellm/pull/10206)
- **Grafana**: Enabled datasource selection via templating in Grafana dashboard [PR](https://github.com/BerriAI/litellm/pull/10257)
- **New Integrations**:
    1. Added AgentOps Integration [PR](https://github.com/BerriAI/litellm/pull/9685)
    2. Added missing attributes for Arize & Phoenix Integration [PR](https://github.com/BerriAI/litellm/pull/10215)
- **Session Logs**: Added UI Session Logs documentation [PR](https://github.com/BerriAI/litellm/pull/10334)

## General Proxy Improvements

- **Caching**: Fixed caching to account for thinking or reasoning_effort config [PR](https://github.com/BerriAI/litellm/pull/10140)
- **Model Groups**: Fixed handling for cases where user sets model_group inside model_info [PR](https://github.com/BerriAI/litellm/pull/10191)


## Security Fixes

- **SQL Injection**: Fixed potential SQL injection vulnerability in spend_management_endpoints.py [PR](https://github.com/BerriAI/litellm/pull/9878)
- **Authentication**: Fixed multiple authentication and token security issues [PR](https://github.com/BerriAI/litellm/pull/10302, https://github.com/BerriAI/litellm/pull/10326)
- **Auth Check**: Fixed typing to ensure cases where model is None are handled properly [PR](https://github.com/BerriAI/litellm/pull/10170)

## Helm

- Fixed serviceAccountName on migration job [PR](https://github.com/BerriAI/litellm/pull/10258)

## Full Changelog

The complete list of changes can be found in the [GitHub release notes](https://github.com/BerriAI/litellm/compare/v1.67.0-stable...v1.67.4-stable).